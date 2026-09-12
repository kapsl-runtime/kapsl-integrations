from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

import parity  # noqa: E402


def tensor(values: list[float], dtype: str = "float32") -> dict[str, object]:
    code = {"float16": "e", "float32": "f", "float64": "d"}[dtype]
    raw = struct.pack(
        ("<" if sys.byteorder == "little" else ">") + code * len(values), *values
    )
    return {
        "shape": [1, len(values)],
        "dtype": dtype,
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "byte_len": len(raw),
        "sha256": parity.sha256_bytes(raw),
    }


def capture(
    variant: str,
    values: list[float],
    *,
    throughput: float = 100.0,
    p95: float = 10.0,
    p99: float = 11.0,
    memory: int = 1_000_000,
    rss: int | None = 10_000_000,
) -> dict[str, object]:
    successes = 10
    duration = successes / throughput
    latencies = [p95] * 9 + [p99]
    return {
        "schema_version": 1,
        "kind": parity.CAPTURE_KIND,
        "suite_id": "unit",
        "task_profile": "forward",
        "variant": variant,
        "session_index": 1,
        "identity": {
            "engine_commit": "1" * 40,
            "integrations_commit": "2" * 40,
            "model_path": "/tmp/model.onnx",
            "model_sha256": "3" * 64,
        },
        "route_evidence": {"verified": True},
        "startup_seconds": 1.0,
        "correctness": {"payload": {"tensor": tensor(values), "latency_ms": 1.0}},
        "warmup_failures": [],
        "trials": [
            {
                "concurrency": 1,
                "trial": 1,
                "requests": successes,
                "successes": successes,
                "failures": 0,
                "failure_samples": [],
                "duration_seconds": duration,
                "throughput_rps": throughput,
                "latencies_ms": latencies,
                "latency_ms": {
                    "mean": p95,
                    "p50": p95,
                    "p95": p95,
                    "p99": p99,
                    "max": p99,
                },
            }
        ],
        "memory": {
            "peak_rss_bytes": rss,
            "rss_samples": 1 if rss is not None else 0,
            "model_memory_usage_max": memory,
            "model_snapshots": [],
        },
    }


def report_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": "unit",
        "task_profile": "forward",
        "identity": {
            "engine_commit": "1" * 40,
            "integrations_commit": "2" * 40,
            "model_path": "/tmp/model.onnx",
            "model_sha256": "3" * 64,
        },
        "payloads": [{"id": "payload", "request": {}}],
        "sequence": ["baseline", "candidate"],
        "gates": {
            "max_abs_error": 1e-5,
            "max_rel_error": 1e-4,
            "min_throughput_ratio": 0.95,
            "max_p95_latency_ratio": 1.05,
            "max_p99_latency_ratio": 1.10,
            "max_model_memory_ratio": 1.05,
            "max_model_memory_increase_bytes": 0.0,
            "max_peak_rss_ratio": 1.05,
            "max_peak_rss_increase_bytes": 0.0,
            "max_startup_ratio": 1.20,
            "require_zero_failures": True,
            "require_model_memory": True,
            "require_process_identity": False,
            "require_process_rss": True,
            "require_route_evidence": True,
            "require_startup_evidence": True,
        },
    }


class TensorComparisonTests(unittest.TestCase):
    def test_float_comparison_uses_absolute_and_relative_tolerance(self) -> None:
        result = parity.compare_tensors(
            tensor([1.0, 1000.0]), tensor([1.000001, 1000.05]), 1e-5, 1e-4
        )
        self.assertTrue(result["passed"])
        failed = parity.compare_tensors(tensor([1.0]), tensor([1.1]), 1e-5, 1e-4)
        self.assertFalse(failed["passed"])

    def test_non_float_comparison_is_exact(self) -> None:
        left = {
            "shape": [2],
            "dtype": "uint8",
            "data_base64": base64.b64encode(b"ab").decode(),
        }
        right = dict(left)
        self.assertTrue(parity.compare_tensors(left, right, 0, 0)["passed"])
        right["data_base64"] = base64.b64encode(b"ac").decode()
        self.assertFalse(parity.compare_tensors(left, right, 0, 0)["passed"])


class ReportTests(unittest.TestCase):
    def test_startup_comparison_rejects_mixed_measurement_boundaries(self) -> None:
        baseline = capture("baseline", [1.0])
        candidate = capture("candidate", [1.0])
        candidate["startup_measurement"] = {
            "method": "process-launch-to-model-ready",
            "clock": "perf_counter",
            "readiness_poll_seconds": 0.005,
        }
        report = parity.build_report(report_config(), [baseline], [candidate])
        self.assertIn(
            "startup measurement methods or polling intervals differ",
            report["failures"],
        )

    def test_startup_samples_keep_outliers_and_enforce_unchanged_limit(self) -> None:
        config = report_config()
        config["gates"]["max_startup_ratio"] = 1.5
        baseline = [capture("baseline", [1.0]) for _ in range(20)]
        candidate = [capture("candidate", [1.0]) for _ in range(20)]
        for index, item in enumerate(candidate):
            item["startup_seconds"] = 1.5 if index < 19 else 8.0
        report = parity.build_report(config, baseline, candidate)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["startup"]["ratio"], 1.5)
        self.assertEqual(
            report["startup"]["candidate_seconds_samples"], [1.5] * 19 + [8.0]
        )
        self.assertIn("8000.000", parity.markdown_report(report))
        for item in candidate:
            item["startup_seconds"] = 1.501
        report = parity.build_report(config, baseline, candidate)
        self.assertEqual(report["failures"], ["candidate startup latency exceeds gate"])

    def test_report_passes_within_all_gates(self) -> None:
        report = parity.build_report(
            report_config(),
            [capture("baseline", [1.0, 2.0])],
            [capture("candidate", [1.0, 2.000001], throughput=98.0, p95=10.2)],
        )
        self.assertEqual(report["status"], "passed")
        self.assertAlmostEqual(report["performance"]["1"]["throughput_ratio"], 0.98)

    def test_report_fails_output_performance_memory_and_route_drift(self) -> None:
        candidate = capture(
            "candidate",
            [1.0, 3.0],
            throughput=80.0,
            p95=15.0,
            p99=20.0,
            memory=2_000_000,
            rss=20_000_000,
        )
        candidate["route_evidence"] = {"verified": False}
        report = parity.build_report(
            report_config(), [capture("baseline", [1.0, 2.0])], [candidate]
        )
        self.assertEqual(report["status"], "failed")
        joined = "\n".join(report["failures"])
        self.assertIn("tensor parity", joined)
        self.assertIn("throughput", joined)
        self.assertIn("model memory", joined)
        self.assertIn("route evidence", joined)


class RouteLogTests(unittest.TestCase):
    def test_provider_registration_failure_rejects_reference_and_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.log"
            for route in [
                "Using embedded ORT rollback",
                "Activated signed native backend pack",
            ]:
                variant = {"required_log_markers": [route], "forbidden_log_markers": []}
                path.write_text(route + "\n", encoding="utf-8")
                self.assertTrue(parity.verify_log(path, variant)["verified"])
                for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                    with self.subTest(route=route, provider=provider):
                        path.write_text(
                            route
                            + "\nERROR ort::ep: "
                            + parity.PROVIDER_REGISTRATION_FAILURE
                            + provider
                            + "`: libnvinfer.so.10: cannot open shared object file\n",
                            encoding="utf-8",
                        )
                        evidence = parity.verify_log(path, variant)
                        self.assertFalse(evidence["verified"])
                        self.assertIn(
                            parity.PROVIDER_REGISTRATION_FAILURE,
                            evidence["observed_forbidden_markers"],
                        )


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0.0
        self.clock = mock.patch.object(
            parity.time, "perf_counter", side_effect=lambda: self.now
        )
        self.sleep = mock.patch.object(parity.time, "sleep", side_effect=self.advance)
        self.clock.start()
        self.sleep_mock = self.sleep.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(self.sleep.stop)

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def test_sub_100ms_readiness_and_real_slower_starts_remain_distinct(self) -> None:
        for ready_after in (0.086, 0.094, 0.167, 0.168):
            with self.subTest(ready_after=ready_after):
                self.now = 0.0

                def snapshot(*_):
                    self.advance(0.0002)  # HTTP observation time is included.
                    return {
                        "status": "active" if self.now >= ready_after else "loading"
                    }

                with mock.patch.object(parity, "model_snapshot", side_effect=snapshot):
                    measured, _ = parity.wait_ready("http://runtime", 0, 1.0, None)
                self.assertGreaterEqual(measured, ready_after)
                self.assertLess(measured - ready_after, 0.006)

    def test_launch_and_sampler_time_count_toward_startup(self) -> None:
        self.now = 0.04
        with mock.patch.object(
            parity, "model_snapshot", return_value={"status": "active"}
        ):
            measured, _ = parity.wait_ready("http://runtime", 0, 1.0, None, started=0.0)
        self.assertEqual(measured, 0.04)

    def test_slow_probe_cannot_exceed_remaining_deadline(self) -> None:
        self.now = 0.04

        def snapshot(_url, _model, timeout):
            self.assertAlmostEqual(timeout, 0.01)
            self.advance(timeout + 0.001)
            return {"status": "active"}

        with mock.patch.object(parity, "model_snapshot", side_effect=snapshot):
            with self.assertRaisesRegex(parity.ParityError, "did not become ready"):
                parity.wait_ready("http://runtime", 0, 0.05, None, started=0.0)
        self.sleep_mock.assert_not_called()

    def test_sleep_uses_remaining_deadline_and_requires_healthy_model(self) -> None:
        with mock.patch.object(
            parity,
            "model_snapshot",
            return_value={"status": "active", "healthy": False},
        ):
            with self.assertRaisesRegex(parity.ParityError, "healthy=False"):
                parity.wait_ready("http://runtime", 0, 0.003, None)
        self.sleep_mock.assert_called_once_with(0.003)

    def test_runtime_exit_fails_without_waiting(self) -> None:
        process = mock.Mock(returncode=7)
        process.poll.return_value = 7
        with mock.patch.object(parity, "model_snapshot") as snapshot:
            with self.assertRaisesRegex(parity.ParityError, "status 7"):
                parity.wait_ready("http://runtime", 0, 1.0, process)
        snapshot.assert_not_called()
        self.sleep_mock.assert_not_called()


FAKE_RUNTIME = r"""
import argparse
import base64
import json
import os
import signal
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
candidate = os.environ.get("KAPSL_GENERIC_NATIVE_PACKS") == "1"
if candidate:
    print("Activated signed native backend pack onnx/cpu", flush=True)
else:
    print("Using embedded ORT rollback", flush=True)
raw = struct.pack("=ff", 1.0, 2.0)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def _send(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/api/models":
            self._send([{"id": 0, "status": "active", "healthy": True, "memory_usage": 1024,
                         "onnx_session_pool_total": 1, "onnx_session_pool_idle": 1}])
        else:
            self.send_error(404)
    def do_POST(self):
        if self.path == "/api/models/0/infer":
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self._send({"shape": [1, 2], "dtype": "float32",
                        "data_base64": base64.b64encode(raw).decode()})
        else:
            self.send_error(404)

server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
server.serve_forever()
"""


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CertificationProcessTests(unittest.TestCase):
    def test_owned_process_sequence_certifies_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.write_bytes(b"model")
            fake = root / "fake_runtime.py"
            fake.write_text(FAKE_RUNTIME, encoding="utf-8")
            port = unused_port()
            base_url = f"http://127.0.0.1:{port}"

            def variant(switch: str) -> dict[str, object]:
                return {
                    "base_url": base_url,
                    "command": [sys.executable, str(fake), "--port", str(port)],
                    "cwd": str(root),
                    "env": {"KAPSL_GENERIC_NATIVE_PACKS": switch},
                    "required_log_markers": (
                        ["Activated signed native backend pack"]
                        if switch == "1"
                        else ["Using embedded ORT rollback"]
                    ),
                    "forbidden_log_markers": (
                        ["Activated signed native backend pack"]
                        if switch == "0"
                        else []
                    ),
                }

            config = {
                "schema_version": 1,
                "suite_id": "fake-ab",
                "task_profile": "forward",
                "identity": {
                    "engine_commit": "1" * 40,
                    "integrations_commit": "2" * 40,
                    "model_path": str(model),
                    "model_sha256": parity.sha256_file(model),
                },
                "payloads": [{"id": "one", "request": {"input": {}}}],
                "workload": {
                    "model_id": 0,
                    "warmup_requests": 2,
                    "requests_per_payload": 2,
                    "concurrency": [1, 2],
                    "trials": 1,
                    "timeout_seconds": 5,
                    "readiness_timeout_seconds": 5,
                    "cooldown_seconds": 0,
                    "rss_sample_seconds": 0.01,
                },
                "gates": {
                    "max_abs_error": 0,
                    "max_rel_error": 0,
                    "min_throughput_ratio": 0,
                    "max_p95_latency_ratio": 1000,
                    "max_p99_latency_ratio": 1000,
                    "max_model_memory_ratio": 1000,
                    "max_model_memory_increase_bytes": 0,
                    "max_peak_rss_ratio": 1000,
                    "max_peak_rss_increase_bytes": 0,
                    "max_startup_ratio": 1000,
                    "require_zero_failures": True,
                    "require_model_memory": True,
                    "require_process_identity": True,
                    "require_process_rss": False,
                    "require_route_evidence": True,
                    "require_startup_evidence": True,
                },
                "sequence": ["baseline", "candidate", "candidate", "baseline"],
                "baseline": variant("0"),
                "candidate": variant("1"),
            }
            config_path = root / "config.json"
            for invalid in (0, -0.01, True, "0.005", float("nan"), float("inf")):
                with self.subTest(readiness_poll_seconds=invalid):
                    config["workload"]["readiness_poll_seconds"] = invalid
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(
                        parity.ParityError, "readiness_poll_seconds"
                    ):
                        parity.resolve_config(config_path, require_commands=True)
            del config["workload"]["readiness_poll_seconds"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output_dir = root / "artifacts"
            status = parity.certify_command(
                argparse.Namespace(config=config_path, output_dir=output_dir)
            )
            self.assertEqual(status, 0)
            report = json.loads((output_dir / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["captures"]["baseline_sessions"], 2)
            self.assertEqual(report["captures"]["candidate_sessions"], 2)
            self.assertEqual(len(report["startup"]["baseline_seconds_samples"]), 2)
            self.assertEqual(len(report["startup"]["candidate_seconds_samples"]), 2)
            self.assertTrue((output_dir / "REPORT.md").is_file())
            candidate_capture = json.loads(
                (output_dir / "02-candidate.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("argv", candidate_capture["process"])
            self.assertEqual(candidate_capture["process"]["generic_native_packs"], "1")
            self.assertEqual(
                candidate_capture["startup_measurement"],
                {
                    "method": "process-launch-to-model-ready",
                    "clock": "perf_counter",
                    "readiness_poll_seconds": 0.005,
                },
            )
            self.assertEqual(
                [warmup["concurrency"] for warmup in candidate_capture["warmups"]],
                [1, 2],
            )
            self.assertTrue(
                all(warmup["successes"] == 2 for warmup in candidate_capture["warmups"])
            )
            self.assertEqual(candidate_capture["warmup_failures"], [])
            with socket.socket() as sock:
                sock.settimeout(0.2)
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)

            config["workload"]["model_id"] = 99
            config["workload"]["readiness_timeout_seconds"] = 0.25
            config_path.write_text(json.dumps(config), encoding="utf-8")
            failed_output_dir = root / "failed-artifacts"
            with self.assertRaises(parity.ParityError):
                parity.certify_command(
                    argparse.Namespace(config=config_path, output_dir=failed_output_dir)
                )
            self.assertTrue((failed_output_dir / "01-baseline.log").is_file())
            with socket.socket() as sock:
                sock.settimeout(0.2)
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    unittest.main()
