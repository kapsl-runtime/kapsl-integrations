#!/usr/bin/env python3
"""Reproducible embedded-vs-native-pack ORT CPU integration certification.

The harness uses only the Python standard library. It can capture an already
running Kapsl endpoint, compare existing captures, or own an ABBA sequence of
baseline/candidate runtime processes and clean them up unconditionally.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CAPTURE_KIND = "kapsl-ort-cpu-capture"
REPORT_KIND = "kapsl-ort-cpu-parity-report"
DEFAULT_SEQUENCE = ["baseline", "candidate", "candidate", "baseline"]
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
FLOAT_FORMATS = {
    "float16": "e",
    "f16": "e",
    "float32": "f",
    "f32": "f",
    "float64": "d",
    "f64": "d",
}
NUMERIC_DTYPE_SIZES = {
    "float16": 2,
    "f16": 2,
    "float32": 4,
    "f32": 4,
    "float64": 8,
    "f64": 8,
    "int32": 4,
    "i32": 4,
    "int64": 8,
    "i64": 8,
    "uint8": 1,
    "u8": 1,
}


class ParityError(RuntimeError):
    """A configuration, capture, or comparison failure."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParityError(f"read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ParityError(f"{path} must contain a JSON object")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ParityError(f"hash {path}: {error}") from error
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be an object")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParityError(f"{label} must be a positive integer")
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParityError(f"{label} must be a non-negative integer")
    return value


def require_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ParityError(f"{label} must be a finite non-negative number")
    return number


def resolve_config(path: Path, *, require_commands: bool) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ParityError(f"schema_version must be {SCHEMA_VERSION}")
    for field in ("suite_id", "task_profile"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ParityError(f"{field} must be a non-empty string")

    identity = require_mapping(config.get("identity"), "identity")
    for field in ("engine_commit", "integrations_commit"):
        value = str(identity.get(field, "")).lower()
        if not HEX_COMMIT.fullmatch(value):
            raise ParityError(f"identity.{field} must be an exact 40-character commit")
        identity[field] = value
    model_path = Path(str(identity.get("model_path", "")))
    if not model_path.is_absolute():
        model_path = (path.parent / model_path).resolve()
    if not model_path.is_file():
        raise ParityError(f"identity.model_path is not a file: {model_path}")
    identity["model_path"] = str(model_path)
    actual_model_sha256 = sha256_file(model_path)
    expected_model_sha256 = str(identity.get("model_sha256", "")).lower()
    if expected_model_sha256:
        if not HEX_SHA256.fullmatch(expected_model_sha256):
            raise ParityError(
                "identity.model_sha256 must be a lowercase SHA-256 digest"
            )
        if expected_model_sha256 != actual_model_sha256:
            raise ParityError(
                "identity.model_sha256 does not match the selected model artifact"
            )
    identity["model_sha256"] = actual_model_sha256

    payloads = config.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ParityError("payloads must be a non-empty array")
    seen_payloads: set[str] = set()
    for index, payload in enumerate(payloads):
        payload = require_mapping(payload, f"payloads[{index}]")
        payload_id = payload.get("id")
        if not isinstance(payload_id, str) or not payload_id:
            raise ParityError(f"payloads[{index}].id must be a non-empty string")
        if payload_id in seen_payloads:
            raise ParityError(f"duplicate payload id: {payload_id}")
        seen_payloads.add(payload_id)
        require_mapping(payload.get("request"), f"payloads[{index}].request")

    workload = require_mapping(config.get("workload"), "workload")
    workload["model_id"] = require_nonnegative_int(
        workload.get("model_id", 0), "workload.model_id"
    )
    workload["warmup_requests"] = require_nonnegative_int(
        workload.get("warmup_requests", 5), "workload.warmup_requests"
    )
    workload["requests_per_payload"] = require_positive_int(
        workload.get("requests_per_payload", 25), "workload.requests_per_payload"
    )
    workload["trials"] = require_positive_int(
        workload.get("trials", 3), "workload.trials"
    )
    concurrency = workload.get("concurrency", [1, 4])
    if not isinstance(concurrency, list) or not concurrency:
        raise ParityError("workload.concurrency must be a non-empty array")
    points = [
        require_positive_int(point, "workload.concurrency entry")
        for point in concurrency
    ]
    if len(points) != len(set(points)):
        raise ParityError("workload.concurrency entries must be unique")
    workload["concurrency"] = points
    workload["timeout_seconds"] = require_nonnegative_number(
        workload.get("timeout_seconds", 30), "workload.timeout_seconds"
    )
    if workload["timeout_seconds"] == 0:
        raise ParityError("workload.timeout_seconds must be greater than zero")
    workload["readiness_timeout_seconds"] = require_nonnegative_number(
        workload.get("readiness_timeout_seconds", 120),
        "workload.readiness_timeout_seconds",
    )
    if workload["readiness_timeout_seconds"] == 0:
        raise ParityError(
            "workload.readiness_timeout_seconds must be greater than zero"
        )
    workload["cooldown_seconds"] = require_nonnegative_number(
        workload.get("cooldown_seconds", 0.25), "workload.cooldown_seconds"
    )
    workload["rss_sample_seconds"] = require_nonnegative_number(
        workload.get("rss_sample_seconds", 0.1), "workload.rss_sample_seconds"
    )

    gates = require_mapping(config.get("gates"), "gates")
    gate_defaults: dict[str, Any] = {
        "max_abs_error": 1e-5,
        "max_rel_error": 1e-4,
        "min_throughput_ratio": 0.95,
        "max_p95_latency_ratio": 1.05,
        "max_p99_latency_ratio": 1.10,
        "max_model_memory_ratio": 1.05,
        "max_model_memory_increase_bytes": 16 * 1024 * 1024,
        "max_peak_rss_ratio": 1.05,
        "max_peak_rss_increase_bytes": 32 * 1024 * 1024,
        "max_startup_ratio": 1.20,
        "require_zero_failures": True,
        "require_model_memory": True,
        "require_process_identity": True,
        "require_process_rss": True,
        "require_route_evidence": True,
        "require_startup_evidence": True,
    }
    for field, default in gate_defaults.items():
        gates.setdefault(field, default)
    for field in (
        "max_abs_error",
        "max_rel_error",
        "min_throughput_ratio",
        "max_p95_latency_ratio",
        "max_p99_latency_ratio",
        "max_model_memory_ratio",
        "max_model_memory_increase_bytes",
        "max_peak_rss_ratio",
        "max_peak_rss_increase_bytes",
        "max_startup_ratio",
    ):
        gates[field] = require_nonnegative_number(gates[field], f"gates.{field}")
    for field in (
        "require_zero_failures",
        "require_model_memory",
        "require_process_identity",
        "require_process_rss",
        "require_route_evidence",
        "require_startup_evidence",
    ):
        if not isinstance(gates[field], bool):
            raise ParityError(f"gates.{field} must be boolean")

    sequence = config.setdefault("sequence", list(DEFAULT_SEQUENCE))
    if not isinstance(sequence, list) or not sequence:
        raise ParityError("sequence must be a non-empty array")
    if any(item not in ("baseline", "candidate") for item in sequence):
        raise ParityError("sequence entries must be baseline or candidate")
    if len(sequence) % len(DEFAULT_SEQUENCE) != 0 or any(
        sequence[index : index + len(DEFAULT_SEQUENCE)] != DEFAULT_SEQUENCE
        for index in range(0, len(sequence), len(DEFAULT_SEQUENCE))
    ):
        raise ParityError(
            "sequence must contain one or more baseline/candidate/candidate/baseline blocks"
        )

    for variant_name, switch in (("baseline", "0"), ("candidate", "1")):
        variant = require_mapping(config.get(variant_name), variant_name)
        base_url = variant.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ParityError(f"{variant_name}.base_url must be an HTTP(S) URL")
        variant["base_url"] = base_url.rstrip("/")
        env = require_mapping(variant.setdefault("env", {}), f"{variant_name}.env")
        env = {str(key): str(value) for key, value in env.items()}
        if env.get("KAPSL_GENERIC_NATIVE_PACKS") != switch:
            raise ParityError(
                f"{variant_name}.env.KAPSL_GENERIC_NATIVE_PACKS must be {switch}"
            )
        variant["env"] = env
        for marker_field in ("required_log_markers", "forbidden_log_markers"):
            markers = variant.setdefault(marker_field, [])
            if not isinstance(markers, list) or not all(
                isinstance(marker, str) and marker for marker in markers
            ):
                raise ParityError(f"{variant_name}.{marker_field} must be strings")
        if require_commands:
            command = variant.get("command")
            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(argument, str) and argument for argument in command
                )
            ):
                raise ParityError(
                    f"{variant_name}.command must be a non-empty argv array"
                )
            cwd = Path(str(variant.get("cwd", path.parent)))
            if not cwd.is_absolute():
                cwd = (path.parent / cwd).resolve()
            if not cwd.is_dir():
                raise ParityError(f"{variant_name}.cwd is not a directory: {cwd}")
            variant["cwd"] = str(cwd)

    if require_commands:
        for variant_name in ("baseline", "candidate"):
            if not config[variant_name]["required_log_markers"]:
                raise ParityError(
                    f"{variant_name}.required_log_markers must prove route activation"
                )
        if config["baseline"]["command"] != config["candidate"]["command"]:
            raise ParityError(
                "baseline.command and candidate.command must be identical"
            )
        if config["baseline"]["cwd"] != config["candidate"]["cwd"]:
            raise ParityError("baseline.cwd and candidate.cwd must be identical")
        if config["baseline"]["base_url"] != config["candidate"]["base_url"]:
            raise ParityError(
                "baseline.base_url and candidate.base_url must be identical"
            )
        allowed_env_differences = config.setdefault(
            "allowed_variant_env_differences",
            [
                "KAPSL_BACKEND_CACHE_DIR",
                "KAPSL_BACKEND_INDEX_PATH",
                "KAPSL_BACKEND_PUBLIC_KEYS",
                "KAPSL_GENERIC_NATIVE_PACKS",
                "KAPSL_LAZY_ONNX_PACKS",
            ],
        )
        if not isinstance(allowed_env_differences, list) or not all(
            isinstance(name, str) and name for name in allowed_env_differences
        ):
            raise ParityError("allowed_variant_env_differences must be strings")
        differing_env = {
            name
            for name in set(config["baseline"]["env"]) | set(config["candidate"]["env"])
            if config["baseline"]["env"].get(name)
            != config["candidate"]["env"].get(name)
        }
        unexpected_env = differing_env - set(allowed_env_differences)
        if unexpected_env:
            raise ParityError(
                "variant environments differ outside the explicit allowlist: "
                + ", ".join(sorted(unexpected_env))
            )
    if gates["require_process_rss"] and workload["rss_sample_seconds"] == 0:
        raise ParityError(
            "workload.rss_sample_seconds must be greater than zero when process RSS is required"
        )
    return config


def auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("KAPSL_BENCHMARK_TOKEN") or os.getenv("KAPSL_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers=auth_headers(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise ParityError(f"HTTP request returned {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ParityError(f"HTTP request failed: {error}") from error


def post_json(
    url: str, payload: Mapping[str, Any], timeout: float
) -> tuple[Any, float]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers=auth_headers(), method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            if response.status < 200 or response.status >= 300:
                raise ParityError(f"HTTP inference returned {response.status}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise ParityError(f"HTTP inference returned {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ParityError(f"HTTP inference failed: {error}") from error
    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        return json.loads(response_body), latency_ms
    except json.JSONDecodeError as error:
        raise ParityError(f"HTTP inference returned invalid JSON: {error}") from error


def model_snapshot(base_url: str, model_id: int, timeout: float) -> dict[str, Any]:
    models = get_json(f"{base_url}/api/models", timeout)
    if isinstance(models, dict):
        models = models.get("models", models.get("data", []))
    if not isinstance(models, list):
        raise ParityError("/api/models did not return an array")
    for model in models:
        if not isinstance(model, dict):
            continue
        try:
            observed_id = int(model.get("id", -1))
        except (TypeError, ValueError):
            continue
        if observed_id == model_id:
            return model
    raise ParityError(f"model id {model_id} is not present at {base_url}")


def wait_ready(
    base_url: str,
    model_id: int,
    timeout: float,
    process: subprocess.Popen[Any] | None,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    deadline = started + timeout
    last_error = "not ready"
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise ParityError(
                f"runtime exited before readiness with status {process.returncode}"
            )
        try:
            snapshot = model_snapshot(base_url, model_id, min(2.0, timeout))
            if snapshot.get("status") == "active" and snapshot.get("healthy", True):
                return time.perf_counter() - started, snapshot
            last_error = f"model status={snapshot.get('status')} healthy={snapshot.get('healthy')}"
        except Exception as error:  # noqa: BLE001 - surfaced after bounded retry
            last_error = str(error)
        time.sleep(0.1)
    raise ParityError(
        f"runtime did not become ready within {timeout:.1f}s: {last_error}"
    )


def tensor_from_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ParityError("inference response must be an object")
    if isinstance(response.get("output"), dict):
        response = response["output"]
    shape = response.get("shape")
    dtype = str(response.get("dtype", "")).lower()
    if not isinstance(shape, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in shape
    ):
        raise ParityError("inference response has no non-negative integer tensor shape")
    if not dtype:
        raise ParityError("inference response has no tensor dtype")
    raw: bytes | None = None
    encoded = response.get("data_base64")
    if isinstance(encoded, str):
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ParityError(f"invalid response data_base64: {error}") from error
    if raw is None:
        data = response.get("data")
        if isinstance(data, list):
            try:
                raw = bytes(data)
            except (TypeError, ValueError) as error:
                raise ParityError(f"response byte array is invalid: {error}") from error
        elif isinstance(data, str):
            try:
                raw = base64.b64decode(data, validate=True)
            except ValueError:
                raw = data.encode("utf-8")
    if raw is None:
        raise ParityError("inference response has no tensor bytes")
    element_size = NUMERIC_DTYPE_SIZES.get(dtype)
    if element_size is not None:
        element_count = math.prod(shape)
        expected_bytes = element_count * element_size
        if len(raw) != expected_bytes:
            raise ParityError(
                f"inference tensor has {len(raw)} bytes; shape and dtype require {expected_bytes}"
            )
    return {
        "shape": shape,
        "dtype": dtype,
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "byte_len": len(raw),
        "sha256": sha256_bytes(raw),
    }


def tensor_bytes(tensor: Mapping[str, Any]) -> bytes:
    try:
        return base64.b64decode(str(tensor["data_base64"]), validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ParityError(f"capture tensor bytes are invalid: {error}") from error


def compare_tensors(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    max_abs_error: float,
    max_rel_error: float,
) -> dict[str, Any]:
    if baseline.get("shape") != candidate.get("shape"):
        return {"passed": False, "reason": "shape mismatch"}
    if baseline.get("dtype") != candidate.get("dtype"):
        return {"passed": False, "reason": "dtype mismatch"}
    baseline_bytes = tensor_bytes(baseline)
    candidate_bytes = tensor_bytes(candidate)
    if len(baseline_bytes) != len(candidate_bytes):
        return {"passed": False, "reason": "byte-length mismatch"}
    dtype = str(baseline["dtype"]).lower()
    format_code = FLOAT_FORMATS.get(dtype)
    if format_code is None:
        passed = baseline_bytes == candidate_bytes
        return {
            "passed": passed,
            "reason": "exact" if passed else "non-floating tensor bytes differ",
            "max_abs_error": 0.0 if passed else None,
            "max_rel_error": 0.0 if passed else None,
        }
    element_size = struct.calcsize(format_code)
    if len(baseline_bytes) % element_size:
        return {"passed": False, "reason": "floating tensor is not element-aligned"}
    endian = "<" if sys.byteorder == "little" else ">"
    baseline_values = struct.iter_unpack(endian + format_code, baseline_bytes)
    candidate_values = struct.iter_unpack(endian + format_code, candidate_bytes)
    observed_abs = 0.0
    observed_rel = 0.0
    for index, (baseline_item, candidate_item) in enumerate(
        zip(baseline_values, candidate_values)
    ):
        left = float(baseline_item[0])
        right = float(candidate_item[0])
        if math.isnan(left) or math.isnan(right):
            if not (math.isnan(left) and math.isnan(right)):
                return {"passed": False, "reason": f"NaN mismatch at element {index}"}
            continue
        if math.isinf(left) or math.isinf(right):
            if left != right:
                return {
                    "passed": False,
                    "reason": f"infinity mismatch at element {index}",
                }
            continue
        absolute = abs(left - right)
        scale = max(abs(left), abs(right))
        relative = absolute / scale if scale else 0.0
        observed_abs = max(observed_abs, absolute)
        observed_rel = max(observed_rel, relative)
        if absolute > max_abs_error + max_rel_error * scale:
            return {
                "passed": False,
                "reason": f"numeric tolerance exceeded at element {index}",
                "max_abs_error": observed_abs,
                "max_rel_error": observed_rel,
            }
    return {
        "passed": True,
        "reason": "within tolerance",
        "max_abs_error": observed_abs,
        "max_rel_error": observed_rel,
    }


class RssSampler:
    def __init__(self, pid: int, interval: float):
        self.pid = pid
        self.interval = interval
        self.peak_bytes: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="ort-parity-rss", daemon=True
        )

    def start(self) -> None:
        if self.interval > 0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval * 4.0))

    def _sample(self) -> int | None:
        status_path = Path(f"/proc/{self.pid}/status")
        if status_path.is_file():
            try:
                for line in status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
            except (OSError, ValueError, IndexError):
                return None
        if os.name != "nt":
            try:
                output = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                ).strip()
                return int(output) * 1024 if output else None
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._sample()
            if value is not None:
                self.samples += 1
                self.peak_bytes = (
                    value if self.peak_bytes is None else max(self.peak_bytes, value)
                )
            self._stop.wait(self.interval)


def cpu_description() -> str | None:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except (OSError, IndexError):
            pass
    if platform.system() == "Darwin":
        try:
            return (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                ).strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return os.getenv("PROCESSOR_IDENTIFIER")


def run_requests(
    base_url: str,
    model_id: int,
    payloads: Sequence[Mapping[str, Any]],
    request_count_per_payload: int,
    concurrency: int,
    timeout: float,
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    jobs: list[tuple[str, Mapping[str, Any]]] = []
    for _ in range(request_count_per_payload):
        for payload in payloads:
            jobs.append(
                (str(payload["id"]), require_mapping(payload["request"], "request"))
            )
    url = f"{base_url}/api/models/{model_id}/infer"
    latencies: list[float] = []
    failures: list[str] = []

    def invoke(payload_id: str, request: Mapping[str, Any]) -> tuple[str, Any, float]:
        response, latency = post_json(url, request, timeout)
        return payload_id, response, latency

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="ort-parity"
    ) as pool:
        futures = [
            pool.submit(invoke, payload_id, request) for payload_id, request in jobs
        ]
        for future in as_completed(futures):
            try:
                payload_id, response, latency = future.result()
                tensor = tensor_from_response(response)
                if tensor["sha256"] != references[payload_id]["sha256"]:
                    failures.append(
                        f"{payload_id}: output changed within one runtime capture"
                    )
                else:
                    latencies.append(latency)
            except Exception as error:  # noqa: BLE001 - recorded as benchmark evidence
                failures.append(str(error))
    duration = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests": len(jobs),
        "successes": len(latencies),
        "failures": len(failures),
        "failure_samples": failures[:10],
        "duration_seconds": duration,
        "throughput_rps": len(latencies) / duration if duration > 0 else 0.0,
        "latencies_ms": sorted(latencies),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "p99": percentile(latencies, 0.99) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def capture_running_endpoint(
    config: Mapping[str, Any],
    variant_name: str,
    session_index: int,
    *,
    process: subprocess.Popen[Any] | None = None,
    sampler: RssSampler | None = None,
    startup_seconds: float | None = None,
    route_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    variant = require_mapping(config[variant_name], variant_name)
    workload = require_mapping(config["workload"], "workload")
    base_url = str(variant["base_url"])
    model_id = int(workload["model_id"])
    if startup_seconds is None:
        _, initial_snapshot = wait_ready(
            base_url,
            model_id,
            float(workload["readiness_timeout_seconds"]),
            process,
        )
    else:
        initial_snapshot = model_snapshot(
            base_url, model_id, float(workload["timeout_seconds"])
        )

    references: dict[str, Any] = {}
    infer_url = f"{base_url}/api/models/{model_id}/infer"
    for payload in config["payloads"]:
        response, latency = post_json(
            infer_url,
            require_mapping(payload["request"], "payload request"),
            float(workload["timeout_seconds"]),
        )
        references[str(payload["id"])] = {
            "tensor": tensor_from_response(response),
            "latency_ms": latency,
        }
    reference_tensors = {
        payload_id: value["tensor"] for payload_id, value in references.items()
    }

    warmup_failures: list[str] = []
    for index in range(int(workload["warmup_requests"])):
        payload = config["payloads"][index % len(config["payloads"])]
        try:
            response, _ = post_json(
                infer_url,
                require_mapping(payload["request"], "payload request"),
                float(workload["timeout_seconds"]),
            )
            tensor = tensor_from_response(response)
            if tensor["sha256"] != reference_tensors[str(payload["id"])]["sha256"]:
                warmup_failures.append(f"{payload['id']}: warmup output changed")
        except Exception as error:  # noqa: BLE001 - retained in capture
            warmup_failures.append(str(error))

    trials: list[dict[str, Any]] = []
    snapshots = [initial_snapshot]
    for concurrency in workload["concurrency"]:
        for trial_index in range(int(workload["trials"])):
            trial = run_requests(
                base_url,
                model_id,
                config["payloads"],
                int(workload["requests_per_payload"]),
                int(concurrency),
                float(workload["timeout_seconds"]),
                reference_tensors,
            )
            trial["trial"] = trial_index + 1
            trials.append(trial)
            snapshots.append(
                model_snapshot(base_url, model_id, float(workload["timeout_seconds"]))
            )
            if float(workload["cooldown_seconds"]) > 0:
                time.sleep(float(workload["cooldown_seconds"]))

    memory_values = [
        int(snapshot.get("memory_usage", 0) or 0)
        for snapshot in snapshots
        if isinstance(snapshot, dict)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CAPTURE_KIND,
        "suite_id": config["suite_id"],
        "task_profile": config["task_profile"],
        "variant": variant_name,
        "session_index": session_index,
        "captured_at_unix_seconds": time.time(),
        "identity": copy.deepcopy(config["identity"]),
        "configuration_sha256": sha256_bytes(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_description": cpu_description(),
        },
        "route_evidence": dict(
            route_evidence or {"verified": False, "mode": "external"}
        ),
        "startup_seconds": startup_seconds,
        "correctness": references,
        "warmup_failures": warmup_failures,
        "trials": trials,
        "memory": {
            "peak_rss_bytes": sampler.peak_bytes if sampler is not None else None,
            "rss_samples": sampler.samples if sampler is not None else 0,
            "model_memory_usage_max": max(memory_values, default=0),
            "model_snapshots": snapshots,
        },
    }


def format_command(
    arguments: Sequence[str], config: Mapping[str, Any], output_dir: Path
) -> list[str]:
    replacements = {
        "model_path": str(config["identity"]["model_path"]),
        "output_dir": str(output_dir),
    }
    try:
        return [argument.format_map(replacements) for argument in arguments]
    except KeyError as error:
        raise ParityError(f"unsupported command placeholder: {error}") from error


def stop_process(process: subprocess.Popen[Any], timeout: float = 10.0) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            if process.poll() is None:
                process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=timeout)
        return
    except OSError:
        pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        except OSError:
            break
        time.sleep(0.05)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def verify_log(log_path: Path, variant: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ParityError(f"read runtime log {log_path}: {error}") from error
    missing = [
        marker for marker in variant["required_log_markers"] if marker not in text
    ]
    forbidden = [
        marker for marker in variant["forbidden_log_markers"] if marker in text
    ]
    return {
        "verified": not missing and not forbidden,
        "mode": "owned-process-log",
        "log_path": str(log_path),
        "log_sha256": sha256_bytes(text.encode("utf-8")),
        "required_markers": variant["required_log_markers"],
        "forbidden_markers": variant["forbidden_log_markers"],
        "missing_markers": missing,
        "observed_forbidden_markers": forbidden,
    }


def run_variant_session(
    config: Mapping[str, Any],
    variant_name: str,
    session_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    variant = require_mapping(config[variant_name], variant_name)
    command = format_command(variant["command"], config, output_dir)
    environment = os.environ.copy()
    environment.update(variant["env"])
    log_path = output_dir / f"{session_index:02d}-{variant_name}.log"
    process: subprocess.Popen[Any] | None = None
    sampler: RssSampler | None = None
    capture: dict[str, Any] | None = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        try:
            process_options: dict[str, Any] = {}
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=variant["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                **process_options,
            )
            sampler = RssSampler(
                process.pid, float(config["workload"]["rss_sample_seconds"])
            )
            sampler.start()
            startup_seconds, _ = wait_ready(
                variant["base_url"],
                int(config["workload"]["model_id"]),
                float(config["workload"]["readiness_timeout_seconds"]),
                process,
            )
            capture = capture_running_endpoint(
                config,
                variant_name,
                session_index,
                process=process,
                sampler=sampler,
                startup_seconds=startup_seconds,
            )
        finally:
            if sampler is not None:
                sampler.stop()
            if process is not None:
                stop_process(process)
    evidence = verify_log(log_path, variant)
    if capture is None:
        raise ParityError(f"{variant_name} session {session_index} produced no capture")
    capture["route_evidence"] = evidence
    command_executable = Path(command[0])
    if command_executable.is_absolute():
        executable_path = command_executable
    elif os.path.dirname(command[0]):
        executable_path = Path(str(variant["cwd"])) / command_executable
    else:
        executable_path = Path(
            shutil.which(command[0], path=environment.get("PATH")) or command[0]
        )
    executable_path = executable_path.resolve()
    capture["process"] = {
        "executable": str(executable_path),
        "executable_sha256": sha256_file(executable_path)
        if executable_path.is_file()
        else None,
        "argv_sha256": sha256_bytes(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ),
        "cwd": variant["cwd"],
        "generic_native_packs": variant["env"]["KAPSL_GENERIC_NATIVE_PACKS"],
        "wall_seconds": time.perf_counter() - started,
        "exit_status": process.returncode if process is not None else None,
    }
    capture["memory"]["peak_rss_bytes"] = (
        sampler.peak_bytes if sampler is not None else None
    )
    capture["memory"]["rss_samples"] = sampler.samples if sampler is not None else 0
    return capture


def aggregate_captures(captures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points: dict[int, list[Mapping[str, Any]]] = {}
    for capture in captures:
        for trial in capture.get("trials", []):
            points.setdefault(int(trial["concurrency"]), []).append(trial)
    aggregates: dict[str, Any] = {}
    for concurrency, trials in sorted(points.items()):
        latencies = [
            float(latency)
            for trial in trials
            for latency in trial.get("latencies_ms", [])
        ]
        successes = sum(int(trial.get("successes", 0)) for trial in trials)
        failures = sum(int(trial.get("failures", 0)) for trial in trials)
        duration = sum(float(trial.get("duration_seconds", 0.0)) for trial in trials)
        aggregates[str(concurrency)] = {
            "sessions_and_trials": len(trials),
            "successes": successes,
            "failures": failures,
            "duration_seconds": duration,
            "throughput_rps": successes / duration if duration > 0 else 0.0,
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": percentile(latencies, 0.50) if latencies else None,
                "p95": percentile(latencies, 0.95) if latencies else None,
                "p99": percentile(latencies, 0.99) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
        }
    peak_rss = [capture.get("memory", {}).get("peak_rss_bytes") for capture in captures]
    peak_rss = [int(value) for value in peak_rss if value is not None]
    model_memory = [
        int(capture.get("memory", {}).get("model_memory_usage_max", 0))
        for capture in captures
    ]
    startups = [
        float(capture["startup_seconds"])
        for capture in captures
        if capture.get("startup_seconds") is not None
    ]
    return {
        "performance": aggregates,
        "memory": {
            "peak_rss_bytes": max(peak_rss) if peak_rss else None,
            "model_memory_usage_max": max(model_memory, default=0),
        },
        "startup_seconds_median": statistics.median(startups) if startups else None,
        "failures": sum(
            len(capture.get("warmup_failures", []))
            + sum(int(trial.get("failures", 0)) for trial in capture.get("trials", []))
            for capture in captures
        ),
    }


def ratio(
    numerator: float | int | None, denominator: float | int | None
) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def capture_tensor(capture: Mapping[str, Any], payload_id: str) -> Mapping[str, Any]:
    correctness = capture.get("correctness")
    if not isinstance(correctness, dict):
        raise ParityError("capture correctness evidence must be an object")
    payload = correctness.get(payload_id)
    if not isinstance(payload, dict) or not isinstance(payload.get("tensor"), dict):
        raise ParityError(f"capture has no tensor evidence for payload {payload_id}")
    return payload["tensor"]


def build_report(
    config: Mapping[str, Any],
    baseline_captures: Sequence[Mapping[str, Any]],
    candidate_captures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = config["gates"]
    failures: list[str] = []
    correctness: dict[str, Any] = {}
    if not baseline_captures or not candidate_captures:
        raise ParityError("comparison requires at least one capture for each variant")
    captures_by_variant = {
        "baseline": baseline_captures,
        "candidate": candidate_captures,
    }
    for expected_variant, captures in captures_by_variant.items():
        for capture in captures:
            if capture.get("kind") != CAPTURE_KIND:
                raise ParityError("comparison input is not an ORT CPU capture")
            if capture.get("suite_id") != config["suite_id"]:
                raise ParityError("capture suite_id does not match the configuration")
            if capture.get("identity") != config["identity"]:
                raise ParityError("capture identity does not match the configuration")
            if capture.get("variant") != expected_variant:
                raise ParityError(
                    f"{expected_variant} input contains a {capture.get('variant')} capture"
                )
            if gates["require_route_evidence"] and not capture.get(
                "route_evidence", {}
            ).get("verified", False):
                failures.append(
                    f"{capture.get('variant')} session "
                    f"{capture.get('session_index')} lacks route evidence"
                )

    all_captures = [*baseline_captures, *candidate_captures]
    process_records = [
        capture.get("process")
        for capture in all_captures
        if isinstance(capture.get("process"), dict)
    ]
    executable_hashes = {
        str(record["executable_sha256"])
        for record in process_records
        if record.get("executable_sha256")
    }
    argument_hashes = {
        str(record["argv_sha256"])
        for record in process_records
        if record.get("argv_sha256")
    }
    working_directories = {
        str(record["cwd"]) for record in process_records if record.get("cwd")
    }
    process_identity_verified = (
        len(process_records) == len(all_captures)
        and len(executable_hashes) == 1
        and len(argument_hashes) == 1
        and len(working_directories) == 1
    )
    for variant_name, captures in captures_by_variant.items():
        expected_switch = "0" if variant_name == "baseline" else "1"
        for capture in captures:
            process_record = capture.get("process")
            if (
                isinstance(process_record, dict)
                and process_record.get("generic_native_packs") != expected_switch
            ):
                process_identity_verified = False
    if gates["require_process_identity"] and not process_identity_verified:
        failures.append(
            "owned-process binary, arguments, working directory, or route switch differ"
        )

    for payload in config["payloads"]:
        payload_id = str(payload["id"])
        baseline_tensor = capture_tensor(baseline_captures[0], payload_id)
        comparisons: list[dict[str, Any]] = []
        for capture in baseline_captures[1:]:
            comparisons.append(
                compare_tensors(
                    baseline_tensor,
                    capture_tensor(capture, payload_id),
                    float(gates["max_abs_error"]),
                    float(gates["max_rel_error"]),
                )
            )
        for capture in candidate_captures:
            comparisons.append(
                compare_tensors(
                    baseline_tensor,
                    capture_tensor(capture, payload_id),
                    float(gates["max_abs_error"]),
                    float(gates["max_rel_error"]),
                )
            )
        passed = all(comparison["passed"] for comparison in comparisons)
        if not passed:
            failures.append(f"payload {payload_id} failed tensor parity")
        correctness[payload_id] = {"passed": passed, "comparisons": comparisons}

    baseline = aggregate_captures(baseline_captures)
    candidate = aggregate_captures(candidate_captures)
    performance: dict[str, Any] = {}
    if set(baseline["performance"]) != set(candidate["performance"]):
        failures.append("baseline and candidate concurrency points differ")
    for concurrency in sorted(
        set(baseline["performance"]) & set(candidate["performance"]), key=int
    ):
        left = baseline["performance"][concurrency]
        right = candidate["performance"][concurrency]
        throughput_ratio = ratio(right["throughput_rps"], left["throughput_rps"])
        p95_ratio = ratio(right["latency_ms"]["p95"], left["latency_ms"]["p95"])
        p99_ratio = ratio(right["latency_ms"]["p99"], left["latency_ms"]["p99"])
        point_failures: list[str] = []
        if throughput_ratio is None or throughput_ratio < gates["min_throughput_ratio"]:
            point_failures.append("throughput ratio below gate")
        if p95_ratio is None or p95_ratio > gates["max_p95_latency_ratio"]:
            point_failures.append("p95 latency ratio above gate")
        if p99_ratio is None or p99_ratio > gates["max_p99_latency_ratio"]:
            point_failures.append("p99 latency ratio above gate")
        if gates["require_zero_failures"] and (left["failures"] or right["failures"]):
            point_failures.append("request failures observed")
        if point_failures:
            failures.extend(
                f"concurrency {concurrency}: {item}" for item in point_failures
            )
        performance[concurrency] = {
            "passed": not point_failures,
            "baseline": left,
            "candidate": right,
            "throughput_ratio": throughput_ratio,
            "p95_latency_ratio": p95_ratio,
            "p99_latency_ratio": p99_ratio,
            "failures": point_failures,
        }

    baseline_model_memory = baseline["memory"]["model_memory_usage_max"]
    candidate_model_memory = candidate["memory"]["model_memory_usage_max"]
    model_memory_limit = (
        baseline_model_memory * gates["max_model_memory_ratio"]
        + gates["max_model_memory_increase_bytes"]
    )
    if gates["require_model_memory"] and (
        baseline_model_memory <= 0 or candidate_model_memory <= 0
    ):
        failures.append("model memory evidence is unavailable")
    elif candidate_model_memory > model_memory_limit:
        failures.append("candidate model memory exceeds gate")
    baseline_rss = baseline["memory"]["peak_rss_bytes"]
    candidate_rss = candidate["memory"]["peak_rss_bytes"]
    rss_limit = None
    if baseline_rss is None or candidate_rss is None:
        if gates["require_process_rss"]:
            failures.append("process RSS evidence is unavailable")
    else:
        rss_limit = (
            baseline_rss * gates["max_peak_rss_ratio"]
            + gates["max_peak_rss_increase_bytes"]
        )
        if candidate_rss > rss_limit:
            failures.append("candidate peak RSS exceeds gate")
    startup_ratio = ratio(
        candidate["startup_seconds_median"], baseline["startup_seconds_median"]
    )
    if startup_ratio is None and gates["require_startup_evidence"]:
        failures.append("process startup evidence is unavailable")
    elif startup_ratio is not None and startup_ratio > gates["max_startup_ratio"]:
        failures.append("candidate startup latency exceeds gate")
    if gates["require_zero_failures"] and (
        baseline["failures"] or candidate["failures"]
    ):
        failures.append("capture contains warmup or measured request failures")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "suite_id": config["suite_id"],
        "task_profile": config["task_profile"],
        "status": "passed" if not failures else "failed",
        "generated_at_unix_seconds": time.time(),
        "identity": copy.deepcopy(config["identity"]),
        "sequence": list(config["sequence"]),
        "gates": copy.deepcopy(gates),
        "correctness": correctness,
        "performance": performance,
        "memory": {
            "baseline": baseline["memory"],
            "candidate": candidate["memory"],
            "candidate_model_memory_limit": model_memory_limit,
            "candidate_peak_rss_limit": rss_limit,
        },
        "startup": {
            "baseline_seconds_median": baseline["startup_seconds_median"],
            "candidate_seconds_median": candidate["startup_seconds_median"],
            "ratio": startup_ratio,
        },
        "process_identity": {
            "verified": process_identity_verified,
            "executable_sha256": sorted(executable_hashes),
            "argv_sha256": sorted(argument_hashes),
            "working_directories": sorted(working_directories),
        },
        "failures": failures,
        "captures": {
            "baseline_sessions": len(baseline_captures),
            "candidate_sessions": len(candidate_captures),
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# ORT CPU parity: {report['suite_id']}",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        f"Task profile: `{report['task_profile']}`",
        f"Engine commit: `{report['identity']['engine_commit']}`",
        f"Integrations commit: `{report['identity']['integrations_commit']}`",
        f"Model SHA-256: `{report['identity']['model_sha256']}`",
        "",
        "## Performance",
        "",
        "| Concurrency | Baseline req/s | Candidate req/s | Ratio | "
        "Baseline p95 ms | Candidate p95 ms | p95 ratio | Result |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for concurrency, point in sorted(
        report["performance"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            (
                "| {concurrency} | {br:.3f} | {cr:.3f} | {tr:.4f} | "
                "{bp:.3f} | {cp:.3f} | {pr:.4f} | {result} |"
            ).format(
                concurrency=concurrency,
                br=point["baseline"]["throughput_rps"],
                cr=point["candidate"]["throughput_rps"],
                tr=point["throughput_ratio"] or 0.0,
                bp=point["baseline"]["latency_ms"]["p95"] or 0.0,
                cp=point["candidate"]["latency_ms"]["p95"] or 0.0,
                pr=point["p95_latency_ratio"] or 0.0,
                result="PASS" if point["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            *[
                f"- `{payload_id}`: {'PASS' if result['passed'] else 'FAIL'}"
                for payload_id, result in report["correctness"].items()
            ],
            "",
            "## Memory and startup",
            "",
            "- Baseline model memory: "
            f"{report['memory']['baseline']['model_memory_usage_max']} bytes",
            "- Candidate model memory: "
            f"{report['memory']['candidate']['model_memory_usage_max']} bytes",
            f"- Baseline peak RSS: {report['memory']['baseline']['peak_rss_bytes']} bytes",
            f"- Candidate peak RSS: {report['memory']['candidate']['peak_rss_bytes']} bytes",
            f"- Startup ratio: {report['startup']['ratio']}",
            "- Process identity: "
            f"{'PASS' if report['process_identity']['verified'] else 'UNVERIFIED'}",
            "",
            "## Gate failures",
            "",
        ]
    )
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def compare_command(args: argparse.Namespace) -> int:
    config = resolve_config(args.config.resolve(), require_commands=False)
    baseline = [load_json(path.resolve()) for path in args.baseline]
    candidate = [load_json(path.resolve()) for path in args.candidate]
    report = build_report(config, baseline, candidate)
    atomic_write_json(args.output.resolve(), report)
    atomic_write_text(args.markdown.resolve(), markdown_report(report))
    print(args.output.resolve())
    print(args.markdown.resolve())
    return 0 if report["status"] == "passed" else 1


def capture_command(args: argparse.Namespace) -> int:
    config = resolve_config(args.config.resolve(), require_commands=False)
    capture = capture_running_endpoint(config, args.variant, 1)
    if args.log_file is not None:
        capture["route_evidence"] = verify_log(
            args.log_file.resolve(), require_mapping(config[args.variant], args.variant)
        )
    atomic_write_json(args.output.resolve(), capture)
    print(args.output.resolve())
    return 0


def certify_command(args: argparse.Namespace) -> int:
    config = resolve_config(args.config.resolve(), require_commands=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ParityError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ParityError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    captures: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    for session_index, variant_name in enumerate(config["sequence"], start=1):
        print(f"[{session_index}/{len(config['sequence'])}] {variant_name}", flush=True)
        capture = run_variant_session(config, variant_name, session_index, output_dir)
        captures[variant_name].append(capture)
        atomic_write_json(
            output_dir / f"{session_index:02d}-{variant_name}.json", capture
        )
    report = build_report(config, captures["baseline"], captures["candidate"])
    atomic_write_json(output_dir / "report.json", report)
    atomic_write_text(output_dir / "REPORT.md", markdown_report(report))
    print(output_dir / "report.json")
    print(output_dir / "REPORT.md")
    return 0 if report["status"] == "passed" else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="Capture an already-running endpoint")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    capture.add_argument("--log-file", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    capture.set_defaults(function=capture_command)

    compare = commands.add_parser("compare", help="Compare existing capture artifacts")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, action="append", required=True)
    compare.add_argument("--candidate", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--markdown", type=Path, required=True)
    compare.set_defaults(function=compare_command)

    certify = commands.add_parser(
        "certify", help="Own and compare an ABBA process sequence"
    )
    certify.add_argument("--config", type=Path, required=True)
    certify.add_argument("--output-dir", type=Path, required=True)
    certify.set_defaults(function=certify_command)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.function(args))
    except ParityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
