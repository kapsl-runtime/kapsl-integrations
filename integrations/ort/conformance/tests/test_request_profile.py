import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "request_profile_analysis",
    Path(__file__).resolve().parents[1] / "analyze_request_profile.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RequestProfileTests(unittest.TestCase):
    def report(self):
        return {
            "schema_version": 1,
            "component": "ort",
            "process_id": 1,
            "model_id": 2,
            "replica_id": 3,
            "origin_unix_ns": "100",
            "sample_limit": 8192,
            "records": [
                {
                    "operation": "infer",
                    "request_id": 7,
                    "sequence": 0,
                    "started_ns": 1,
                    "wall_ns": 1000,
                    "phases": [{"name": "ort_run", "wall_ns": 900}],
                }
            ],
        }

    def analyze(self, lines):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.log"
            path.write_text("\n".join(lines))
            return analysis.analyze(path, 0, 1000)

    def line(self, report):
        return "INFO KAPSL_REQUEST_PROFILE " + json.dumps(report)

    def test_plain_and_structured_logs_keep_parent_and_phase_separate(self):
        line = self.line(self.report())
        for source in [line, json.dumps({"fields": {"message": line}})]:
            result = self.analyze([source])[0]
            self.assertEqual(result["wall"]["mean_us"], 1)
            self.assertEqual(result["phases"]["ort_run"]["mean_us"], 0.9)
            self.assertEqual(result["replica_id"], 3)

    def test_duplicate_and_impossible_samples_are_rejected(self):
        line = self.line(self.report())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze([line, line])
        report = self.report()
        report["records"][0]["phases"][0]["wall_ns"] = 1001
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.analyze([self.line(report)])
        with self.assertRaisesRegex(ValueError, "no request profiles"):
            self.analyze(["runtime exited before model unload"])

    def correlate(self, reports):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.log"
            path.write_text("\n".join(self.line(report) for report in reports))
            return analysis.correlate_allocator(path)

    def callback_report(self):
        report = self.report()
        report["component"] = "engine.native.allocator"
        report["origin_unix_ns"] = "150"
        report["records"][0].update(
            operation="free",
            started_ns=51,
            wall_ns=200,
            phases=[{"name": "synchronize_before_free", "wall_ns": 100}],
        )
        return report

    def test_callbacks_use_clock_origins_and_interval_union_not_count_windows(self):
        host = self.report()
        host["component"] = "engine.native"
        callbacks = self.callback_report()
        callbacks["records"].append(
            {
                "operation": "synchronize_callback",
                "sequence": 1,
                "request_id": 0,
                "started_ns": 101,
                "wall_ns": 200,
                "phases": [{"name": "device_synchronize", "wall_ns": 200}],
            }
        )
        result = self.correlate([callbacks, host])
        self.assertEqual(result["matched_callbacks"], 2)
        request = result["requests"][0]
        self.assertEqual(request["callback_count"], 2)
        self.assertEqual(request["allocator_covered_wall_us"], 0.25)
        self.assertEqual(
            request["phases"]["free.synchronize_before_free"]["mean_us"], 0.1
        )
        summary = self.analyze([self.line(callbacks)])
        self.assertTrue(all(row["window"] == "unassigned_callbacks" for row in summary))

    def test_concurrent_zero_id_is_ambiguous_but_explicit_owner_is_matched(self):
        import copy

        host = self.report()
        host["component"] = "engine.native"
        second = copy.deepcopy(host["records"][0])
        second.update(sequence=1, request_id=8)
        host["records"].append(second)
        callbacks = self.callback_report()
        zero = copy.deepcopy(callbacks["records"][0])
        zero.update(sequence=1, request_id=0, operation="synchronize_callback")
        callbacks["records"].append(zero)
        result = self.correlate([host, callbacks])
        self.assertEqual(result["matched_callbacks"], 1)
        self.assertEqual(result["requests"][0]["callback_count"], 1)
        self.assertEqual(result["requests"][1]["callback_count"], 0)
        self.assertEqual(
            result["unmatched_callbacks"][0]["reason"], "ambiguous_enclosing_infer"
        )

    def test_request_owner_lifecycle_and_collector_gaps_are_not_guessed(self):
        import copy

        host = self.report()
        host["component"] = "engine.native"
        for field, value in [("process_id", 9), ("model_id", 9), ("replica_id", 9)]:
            callback = self.callback_report()
            callback[field] = value
            result = self.correlate([host, callback])
            self.assertEqual(result["matched_callbacks"], 0)
        for changes in [{"request_id": 99}, {"started_ns": 1001}, {"wall_ns": 2000}]:
            callback = self.callback_report()
            callback["records"][0].update(changes)
            result = self.correlate([host, callback])
            self.assertEqual(result["matched_callbacks"], 0)
            self.assertEqual(
                result["unmatched_callbacks"][0]["reason"], "no_enclosing_infer"
            )
        second = copy.deepcopy(host["records"][0])
        second.update(sequence=1, started_ns=2001)
        host["records"].append(second)
        result = self.correlate([host, self.callback_report()])
        self.assertEqual(result["matched_callbacks"], 1)
        self.assertFalse(result["qualification_passed"])
        result = self.correlate([self.callback_report()])
        self.assertEqual(result["matched_callbacks"], 0)

    def test_invalid_clock_origins_cannot_create_false_correlations(self):
        for value in ["-1", "not-a-clock", 100, True]:
            report = self.report()
            report["origin_unix_ns"] = value
            with self.assertRaisesRegex(ValueError, "clock origin"):
                self.correlate([report])
