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
