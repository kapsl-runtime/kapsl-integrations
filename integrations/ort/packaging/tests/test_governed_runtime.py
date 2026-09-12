from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import governed_runtime  # noqa: E402
from package_cpu import PackageError  # noqa: E402


class GovernedRuntimeTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, dict]:
        ort = root / "ort"
        (ort / "runtime").mkdir(parents=True)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        files = {}
        for name in governed_runtime.runtime_names("cuda12").values():
            data = name.encode()
            (artifacts / name).write_bytes(data)
            files[name] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        provenance = {
            "schema_version": 1,
            "profile": "cuda12",
            "source_commit": governed_runtime.source_lock()["commit"],
            "recipe": governed_runtime.recipe_identity(),
            "files": files,
        }
        data = json.dumps(provenance).encode()
        (artifacts / "governed-runtime.json").write_bytes(data)
        locks = {
            "schema_version": 1,
            "profiles": {
                "cuda12": {
                    "files": files,
                    "provenance_sha256": hashlib.sha256(data).hexdigest(),
                }
            },
        }
        (ort / "runtime/governed-runtimes.lock.json").write_text(json.dumps(locks))
        return artifacts, provenance

    def test_missing_reviewed_artifact_never_accepts_stock_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            (root / "runtime/governed-runtimes.lock.json").write_text(
                '{"schema_version": 1, "profiles": {}}'
            )
            with patch.object(governed_runtime, "ORT_ROOT", root):
                with self.assertRaisesRegex(PackageError, "no reviewed"):
                    governed_runtime.verify(root, "cuda12")

    def test_locked_artifacts_reject_binary_provenance_and_profile_substitution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts, provenance = self.fixture(root)
            with patch.object(governed_runtime, "ORT_ROOT", root / "ort"):
                self.assertEqual(
                    governed_runtime.verify(artifacts, "cuda12"), provenance
                )
                with self.assertRaisesRegex(PackageError, "no reviewed"):
                    governed_runtime.verify(artifacts, "tensorrt10")
                library = artifacts / next(iter(provenance["files"]))
                original = library.read_bytes()
                library.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(PackageError, "object differs"):
                    governed_runtime.verify(artifacts, "cuda12")
                library.write_bytes(original)
                manifest = artifacts / "governed-runtime.json"
                manifest.write_bytes(manifest.read_bytes() + b" ")
                with self.assertRaisesRegex(PackageError, "provenance differs"):
                    governed_runtime.verify(artifacts, "cuda12")

    def test_valid_lock_cannot_select_a_different_build_recipe_or_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts, provenance = self.fixture(root)
            with patch.object(governed_runtime, "ORT_ROOT", root / "ort"):
                with patch.object(governed_runtime, "recipe_identity", return_value={}):
                    with self.assertRaisesRegex(PackageError, "source/build recipe"):
                        governed_runtime.verify(artifacts, "cuda12")
                library = artifacts / next(iter(provenance["files"]))
                outside = root / "foreign-library"
                library.rename(outside)
                library.symlink_to(outside)
                with self.assertRaisesRegex(PackageError, "object differs"):
                    governed_runtime.verify(artifacts, "cuda12")


if __name__ == "__main__":
    unittest.main()
