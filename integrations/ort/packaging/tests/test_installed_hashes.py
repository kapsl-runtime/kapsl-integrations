import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from blake3 import blake3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import installed_hashes  # noqa: E402
import package_cpu  # noqa: E402
import release  # noqa: E402
import test_packaging as fixtures  # noqa: E402


class InstalledHashTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "pack.tar.gz"
        self.members = {"lib/adapter.so": b"adapter", "licenses/LICENSE": b"license"}
        self.archive.write_bytes(fixtures.fixture_archive(self.members))
        self.manifest = {
            "entrypoint": "lib/adapter.so",
            "files": {
                p: hashlib.sha256(b).hexdigest() for p, b in self.members.items()
            },
        }

    def test_complete_map_matches_exact_archive_and_is_repeatable(self):
        expected = {p: blake3(b).hexdigest() for p, b in self.members.items()}
        self.assertEqual(
            installed_hashes.archive_blake3(self.archive, self.manifest), expected
        )
        self.manifest["files_blake3"] = expected
        self.assertEqual(
            installed_hashes.archive_blake3(self.archive, self.manifest), expected
        )

    def test_rejects_bad_sha_incomplete_map_and_changed_existing_blake3(self):
        for variant in ["sha", "missing", "extra", "blake3", "path", "bootstrap"]:
            with self.subTest(variant=variant):
                manifest = copy.deepcopy(self.manifest)
                if variant == "sha":
                    manifest["files"]["lib/adapter.so"] = "0" * 64
                elif variant == "missing":
                    del manifest["files"]["licenses/LICENSE"]
                elif variant == "extra":
                    manifest["files"]["missing"] = "0" * 64
                elif variant == "blake3":
                    manifest["files_blake3"] = {}
                elif variant == "path":
                    manifest["files"]["../escape"] = "0" * 64
                else:
                    manifest["installer"] = {"kind": "bootstrap", "path": "bin/run"}
                with self.assertRaises(package_cpu.PackageError):
                    installed_hashes.archive_blake3(self.archive, manifest)

    def test_rejects_duplicate_symlink_and_escaping_archive_members(self):
        for variant in ["duplicate", "symlink", "escape"]:
            with self.subTest(variant=variant):
                with tarfile.open(self.archive, "w:gz") as archive:
                    for _ in range(2 if variant == "duplicate" else 1):
                        member = tarfile.TarInfo(
                            "../escape" if variant == "escape" else "lib/adapter.so"
                        )
                        member.size = len(b"adapter")
                        if variant == "symlink":
                            member.type = tarfile.SYMTYPE
                            member.linkname = "../external"
                            member.size = 0
                        archive.addfile(member, io.BytesIO(b"adapter"))
                with self.assertRaises(package_cpu.PackageError):
                    installed_hashes.archive_blake3(self.archive, self.manifest)

    def test_signed_catalog_binds_enriched_manifest_without_changing_archive(self):
        key, public_key = fixtures.ReleasePackagingTests.signing_key(self.root)
        for profile in release.PROFILES:
            with self.subTest(profile=profile):
                handoff = self.root / profile
                handoff.mkdir()
                filename = f"kapsl-backend-onnx-{profile}-0.2.4-linux-x86_64.tar.gz"
                archive = handoff / filename
                archive.write_bytes(self.archive.read_bytes())
                digest = release.sha256_file(archive)
                manifest = {
                    **self.manifest,
                    "backend": "onnx",
                    "profile": profile,
                    "pack_version": "0.2.0",
                    "compatible_kapsl": "=0.2.4",
                    "platform": "linux-x86_64",
                }
                manifest_path = handoff / f"{filename}.manifest.json"
                manifest_bytes = json.dumps(manifest).encode()
                manifest_path.write_bytes(manifest_bytes)
                (handoff / f"{filename}.sha256").write_text(f"{digest}  {filename}\n")
                (handoff / f"{filename}.sig").write_text(
                    package_cpu.sign_artifact(key, public_key, digest) + "\n"
                )
                output = self.root / f"release-{profile}"
                release.prepare_profile(
                    argparse.Namespace(
                        profile=profile,
                        adapter_version="0.2.0",
                        kapsl_version="0.2.4",
                        release_tag="kapsl-ort-packs-v0.2.0-kapsl-v0.2.4",
                        repository="kapsl-runtime/kapsl-integrations",
                        source_commit="1" * 40,
                        signing_key=key,
                        expected_public_key=public_key,
                        directory=handoff,
                        output_dir=output,
                        part_bytes=4096,
                        consume_archive=False,
                        installed_blake3=True,
                    )
                )
                self.assertEqual(release.sha256_file(archive), digest)
                self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
                published = output / manifest_path.name
                metadata = json.loads(published.read_text())
                self.assertEqual(metadata["files"], manifest["files"])
                self.assertEqual(
                    metadata["files_blake3"],
                    {p: blake3(b).hexdigest() for p, b in self.members.items()},
                )
                catalog_path = output / f"{filename}.release.json"
                catalog = json.loads(catalog_path.read_text())
                self.assertEqual(catalog["archive"]["sha256"], digest)
                self.assertEqual(
                    catalog["archive"]["manifest"]["sha256"],
                    release.sha256_file(published),
                )
                self.assertEqual(
                    catalog["archive"]["manifest"]["size"], published.stat().st_size
                )
                release.verify_signature(
                    public_key,
                    release.sha256_file(catalog_path),
                    release.parse_signature(output / f"{catalog_path.name}.sig"),
                )


if __name__ == "__main__":
    unittest.main()
