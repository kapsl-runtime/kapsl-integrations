from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGING_ROOT))

import fetch_ort_notices  # noqa: E402
import generate_cargo_notices  # noqa: E402
import package_cpu  # noqa: E402


def fake_elf() -> bytes:
    payload = bytearray(128)
    payload[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", payload, 18, 62)
    return bytes(payload)


def write_pack_inputs(root: Path) -> dict[str, Path]:
    values = {
        "library": ("adapter.so", fake_elf()),
        "cargo_lock": ("Cargo.lock", b"lock\n"),
        "rust_toolchain": ("rust-toolchain.toml", b"toolchain\n"),
        "kapsl_license": ("LICENSE", b"Apache-2.0\n"),
        "kapsl_notice": ("NOTICE", b"Copyright Kapsl\n"),
        "ort_license": ("ORT-LICENSE", b"MIT License\n"),
        "ort_notices": ("ORT-NOTICES", b"fixture ORT notices\n"),
        "cargo_notices": (
            "RUST-NOTICES",
            b"KAPSL ORT ADAPTER RUST DEPENDENCY NOTICES\nfixture\n",
        ),
    }
    result: dict[str, Path] = {}
    for key, (name, payload) in values.items():
        path = root / name
        path.write_bytes(payload)
        result[key] = path
    return result


class PackArchiveTests(unittest.TestCase):
    def test_pack_is_deterministic_and_matches_engine_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_pack_inputs(root)
            arguments = {
                "library_path": inputs["library"],
                "kapsl_version": "0.2.3",
                "source_commit": "1" * 40,
                "source_date_epoch": 1_700_000_000,
                "cargo_lock_path": inputs["cargo_lock"],
                "rust_toolchain_path": inputs["rust_toolchain"],
                "kapsl_license_path": inputs["kapsl_license"],
                "kapsl_notice_path": inputs["kapsl_notice"],
                "ort_license_path": inputs["ort_license"],
                "ort_notices_path": inputs["ort_notices"],
                "cargo_notices_path": inputs["cargo_notices"],
                "needed_libraries": ["libc.so.6", "libm.so.6"],
            }
            with mock.patch.object(package_cpu, "validate_notice"):
                first = package_cpu.create_pack(output_dir=root / "first", **arguments)
                second = package_cpu.create_pack(
                    output_dir=root / "second", **arguments
                )

            self.assertEqual(
                first["archive"].read_bytes(), second["archive"].read_bytes()
            )
            self.assertEqual(
                first["manifest"].read_bytes(), second["manifest"].read_bytes()
            )
            self.assertEqual(
                first["checksum"].read_bytes(), second["checksum"].read_bytes()
            )
            manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["backend"], "onnx")
            self.assertEqual(manifest["profile"], "cpu")
            self.assertEqual(manifest["compatible_kapsl"], "=0.2.3")
            self.assertEqual(manifest["accelerator_profile"], "cpu")
            self.assertEqual(manifest["execution_mode"], "native")
            self.assertEqual(manifest["entrypoint"], package_cpu.ENTRYPOINT)
            self.assertIn(package_cpu.ENTRYPOINT, manifest["files"])
            self.assertGreaterEqual(len(manifest["licenses"]), 5)

            with tarfile.open(first["archive"], "r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
                self.assertIn("backend-pack.json", names)
                self.assertIn("provenance.json", names)
                self.assertIn(package_cpu.ENTRYPOINT, names)
                provenance = json.load(archive.extractfile("provenance.json"))
            self.assertEqual(provenance["source_commit"], "1" * 40)
            self.assertEqual(
                provenance["onnx_runtime"]["version"],
                fetch_ort_notices.ORT_RUNTIME_VERSION,
            )
            self.assertEqual(
                provenance["entrypoint"]["needed_libraries"],
                ["libc.so.6", "libm.so.6"],
            )
            checksum = first["checksum"].read_text(encoding="ascii").split()[0]
            self.assertEqual(checksum, package_cpu.sha256_file(first["archive"]))

    def test_linux_library_contract_rejects_non_system_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "adapter.so"
            library.write_bytes(fake_elf())
            with mock.patch.object(
                package_cpu,
                "run_tool",
                side_effect=[
                    "0000000000001000 T kapsl_backend_v1\n",
                    " 0x1 (NEEDED) Shared library: [libonnxruntime.so.1]\n",
                ],
            ):
                with self.assertRaisesRegex(
                    package_cpu.PackageError, "unpackaged non-system dependencies"
                ):
                    package_cpu.inspect_linux_library(library)

    def test_linux_library_contract_accepts_allowlisted_system_dependencies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "adapter.so"
            library.write_bytes(fake_elf())
            with mock.patch.object(
                package_cpu,
                "run_tool",
                side_effect=[
                    "0000000000001000 T kapsl_backend_v1\n",
                    "\n".join(
                        [
                            " 0x1 (NEEDED) Shared library: [libc.so.6]",
                            " 0x1 (NEEDED) Shared library: [libm.so.6]",
                            " 0x1d (RUNPATH) Library runpath: [$ORIGIN]",
                        ]
                    ),
                ],
            ):
                self.assertEqual(
                    package_cpu.inspect_linux_library(library),
                    ["libc.so.6", "libm.so.6"],
                )


class SignatureTests(unittest.TestCase):
    def test_artifact_signature_is_domain_separated_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = root / "key.pem"
            public_key = root / "public.pem"
            subprocess.run(
                ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "openssl",
                    "pkey",
                    "-in",
                    str(key),
                    "-pubout",
                    "-out",
                    str(public_key),
                ],
                check=True,
                capture_output=True,
            )
            expected = package_cpu.signing_public_key(key)
            digest = "a" * 64
            encoded = package_cpu.sign_artifact(key, expected, digest)
            signature = base64.b64decode(encoded.removeprefix("ed25519:"))
            message = package_cpu.ARTIFACT_DOMAIN + f"sha256:{digest}".encode()
            message_path = root / "message"
            signature_path = root / "signature"
            message_path.write_bytes(message)
            signature_path.write_bytes(signature)
            subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_key),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ],
                check=True,
                capture_output=True,
            )
            with self.assertRaisesRegex(
                package_cpu.PackageError, "does not match the expected public key"
            ):
                package_cpu.sign_artifact(
                    key, base64.b64encode(b"x" * 32).decode(), digest
                )


class CargoNoticeTests(unittest.TestCase):
    def test_notices_include_only_normal_linked_dependencies_without_host_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            dependency = root / "dependency"
            dev_dependency = root / "dev-dependency"
            for directory in (workspace, dependency, dev_dependency):
                directory.mkdir()
                (directory / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
            (dependency / "LICENSE-MIT").write_text("MIT fixture\n", encoding="utf-8")
            (dev_dependency / "LICENSE").write_text("dev fixture\n", encoding="utf-8")
            workspace_license = root / "LICENSE"
            workspace_license.write_text("Apache fixture\n", encoding="utf-8")
            metadata = {
                "workspace_members": ["root"],
                "packages": [
                    {
                        "id": "root",
                        "name": "kapsl-backend-ort",
                        "version": "0.1.0",
                        "manifest_path": str(workspace / "Cargo.toml"),
                        "license": "Apache-2.0",
                    },
                    {
                        "id": "dep",
                        "name": "example",
                        "version": "1.0.0",
                        "manifest_path": str(dependency / "Cargo.toml"),
                        "license": "MIT",
                        "source": "registry+https://example.invalid/index",
                        "repository": "https://example.invalid/example",
                    },
                    {
                        "id": "dev",
                        "name": "dev-only",
                        "version": "1.0.0",
                        "manifest_path": str(dev_dependency / "Cargo.toml"),
                        "license": "MIT",
                    },
                ],
                "resolve": {
                    "nodes": [
                        {
                            "id": "root",
                            "deps": [
                                {
                                    "pkg": "dep",
                                    "dep_kinds": [{"kind": None, "target": None}],
                                },
                                {
                                    "pkg": "dev",
                                    "dep_kinds": [{"kind": "dev", "target": None}],
                                },
                            ],
                        },
                        {"id": "dep", "deps": []},
                        {"id": "dev", "deps": []},
                    ]
                },
            }
            notices = generate_cargo_notices.render_notices(
                metadata,
                "kapsl-backend-ort",
                "x86_64-unknown-linux-gnu",
                workspace_license,
            )
            self.assertIn("example 1.0.0", notices)
            self.assertIn("MIT fixture", notices)
            self.assertNotIn("dev-only", notices)
            self.assertNotIn(str(root), notices)


class FetchedNoticeTests(unittest.TestCase):
    def test_notice_digest_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(fetch_ort_notices.NoticeFetchError, "SHA-256"):
            fetch_ort_notices.validate_notice(b"tampered")


if __name__ == "__main__":
    unittest.main()
