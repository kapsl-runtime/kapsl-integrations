from __future__ import annotations

import base64
import gzip
import hashlib
import io
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
import fetch_ort_runtime  # noqa: E402
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
        "runtime_library": (fetch_ort_runtime.RUNTIME_SONAME, fake_elf()),
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
                "runtime_library_path": inputs["runtime_library"],
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
                "adapter_inspection": {
                    "needed_libraries": [
                        "libc.so.6",
                        "libm.so.6",
                        fetch_ort_runtime.RUNTIME_SONAME,
                    ],
                    "maximum_required_glibc": "2.34",
                },
                "runtime_inspection": {
                    "needed_libraries": ["libc.so.6", "libm.so.6"],
                    "soname": fetch_ort_runtime.RUNTIME_SONAME,
                    "sha256": package_cpu.sha256_file(inputs["runtime_library"]),
                    "maximum_required_glibc": "2.27",
                },
            }
            runtime_digest = package_cpu.sha256_file(inputs["runtime_library"])
            with (
                mock.patch.object(package_cpu, "validate_notice"),
                mock.patch.object(
                    package_cpu, "RUNTIME_LIBRARY_SHA256", runtime_digest
                ),
            ):
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
            self.assertEqual(manifest["adapter_abi"], package_cpu.ADAPTER_ABI)
            self.assertEqual(manifest["compatible_kapsl"], "=0.2.3")
            self.assertEqual(manifest["accelerator_profile"], "cpu")
            self.assertEqual(manifest["execution_mode"], "native")
            self.assertEqual(manifest["entrypoint"], package_cpu.ENTRYPOINT)
            self.assertIn(package_cpu.ENTRYPOINT, manifest["files"])
            self.assertIn(fetch_ort_runtime.RUNTIME_SONAME, manifest["files"])
            self.assertGreaterEqual(len(manifest["licenses"]), 5)

            with tarfile.open(first["archive"], "r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
                self.assertIn("backend-pack.json", names)
                self.assertIn("provenance.json", names)
                self.assertIn(package_cpu.ENTRYPOINT, names)
                self.assertIn(fetch_ort_runtime.RUNTIME_SONAME, names)
                provenance = json.load(archive.extractfile("provenance.json"))
                payload = json.load(archive.extractfile("backend-pack.json"))
            self.assertEqual(provenance["source_commit"], "1" * 40)
            self.assertEqual(payload["adapter_abi"], package_cpu.ADAPTER_ABI)
            self.assertEqual(
                provenance["adapter"]["adapter_abi"], package_cpu.ADAPTER_ABI
            )
            self.assertEqual(
                provenance["onnx_runtime"]["version"],
                fetch_ort_notices.ORT_RUNTIME_VERSION,
            )
            self.assertEqual(
                provenance["entrypoint"]["needed_libraries"],
                ["libc.so.6", "libm.so.6", fetch_ort_runtime.RUNTIME_SONAME],
            )
            self.assertEqual(
                provenance["onnx_runtime"]["library"]["path"],
                fetch_ort_runtime.RUNTIME_SONAME,
            )
            self.assertEqual(
                provenance["onnx_runtime"]["distribution_sha256"],
                fetch_ort_runtime.RUNTIME_ARCHIVE_SHA256,
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
                    "\n".join(
                        [
                            " 0x1 (NEEDED) Shared library: [libonnxruntime.so.1]",
                            " 0x1 (NEEDED) Shared library: [libunexpected.so.1]",
                            " 0x1d (RUNPATH) Library runpath: [$ORIGIN]",
                        ]
                    ),
                ],
            ):
                with self.assertRaisesRegex(
                    package_cpu.PackageError, "unpackaged non-system dependencies"
                ):
                    package_cpu.inspect_linux_adapter(library)

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
                            " 0x1 (NEEDED) Shared library: [ld-linux-x86-64.so.2]",
                            " 0x1 (NEEDED) Shared library: [libc.so.6]",
                            " 0x1 (NEEDED) Shared library: [libm.so.6]",
                            " 0x1 (NEEDED) Shared library: [libonnxruntime.so.1]",
                            " 0x1d (RUNPATH) Library runpath: [$ORIGIN]",
                        ]
                    ),
                    "Name: GLIBC_2.2.5\nName: GLIBC_2.34\n",
                    " 1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND malloc@GLIBC_2.2.5\n",
                ],
            ):
                self.assertEqual(
                    package_cpu.inspect_linux_adapter(library),
                    {
                        "needed_libraries": [
                            "ld-linux-x86-64.so.2",
                            "libc.so.6",
                            "libm.so.6",
                            "libonnxruntime.so.1",
                        ],
                        "maximum_required_glibc": "2.34",
                    },
                )

    def test_linux_library_contract_rejects_c23_glibc_imports(self) -> None:
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
                            " 0x1 (NEEDED) Shared library: [libonnxruntime.so.1]",
                            " 0x1d (RUNPATH) Library runpath: [$ORIGIN]",
                        ]
                    ),
                    "Name: GLIBC_2.34\n",
                    " 1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND __isoc23_strtoll\n",
                ],
            ):
                with self.assertRaisesRegex(
                    package_cpu.PackageError, "unsupported C23 glibc symbols"
                ):
                    package_cpu.inspect_linux_adapter(library)

    def test_runtime_contract_requires_pinned_soname_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / fetch_ort_runtime.RUNTIME_SONAME
            library.write_bytes(fake_elf())
            digest = package_cpu.sha256_file(library)
            with (
                mock.patch.object(package_cpu, "RUNTIME_LIBRARY_SHA256", digest),
                mock.patch.object(
                    package_cpu,
                    "run_tool",
                    side_effect=[
                        "\n".join(
                            [
                                " 0x1 (NEEDED) Shared library: [libc.so.6]",
                                " 0xe (SONAME) Library soname: [libonnxruntime.so.1]",
                                " 0x1d (RUNPATH) Library runpath: [$ORIGIN]",
                            ]
                        ),
                        "Name: GLIBC_2.2.5\nName: GLIBC_2.27\n",
                        " 1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND malloc@GLIBC_2.2.5\n",
                    ],
                ),
            ):
                self.assertEqual(
                    package_cpu.inspect_linux_runtime(library),
                    {
                        "needed_libraries": ["libc.so.6"],
                        "soname": fetch_ort_runtime.RUNTIME_SONAME,
                        "sha256": digest,
                        "maximum_required_glibc": "2.27",
                    },
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


class FetchedRuntimeTests(unittest.TestCase):
    def test_runtime_archive_is_authenticated_and_extracts_one_regular_file(
        self,
    ) -> None:
        runtime = fake_elf()
        buffer = io.BytesIO()
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=buffer, mtime=0
        ) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w") as archive:
                info = tarfile.TarInfo(fetch_ort_runtime.RUNTIME_ARCHIVE_MEMBER)
                info.size = len(runtime)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(runtime))
        archive_payload = buffer.getvalue()
        with (
            mock.patch.object(
                fetch_ort_runtime,
                "RUNTIME_ARCHIVE_SHA256",
                hashlib.sha256(archive_payload).hexdigest(),
            ),
            mock.patch.object(
                fetch_ort_runtime,
                "RUNTIME_LIBRARY_SHA256",
                hashlib.sha256(runtime).hexdigest(),
            ),
            mock.patch.object(fetch_ort_runtime, "RUNTIME_LIBRARY_BYTES", len(runtime)),
        ):
            self.assertEqual(
                fetch_ort_runtime.extract_runtime(archive_payload), runtime
            )

    def test_runtime_archive_digest_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(fetch_ort_runtime.RuntimeFetchError, "SHA-256"):
            fetch_ort_runtime.validate_archive(b"tampered")


class PackagingEntrypointTests(unittest.TestCase):
    def test_disables_bytecode_before_first_python_invocation(self) -> None:
        script = (Path(__file__).parents[1] / "build_cpu_pack.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            script.index("export PYTHONDONTWRITEBYTECODE=1"),
            script.index(
                'python3 "$repo_root/integrations/ort/packaging/fetch_ort_runtime.py"'
            ),
        )


if __name__ == "__main__":
    unittest.main()
