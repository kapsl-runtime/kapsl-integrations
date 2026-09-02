from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PACKAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGING_ROOT))

import fetch_ort_gpu_runtime  # noqa: E402
import fetch_ort_notices  # noqa: E402
import fetch_ort_runtime  # noqa: E402
import fetch_tensorrt_runtime  # noqa: E402
import generate_cargo_notices  # noqa: E402
import package_accelerator  # noqa: E402
import package_cpu  # noqa: E402
import release as release_packaging  # noqa: E402


def fixture_archive(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=1) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, payload in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class GpuRuntimeFetchTests(unittest.TestCase):
    def test_extracts_only_exact_pinned_runtime_files(self) -> None:
        members = {
            "fixture/core.so": b"core",
            "fixture/cuda.so": b"cuda",
        }
        archive = fixture_archive(members)
        files = {
            "core.so": fetch_ort_gpu_runtime.RuntimeFile(
                "fixture/core.so", hashlib.sha256(b"core").hexdigest(), 4
            ),
            "cuda.so": fetch_ort_gpu_runtime.RuntimeFile(
                "fixture/cuda.so", hashlib.sha256(b"cuda").hexdigest(), 4
            ),
        }
        with (
            mock.patch.object(
                fetch_ort_gpu_runtime,
                "GPU_RUNTIME_ARCHIVE_BYTES",
                len(archive),
            ),
            mock.patch.object(
                fetch_ort_gpu_runtime,
                "GPU_RUNTIME_ARCHIVE_SHA256",
                hashlib.sha256(archive).hexdigest(),
            ),
            mock.patch.object(fetch_ort_gpu_runtime, "GPU_RUNTIME_FILES", files),
        ):
            self.assertEqual(
                fetch_ort_gpu_runtime.extract_runtime(archive),
                {"core.so": b"core", "cuda.so": b"cuda"},
            )

    def test_rejects_unpinned_archive(self) -> None:
        with self.assertRaises(fetch_ort_gpu_runtime.GpuRuntimeFetchError):
            fetch_ort_gpu_runtime.validate_archive(b"not-the-release")


class TensorRtRuntimeFetchTests(unittest.TestCase):
    def test_authenticates_and_extracts_only_pinned_zip_members(self) -> None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("runtime/libnvinfer.so.10", b"runtime")
            output.writestr("runtime/windows-only.so", b"windows")
        payload = archive.getvalue()
        source = fetch_tensorrt_runtime.MemoryRangeSource(payload)
        expected = fetch_tensorrt_runtime.RuntimeFile(
            member="runtime/libnvinfer.so.10",
            output_name="libnvinfer.so.10",
            sha256=hashlib.sha256(b"runtime").hexdigest(),
            size=len(b"runtime"),
            kind="runtime",
        )

        fetch_tensorrt_runtime.verify_distribution(
            source, hashlib.sha256(payload).hexdigest()
        )
        members = fetch_tensorrt_runtime.parse_central_directory(source)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / expected.output_name
            fetch_tensorrt_runtime.extract_member(
                source, members[expected.member], expected, destination
            )
            self.assertEqual(destination.read_bytes(), b"runtime")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)

    def test_rejects_a_distribution_digest_mismatch(self) -> None:
        source = fetch_tensorrt_runtime.MemoryRangeSource(b"not-the-wheel")
        with self.assertRaises(fetch_tensorrt_runtime.TensorRtFetchError):
            fetch_tensorrt_runtime.verify_distribution(source, "0" * 64)

    def test_resolves_zip64_local_offsets(self) -> None:
        extra = struct.pack("<HHQ", 0x0001, 8, 3_103_274_258)
        self.assertEqual(
            fetch_tensorrt_runtime.resolve_zip64_fields(
                name="LICENSE.txt",
                size=46_961,
                compressed_size=14_720,
                local_header_offset=0xFFFF_FFFF,
                disk=0,
                extra=extra,
            ),
            (46_961, 14_720, 3_103_274_258, 0),
        )


class AcceleratorPackagingTests(unittest.TestCase):
    @staticmethod
    def candidate(
        name: str, origin: str = "fixture"
    ) -> package_accelerator.CandidateLibrary:
        return package_accelerator.CandidateLibrary(
            Path("/fixture") / name,
            origin,
            hashlib.sha256(name.encode()).hexdigest(),
        )

    def test_dependency_closure_keeps_pack_libraries_and_host_driver_external(
        self,
    ) -> None:
        names = {
            package_accelerator.ENTRYPOINT,
            package_accelerator.RUNTIME_SONAME,
            "libonnxruntime_providers_cuda.so",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libz.so.1",
        }
        candidates = {name: self.candidate(name) for name in names}
        dependencies = {
            package_accelerator.ENTRYPOINT: [package_accelerator.RUNTIME_SONAME],
            "libonnxruntime_providers_cuda.so": [
                "libcublas.so.12",
                "libcudnn.so.9",
                "libcuda.so.1",
                "libc.so.6",
            ],
            "libcublas.so.12": ["libc.so.6"],
            "libcudnn.so.9": ["libz.so.1", "libc.so.6"],
            "libz.so.1": ["libc.so.6"],
            package_accelerator.RUNTIME_SONAME: ["libc.so.6"],
        }

        selected = package_accelerator.resolve_dependency_closure(
            candidates,
            {
                package_accelerator.ENTRYPOINT,
                "libonnxruntime_providers_cuda.so",
            },
            lambda candidate: dependencies[candidate.path.name],
        )

        self.assertEqual(selected, names)
        self.assertNotIn("libcuda.so.1", selected)

    def test_dependency_closure_rejects_missing_user_space_library(self) -> None:
        candidate = self.candidate("provider.so")
        with self.assertRaisesRegex(package_accelerator.PackageError, "libmissing"):
            package_accelerator.resolve_dependency_closure(
                {"provider.so": candidate},
                {"provider.so"},
                lambda _: ["libmissing.so.1"],
            )

    def test_staging_is_owner_only_while_archive_mode_remains_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "provider.so"
            source.write_bytes(b"provider")
            candidate = package_accelerator.CandidateLibrary(
                source,
                "fixture",
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )

            with mock.patch.object(package_accelerator, "run_tool"):
                staged = package_accelerator.stage_and_normalize(
                    {source.name: candidate},
                    {source.name},
                    root / "staged",
                )

            target = staged[source.name]
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                package_accelerator.PackEntry.from_path(target).mode,
                0o755,
            )

    def test_release_staging_can_consume_scratch_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "provider.so"
            source.write_bytes(b"provider")
            candidate = package_accelerator.CandidateLibrary(
                source,
                "fixture",
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )

            with mock.patch.object(package_accelerator, "run_tool"):
                staged = package_accelerator.stage_and_normalize(
                    {source.name: candidate},
                    {source.name},
                    root / "staged",
                    consume_sources=True,
                )

            self.assertFalse(source.exists())
            self.assertEqual(staged[source.name].read_bytes(), b"provider")

    def test_tensorrt_roots_include_linux_builder_resource_only(self) -> None:
        names = {
            package_accelerator.ENTRYPOINT,
            *package_accelerator.PROFILES["tensorrt10"].ort_libraries,
            "libcudnn.so.9",
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
            "libnvinfer_builder_resource.so.10.9.0",
            "libnvinfer_builder_resource_win.so.10.9.0",
        }
        candidates = {
            name: self.candidate(
                name,
                "tensorrt"
                if name.startswith(("libnvinfer", "libnvonnx"))
                else "fixture",
            )
            for name in names
        }

        roots = package_accelerator.root_library_names(
            package_accelerator.PROFILES["tensorrt10"], candidates
        )

        self.assertIn("libnvinfer_builder_resource.so.10.9.0", roots)
        self.assertNotIn("libnvinfer_builder_resource_win.so.10.9.0", roots)

    def test_runtime_provenance_authenticates_every_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "libcudart.so.12"
            library.write_bytes(b"cuda")
            candidate = package_accelerator.CandidateLibrary(
                library,
                "cuda",
                hashlib.sha256(b"cuda").hexdigest(),
            )
            provenance = root / "source.json"
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": {
                            library.name: {
                                "sha256": candidate.sha256,
                                "size": library.stat().st_size,
                            }
                        },
                    }
                )
            )

            payload = package_accelerator.validate_runtime_provenance(
                provenance, "cuda", {library.name: candidate}
            )

            self.assertEqual(payload["schema_version"], 1)

    def test_runtime_provenance_authenticates_packaged_licenses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            license_path = root / "ZLIB-COPYRIGHT"
            license_path.write_bytes(b"zlib license")
            provenance = root / "source.json"
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": {
                            license_path.name: {
                                "sha256": hashlib.sha256(
                                    license_path.read_bytes()
                                ).hexdigest(),
                                "size": license_path.stat().st_size,
                            }
                        },
                    }
                )
            )

            package_accelerator.validate_runtime_provenance(
                provenance,
                "cuda",
                {},
                {license_path.name: license_path},
            )
            license_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                package_accelerator.PackageError, "does not authenticate ZLIB-COPYRIGHT"
            ):
                package_accelerator.validate_runtime_provenance(
                    provenance,
                    "cuda",
                    {},
                    {license_path.name: license_path},
                )

    def test_profile_manifest_is_exact_and_governed(self) -> None:
        entries = {
            package_accelerator.ENTRYPOINT: package_accelerator.PackEntry.from_bytes(
                b"adapter", 0o755
            ),
            "backend-pack.json": package_accelerator.PackEntry.from_bytes(b"{}"),
            package_cpu.PROVENANCE_PATH: package_accelerator.PackEntry.from_bytes(
                b"{}"
            ),
        }
        manifest = package_accelerator.manifest_template(
            package_accelerator.PROFILES["tensorrt10"], entries, "0.2.3"
        )
        self.assertEqual(manifest["profile"], "tensorrt10")
        self.assertEqual(manifest["accelerator_profile"], "tensorrt")
        self.assertEqual(manifest["adapter_abi"], "kapsl-backend-v1")
        self.assertEqual(manifest["minimum_cuda"], "12.0")
        self.assertEqual(manifest["minimum_tensorrt"], "10.9")
        self.assertEqual(manifest["formats"], ["onnx"])
        self.assertIn("generate", manifest["tasks"])
        self.assertTrue(manifest["capabilities"]["governed_device_allocator"])
        self.assertTrue(manifest["capabilities"]["scoped_device_allocator"])
        self.assertFalse(manifest["capabilities"]["kv_participation"])
        self.assertEqual(
            manifest["accelerator_requirements"]["execution_providers"],
            ["tensorrt", "cuda"],
        )
        self.assertFalse(manifest["accelerator_requirements"]["implicit_cpu_fallback"])
        self.assertEqual(
            manifest["memory_behavior"]["allocation_scope"],
            "kapsl-scoped-device-allocator-v1",
        )
        self.assertEqual(
            manifest["artifact_authentication"]["signature_location"], "detached"
        )
        self.assertEqual(manifest["provenance"]["path"], package_cpu.PROVENANCE_PATH)
        self.assertGreater(manifest["memory"]["accelerator_bytes"], 0)

    def test_streaming_archive_is_deterministic_and_validates(self) -> None:
        profile = package_accelerator.PROFILES["cuda12"]
        payload = {
            "schema_version": 1,
            "backend": "onnx",
            "profile": "cuda12",
            "pack_version": package_cpu.ADAPTER_VERSION,
            "runtime_abi": 1,
            "adapter_abi": "kapsl-backend-v1",
            "platform": "linux-x86_64",
            "execution_mode": "native",
            "entrypoint": package_accelerator.ENTRYPOINT,
            "accelerator_requirements": package_accelerator.accelerator_requirements(
                profile
            ),
            **package_cpu.common_pack_contract(True),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / package_accelerator.ENTRYPOINT
            library.write_bytes(b"streamed adapter")
            entries = {
                package_accelerator.ENTRYPOINT: package_accelerator.PackEntry.from_path(
                    library
                ),
                "backend-pack.json": package_accelerator.PackEntry.from_bytes(
                    (json.dumps(payload, sort_keys=True) + "\n").encode()
                ),
                package_cpu.PROVENANCE_PATH: package_accelerator.PackEntry.from_bytes(
                    b"{}"
                ),
            }
            manifest = package_accelerator.manifest_template(profile, entries, "0.2.3")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            for archive in (first, second):
                package_accelerator.write_streaming_archive(
                    archive, entries, 1_700_000_000
                )
                package_accelerator.validate_streaming_archive(
                    archive, entries, manifest, 1_700_000_000
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())

            consumed = root / "consumed.tar.gz"
            package_accelerator.write_streaming_archive(
                consumed,
                entries,
                1_700_000_000,
                consume_paths=True,
            )
            self.assertFalse(library.exists())
            package_accelerator.validate_streaming_archive(
                consumed, entries, manifest, 1_700_000_000
            )
            self.assertEqual(first.read_bytes(), consumed.read_bytes())

    def test_driver_library_names_are_host_owned(self) -> None:
        self.assertTrue(package_accelerator.is_driver_library("libcuda.so.1"))
        self.assertTrue(
            package_accelerator.is_driver_library("libnvidia-ml.so.560.28.03")
        )
        self.assertFalse(package_accelerator.is_driver_library("libcudart.so.12"))


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
    def test_elf_inspection_reads_only_the_fixed_header_from_large_libraries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "large.so"
            with library.open("wb") as stream:
                stream.write(fake_elf())
                stream.truncate(package_cpu.MAX_INPUT_BYTES + 1)

            package_cpu.inspect_elf_header(library, "large library")

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
            self.assertEqual(manifest["pack_version"], "0.2.0")
            self.assertEqual(manifest["formats"], ["onnx"])
            self.assertIn("generate", manifest["tasks"])
            self.assertFalse(manifest["capabilities"]["governed_device_allocator"])
            self.assertFalse(manifest["capabilities"]["scoped_device_allocator"])
            self.assertEqual(
                manifest["accelerator_requirements"]["execution_providers"],
                ["cpu"],
            )
            self.assertEqual(manifest["memory_behavior"]["device_allocation"], "none")
            self.assertIn(package_cpu.PROVENANCE_PATH, manifest["files"])
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
            for field in package_cpu.PACK_CONTRACT_FIELDS:
                self.assertEqual(payload[field], manifest[field], field)
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
            release_packaging.verify_signature(expected, digest, encoded)
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


class ReleasePackagingTests(unittest.TestCase):
    @staticmethod
    def signing_key(root: Path) -> tuple[Path, str]:
        key = root / "key.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)],
            check=True,
            capture_output=True,
        )
        return key, package_cpu.signing_public_key(key)

    def test_stable_release_ref_drives_all_publish_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "github-output"
            args = argparse.Namespace(
                repository_root=PACKAGING_ROOT.parents[2],
                event_name="push",
                ref_type="tag",
                ref_name="kapsl-ort-packs-v0.2.0-kapsl-v0.2.4",
                requested_kapsl_version="",
                requested_profile="all",
                requested_publish="false",
                github_output=output,
            )

            release_packaging.validate_release_ref(args)

            values = dict(
                line.split("=", 1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(values["publish"], "true")
            self.assertEqual(values["kapsl_version"], "0.2.4")
            self.assertEqual(
                json.loads(values["matrix"]),
                {"profile": ["cpu", "cuda12", "tensorrt10"]},
            )

    def test_prerelease_ref_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                repository_root=PACKAGING_ROOT.parents[2],
                event_name="push",
                ref_type="tag",
                ref_name="kapsl-ort-packs-v0.2.0-beta.1-kapsl-v0.2.4",
                requested_kapsl_version="",
                requested_profile="all",
                requested_publish="false",
                github_output=Path(temporary) / "github-output",
            )
            with self.assertRaises(package_cpu.PackageError):
                release_packaging.validate_release_ref(args)

    def test_release_archive_is_split_without_changing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "pack.tar.gz"
            payload = b"0123456789abcdef"
            archive.write_bytes(payload)
            output = root / "parts"
            output.mkdir()

            parts = release_packaging.split_archive(archive, output, 7)

            self.assertEqual([item["size"] for item in parts], [7, 7, 2])
            reconstructed = b"".join(
                (output / item["name"]).read_bytes() for item in parts
            )
            self.assertEqual(reconstructed, payload)

    def test_signed_profile_catalogs_assemble_into_signed_release_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key, public_key = self.signing_key(root)
            catalogs = root / "catalogs"
            catalogs.mkdir()
            github_assets = []
            release_tag = "kapsl-ort-packs-v0.2.0-kapsl-v0.2.4"
            source_commit = "1" * 40
            for profile in release_packaging.PROFILES:
                handoff = root / f"handoff-{profile}"
                handoff.mkdir()
                filename = f"kapsl-backend-onnx-{profile}-0.2.4-linux-x86_64.tar.gz"
                archive = handoff / filename
                archive.write_bytes((profile * 8).encode())
                digest = release_packaging.sha256_file(archive)
                (handoff / f"{filename}.manifest.json").write_text(
                    json.dumps(
                        {
                            "backend": "onnx",
                            "profile": profile,
                            "pack_version": "0.2.0",
                            "compatible_kapsl": "=0.2.4",
                            "platform": "linux-x86_64",
                        }
                    ),
                    encoding="utf-8",
                )
                (handoff / f"{filename}.sha256").write_text(
                    f"{digest}  {filename}\n", encoding="ascii"
                )
                (handoff / f"{filename}.sig").write_text(
                    package_cpu.sign_artifact(key, public_key, digest) + "\n",
                    encoding="ascii",
                )
                output = root / f"release-{profile}"
                release_packaging.prepare_profile(
                    argparse.Namespace(
                        profile=profile,
                        adapter_version="0.2.0",
                        kapsl_version="0.2.4",
                        release_tag=release_tag,
                        repository="kapsl-runtime/kapsl-integrations",
                        source_commit=source_commit,
                        signing_key=key,
                        expected_public_key=public_key,
                        directory=handoff,
                        output_dir=output,
                        part_bytes=7,
                        consume_archive=True,
                    )
                )
                self.assertFalse(archive.exists())
                for path in output.iterdir():
                    github_assets.append(
                        {
                            "name": path.name,
                            "size": path.stat().st_size,
                            "digest": f"sha256:{release_packaging.sha256_file(path)}",
                            "state": "uploaded",
                        }
                    )
                for path in output.glob("*.release.json*"):
                    shutil.copyfile(path, catalogs / path.name)

            index_dir = root / "index"
            github_assets_path = root / "github-assets.json"
            github_assets_path.write_text(
                json.dumps({"assets": github_assets}), encoding="utf-8"
            )
            release_packaging.assemble_index(
                argparse.Namespace(
                    adapter_version="0.2.0",
                    kapsl_version="0.2.4",
                    release_tag=release_tag,
                    repository="kapsl-runtime/kapsl-integrations",
                    source_commit=source_commit,
                    signing_key=key,
                    expected_public_key=public_key,
                    input_dir=catalogs,
                    github_assets=github_assets_path,
                    output_dir=index_dir,
                )
            )
            index_path = next(index_dir.glob("*.release.json"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(list(index["profiles"]), list(release_packaging.PROFILES))
            release_packaging.verify_signature(
                public_key,
                release_packaging.sha256_file(index_path),
                release_packaging.parse_signature(
                    index_path.with_name(f"{index_path.name}.sig")
                ),
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

    def test_missing_crate_license_uses_only_an_exact_digest_pinned_supplement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "dependency"
            dependency.mkdir()
            (dependency / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
            license_path = root / "licenses" / "example-LICENSE"
            license_path.parent.mkdir()
            license_payload = b"MIT fixture supplement\n"
            license_path.write_bytes(license_payload)
            index_path = root / "rust-license-supplements.json"
            index_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "licenses": [
                            {
                                "name": "example",
                                "version": "1.2.3",
                                "license": "MIT",
                                "path": "licenses/example-LICENSE",
                                "sha256": hashlib.sha256(license_payload).hexdigest(),
                                "source": (
                                    "https://example.invalid/"
                                    "0123456789abcdef0123456789abcdef01234567/LICENSE"
                                ),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            supplements = generate_cargo_notices.load_supplemental_licenses(index_path)
            package = {
                "name": "example",
                "version": "1.2.3",
                "license": "MIT",
                "manifest_path": str(dependency / "Cargo.toml"),
            }
            self.assertEqual(
                generate_cargo_notices.license_paths(
                    package, root / "workspace-LICENSE", supplements
                ),
                [license_path.resolve()],
            )

            package["license"] = "Apache-2.0"
            with self.assertRaisesRegex(
                generate_cargo_notices.NoticeError, "no packaged license text"
            ):
                generate_cargo_notices.license_paths(
                    package, root / "workspace-LICENSE", supplements
                )

    def test_supplemental_license_index_rejects_digest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            license_path = root / "LICENSE"
            license_path.write_text("changed\n", encoding="utf-8")
            index_path = root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "licenses": [
                            {
                                "name": "example",
                                "version": "1.0.0",
                                "license": "MIT",
                                "path": "LICENSE",
                                "sha256": "0" * 64,
                                "source": "https://example.invalid/LICENSE",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                generate_cargo_notices.NoticeError, "SHA-256 mismatch"
            ):
                generate_cargo_notices.load_supplemental_licenses(index_path)


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

    def test_pack_release_workflow_is_host_only_and_fail_closed(self) -> None:
        workflow = (
            PACKAGING_ROOT.parents[2] / ".github/workflows/publish-ort-packs.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("gpu-device-pool", workflow)
        self.assertNotIn("--gpus", workflow)
        self.assertNotIn("self-hosted", workflow)
        self.assertIn("runs-on: ubuntu-22.04", workflow)
        self.assertIn("environment: ort-pack-release", workflow)
        self.assertIn("if: always()", workflow)
        self.assertIn("--draft=false", workflow)

    def test_cuda_runtime_collection_includes_pack_local_zlib_and_license(self) -> None:
        collector = (PACKAGING_ROOT / "collect_cuda_runtime.sh").read_text(
            encoding="utf-8"
        )
        builder = (PACKAGING_ROOT / "build_accelerator_packs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("/lib/x86_64-linux-gnu/libz.so.1", collector)
        self.assertIn("/usr/share/doc/zlib1g/copyright", collector)
        self.assertIn('--zlib-license "$zlib_license"', builder)


if __name__ == "__main__":
    unittest.main()
