#!/usr/bin/env python3
"""Build reproducible Linux x86_64 ORT CUDA and TensorRT backend packs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from fetch_ort_gpu_runtime import (
    GPU_RUNTIME_ARCHIVE_SHA256,
    GPU_RUNTIME_ARCHIVE_URL,
    GPU_RUNTIME_FILES,
)
from fetch_ort_notices import (
    NOTICE_SHA256,
    ORT_RUNTIME_VERSION,
    NoticeFetchError,
    validate_notice,
)
from package_cpu import (
    ADAPTER_ABI,
    ADAPTER_VERSION,
    ALLOWED_SYSTEM_LIBRARIES,
    ENTRYPOINT,
    HEX_COMMIT,
    MAX_GLIBC_VERSION,
    ORT_BINDING_VERSION,
    PACK_CONTRACT_FIELDS,
    PLATFORM,
    PROVENANCE_PATH,
    RUNTIME_ABI,
    RUNTIME_SONAME,
    RUNTIME_VERSION,
    RUST_TOOLCHAIN,
    SCHEMA_VERSION,
    TARGET,
    PackageError,
    atomic_write,
    common_pack_contract,
    inspect_elf_header,
    inspect_glibc_contract,
    json_bytes,
    parse_dynamic_contract,
    read_bounded,
    run_tool,
    sha256_bytes,
    sha256_file,
    sign_artifact,
    validate_source_contract,
)


MINIMUM_CUDA = "12.0"
MINIMUM_DRIVER = "560.28.03"
MINIMUM_TENSORRT = "10.9"
RUNPATH = "$ORIGIN"
DRIVER_LIBRARY = re.compile(r"(?:libcuda\.so(?:\..*)?|libnvidia-[^/]+\.so(?:\..*)?)")


@dataclass(frozen=True)
class AcceleratorProfile:
    name: str
    feature: str
    accelerator: str
    ort_libraries: tuple[str, ...]


PROFILES: Mapping[str, AcceleratorProfile] = {
    "cuda12": AcceleratorProfile(
        name="cuda12",
        feature="profile-cuda12",
        accelerator="cuda",
        ort_libraries=(
            RUNTIME_SONAME,
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_cuda.so",
        ),
    ),
    "tensorrt10": AcceleratorProfile(
        name="tensorrt10",
        feature="profile-tensorrt10",
        accelerator="tensorrt",
        ort_libraries=(
            RUNTIME_SONAME,
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_tensorrt.so",
        ),
    ),
}


def accelerator_requirements(profile: AcceleratorProfile) -> dict[str, Any]:
    providers = ["cuda"]
    requirements: dict[str, Any] = {
        "execution_providers": providers,
        "implicit_cpu_fallback": False,
        "kind": profile.accelerator,
        "minimum_cuda": MINIMUM_CUDA,
        "minimum_driver": MINIMUM_DRIVER,
    }
    if profile.name == "tensorrt10":
        requirements["execution_providers"] = ["tensorrt", "cuda"]
        requirements["minimum_tensorrt"] = MINIMUM_TENSORRT
    return requirements


@dataclass(frozen=True)
class CandidateLibrary:
    path: Path
    origin: str
    sha256: str


@dataclass(frozen=True)
class PackEntry:
    mode: int
    size: int
    sha256: str
    path: Path | None = None
    payload: bytes | None = None

    @classmethod
    def from_path(cls, path: Path, mode: int = 0o755) -> PackEntry:
        return cls(
            mode=mode,
            size=path.stat().st_size,
            sha256=sha256_file(path),
            path=path,
        )

    @classmethod
    def from_bytes(cls, payload: bytes, mode: int = 0o644) -> PackEntry:
        return cls(
            mode=mode,
            size=len(payload),
            sha256=sha256_bytes(payload),
            payload=payload,
        )

    def stream(self):
        if self.path is not None:
            return self.path.open("rb")
        if self.payload is None:
            raise PackageError("pack entry has neither a path nor an in-memory payload")
        return io.BytesIO(self.payload)


def is_driver_library(name: str) -> bool:
    return DRIVER_LIBRARY.fullmatch(name) is not None


def is_host_library(name: str) -> bool:
    return name in ALLOWED_SYSTEM_LIBRARIES or is_driver_library(name)


def runtime_library_paths(directory: Path, origin: str) -> list[CandidateLibrary]:
    if not directory.is_dir():
        raise PackageError(f"{origin} runtime directory is missing: {directory}")
    result: list[CandidateLibrary] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or ".so" not in path.name:
            continue
        if path.is_symlink():
            raise PackageError(
                f"{origin} runtime input must be flattened to regular files: {path}"
            )
        if path.name.startswith("libonnxruntime"):
            continue
        if is_driver_library(path.name):
            raise PackageError(
                f"{origin} runtime input contains host NVIDIA driver library {path.name}"
            )
        result.append(
            CandidateLibrary(
                path=path.absolute(), origin=origin, sha256=sha256_file(path)
            )
        )
    if not result:
        raise PackageError(f"{origin} runtime directory contains no shared libraries")
    return result


def verified_ort_libraries(
    directory: Path, profile: AcceleratorProfile
) -> list[CandidateLibrary]:
    result: list[CandidateLibrary] = []
    for name in profile.ort_libraries:
        expected = GPU_RUNTIME_FILES[name]
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"official ONNX Runtime GPU library is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected.size or digest != expected.sha256:
            raise PackageError(
                f"official ONNX Runtime GPU library {name} is not the pinned "
                f"release object: size={size}, sha256={digest}"
            )
        result.append(CandidateLibrary(path.absolute(), "onnx-runtime", digest))
    return result


def merge_candidates(
    groups: Sequence[Sequence[CandidateLibrary]],
) -> dict[str, CandidateLibrary]:
    result: dict[str, CandidateLibrary] = {}
    for group in groups:
        for candidate in group:
            name = candidate.path.name
            existing = result.get(name)
            if existing is not None:
                if existing.sha256 != candidate.sha256:
                    raise PackageError(
                        f"conflicting runtime library basename {name}: "
                        f"{existing.origin} differs from {candidate.origin}"
                    )
                continue
            result[name] = candidate
    return result


def root_library_names(
    profile: AcceleratorProfile,
    candidates: Mapping[str, CandidateLibrary],
) -> set[str]:
    roots = {ENTRYPOINT, *profile.ort_libraries}
    cudnn = {
        name
        for name in candidates
        if re.fullmatch(r"libcudnn(?:_[A-Za-z0-9_]+)?\.so\.9", name)
    }
    if "libcudnn.so.9" not in cudnn:
        raise PackageError("CUDA runtime closure is missing libcudnn.so.9")
    roots.update(cudnn)
    if profile.name == "tensorrt10":
        required = {
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
        }
        missing = {
            name
            for name in required
            if name not in candidates or candidates[name].origin != "tensorrt"
        }
        if missing:
            raise PackageError(
                "TensorRT closure is missing required libraries: "
                + ", ".join(sorted(missing))
            )
        builder_resources = {
            name
            for name, candidate in candidates.items()
            if candidate.origin == "tensorrt"
            and re.fullmatch(r"libnvinfer_builder_resource\.so\.10(?:\.[0-9]+)*", name)
        }
        if len(builder_resources) != 1:
            raise PackageError(
                "TensorRT closure must contain exactly one Linux "
                "libnvinfer_builder_resource.so.10 release object"
            )
        roots.update(required)
        roots.update(builder_resources)
    return roots


def resolve_dependency_closure(
    candidates: Mapping[str, CandidateLibrary],
    roots: set[str],
    needed_libraries: Callable[[CandidateLibrary], Sequence[str]],
) -> set[str]:
    selected: set[str] = set()
    pending = sorted(roots, reverse=True)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        candidate = candidates.get(name)
        if candidate is None:
            raise PackageError(f"runtime dependency closure is missing {name}")
        selected.add(name)
        for dependency in sorted(set(needed_libraries(candidate)), reverse=True):
            if is_host_library(dependency) or dependency in selected:
                continue
            if dependency not in candidates:
                raise PackageError(
                    f"{name} requires unpackaged user-space library {dependency}"
                )
            pending.append(dependency)
    return selected


def needed_for(candidate: CandidateLibrary) -> Sequence[str]:
    needed, _, _ = parse_dynamic_contract(candidate.path, candidate.path.name)
    return needed


def stage_and_normalize(
    candidates: Mapping[str, CandidateLibrary],
    selected: set[str],
    destination: Path,
    *,
    consume_sources: bool = False,
) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[str, Path] = {}
    for name in sorted(selected):
        source = candidates[name].path
        target = destination / name
        shutil.copyfile(source, target)
        if consume_sources:
            source.unlink()
        os.chmod(target, 0o700)
        run_tool(
            ["patchelf", "--set-rpath", RUNPATH, str(target)],
            f"normalize {name} runtime path",
        )
        result[name] = target
    return result


def inspect_staged_libraries(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        inspect_elf_header(path, name)
        needed, soname, runpaths = parse_dynamic_contract(path, name)
        if RUNPATH not in runpaths:
            raise PackageError(
                f"{name} does not resolve dependencies through {RUNPATH}"
            )
        missing = sorted(
            dependency
            for dependency in needed
            if not is_host_library(dependency) and dependency not in paths
        )
        if missing:
            raise PackageError(
                f"{name} has an incomplete pack-local closure: {', '.join(missing)}"
            )
        result[name] = {
            "sha256": sha256_file(path),
            "soname": soname,
            "needed_libraries": needed,
            "maximum_required_glibc": inspect_glibc_contract(path, name),
            "runpath": RUNPATH,
        }
    symbols = run_tool(
        ["nm", "-D", "--defined-only", str(paths[ENTRYPOINT])],
        "inspect ORT accelerator adapter symbols",
    )
    if not re.search(r"(?:^|\s)kapsl_backend_v1$", symbols, re.MULTILINE):
        raise PackageError(
            "ORT accelerator entrypoint does not export kapsl_backend_v1"
        )
    if RUNTIME_SONAME not in result[ENTRYPOINT]["needed_libraries"]:
        raise PackageError(
            f"ORT accelerator entrypoint must link the pack-local {RUNTIME_SONAME}"
        )
    return result


def validate_runtime_provenance(
    path: Path,
    origin: str,
    candidates: Mapping[str, CandidateLibrary],
    authenticated_files: Mapping[str, Path] | None = None,
) -> Mapping[str, Any]:
    try:
        payload = json.loads(read_bounded(path, f"{origin} runtime provenance"))
    except json.JSONDecodeError as error:
        raise PackageError(
            f"{origin} runtime provenance is invalid JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PackageError(f"{origin} runtime provenance has an unsupported schema")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise PackageError(f"{origin} runtime provenance has no file map")
    for name, candidate in candidates.items():
        if candidate.origin != origin:
            continue
        entry = files.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != candidate.sha256
            or entry.get("size") != candidate.path.stat().st_size
        ):
            raise PackageError(
                f"{origin} runtime provenance does not authenticate {name}"
            )
    for name, authenticated_path in (authenticated_files or {}).items():
        entry = files.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != sha256_file(authenticated_path)
            or entry.get("size") != authenticated_path.stat().st_size
        ):
            raise PackageError(
                f"{origin} runtime provenance does not authenticate {name}"
            )
    return payload


def license_entries(
    *,
    repository_root: Path,
    ort_notices: bytes,
    cargo_notices: bytes,
    nvidia_license: bytes,
    zlib_license: bytes,
    tensorrt_license_dir: Path | None,
) -> dict[str, PackEntry]:
    try:
        validate_notice(ort_notices)
    except NoticeFetchError as error:
        raise PackageError(str(error)) from error
    if b"KAPSL ORT ADAPTER RUST DEPENDENCY NOTICES" not in cargo_notices:
        raise PackageError("Rust dependency notices are missing their expected heading")
    result = {
        "licenses/KAPSL-LICENSE": PackEntry.from_bytes(
            read_bounded(repository_root / "LICENSE", "Kapsl license"),
        ),
        "licenses/KAPSL-NOTICE": PackEntry.from_bytes(
            read_bounded(repository_root / "NOTICE", "Kapsl notice"),
        ),
        "licenses/ONNX-RUNTIME-LICENSE": PackEntry.from_bytes(
            read_bounded(
                repository_root / "integrations/ort/third_party/ONNX-RUNTIME-LICENSE",
                "ONNX Runtime license",
            ),
        ),
        "licenses/ONNX-RUNTIME-THIRD-PARTY-NOTICES": PackEntry.from_bytes(ort_notices),
        "licenses/RUST-DEPENDENCY-NOTICES": PackEntry.from_bytes(cargo_notices),
        "licenses/NVIDIA-CONTAINER-LICENSE": PackEntry.from_bytes(nvidia_license),
        "licenses/ZLIB-COPYRIGHT": PackEntry.from_bytes(zlib_license),
    }
    if tensorrt_license_dir is not None:
        if not tensorrt_license_dir.is_dir():
            raise PackageError(
                f"TensorRT license directory is missing: {tensorrt_license_dir}"
            )
        licenses = sorted(
            path for path in tensorrt_license_dir.rglob("*") if path.is_file()
        )
        if not licenses:
            raise PackageError("TensorRT license directory contains no files")
        for index, path in enumerate(licenses, 1):
            result[f"licenses/TENSORRT-{index:03d}-{path.name}"] = PackEntry.from_bytes(
                read_bounded(path, "TensorRT license")
            )
    return result


def build_entries(
    *,
    profile: AcceleratorProfile,
    staged: Mapping[str, Path],
    inspections: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, CandidateLibrary],
    repository_root: Path,
    source_commit: str,
    source_date_epoch: int,
    ort_notices: bytes,
    cargo_notices: bytes,
    nvidia_license: bytes,
    zlib_license: bytes,
    tensorrt_license_dir: Path | None,
    runtime_distributions: Mapping[str, Mapping[str, Any]],
) -> dict[str, PackEntry]:
    binary_entries = {
        name: PackEntry.from_path(path) for name, path in sorted(staged.items())
    }
    payload_manifest = {
        "schema_version": SCHEMA_VERSION,
        "backend": "onnx",
        "profile": profile.name,
        "pack_version": ADAPTER_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "adapter_abi": ADAPTER_ABI,
        "platform": PLATFORM,
        "execution_mode": "native",
        "entrypoint": ENTRYPOINT,
        "accelerator_requirements": accelerator_requirements(profile),
        **common_pack_contract(True),
    }
    official_files = {
        name: {
            "archive_member": GPU_RUNTIME_FILES[name].member,
            "upstream_sha256": GPU_RUNTIME_FILES[name].sha256,
            "packaged_sha256": inspections[name]["sha256"],
        }
        for name in profile.ort_libraries
    }
    provenance = {
        "schema_version": 1,
        "source_repository": "https://github.com/kapsl-runtime/kapsl-integrations",
        "source_commit": source_commit,
        "source_date_epoch": source_date_epoch,
        "profile": {
            "pack": profile.name,
            "cargo_feature": profile.feature,
            "accelerator": profile.accelerator,
        },
        "adapter": {
            "crate": "kapsl-backend-ort",
            "version": ADAPTER_VERSION,
            "backend_abi": RUNTIME_ABI,
            "adapter_abi": ADAPTER_ABI,
        },
        "onnx_runtime": {
            "version": ORT_RUNTIME_VERSION,
            "binding_crate": "ort",
            "binding_version": ORT_BINDING_VERSION,
            "distribution_url": GPU_RUNTIME_ARCHIVE_URL,
            "distribution_sha256": GPU_RUNTIME_ARCHIVE_SHA256,
            "official_files": official_files,
        },
        "accelerator_runtime_distributions": dict(runtime_distributions),
        "build": {
            "target": TARGET,
            "rust_toolchain": RUST_TOOLCHAIN,
            "cargo_lock_sha256": sha256_file(repository_root / "Cargo.lock"),
            "rust_toolchain_sha256": sha256_file(
                repository_root / "rust-toolchain.toml"
            ),
            "maximum_permitted_glibc": ".".join(
                str(component) for component in MAX_GLIBC_VERSION
            ),
            "runtime_path_normalization": "patchelf --set-rpath $ORIGIN",
        },
        "libraries": {
            name: {
                **dict(inspections[name]),
                "source_origin": candidates[name].origin,
                "source_sha256": candidates[name].sha256,
            }
            for name in sorted(staged)
        },
        "notices": {
            "onnx_runtime_third_party_sha256": NOTICE_SHA256,
            "rust_dependencies_sha256": sha256_bytes(cargo_notices),
            "nvidia_license_sha256": sha256_bytes(nvidia_license),
            "zlib_license_sha256": sha256_bytes(zlib_license),
        },
    }
    entries: dict[str, PackEntry] = {
        **binary_entries,
        "backend-pack.json": PackEntry.from_bytes(json_bytes(payload_manifest)),
        "provenance.json": PackEntry.from_bytes(json_bytes(provenance)),
    }
    entries.update(
        license_entries(
            repository_root=repository_root,
            ort_notices=ort_notices,
            cargo_notices=cargo_notices,
            nvidia_license=nvidia_license,
            zlib_license=zlib_license,
            tensorrt_license_dir=tensorrt_license_dir,
        )
    )
    return entries


def manifest_template(
    profile: AcceleratorProfile,
    entries: Mapping[str, PackEntry],
    kapsl_version: str,
) -> dict[str, Any]:
    missing = {ENTRYPOINT, "backend-pack.json", PROVENANCE_PATH} - set(entries)
    if missing:
        raise PackageError(
            f"{profile.name} pack is missing required entries: "
            + ", ".join(sorted(missing))
        )
    files = {name: entry.sha256 for name, entry in sorted(entries.items())}
    licenses = [
        {"name": PurePosixPath(name).name, "path": name}
        for name in sorted(entries)
        if name.startswith("licenses/")
    ]
    template = {
        "schema_version": SCHEMA_VERSION,
        "backend": "onnx",
        "profile": profile.name,
        "pack_version": ADAPTER_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "adapter_abi": ADAPTER_ABI,
        "compatible_kapsl": f"={kapsl_version}",
        "platform": PLATFORM,
        "architecture": "x86_64",
        "accelerator_profile": profile.accelerator,
        "minimum_cuda": MINIMUM_CUDA,
        "minimum_driver": MINIMUM_DRIVER,
        "execution_mode": "native",
        "entrypoint": ENTRYPOINT,
        "accelerator_requirements": accelerator_requirements(profile),
        **common_pack_contract(True),
        "installed_bytes": sum(entry.size for entry in entries.values()),
        "memory": {
            "host_bytes": 64 * 1024 * 1024,
            "accelerator_bytes": 128 * 1024 * 1024,
            "workspace_weight_ppm": 250_000,
            "minimum_workspace_bytes": 256 * 1024 * 1024,
        },
        "installer": {"kind": "extract"},
        "files": files,
        "licenses": licenses,
        "priority": 200,
    }
    if profile.name == "tensorrt10":
        template["minimum_tensorrt"] = MINIMUM_TENSORRT
    return template


def write_streaming_archive(
    path: Path,
    entries: Mapping[str, PackEntry],
    source_date_epoch: int,
    *,
    consume_paths: bool = False,
) -> None:
    directories = sorted(
        {
            str(parent)
            for name in entries
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=output,
                mtime=source_date_epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    for directory in directories:
                        info = tarfile.TarInfo(directory)
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.mtime = source_date_epoch
                        archive.addfile(info)
                    for name, entry in sorted(entries.items()):
                        info = tarfile.TarInfo(name)
                        info.size = entry.size
                        info.mode = entry.mode
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.mtime = source_date_epoch
                        with entry.stream() as stream:
                            archive.addfile(info, stream)
                        if consume_paths and entry.path is not None:
                            entry.path.unlink()
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_streaming_archive(
    archive_path: Path,
    entries: Mapping[str, PackEntry],
    template: Mapping[str, Any],
    source_date_epoch: int,
) -> None:
    expected_files = set(entries)
    observed_files: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or any(
                    part in ("", ".", "..") for part in path.parts
                ):
                    raise PackageError(f"archive contains unsafe path {member.name}")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.mtime != source_date_epoch
                ):
                    raise PackageError(
                        f"archive metadata is not reproducible: {member.name}"
                    )
                if member.isdir():
                    if member.mode != 0o755:
                        raise PackageError(
                            f"archive directory mode is invalid: {member.name}"
                        )
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise PackageError(
                        f"archive member is not a regular file: {member.name}"
                    )
                if member.name not in entries or member.name in observed_files:
                    raise PackageError(
                        f"archive has unexpected or duplicate file: {member.name}"
                    )
                expected = entries[member.name]
                if member.mode != expected.mode or member.size != expected.size:
                    raise PackageError(
                        f"archive metadata differs from input: {member.name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError(f"archive file cannot be read: {member.name}")
                digest = hashlib.sha256()
                observed_size = 0
                while block := stream.read(1024 * 1024):
                    observed_size += len(block)
                    digest.update(block)
                if (
                    observed_size != expected.size
                    or digest.hexdigest() != expected.sha256
                ):
                    raise PackageError(
                        f"archive file differs from signed input: {member.name}"
                    )
                observed_files.add(member.name)
    except (OSError, tarfile.TarError) as error:
        raise PackageError(f"inspect archive {archive_path}: {error}") from error
    if observed_files != expected_files:
        raise PackageError(
            "archive is missing files: "
            + ", ".join(sorted(expected_files - observed_files))
        )
    payload_entry = entries["backend-pack.json"]
    if payload_entry.payload is None:
        raise PackageError("backend-pack.json must be an in-memory metadata entry")
    payload = json.loads(payload_entry.payload)
    for field in PACK_CONTRACT_FIELDS:
        if payload.get(field) != template.get(field):
            raise PackageError(f"payload/template mismatch for {field}")
    if template.get("files") != {
        name: entry.sha256 for name, entry in sorted(entries.items())
    }:
        raise PackageError("manifest installed-file hashes do not match archive inputs")


def create_pack(
    *,
    profile: AcceleratorProfile,
    library_path: Path,
    ort_runtime_dir: Path,
    cuda_runtime_dir: Path,
    tensorrt_runtime_dir: Path | None,
    tensorrt_license_dir: Path | None,
    nvidia_license_path: Path,
    zlib_license_path: Path,
    cuda_runtime_provenance_path: Path,
    tensorrt_runtime_provenance_path: Path | None,
    output_dir: Path,
    kapsl_version: str,
    source_commit: str,
    source_date_epoch: int,
    repository_root: Path,
    cargo_notices_path: Path,
    ort_notices_path: Path,
    signing_key: Path | None = None,
    expected_public_key: str | None = None,
    consume_input_libraries: bool = False,
) -> dict[str, Path]:
    if not RUNTIME_VERSION.fullmatch(kapsl_version):
        raise PackageError("kapsl-version must be an exact semantic version")
    if not HEX_COMMIT.fullmatch(source_commit):
        raise PackageError("source-commit must be a lowercase 40-character commit")
    if source_date_epoch <= 0 or source_date_epoch > 0xFFFF_FFFF:
        raise PackageError("source-date-epoch must fit the gzip timestamp field")

    adapter = CandidateLibrary(
        library_path.resolve(), "adapter", sha256_file(library_path)
    )
    ort = verified_ort_libraries(ort_runtime_dir, profile)
    cuda = runtime_library_paths(cuda_runtime_dir, "cuda")
    tensorrt: list[CandidateLibrary] = []
    if profile.name == "tensorrt10":
        if tensorrt_runtime_dir is None:
            raise PackageError("TensorRT profile requires a TensorRT runtime directory")
        tensorrt = runtime_library_paths(tensorrt_runtime_dir, "tensorrt")
    candidates = merge_candidates([[adapter], ort, cuda, tensorrt])
    runtime_distributions = {
        "cuda": validate_runtime_provenance(
            cuda_runtime_provenance_path,
            "cuda",
            candidates,
            {
                "NVIDIA-CONTAINER-LICENSE": nvidia_license_path,
                "ZLIB-COPYRIGHT": zlib_license_path,
            },
        )
    }
    if profile.name == "tensorrt10":
        if tensorrt_runtime_provenance_path is None:
            raise PackageError(
                "TensorRT profile requires authenticated runtime provenance"
            )
        runtime_distributions["tensorrt"] = validate_runtime_provenance(
            tensorrt_runtime_provenance_path, "tensorrt", candidates
        )
    selected = resolve_dependency_closure(
        candidates, root_library_names(profile, candidates), needed_for
    )
    if consume_input_libraries:
        for name in sorted(set(candidates) - selected):
            candidates[name].path.unlink()
    filename = f"kapsl-backend-onnx-{profile.name}-{kapsl_version}-{PLATFORM}.tar.gz"
    archive_path = output_dir / filename
    template_path = output_dir / f"{filename}.manifest.json"
    checksum_path = output_dir / f"{filename}.sha256"
    signature_path = output_dir / f"{filename}.sig"
    with tempfile.TemporaryDirectory(prefix=f"kapsl-ort-{profile.name}-") as temporary:
        staged = stage_and_normalize(
            candidates,
            selected,
            Path(temporary) / "normalized",
            consume_sources=consume_input_libraries,
        )
        inspections = inspect_staged_libraries(staged)
        entries = build_entries(
            profile=profile,
            staged=staged,
            inspections=inspections,
            candidates=candidates,
            repository_root=repository_root,
            source_commit=source_commit,
            source_date_epoch=source_date_epoch,
            ort_notices=read_bounded(ort_notices_path, "ONNX Runtime notices"),
            cargo_notices=read_bounded(cargo_notices_path, "Rust dependency notices"),
            nvidia_license=read_bounded(nvidia_license_path, "NVIDIA license"),
            zlib_license=read_bounded(zlib_license_path, "zlib copyright/license"),
            tensorrt_license_dir=tensorrt_license_dir,
            runtime_distributions=runtime_distributions,
        )
        template = manifest_template(profile, entries, kapsl_version)
        write_streaming_archive(
            archive_path,
            entries,
            source_date_epoch,
            consume_paths=consume_input_libraries,
        )
        validate_streaming_archive(archive_path, entries, template, source_date_epoch)
    atomic_write(template_path, json_bytes(template))
    digest = sha256_file(archive_path)
    atomic_write(checksum_path, f"{digest}  {filename}\n".encode("ascii"))
    result = {
        "archive": archive_path,
        "manifest": template_path,
        "checksum": checksum_path,
    }
    if signing_key is not None:
        if expected_public_key is None:
            raise PackageError("expected-public-key is required with signing-key")
        signature = sign_artifact(signing_key, expected_public_key, digest)
        atomic_write(signature_path, f"{signature}\n".encode("ascii"))
        result["signature"] = signature_path
    elif signature_path.exists():
        signature_path.unlink()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=sorted(PROFILES), required=True)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--ort-runtime-dir", type=Path, required=True)
    result.add_argument("--cuda-runtime-dir", type=Path, required=True)
    result.add_argument("--tensorrt-runtime-dir", type=Path)
    result.add_argument("--tensorrt-license-dir", type=Path)
    result.add_argument("--nvidia-license", type=Path, required=True)
    result.add_argument("--zlib-license", type=Path, required=True)
    result.add_argument("--cuda-runtime-provenance", type=Path, required=True)
    result.add_argument("--tensorrt-runtime-provenance", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--kapsl-version", required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--source-date-epoch", type=int, required=True)
    result.add_argument("--repository-root", type=Path, required=True)
    result.add_argument("--cargo-notices", type=Path, required=True)
    result.add_argument("--ort-notices", type=Path, required=True)
    result.add_argument("--signing-key", type=Path)
    result.add_argument("--expected-public-key")
    result.add_argument(
        "--consume-input-libraries",
        action="store_true",
        help=(
            "delete each selected input library after staging it; intended only "
            "for isolated release-runner scratch directories"
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if platform.system() != "Linux" or platform.machine() not in (
            "x86_64",
            "amd64",
        ):
            raise PackageError(
                "ORT accelerator packs are currently built only on Linux x86_64"
            )
        source_commit = args.source_commit.lower()
        repository_root = args.repository_root.resolve()
        validate_source_contract(repository_root, source_commit, args.source_date_epoch)
        paths = create_pack(
            profile=PROFILES[args.profile],
            library_path=args.library.resolve(),
            ort_runtime_dir=args.ort_runtime_dir.resolve(),
            cuda_runtime_dir=args.cuda_runtime_dir.resolve(),
            tensorrt_runtime_dir=(
                args.tensorrt_runtime_dir.resolve()
                if args.tensorrt_runtime_dir
                else None
            ),
            tensorrt_license_dir=(
                args.tensorrt_license_dir.resolve()
                if args.tensorrt_license_dir
                else None
            ),
            nvidia_license_path=args.nvidia_license.resolve(),
            zlib_license_path=args.zlib_license.resolve(),
            cuda_runtime_provenance_path=args.cuda_runtime_provenance.resolve(),
            tensorrt_runtime_provenance_path=(
                args.tensorrt_runtime_provenance.resolve()
                if args.tensorrt_runtime_provenance
                else None
            ),
            output_dir=args.output_dir.resolve(),
            kapsl_version=args.kapsl_version,
            source_commit=source_commit,
            source_date_epoch=args.source_date_epoch,
            repository_root=repository_root,
            cargo_notices_path=args.cargo_notices.resolve(),
            ort_notices_path=args.ort_notices.resolve(),
            signing_key=args.signing_key.resolve() if args.signing_key else None,
            expected_public_key=args.expected_public_key,
            consume_input_libraries=args.consume_input_libraries,
        )
        for path in paths.values():
            print(path)
        return 0
    except PackageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
