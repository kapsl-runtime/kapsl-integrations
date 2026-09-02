#!/usr/bin/env python3
"""Build and optionally sign the reproducible Linux x86_64 ORT CPU pack."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from fetch_ort_notices import (
    NOTICE_SHA256,
    ORT_RUNTIME_VERSION,
    NoticeFetchError,
    validate_notice,
)
from fetch_ort_runtime import (
    RUNTIME_ARCHIVE_SHA256,
    RUNTIME_ARCHIVE_URL,
    RUNTIME_LIBRARY_SHA256,
    RUNTIME_SONAME,
)


SCHEMA_VERSION = 1
RUNTIME_ABI = 1
ADAPTER_ABI = "kapsl-backend-v1"
ADAPTER_VERSION = "0.2.0"
ORT_BINDING_VERSION = "2.0.0-rc.11"
RUST_TOOLCHAIN = "1.92.0"
TARGET = "x86_64-unknown-linux-gnu"
PLATFORM = "linux-x86_64"
ENTRYPOINT = "libkapsl_backend_ort.so"
SUPPORTED_FORMATS = ("onnx",)
SUPPORTED_TASKS = (
    "forward",
    "embed",
    "classify",
    "detect",
    "transcribe",
    "generate",
)
PROVENANCE_PATH = "provenance.json"
ARTIFACT_SIGNATURE_ALGORITHM = "ed25519"
ARTIFACT_SIGNATURE_DOMAIN = "kapsl-backend-artifact-v1"
ARTIFACT_DOMAIN = b"kapsl-backend-artifact-v1\0"
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
RUNTIME_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
MAX_INPUT_BYTES = 1024 * 1024 * 1024
MAX_GLIBC_VERSION = (2, 35)
ALLOWED_SYSTEM_LIBRARIES = {
    "ld-linux-x86-64.so.2",
    "libatomic.so.1",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libgomp.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libstdc++.so.6",
    "libutil.so.1",
}


def capability_contract(governed_device_allocator: bool) -> dict[str, bool]:
    """Return the backend-neutral capabilities repeated in both manifests."""
    return {
        "batching": True,
        "cancellation": True,
        "concurrent_inference": True,
        "governed_device_allocator": governed_device_allocator,
        "kv_participation": False,
        "memory_reporting": True,
        "scoped_device_allocator": governed_device_allocator,
        "streaming": True,
    }


def memory_behavior(governed_device_allocator: bool) -> dict[str, Any]:
    """Describe memory ownership without embedding backend implementation types."""
    return {
        "allocation_scope": (
            "kapsl-scoped-device-allocator-v1" if governed_device_allocator else None
        ),
        "device_allocation": (
            "host-governed-scoped" if governed_device_allocator else "none"
        ),
        "live_reporting": True,
        "planned_reporting": True,
        "request_reporting": True,
        "synchronize_before_free": governed_device_allocator,
    }


def artifact_authentication() -> dict[str, Any]:
    """Declare the detached artifact authentication contract; never key material."""
    return {
        "hash_algorithm": "sha256",
        "signature_algorithm": ARTIFACT_SIGNATURE_ALGORITHM,
        "signature_domain": ARTIFACT_SIGNATURE_DOMAIN,
        "signature_encoding": "ed25519:base64-raw",
        "signature_location": "detached",
    }


def common_pack_contract(governed_device_allocator: bool) -> dict[str, Any]:
    return {
        "formats": list(SUPPORTED_FORMATS),
        "tasks": list(SUPPORTED_TASKS),
        "capabilities": capability_contract(governed_device_allocator),
        "memory_behavior": memory_behavior(governed_device_allocator),
        "artifact_authentication": artifact_authentication(),
        "provenance": {"path": PROVENANCE_PATH, "schema_version": 1},
    }


PACK_CONTRACT_FIELDS = (
    "schema_version",
    "backend",
    "profile",
    "pack_version",
    "runtime_abi",
    "adapter_abi",
    "platform",
    "execution_mode",
    "entrypoint",
    "formats",
    "tasks",
    "capabilities",
    "accelerator_requirements",
    "memory_behavior",
    "artifact_authentication",
    "provenance",
)


class PackageError(RuntimeError):
    """The pack inputs or resulting archive violate the release contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise PackageError(f"hash {path}: {error}") from error
    return digest.hexdigest()


def read_bounded(path: Path, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_INPUT_BYTES:
            raise PackageError(
                f"{label} must contain 1..{MAX_INPUT_BYTES} bytes: {path}"
            )
        return path.read_bytes()
    except OSError as error:
        raise PackageError(f"read {label} {path}: {error}") from error


def atomic_write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_tool(arguments: Sequence[str], label: str) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "")
        raise PackageError(f"{label}: {error}: {detail}") from error
    return completed.stdout


def inspect_elf_header(path: Path, label: str) -> None:
    payload = read_bounded(path, label)
    if len(payload) < 20 or payload[:4] != b"\x7fELF":
        raise PackageError(f"{label} must be an ELF library")
    if payload[4] != 2 or payload[5] != 1:
        raise PackageError(f"{label} must be little-endian ELF64")
    machine = struct.unpack_from("<H", payload, 18)[0]
    if machine != 62:
        raise PackageError(f"{label} has ELF machine {machine}; expected x86_64 (62)")


def inspect_glibc_contract(path: Path, label: str) -> str:
    version_info = run_tool(
        ["readelf", "--version-info", "--wide", str(path)],
        f"inspect {label} glibc versions",
    )
    versions = {
        tuple(int(component) for component in raw.split("."))
        for raw in re.findall(r"\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b", version_info)
    }
    if not versions:
        raise PackageError(f"{label} declares no versioned glibc requirements")
    maximum = max(versions)
    if maximum > MAX_GLIBC_VERSION:
        rendered = ".".join(str(component) for component in maximum)
        policy = ".".join(str(component) for component in MAX_GLIBC_VERSION)
        raise PackageError(
            f"{label} requires GLIBC_{rendered}; release policy permits at most "
            f"GLIBC_{policy}"
        )
    dynamic_symbols = run_tool(
        ["readelf", "--dyn-syms", "--wide", str(path)],
        f"inspect {label} dynamic symbols",
    )
    c23_symbols = sorted(
        set(re.findall(r"\b(__isoc23_[A-Za-z0-9_]+)", dynamic_symbols))
    )
    if c23_symbols:
        raise PackageError(
            f"{label} imports unsupported C23 glibc symbols: " + ", ".join(c23_symbols)
        )
    return ".".join(str(component) for component in maximum)


def parse_dynamic_contract(
    path: Path, label: str
) -> tuple[list[str], str | None, list[str]]:
    dynamic = run_tool(["readelf", "-d", str(path)], f"inspect {label} dependencies")
    needed = sorted(set(re.findall(r"\(NEEDED\).*\[([^]]+)]", dynamic)))
    sonames = sorted(set(re.findall(r"\(SONAME\).*\[([^]]+)]", dynamic)))
    if len(sonames) > 1:
        raise PackageError(f"{label} declares multiple ELF SONAME values")
    runpaths: list[str] = []
    for value in re.findall(r"\((?:RPATH|RUNPATH)\).*\[([^]]*)]", dynamic):
        for component in value.split(":"):
            if (
                component
                and component != "$ORIGIN"
                and not component.startswith("$ORIGIN/")
            ):
                raise PackageError(
                    f"{label} contains a non-pack-local runtime path: {component}"
                )
            if component:
                runpaths.append(component)
    return needed, sonames[0] if sonames else None, runpaths


def inspect_linux_adapter(path: Path) -> dict[str, Any]:
    inspect_elf_header(path, "ORT adapter library")
    symbols = run_tool(["nm", "-D", "--defined-only", str(path)], "inspect symbols")
    if not re.search(r"(?:^|\s)kapsl_backend_v1$", symbols, re.MULTILINE):
        raise PackageError("ORT CPU pack entrypoint does not export kapsl_backend_v1")

    needed, _, runpaths = parse_dynamic_contract(path, "ORT adapter library")
    unexpected = sorted(set(needed) - ALLOWED_SYSTEM_LIBRARIES - {RUNTIME_SONAME})
    if unexpected:
        raise PackageError(
            "ORT CPU entrypoint has unpackaged non-system dependencies: "
            + ", ".join(unexpected)
        )
    if RUNTIME_SONAME not in needed:
        raise PackageError(
            f"ORT CPU entrypoint must dynamically link the pack-local {RUNTIME_SONAME}"
        )
    for name in needed:
        lowered = name.lower()
        if name != RUNTIME_SONAME and any(
            token in lowered
            for token in ("onnxruntime", "cuda", "cudnn", "nvinfer", "tensorrt")
        ):
            raise PackageError(
                f"CPU entrypoint unexpectedly links accelerator/runtime {name}"
            )
    if "$ORIGIN" not in runpaths:
        raise PackageError(
            "ORT CPU entrypoint must resolve its runtime through $ORIGIN"
        )
    return {
        "needed_libraries": needed,
        "maximum_required_glibc": inspect_glibc_contract(path, "ORT adapter library"),
    }


def inspect_linux_runtime(path: Path) -> dict[str, Any]:
    inspect_elf_header(path, "ONNX Runtime library")
    digest = sha256_file(path)
    if digest != RUNTIME_LIBRARY_SHA256:
        raise PackageError(
            f"ONNX Runtime library has SHA-256 {digest}; expected {RUNTIME_LIBRARY_SHA256}"
        )
    needed, soname, _ = parse_dynamic_contract(path, "ONNX Runtime library")
    if soname != RUNTIME_SONAME:
        raise PackageError(
            f"ONNX Runtime library SONAME is {soname!r}; expected {RUNTIME_SONAME!r}"
        )
    unexpected = sorted(set(needed) - ALLOWED_SYSTEM_LIBRARIES)
    if unexpected:
        raise PackageError(
            "ONNX Runtime library has unexpected non-system dependencies: "
            + ", ".join(unexpected)
        )
    return {
        "needed_libraries": needed,
        "soname": soname,
        "sha256": digest,
        "maximum_required_glibc": inspect_glibc_contract(path, "ONNX Runtime library"),
    }


def validate_source_contract(
    repository_root: Path, source_commit: str, epoch: int
) -> None:
    if not repository_root.is_dir():
        raise PackageError(f"repository root is not a directory: {repository_root}")
    head = run_tool(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        "resolve source commit",
    ).strip()
    if head != source_commit:
        raise PackageError(
            f"source commit is {source_commit}, but checkout HEAD is {head}"
        )
    status = run_tool(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        "inspect source checkout",
    )
    if status.strip():
        raise PackageError("refusing to package a dirty source checkout")
    commit_epoch = run_tool(
        ["git", "-C", str(repository_root), "show", "-s", "--format=%ct", "HEAD"],
        "resolve source epoch",
    ).strip()
    if commit_epoch != str(epoch):
        raise PackageError(
            f"source-date-epoch is {epoch}, but source commit timestamp is {commit_epoch}"
        )

    manifest = read_bounded(
        repository_root / "integrations/ort/Cargo.toml", "ORT Cargo manifest"
    ).decode("utf-8")
    required_literals = (
        f'version = "{ADAPTER_VERSION}"',
        f'ort = {{ version = "={ORT_BINDING_VERSION}"',
        'kapsl-backend-abi = "=0.2.0"',
        'kapsl-core = "=0.3.0"',
        'kapsl-engine-api = "=0.3.0"',
        'kapsl-llm = { version = "=0.3.4"',
    )
    missing = [literal for literal in required_literals if literal not in manifest]
    if missing:
        raise PackageError(
            "ORT source manifest does not match packaging constants: "
            + ", ".join(missing)
        )


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_entries(
    *,
    library: bytes,
    runtime_library: bytes,
    kapsl_license: bytes,
    kapsl_notice: bytes,
    ort_license: bytes,
    ort_notices: bytes,
    cargo_notices: bytes,
    source_commit: str,
    source_date_epoch: int,
    cargo_lock_sha256: str,
    rust_toolchain_sha256: str,
    adapter_inspection: Mapping[str, Any],
    runtime_inspection: Mapping[str, Any],
) -> dict[str, tuple[bytes, int]]:
    try:
        validate_notice(ort_notices)
    except NoticeFetchError as error:
        raise PackageError(str(error)) from error
    if b"KAPSL ORT ADAPTER RUST DEPENDENCY NOTICES" not in cargo_notices:
        raise PackageError("Rust dependency notices are missing their expected heading")
    binary_sha256 = sha256_bytes(library)
    runtime_sha256 = sha256_bytes(runtime_library)
    if runtime_sha256 != RUNTIME_LIBRARY_SHA256:
        raise PackageError(
            f"ONNX Runtime library has SHA-256 {runtime_sha256}; "
            f"expected {RUNTIME_LIBRARY_SHA256}"
        )
    payload_manifest = {
        "schema_version": SCHEMA_VERSION,
        "backend": "onnx",
        "profile": "cpu",
        "pack_version": ADAPTER_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "adapter_abi": ADAPTER_ABI,
        "platform": PLATFORM,
        "execution_mode": "native",
        "entrypoint": ENTRYPOINT,
        "accelerator_requirements": {
            "execution_providers": ["cpu"],
            "implicit_cpu_fallback": False,
            "kind": "cpu",
        },
        **common_pack_contract(False),
    }
    provenance = {
        "schema_version": 1,
        "source_repository": "https://github.com/kapsl-runtime/kapsl-integrations",
        "source_commit": source_commit,
        "source_date_epoch": source_date_epoch,
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
            "distribution_url": RUNTIME_ARCHIVE_URL,
            "distribution_sha256": RUNTIME_ARCHIVE_SHA256,
            "library": {
                "path": RUNTIME_SONAME,
                "sha256": runtime_sha256,
                "soname": runtime_inspection["soname"],
                "needed_libraries": sorted(runtime_inspection["needed_libraries"]),
                "maximum_required_glibc": runtime_inspection["maximum_required_glibc"],
            },
        },
        "build": {
            "target": TARGET,
            "rust_toolchain": RUST_TOOLCHAIN,
            "cargo_lock_sha256": cargo_lock_sha256,
            "rust_toolchain_sha256": rust_toolchain_sha256,
            "maximum_permitted_glibc": ".".join(
                str(component) for component in MAX_GLIBC_VERSION
            ),
        },
        "entrypoint": {
            "path": ENTRYPOINT,
            "sha256": binary_sha256,
            "needed_libraries": sorted(adapter_inspection["needed_libraries"]),
            "maximum_required_glibc": adapter_inspection["maximum_required_glibc"],
            "dependency_closure": (
                "pack-local official ONNX Runtime shared library plus "
                "allowlisted host system libraries"
            ),
        },
        "notices": {
            "onnx_runtime_third_party_sha256": NOTICE_SHA256,
            "rust_dependencies_sha256": sha256_bytes(cargo_notices),
        },
    }
    return {
        ENTRYPOINT: (library, 0o755),
        RUNTIME_SONAME: (runtime_library, 0o755),
        "backend-pack.json": (json_bytes(payload_manifest), 0o644),
        "provenance.json": (json_bytes(provenance), 0o644),
        "licenses/KAPSL-LICENSE": (kapsl_license, 0o644),
        "licenses/KAPSL-NOTICE": (kapsl_notice, 0o644),
        "licenses/ONNX-RUNTIME-LICENSE": (ort_license, 0o644),
        "licenses/ONNX-RUNTIME-THIRD-PARTY-NOTICES": (ort_notices, 0o644),
        "licenses/RUST-DEPENDENCY-NOTICES": (cargo_notices, 0o644),
    }


def manifest_template(
    entries: Mapping[str, tuple[bytes, int]], kapsl_version: str
) -> dict[str, Any]:
    missing = {ENTRYPOINT, "backend-pack.json", PROVENANCE_PATH} - set(entries)
    if missing:
        raise PackageError(
            "CPU pack is missing required entries: " + ", ".join(sorted(missing))
        )
    files = {
        name: sha256_bytes(payload) for name, (payload, _) in sorted(entries.items())
    }
    licenses = [
        {"name": PurePosixPath(name).name, "path": name}
        for name in sorted(entries)
        if name.startswith("licenses/")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "onnx",
        "profile": "cpu",
        "pack_version": ADAPTER_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "adapter_abi": ADAPTER_ABI,
        "compatible_kapsl": f"={kapsl_version}",
        "platform": PLATFORM,
        "architecture": "x86_64",
        "accelerator_profile": "cpu",
        "execution_mode": "native",
        "entrypoint": ENTRYPOINT,
        "accelerator_requirements": {
            "execution_providers": ["cpu"],
            "implicit_cpu_fallback": False,
            "kind": "cpu",
        },
        **common_pack_contract(False),
        "installed_bytes": sum(len(payload) for payload, _ in entries.values()),
        "memory": {
            "host_bytes": 64 * 1024 * 1024,
            "accelerator_bytes": 0,
            "workspace_weight_ppm": 250_000,
            "minimum_workspace_bytes": 256 * 1024 * 1024,
        },
        "installer": {"kind": "extract"},
        "files": files,
        "licenses": licenses,
        "priority": 200,
    }


def write_archive(
    path: Path, entries: Mapping[str, tuple[bytes, int]], source_date_epoch: int
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
                    for name, (payload, mode) in sorted(entries.items()):
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        info.mode = mode
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.mtime = source_date_epoch
                        archive.addfile(info, io.BytesIO(payload))
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_archive(
    archive_path: Path,
    entries: Mapping[str, tuple[bytes, int]],
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
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError(f"archive file cannot be read: {member.name}")
                payload = stream.read()
                expected_payload, expected_mode = entries[member.name]
                if payload != expected_payload or member.mode != expected_mode:
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
    payload = json.loads(entries["backend-pack.json"][0])
    for field in PACK_CONTRACT_FIELDS:
        if payload.get(field) != template.get(field):
            raise PackageError(f"payload/template mismatch for {field}")
    if template.get("files") != {
        name: sha256_bytes(value) for name, (value, _) in sorted(entries.items())
    }:
        raise PackageError("manifest installed-file hashes do not match archive inputs")


def signing_public_key(signing_key: Path) -> str:
    output = run_tool_bytes(
        ["openssl", "pkey", "-in", str(signing_key), "-pubout", "-outform", "DER"],
        "derive signing public key",
    )
    if len(output) != 44:
        raise PackageError("artifact signing key is not an Ed25519 private key")
    return base64.b64encode(output[-32:]).decode("ascii")


def run_tool_bytes(arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(list(arguments), check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        raise PackageError(f"{label}: {error}: {detail!r}") from error
    return completed.stdout


def sign_artifact(signing_key: Path, expected_public_key: str, digest: str) -> str:
    if not signing_key.is_file():
        raise PackageError(f"artifact signing key is not a file: {signing_key}")
    normalized_expected = expected_public_key.removeprefix("ed25519:")
    try:
        expected_bytes = base64.b64decode(normalized_expected, validate=True)
    except ValueError as error:
        raise PackageError(
            f"expected Ed25519 public key is invalid: {error}"
        ) from error
    if len(expected_bytes) != 32:
        raise PackageError("expected Ed25519 public key must contain 32 bytes")
    actual_public_key = signing_public_key(signing_key)
    if actual_public_key != base64.b64encode(expected_bytes).decode("ascii"):
        raise PackageError(
            "artifact signing key does not match the expected public key"
        )

    message = ARTIFACT_DOMAIN + f"sha256:{digest}".encode("ascii")
    with tempfile.NamedTemporaryFile() as source:
        source.write(message)
        source.flush()
        signature = run_tool_bytes(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(signing_key),
                "-in",
                source.name,
            ],
            "sign artifact digest",
        )
    if len(signature) != 64:
        raise PackageError(
            f"OpenSSL emitted a {len(signature)} byte signature; expected 64"
        )
    return "ed25519:" + base64.b64encode(signature).decode("ascii")


def create_pack(
    *,
    library_path: Path,
    runtime_library_path: Path,
    output_dir: Path,
    kapsl_version: str,
    source_commit: str,
    source_date_epoch: int,
    cargo_lock_path: Path,
    rust_toolchain_path: Path,
    kapsl_license_path: Path,
    kapsl_notice_path: Path,
    ort_license_path: Path,
    ort_notices_path: Path,
    cargo_notices_path: Path,
    adapter_inspection: Mapping[str, Any],
    runtime_inspection: Mapping[str, Any],
    signing_key: Path | None = None,
    expected_public_key: str | None = None,
) -> dict[str, Path]:
    if not RUNTIME_VERSION.fullmatch(kapsl_version):
        raise PackageError("kapsl-version must be an exact semantic version")
    if not HEX_COMMIT.fullmatch(source_commit):
        raise PackageError("source-commit must be a lowercase 40-character commit")
    if source_date_epoch <= 0 or source_date_epoch > 0xFFFF_FFFF:
        raise PackageError("source-date-epoch must fit the gzip timestamp field")

    entries = build_entries(
        library=read_bounded(library_path, "ORT adapter library"),
        runtime_library=read_bounded(runtime_library_path, "ONNX Runtime library"),
        kapsl_license=read_bounded(kapsl_license_path, "Kapsl license"),
        kapsl_notice=read_bounded(kapsl_notice_path, "Kapsl notice"),
        ort_license=read_bounded(ort_license_path, "ONNX Runtime license"),
        ort_notices=read_bounded(ort_notices_path, "ONNX Runtime notices"),
        cargo_notices=read_bounded(cargo_notices_path, "Rust dependency notices"),
        source_commit=source_commit,
        source_date_epoch=source_date_epoch,
        cargo_lock_sha256=sha256_file(cargo_lock_path),
        rust_toolchain_sha256=sha256_file(rust_toolchain_path),
        adapter_inspection=adapter_inspection,
        runtime_inspection=runtime_inspection,
    )
    template = manifest_template(entries, kapsl_version)
    filename = f"kapsl-backend-onnx-cpu-{kapsl_version}-{PLATFORM}.tar.gz"
    archive_path = output_dir / filename
    template_path = output_dir / f"{filename}.manifest.json"
    checksum_path = output_dir / f"{filename}.sha256"
    signature_path = output_dir / f"{filename}.sig"
    write_archive(archive_path, entries, source_date_epoch)
    validate_archive(archive_path, entries, template, source_date_epoch)
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
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--runtime-library", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--kapsl-version", required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--source-date-epoch", type=int, required=True)
    result.add_argument("--repository-root", type=Path, required=True)
    result.add_argument("--cargo-notices", type=Path, required=True)
    result.add_argument("--ort-notices", type=Path, required=True)
    result.add_argument("--signing-key", type=Path)
    result.add_argument("--expected-public-key")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if platform.system() != "Linux" or platform.machine() not in (
            "x86_64",
            "amd64",
        ):
            raise PackageError("ORT CPU packs are currently built only on Linux x86_64")
        source_commit = args.source_commit.lower()
        repository_root = args.repository_root.resolve()
        validate_source_contract(repository_root, source_commit, args.source_date_epoch)
        adapter_inspection = inspect_linux_adapter(args.library.resolve())
        runtime_inspection = inspect_linux_runtime(args.runtime_library.resolve())
        paths = create_pack(
            library_path=args.library.resolve(),
            runtime_library_path=args.runtime_library.resolve(),
            output_dir=args.output_dir.resolve(),
            kapsl_version=args.kapsl_version,
            source_commit=source_commit,
            source_date_epoch=args.source_date_epoch,
            cargo_lock_path=repository_root / "Cargo.lock",
            rust_toolchain_path=repository_root / "rust-toolchain.toml",
            kapsl_license_path=repository_root / "LICENSE",
            kapsl_notice_path=repository_root / "NOTICE",
            ort_license_path=repository_root
            / "integrations/ort/third_party/ONNX-RUNTIME-LICENSE",
            ort_notices_path=args.ort_notices.resolve(),
            cargo_notices_path=args.cargo_notices.resolve(),
            adapter_inspection=adapter_inspection,
            runtime_inspection=runtime_inspection,
            signing_key=args.signing_key.resolve() if args.signing_key else None,
            expected_public_key=args.expected_public_key,
        )
        for path in paths.values():
            print(path)
        return 0
    except PackageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
