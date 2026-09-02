#!/usr/bin/env python3
"""Fetch the exact official ONNX Runtime GPU libraries used by ORT packs."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from fetch_ort_notices import ORT_RUNTIME_VERSION


GPU_RUNTIME_ARCHIVE_NAME = f"onnxruntime-linux-x64-gpu-{ORT_RUNTIME_VERSION}.tgz"
GPU_RUNTIME_ARCHIVE_URL = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    f"v{ORT_RUNTIME_VERSION}/{GPU_RUNTIME_ARCHIVE_NAME}"
)
GPU_RUNTIME_ARCHIVE_SHA256 = (
    "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53"
)
GPU_RUNTIME_ARCHIVE_BYTES = 240_893_669
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
ARCHIVE_PREFIX = f"onnxruntime-linux-x64-gpu-{ORT_RUNTIME_VERSION}/lib"


@dataclass(frozen=True)
class RuntimeFile:
    member: str
    sha256: str
    size: int


GPU_RUNTIME_FILES: Mapping[str, RuntimeFile] = {
    "libonnxruntime.so.1": RuntimeFile(
        member=f"{ARCHIVE_PREFIX}/libonnxruntime.so.{ORT_RUNTIME_VERSION}",
        sha256="dfca180cdcb0d79fb64a5a548da34cdea810d5145520f59c8b12201802966716",
        size=23_921_240,
    ),
    "libonnxruntime_providers_shared.so": RuntimeFile(
        member=f"{ARCHIVE_PREFIX}/libonnxruntime_providers_shared.so",
        sha256="ebbfc73b7da61d56eba58cb3eb76caa6dcba8b3026f1d75626bb580ffe3cb69b",
        size=14_632,
    ),
    "libonnxruntime_providers_cuda.so": RuntimeFile(
        member=f"{ARCHIVE_PREFIX}/libonnxruntime_providers_cuda.so",
        sha256="78760bb32dcebf7997c6b2300b1d3cbcf3aee8e297998bc4f76c7e1b84c116f1",
        size=368_483_592,
    ),
    "libonnxruntime_providers_tensorrt.so": RuntimeFile(
        member=f"{ARCHIVE_PREFIX}/libonnxruntime_providers_tensorrt.so",
        sha256="57b27cbd69e488df5e0bfe0368208926ab699cb9d0e83d3734754febeecb4509",
        size=830_120,
    ),
}


class GpuRuntimeFetchError(RuntimeError):
    """The pinned GPU runtime could not be fetched or authenticated."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_archive(payload: bytes) -> None:
    if len(payload) != GPU_RUNTIME_ARCHIVE_BYTES:
        raise GpuRuntimeFetchError(
            "ONNX Runtime GPU archive contains "
            f"{len(payload)} bytes; expected {GPU_RUNTIME_ARCHIVE_BYTES}"
        )
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise GpuRuntimeFetchError(
            f"ONNX Runtime GPU archive exceeds {MAX_ARCHIVE_BYTES} bytes"
        )
    digest = sha256_bytes(payload)
    if digest != GPU_RUNTIME_ARCHIVE_SHA256:
        raise GpuRuntimeFetchError(
            f"ONNX Runtime GPU archive has SHA-256 {digest}; "
            f"expected {GPU_RUNTIME_ARCHIVE_SHA256}"
        )


def extract_runtime(archive_payload: bytes) -> dict[str, bytes]:
    validate_archive(archive_payload)
    result: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as archive:
            for output_name, expected in GPU_RUNTIME_FILES.items():
                try:
                    member = archive.getmember(expected.member)
                except KeyError as error:
                    raise GpuRuntimeFetchError(
                        f"ONNX Runtime GPU archive is missing {expected.member}"
                    ) from error
                if not member.isfile() or member.issym() or member.islnk():
                    raise GpuRuntimeFetchError(
                        f"ONNX Runtime GPU member is not a regular file: {expected.member}"
                    )
                if member.size != expected.size:
                    raise GpuRuntimeFetchError(
                        f"{expected.member} contains {member.size} bytes; "
                        f"expected {expected.size}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise GpuRuntimeFetchError(
                        f"ONNX Runtime GPU member cannot be read: {expected.member}"
                    )
                payload = stream.read(expected.size + 1)
                if len(payload) != expected.size:
                    raise GpuRuntimeFetchError(
                        f"{expected.member} contains {len(payload)} bytes; "
                        f"expected {expected.size}"
                    )
                digest = sha256_bytes(payload)
                if digest != expected.sha256:
                    raise GpuRuntimeFetchError(
                        f"{expected.member} has SHA-256 {digest}; "
                        f"expected {expected.sha256}"
                    )
                result[output_name] = payload
    except (OSError, tarfile.TarError) as error:
        raise GpuRuntimeFetchError(
            f"inspect ONNX Runtime GPU archive: {error}"
        ) from error
    return result


def fetch_archive() -> bytes:
    request = urllib.request.Request(
        GPU_RUNTIME_ARCHIVE_URL,
        headers={"User-Agent": "kapsl-integrations-ort-packager/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise GpuRuntimeFetchError(
            f"fetch {GPU_RUNTIME_ARCHIVE_URL}: {error}"
        ) from error
    validate_archive(payload)
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o755)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_runtime(output_dir: Path, libraries: Mapping[str, bytes]) -> None:
    for name, payload in sorted(libraries.items()):
        atomic_write(output_dir / name, payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output_dir = args.output_dir.resolve()
        write_runtime(output_dir, extract_runtime(fetch_archive()))
        print(output_dir)
        return 0
    except GpuRuntimeFetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
