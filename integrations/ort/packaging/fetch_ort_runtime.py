#!/usr/bin/env python3
"""Fetch and extract the pinned official ONNX Runtime Linux CPU library."""

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
from pathlib import Path
from typing import Sequence

from fetch_ort_notices import ORT_RUNTIME_VERSION


RUNTIME_SONAME = "libonnxruntime.so.1"
RUNTIME_ARCHIVE_NAME = f"onnxruntime-linux-x64-{ORT_RUNTIME_VERSION}.tgz"
RUNTIME_ARCHIVE_URL = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    f"v{ORT_RUNTIME_VERSION}/{RUNTIME_ARCHIVE_NAME}"
)
RUNTIME_ARCHIVE_SHA256 = (
    "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b"
)
RUNTIME_ARCHIVE_MEMBER = (
    f"onnxruntime-linux-x64-{ORT_RUNTIME_VERSION}/lib/"
    f"libonnxruntime.so.{ORT_RUNTIME_VERSION}"
)
RUNTIME_LIBRARY_SHA256 = (
    "13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b"
)
RUNTIME_LIBRARY_BYTES = 22_326_072
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


class RuntimeFetchError(RuntimeError):
    """The pinned runtime could not be fetched, authenticated, or extracted."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_archive(payload: bytes) -> None:
    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise RuntimeFetchError(
            f"ONNX Runtime archive must contain 1..{MAX_ARCHIVE_BYTES} bytes"
        )
    digest = sha256_bytes(payload)
    if digest != RUNTIME_ARCHIVE_SHA256:
        raise RuntimeFetchError(
            f"ONNX Runtime archive has SHA-256 {digest}; "
            f"expected {RUNTIME_ARCHIVE_SHA256}"
        )


def extract_runtime(archive_payload: bytes) -> bytes:
    validate_archive(archive_payload)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as archive:
            try:
                member = archive.getmember(RUNTIME_ARCHIVE_MEMBER)
            except KeyError as error:
                raise RuntimeFetchError(
                    f"ONNX Runtime archive is missing {RUNTIME_ARCHIVE_MEMBER}"
                ) from error
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeFetchError(
                    "ONNX Runtime archive library is not a regular file"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeFetchError("ONNX Runtime archive library cannot be read")
            payload = stream.read(RUNTIME_LIBRARY_BYTES + 1)
    except (OSError, tarfile.TarError) as error:
        raise RuntimeFetchError(f"inspect ONNX Runtime archive: {error}") from error
    if len(payload) != RUNTIME_LIBRARY_BYTES:
        raise RuntimeFetchError(
            f"ONNX Runtime library contains {len(payload)} bytes; "
            f"expected {RUNTIME_LIBRARY_BYTES}"
        )
    digest = sha256_bytes(payload)
    if digest != RUNTIME_LIBRARY_SHA256:
        raise RuntimeFetchError(
            f"ONNX Runtime library has SHA-256 {digest}; "
            f"expected {RUNTIME_LIBRARY_SHA256}"
        )
    return payload


def fetch_archive() -> bytes:
    request = urllib.request.Request(
        RUNTIME_ARCHIVE_URL,
        headers={"User-Agent": "kapsl-integrations-ort-packager/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeFetchError(f"fetch {RUNTIME_ARCHIVE_URL}: {error}") from error
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = args.output.resolve()
        atomic_write(output, extract_runtime(fetch_archive()))
        print(output)
        return 0
    except RuntimeFetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
