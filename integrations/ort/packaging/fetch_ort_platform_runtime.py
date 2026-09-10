#!/usr/bin/env python3
"""Fetch verified official ORT CPU build inputs for retained engine platforms.

This prepares runtime inputs, not a signed backend pack. Adapter compilation,
dependency closure, signing and platform conformance are separate release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from fetch_ort_notices import ORT_RUNTIME_VERSION
from fetch_ort_runtime import (
    RUNTIME_ARCHIVE_SHA256,
    RUNTIME_LIBRARY_BYTES,
    RUNTIME_LIBRARY_SHA256,
    atomic_write,
)


@dataclass(frozen=True)
class RuntimeFile:
    member: str
    output_name: str
    size: int
    sha256: str
    build_only: bool = False


@dataclass(frozen=True)
class PlatformRuntime:
    platform: str
    target: str
    entrypoint: str
    archive_name: str
    archive_sha256: str
    maximum_archive_bytes: int
    files: tuple[RuntimeFile, ...]

    @property
    def url(self) -> str:
        return (
            "https://github.com/microsoft/onnxruntime/releases/download/"
            f"v{ORT_RUNTIME_VERSION}/{self.archive_name}"
        )


CPU_RUNTIMES = {
    "linux-x86_64": PlatformRuntime(
        "linux-x86_64",
        "x86_64-unknown-linux-gnu",
        "libkapsl_backend_ort.so",
        "onnxruntime-linux-x64-1.23.2.tgz",
        RUNTIME_ARCHIVE_SHA256,
        32 * 1024 * 1024,
        (
            RuntimeFile(
                "onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so.1.23.2",
                "libonnxruntime.so.1",
                RUNTIME_LIBRARY_BYTES,
                RUNTIME_LIBRARY_SHA256,
            ),
        ),
    ),
    "linux-aarch64": PlatformRuntime(
        "linux-aarch64",
        "aarch64-unknown-linux-gnu",
        "libkapsl_backend_ort.so",
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
        16 * 1024 * 1024,
        (
            RuntimeFile(
                "onnxruntime-linux-aarch64-1.23.2/lib/libonnxruntime.so.1.23.2",
                "libonnxruntime.so.1",
                18_693_384,
                "648ffa64fbe027ae27139109410900cf776a030dec2dbbac51053318cc44c286",
            ),
        ),
    ),
    "macos-aarch64": PlatformRuntime(
        "macos-aarch64",
        "aarch64-apple-darwin",
        "libkapsl_backend_ort.dylib",
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
        16 * 1024 * 1024,
        (
            RuntimeFile(
                "./onnxruntime-osx-arm64-1.23.2/lib/libonnxruntime.1.23.2.dylib",
                "libonnxruntime.1.23.2.dylib",
                35_138_784,
                "d306d2bc768540766c7ed8a1e0ff05d2870c77a934ebeee4a7bafa1b732ef299",
            ),
        ),
    ),
    "macos-x86_64": PlatformRuntime(
        "macos-x86_64",
        "x86_64-apple-darwin",
        "libkapsl_backend_ort.dylib",
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
        16 * 1024 * 1024,
        (
            RuntimeFile(
                "./onnxruntime-osx-x86_64-1.23.2/lib/libonnxruntime.1.23.2.dylib",
                "libonnxruntime.1.23.2.dylib",
                39_742_608,
                "8c9c78de65ea3786f987c0d980e9c1b13a3a5fbc6b3e2965ba05b450e6e4c054",
            ),
        ),
    ),
    "windows-x86_64": PlatformRuntime(
        "windows-x86_64",
        "x86_64-pc-windows-msvc",
        "kapsl_backend_ort.dll",
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
        96 * 1024 * 1024,
        (
            RuntimeFile(
                "onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll",
                "onnxruntime.dll",
                14_186_016,
                "dec964ab1ee36cc9b0ae247d13b376627992fc57dec0454354017ab8fd84f1ea",
            ),
            RuntimeFile(
                "onnxruntime-win-x64-1.23.2/lib/onnxruntime.lib",
                "onnxruntime.lib",
                2124,
                "977263ca76e6a9d0f230a198d3b05b2a2bddfed66bc5c4d4fd25293b03cc78b5",
                build_only=True,
            ),
        ),
    ),
}


class PlatformRuntimeError(RuntimeError):
    """A runtime input did not match its selected platform's immutable lock."""


def extract_runtime(payload: bytes, runtime: PlatformRuntime) -> dict[str, bytes]:
    if not payload or len(payload) > runtime.maximum_archive_bytes:
        raise PlatformRuntimeError("ORT runtime archive is empty or oversized")
    if hashlib.sha256(payload).hexdigest() != runtime.archive_sha256:
        raise PlatformRuntimeError(
            "ORT archive SHA-256 does not match the selected platform"
        )
    result = {}
    try:
        if runtime.archive_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for locked in runtime.files:
                    matches = [
                        item
                        for item in archive.infolist()
                        if item.filename == locked.member
                    ]
                    if len(matches) != 1:
                        raise PlatformRuntimeError(
                            f"missing or duplicate ORT archive member: {locked.member}"
                        )
                    member = matches[0]
                    mode = member.external_attr >> 16
                    if (
                        member.is_dir()
                        or stat.S_ISLNK(mode)
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or member.flag_bits & 1
                    ):
                        raise PlatformRuntimeError(
                            "ORT archive member is not an unencrypted regular file"
                        )
                    if member.file_size != locked.size:
                        raise PlatformRuntimeError(
                            "ORT archive member size does not match its lock"
                        )
                    with archive.open(member) as stream:
                        result[locked.output_name] = stream.read(locked.size + 1)
        else:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                for locked in runtime.files:
                    matches = [
                        item
                        for item in archive.getmembers()
                        if item.name == locked.member
                    ]
                    if len(matches) != 1:
                        raise PlatformRuntimeError(
                            f"missing or duplicate ORT archive member: {locked.member}"
                        )
                    member = matches[0]
                    if not member.isfile() or member.size != locked.size:
                        raise PlatformRuntimeError(
                            "ORT archive member is not a regular file of the locked size"
                        )
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise PlatformRuntimeError("ORT archive member cannot be read")
                    result[locked.output_name] = stream.read(locked.size + 1)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, PlatformRuntimeError):
            raise
        raise PlatformRuntimeError(f"inspect ORT runtime archive: {error}") from error
    for locked in runtime.files:
        value = result[locked.output_name]
        if (
            len(value) != locked.size
            or hashlib.sha256(value).hexdigest() != locked.sha256
        ):
            raise PlatformRuntimeError(
                f"ORT runtime bytes do not match the lock: {locked.output_name}"
            )
    return result


def fetch_archive(runtime: PlatformRuntime) -> bytes:
    request = urllib.request.Request(
        runtime.url, headers={"User-Agent": "kapsl-integrations-ort-packager/0.2"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(runtime.maximum_archive_bytes + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise PlatformRuntimeError(f"fetch {runtime.url}: {error}") from error


def prepare(runtime: PlatformRuntime, payload: bytes, output: Path) -> None:
    # Validate all members before writing any output. Never unpack archive paths.
    files = extract_runtime(payload, runtime)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise PlatformRuntimeError(
            "ORT input directory must be empty to avoid mixing platforms"
        )
    for name, value in files.items():
        atomic_write(output / name, value)
    receipt = {
        "schema_version": 1,
        "platform": runtime.platform,
        "target": runtime.target,
        "entrypoint": runtime.entrypoint,
        "ort_version": ORT_RUNTIME_VERSION,
        "archive_url": runtime.url,
        "archive_sha256": runtime.archive_sha256,
        "files": [
            {
                "path": file.output_name,
                "sha256": file.sha256,
                "bytes": file.size,
                "build_only": file.build_only,
            }
            for file in runtime.files
        ],
    }
    atomic_write(
        output / "runtime-inputs.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=tuple(CPU_RUNTIMES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use a local archive, verifying the same immutable lock",
    )
    args = parser.parse_args(argv)
    try:
        runtime = CPU_RUNTIMES[args.platform]
        if args.archive:
            with args.archive.open("rb") as stream:
                payload = stream.read(runtime.maximum_archive_bytes + 1)
        else:
            payload = fetch_archive(runtime)
        prepare(runtime, payload, args.output_dir.resolve())
        return 0
    except (OSError, PlatformRuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
