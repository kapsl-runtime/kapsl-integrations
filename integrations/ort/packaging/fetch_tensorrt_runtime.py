#!/usr/bin/env python3
"""Fetch and authenticate the exact TensorRT 10.9 Linux runtime closure."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import struct
import sys
import tempfile
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Protocol, Sequence


TENSORRT_VERSION = "10.9.0.34"
WHEEL_NAME = "tensorrt_cu12_libs-10.9.0.34-py2.py3-none-manylinux_2_28_x86_64.whl"
WHEEL_URL = f"https://pypi.nvidia.com/tensorrt-cu12-libs/{WHEEL_NAME}"
WHEEL_SHA256 = "4a82f0bda2874596f202f6edc8dae99b86a3c4ec2fa142a9c847c4d3a57864a0"
WHEEL_BYTES = 3_103_291_777
MAX_CENTRAL_DIRECTORY_BYTES = 4 * 1024 * 1024
READ_BLOCK_BYTES = 8 * 1024 * 1024
HASH_RANGE_BYTES = 256 * 1024 * 1024
EOCD = struct.Struct("<4s4H2IH")
CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
LOCAL_HEADER = struct.Struct("<4s5H3I2H")
EXTRA_HEADER = struct.Struct("<HH")
CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")


@dataclass(frozen=True)
class RuntimeFile:
    member: str
    output_name: str
    sha256: str
    size: int
    kind: str


RUNTIME_FILES: tuple[RuntimeFile, ...] = (
    RuntimeFile(
        member="tensorrt_libs/libnvinfer.so.10",
        output_name="libnvinfer.so.10",
        sha256="366342e2c2da994d281237449a16c23a44cbfb5a4806d3de6a8c68d995a7c5de",
        size=672_119_672,
        kind="runtime",
    ),
    RuntimeFile(
        member="tensorrt_libs/libnvinfer_plugin.so.10",
        output_name="libnvinfer_plugin.so.10",
        sha256="fc9ea692cda90b7055c3c6ab5f68d7693772bac47e65280d5466d6368fe2c38e",
        size=55_059_144,
        kind="runtime",
    ),
    RuntimeFile(
        member="tensorrt_libs/libnvonnxparser.so.10",
        output_name="libnvonnxparser.so.10",
        sha256="b067692867444727381976dddad62012c2922ee90e799ec74a17c3e955b13bcd",
        size=4_448_800,
        kind="runtime",
    ),
    RuntimeFile(
        member="tensorrt_libs/libnvinfer_builder_resource.so.10.9.0",
        output_name="libnvinfer_builder_resource.so.10.9.0",
        sha256="70dc6615a7634e0fef346e285b942c20812879873c7112ffcff90a86c638a0a6",
        size=1_966_319_120,
        kind="runtime",
    ),
    RuntimeFile(
        member="tensorrt_cu12_libs-10.9.0.34.dist-info/LICENSE.txt",
        output_name="NVIDIA-TENSORRT-LICENSE.txt",
        sha256="64bd290f0251405f783ba1d2e155c500542be69795e51147a1d9f11a57bda8cc",
        size=46_961,
        kind="license",
    ),
)


class TensorRtFetchError(RuntimeError):
    """The pinned TensorRT distribution could not be authenticated."""


class RangeSource(Protocol):
    size: int

    @contextlib.contextmanager
    def open_range(self, start: int, end: int) -> Iterator[BinaryIO]: ...


@dataclass(frozen=True)
class ZipMember:
    name: str
    flags: int
    compression: int
    crc32: int
    compressed_size: int
    size: int
    local_header_offset: int


class HttpRangeSource:
    def __init__(self, url: str, expected_size: int) -> None:
        self.url = url
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "kapsl-integrations-ort-packager/0.2"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                observed = int(response.headers["Content-Length"])
        except (KeyError, ValueError, urllib.error.URLError, TimeoutError) as error:
            raise TensorRtFetchError(
                f"inspect TensorRT distribution: {error}"
            ) from error
        if observed != expected_size:
            raise TensorRtFetchError(
                f"TensorRT distribution contains {observed} bytes; "
                f"expected {expected_size}"
            )
        self.size = observed

    @contextlib.contextmanager
    def open_range(self, start: int, end: int) -> Iterator[BinaryIO]:
        if start < 0 or end < start or end >= self.size:
            raise TensorRtFetchError(f"invalid TensorRT byte range {start}-{end}")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "kapsl-integrations-ort-packager/0.2",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                match = CONTENT_RANGE.fullmatch(
                    response.headers.get("Content-Range", "")
                )
                if (
                    response.status != 206
                    or match is None
                    or tuple(int(value) for value in match.groups())
                    != (start, end, self.size)
                ):
                    raise TensorRtFetchError(
                        "TensorRT distribution server did not honor the exact byte range"
                    )
                yield response
        except (urllib.error.URLError, TimeoutError) as error:
            raise TensorRtFetchError(
                f"fetch TensorRT distribution range {start}-{end}: {error}"
            ) from error


class MemoryRangeSource:
    """In-memory range source used by the packaging unit tests."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.size = len(payload)

    @contextlib.contextmanager
    def open_range(self, start: int, end: int) -> Iterator[BinaryIO]:
        if start < 0 or end < start or end >= self.size:
            raise TensorRtFetchError(f"invalid fixture byte range {start}-{end}")
        with io.BytesIO(self.payload[start : end + 1]) as stream:
            yield stream


def read_exact_range(source: RangeSource, start: int, end: int) -> bytes:
    expected = end - start + 1
    with source.open_range(start, end) as stream:
        payload = stream.read(expected + 1)
    if len(payload) != expected:
        raise TensorRtFetchError(
            f"TensorRT range {start}-{end} contains {len(payload)} bytes; "
            f"expected {expected}"
        )
    return payload


def verify_distribution(source: RangeSource, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    observed = 0
    for start in range(0, source.size, HASH_RANGE_BYTES):
        end = min(source.size, start + HASH_RANGE_BYTES) - 1
        range_size = 0
        with source.open_range(start, end) as stream:
            while block := stream.read(READ_BLOCK_BYTES):
                range_size += len(block)
                observed += len(block)
                digest.update(block)
        if range_size != end - start + 1:
            raise TensorRtFetchError(
                f"TensorRT distribution range {start}-{end} contains {range_size} bytes"
            )
    if observed != source.size:
        raise TensorRtFetchError(
            f"TensorRT distribution stream contains {observed} bytes; "
            f"expected {source.size}"
        )
    if digest.hexdigest() != expected_sha256:
        raise TensorRtFetchError(
            f"TensorRT distribution has SHA-256 {digest.hexdigest()}; "
            f"expected {expected_sha256}"
        )


def resolve_zip64_fields(
    *,
    name: str,
    size: int,
    compressed_size: int,
    local_header_offset: int,
    disk: int,
    extra: bytes,
) -> tuple[int, int, int, int]:
    needs_zip64 = (
        size == 0xFFFF_FFFF,
        compressed_size == 0xFFFF_FFFF,
        local_header_offset == 0xFFFF_FFFF,
        disk == 0xFFFF,
    )
    if not any(needs_zip64):
        return size, compressed_size, local_header_offset, disk
    offset = 0
    zip64 = None
    while offset < len(extra):
        if offset + EXTRA_HEADER.size > len(extra):
            raise TensorRtFetchError(f"{name} has truncated ZIP extra metadata")
        kind, length = EXTRA_HEADER.unpack_from(extra, offset)
        offset += EXTRA_HEADER.size
        end = offset + length
        if end > len(extra):
            raise TensorRtFetchError(f"{name} has malformed ZIP extra metadata")
        if kind == 0x0001:
            if zip64 is not None:
                raise TensorRtFetchError(f"{name} has duplicate ZIP64 metadata")
            zip64 = extra[offset:end]
        offset = end
    if zip64 is None:
        raise TensorRtFetchError(f"{name} is missing required ZIP64 metadata")

    values = [size, compressed_size, local_header_offset, disk]
    cursor = 0
    for index, required in enumerate(needs_zip64):
        if not required:
            continue
        width = 4 if index == 3 else 8
        if cursor + width > len(zip64):
            raise TensorRtFetchError(f"{name} has truncated ZIP64 values")
        values[index] = int.from_bytes(zip64[cursor : cursor + width], "little")
        cursor += width
    if cursor != len(zip64):
        raise TensorRtFetchError(f"{name} has unexpected ZIP64 values")
    return values[0], values[1], values[2], values[3]


def parse_central_directory(source: RangeSource) -> Mapping[str, ZipMember]:
    tail_size = min(source.size, 65_535 + EOCD.size)
    tail_offset = source.size - tail_size
    tail = read_exact_range(source, tail_offset, source.size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or eocd_offset + EOCD.size > len(tail):
        raise TensorRtFetchError("TensorRT wheel has no valid ZIP end record")
    (
        signature,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = EOCD.unpack_from(tail, eocd_offset)
    if signature != b"PK\x05\x06" or disk != 0 or central_disk != 0:
        raise TensorRtFetchError("TensorRT wheel uses unsupported multi-disk ZIP data")
    if disk_entries != total_entries or total_entries == 0:
        raise TensorRtFetchError("TensorRT wheel has an invalid ZIP member count")
    if eocd_offset + EOCD.size + comment_size != len(tail):
        raise TensorRtFetchError("TensorRT wheel has trailing or malformed ZIP data")
    if (
        total_entries == 0xFFFF
        or central_size == 0xFFFF_FFFF
        or central_offset == 0xFFFF_FFFF
    ):
        raise TensorRtFetchError("TensorRT wheel unexpectedly requires ZIP64 metadata")
    if central_size <= 0 or central_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise TensorRtFetchError("TensorRT wheel central directory exceeds its bound")
    if central_offset + central_size > tail_offset + eocd_offset:
        raise TensorRtFetchError(
            "TensorRT wheel central directory overlaps its end record"
        )

    payload = read_exact_range(
        source, central_offset, central_offset + central_size - 1
    )
    result: dict[str, ZipMember] = {}
    offset = 0
    for _ in range(total_entries):
        if offset + CENTRAL_HEADER.size > len(payload):
            raise TensorRtFetchError("TensorRT wheel central directory is truncated")
        fields = CENTRAL_HEADER.unpack_from(payload, offset)
        if fields[0] != b"PK\x01\x02":
            raise TensorRtFetchError("TensorRT wheel central directory is malformed")
        (
            _,
            _,
            _,
            flags,
            compression,
            _,
            _,
            crc32,
            compressed_size,
            size,
            name_size,
            extra_size,
            member_comment_size,
            member_disk,
            _,
            _,
            local_header_offset,
        ) = fields
        end = (
            offset + CENTRAL_HEADER.size + name_size + extra_size + member_comment_size
        )
        if end > len(payload):
            raise TensorRtFetchError("TensorRT wheel member metadata is truncated")
        name_bytes = payload[
            offset + CENTRAL_HEADER.size : offset + CENTRAL_HEADER.size + name_size
        ]
        extra_start = offset + CENTRAL_HEADER.size + name_size
        extra = payload[extra_start : extra_start + extra_size]
        try:
            name = name_bytes.decode("utf-8" if flags & 0x800 else "ascii")
        except UnicodeDecodeError as error:
            raise TensorRtFetchError(
                "TensorRT wheel has a non-portable member name"
            ) from error
        if flags & 0x1:
            raise TensorRtFetchError(f"TensorRT wheel member is encrypted: {name}")
        if name in result:
            raise TensorRtFetchError(f"TensorRT wheel has duplicate member {name}")
        size, compressed_size, local_header_offset, member_disk = resolve_zip64_fields(
            name=name,
            size=size,
            compressed_size=compressed_size,
            local_header_offset=local_header_offset,
            disk=member_disk,
            extra=extra,
        )
        if member_disk != 0:
            raise TensorRtFetchError(
                f"TensorRT wheel member is on another disk: {name}"
            )
        result[name] = ZipMember(
            name=name,
            flags=flags,
            compression=compression,
            crc32=crc32,
            compressed_size=compressed_size,
            size=size,
            local_header_offset=local_header_offset,
        )
        offset = end
    if offset != len(payload):
        raise TensorRtFetchError("TensorRT wheel central directory has extra data")
    return result


def member_data_offset(source: RangeSource, member: ZipMember) -> int:
    end = member.local_header_offset + LOCAL_HEADER.size - 1
    payload = read_exact_range(source, member.local_header_offset, end)
    fields = LOCAL_HEADER.unpack(payload)
    if fields[0] != b"PK\x03\x04":
        raise TensorRtFetchError(f"TensorRT member has no local header: {member.name}")
    flags, compression, name_size, extra_size = (
        fields[2],
        fields[3],
        fields[9],
        fields[10],
    )
    if flags != member.flags or compression != member.compression:
        raise TensorRtFetchError(
            f"TensorRT local and central metadata differ: {member.name}"
        )
    name_start = member.local_header_offset + LOCAL_HEADER.size
    name = read_exact_range(source, name_start, name_start + name_size - 1)
    expected_name = member.name.encode("utf-8" if flags & 0x800 else "ascii")
    if name != expected_name:
        raise TensorRtFetchError(f"TensorRT local member name differs: {member.name}")
    return name_start + name_size + extra_size


def extract_member(
    source: RangeSource,
    member: ZipMember,
    expected: RuntimeFile,
    destination: Path,
) -> None:
    if member.size != expected.size:
        raise TensorRtFetchError(
            f"{member.name} contains {member.size} bytes; expected {expected.size}"
        )
    if member.compression not in (0, 8):
        raise TensorRtFetchError(
            f"{member.name} uses unsupported ZIP compression {member.compression}"
        )
    data_offset = member_data_offset(source, member)
    data_end = data_offset + member.compressed_size - 1
    if data_end >= source.size:
        raise TensorRtFetchError(f"{member.name} compressed data is out of bounds")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary)
    digest = hashlib.sha256()
    checksum = 0
    written = 0
    decompressor = (
        zlib.decompressobj(-zlib.MAX_WBITS) if member.compression == 8 else None
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            with source.open_range(data_offset, data_end) as stream:
                while block := stream.read(READ_BLOCK_BYTES):
                    payload = decompressor.decompress(block) if decompressor else block
                    if payload:
                        if written + len(payload) > expected.size:
                            raise TensorRtFetchError(
                                f"{member.name} expands beyond its pinned size"
                            )
                        output.write(payload)
                        digest.update(payload)
                        checksum = zlib.crc32(payload, checksum)
                        written += len(payload)
                if decompressor:
                    payload = decompressor.flush()
                    if payload:
                        if written + len(payload) > expected.size:
                            raise TensorRtFetchError(
                                f"{member.name} expands beyond its pinned size"
                            )
                        output.write(payload)
                        digest.update(payload)
                        checksum = zlib.crc32(payload, checksum)
                        written += len(payload)
                    if not decompressor.eof or decompressor.unused_data:
                        raise TensorRtFetchError(
                            f"{member.name} has malformed compressed data"
                        )
            output.flush()
            os.fsync(output.fileno())
        if written != expected.size or digest.hexdigest() != expected.sha256:
            raise TensorRtFetchError(
                f"{member.name} failed its pinned size or SHA-256 validation"
            )
        if checksum & 0xFFFF_FFFF != member.crc32:
            raise TensorRtFetchError(f"{member.name} failed its ZIP CRC validation")
        os.chmod(temporary_path, 0o755 if expected.kind == "runtime" else 0o644)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def fetch_runtime(
    source: RangeSource,
    runtime_dir: Path,
    license_dir: Path,
    provenance_path: Path,
) -> None:
    verify_distribution(source, WHEEL_SHA256)
    members = parse_central_directory(source)
    expected_names = {item.member for item in RUNTIME_FILES}
    missing = expected_names - set(members)
    if missing:
        raise TensorRtFetchError(
            "TensorRT wheel is missing pinned members: " + ", ".join(sorted(missing))
        )
    if runtime_dir.exists() or license_dir.exists() or provenance_path.exists():
        raise TensorRtFetchError("TensorRT output paths must not already exist")
    runtime_dir.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    try:
        for expected in RUNTIME_FILES:
            root = runtime_dir if expected.kind == "runtime" else license_dir
            extract_member(
                source, members[expected.member], expected, root / expected.output_name
            )
        atomic_write(
            provenance_path,
            json_bytes(
                {
                    "schema_version": 1,
                    "name": "NVIDIA TensorRT Linux x86_64 runtime",
                    "version": TENSORRT_VERSION,
                    "distribution": {
                        "url": WHEEL_URL,
                        "sha256": WHEEL_SHA256,
                        "size": WHEEL_BYTES,
                    },
                    "files": {
                        item.output_name: {
                            "member": item.member,
                            "sha256": item.sha256,
                            "size": item.size,
                        }
                        for item in RUNTIME_FILES
                    },
                }
            ),
        )
    except Exception:
        for root in (runtime_dir, license_dir):
            for path in root.glob("*"):
                if path.is_file():
                    path.unlink()
            root.rmdir()
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--license-dir", type=Path, required=True)
    result.add_argument("--provenance", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        source = HttpRangeSource(WHEEL_URL, WHEEL_BYTES)
        fetch_runtime(
            source,
            args.runtime_dir.resolve(),
            args.license_dir.resolve(),
            args.provenance.resolve(),
        )
        print(args.runtime_dir.resolve())
        return 0
    except (OSError, TensorRtFetchError, zlib.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
