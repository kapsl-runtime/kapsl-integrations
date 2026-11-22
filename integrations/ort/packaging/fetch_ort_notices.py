#!/usr/bin/env python3
"""Fetch the pinned ONNX Runtime notices with a mandatory digest check."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from ort_mirror import fetch_first_valid, mirror_url


ORT_RUNTIME_VERSION = "1.23.2"
NOTICE_URL = (
    "https://raw.githubusercontent.com/microsoft/onnxruntime/"
    f"v{ORT_RUNTIME_VERSION}/ThirdPartyNotices.txt"
)
NOTICE_NAME = "ThirdPartyNotices.txt"
NOTICE_SHA256 = "e9e90971a8e75a9a8ac0c6412e29c1202d079998389915aa485f46c816c3b4cc"
MAX_NOTICE_BYTES = 1024 * 1024


class NoticeFetchError(RuntimeError):
    """The pinned notice could not be fetched and authenticated."""


def validate_notice(payload: bytes) -> None:
    if len(payload) > MAX_NOTICE_BYTES:
        raise NoticeFetchError(f"ONNX Runtime notices exceed {MAX_NOTICE_BYTES} bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != NOTICE_SHA256:
        raise NoticeFetchError(
            f"ONNX Runtime notices have SHA-256 {digest}; expected {NOTICE_SHA256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoticeFetchError(
            f"ONNX Runtime notices are not UTF-8: {error}"
        ) from error
    if "THIRD PARTY SOFTWARE NOTICES AND INFORMATION" not in text:
        raise NoticeFetchError(
            "ONNX Runtime notices are missing their expected heading"
        )


def notice_sources() -> list[tuple[str, object]]:
    """R2 mirror first, upstream second; both are held to the pinned digest."""
    def accept(payload: bytes) -> bytes:
        validate_notice(payload)
        return payload

    urls = [mirror_url(ORT_RUNTIME_VERSION, NOTICE_NAME), NOTICE_URL]
    return [(url, accept) for url in urls if url]


def fetch_notice() -> bytes:
    return fetch_first_valid(
        notice_sources(), NoticeFetchError, MAX_NOTICE_BYTES, 60, "ONNX Runtime notices"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
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
        atomic_write(args.output.resolve(), fetch_notice())
        print(args.output.resolve())
        return 0
    except NoticeFetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
