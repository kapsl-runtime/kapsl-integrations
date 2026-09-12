#!/usr/bin/env python3
"""Fetch a reviewed immutable integration runtime; never compile ORT in the engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from governed_runtime import ORT_ROOT, verify
from package_cpu import PackageError


def fetch(profile: str, output: Path) -> None:
    locks = json.loads((ORT_ROOT / "runtime/governed-runtimes.lock.json").read_text())
    entry = locks.get("profiles", {}).get(profile)
    if locks.get("schema_version") != 1 or not entry:
        raise PackageError(
            f"{profile} has no reviewed governed ORT runtime artifact; finish source-build qualification first"
        )
    url = entry.get("archive_url", "")
    if not url.startswith(
        "https://github.com/kapsl-runtime/kapsl-integrations/releases/download/"
    ):
        raise PackageError(
            "governed ORT artifact must be an immutable integrations release asset"
        )
    if output.exists():
        raise PackageError("governed ORT output directory must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="kapsl-governed-ort-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        archive = root / "runtime.tar.gz"
        maximum = (
            sum(item["size"] for item in entry["files"].values()) + 2 * 1024 * 1024
        )
        digest = hashlib.sha256()
        total = 0
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            archive.open("wb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise PackageError(
                        "governed ORT archive exceeds the locked file-size limit"
                    )
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != entry.get("archive_sha256"):
            raise PackageError(
                "governed ORT archive SHA-256 differs from its reviewed lock"
            )
        staged = root / "runtime"
        staged.mkdir()
        allowed = {
            **{name: item["size"] for name, item in entry["files"].items()},
            "governed-runtime.json": 1024 * 1024,
        }
        seen: set[str] = set()
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                if (
                    member.name not in allowed
                    or member.name in seen
                    or not member.isfile()
                    or member.size > allowed[member.name]
                    or Path(member.name).name != member.name
                ):
                    raise PackageError(
                        f"invalid governed ORT archive member: {member.name}"
                    )
                seen.add(member.name)
                stream = source.extractfile(member)
                if stream is None:
                    raise PackageError("governed ORT archive member has no data")
                data = stream.read(member.size + 1)
                if len(data) != member.size:
                    raise PackageError("truncated governed ORT archive member")
                (staged / member.name).write_bytes(data)
        if seen != set(allowed):
            raise PackageError("governed ORT archive is missing locked members")
        verify(staged, profile)
        staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cuda12", "tensorrt10"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
