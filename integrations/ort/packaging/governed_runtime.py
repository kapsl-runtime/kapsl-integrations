"""Verify reviewed source-built ORT inputs before signing an accelerator pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from package_cpu import PackageError, read_bounded, sha256_file

ORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORT_ROOT))
from runtime.prepare_source import recipe_identity, runtime_names, source_lock  # noqa: E402


def verify(directory: Path, profile: str) -> dict:
    locks = json.loads((ORT_ROOT / "runtime/governed-runtimes.lock.json").read_text())
    if locks.get("schema_version") != 1:
        raise PackageError("unsupported governed ORT artifact lock schema")
    expected = locks.get("profiles", {}).get(profile)
    if not expected:
        raise PackageError(
            f"{profile} has no reviewed governed ORT runtime artifact lock; build and qualify the integration runtime before publishing a pack"
        )
    path = directory / "governed-runtime.json"
    if path.is_symlink() or not path.is_file():
        raise PackageError("governed ORT runtime provenance is missing")
    if sha256_file(path) != expected.get("provenance_sha256"):
        raise PackageError(
            "governed ORT provenance differs from its reviewed artifact lock"
        )
    try:
        manifest = json.loads(read_bounded(path, "governed ORT provenance"))
    except (ValueError, OSError) as error:
        raise PackageError(f"invalid governed ORT provenance: {error}") from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("profile") != profile
        or manifest.get("source_commit") != source_lock()["commit"]
        or manifest.get("recipe") != recipe_identity()
    ):
        raise PackageError(
            "governed ORT runtime does not match the reviewed source/build recipe"
        )
    files = manifest.get("files", {})
    if files != expected.get("files") or set(files) != set(
        runtime_names(profile).values()
    ):
        raise PackageError(
            "governed ORT runtime file set differs from its reviewed artifact lock"
        )
    for name, record in files.items():
        library = directory / name
        if (
            library.is_symlink()
            or not library.is_file()
            or library.stat().st_size != record.get("size")
            or sha256_file(library) != record.get("sha256")
        ):
            raise PackageError(
                f"governed ORT runtime object differs from its reviewed artifact lock: {name}"
            )
    return manifest
