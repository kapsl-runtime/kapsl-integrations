#!/usr/bin/env python3
"""Prepare the pinned integration-owned ORT source; never modify an SDK checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = ("cuda12", "tensorrt10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_lock() -> dict:
    return json.loads((ROOT / "source-lock.json").read_text())


def namespace(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unsupported governed ORT profile: {profile}")
    manifest = (ROOT.parent / "Cargo.toml").read_text()
    match = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', manifest, re.MULTILINE)
    if match is None:
        raise ValueError("adapter version must be an exact stable version")
    version = match.group(1).replace(".", "_")
    return f"kapsl_ort_{profile}_scoped_v1_a{version}"


def runtime_names(profile: str) -> dict[str, str]:
    prefix = namespace(profile)
    providers = ["shared", "cuda"]
    if profile == "tensorrt10":
        providers.append("tensorrt")
    return {
        "libonnxruntime.so.1.23.2": f"lib{prefix}.so",
        **{
            f"libonnxruntime_providers_{provider}.so": f"lib{prefix}_providers_{provider}.so"
            for provider in providers
        },
    }


def recipe_identity() -> dict[str, str]:
    paths = [
        ROOT / "source-lock.json",
        ROOT / "prepare_source.py",
        ROOT / "build_runtime.py",
    ]
    paths += sorted((ROOT / "include").glob("*.h"))
    paths += sorted((ROOT / "patches").glob("*.patch"))
    return {
        "adapter.Cargo.toml": sha256(ROOT.parent / "Cargo.toml"),
        **{p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
    }


def verify_inputs(source: Path) -> None:
    lock = source_lock()
    for relative, digest in lock["files"].items():
        path = source / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"ORT source differs from its immutable lock: {relative}")
    for relative, digest in lock["patches"].items():
        if sha256(ROOT / "patches" / relative) != digest:
            raise ValueError(f"ORT patch differs from its reviewed lock: {relative}")


def apply_patches(source: Path, profile: str) -> dict[str, str]:
    """Also used by host CI on authenticated sparse source fixtures."""
    prefix = namespace(profile)
    verify_inputs(source)
    for relative in source_lock()["patches"]:
        patch = ROOT / "patches" / relative
        subprocess.run(
            ["git", "apply", "--check", "--whitespace=error-all", str(patch)],
            cwd=source,
            check=True,
        )
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)

    bridge = source / "onnxruntime/core/session/provider_bridge_ort.cc"
    text = bridge.read_text()
    text = text.replace('"onnxruntime_providers_', f'"{prefix}_providers_')
    text = text.replace('"Provider_SetHost"', f'"{prefix}_Provider_SetHost"')
    bridge.write_text(text)
    # Only the shared provider library is loaded with global symbols by ORT.
    # Namespace its imports/exports so embedded ORT and other profile packs
    # cannot overwrite this runtime's ProviderHost pointer.
    common = source / "onnxruntime/core/providers/shared/common.h"
    common.write_text(
        f"#define Provider_GetHost {prefix}_Provider_GetHost\n"
        f"#define Provider_SetHost {prefix}_Provider_SetHost\n" + common.read_text()
    )
    exports = source / "onnxruntime/core/providers/shared/version_script.lds"
    exports.write_text(
        exports.read_text()
        .replace("Provider_GetHost", f"{prefix}_Provider_GetHost")
        .replace("Provider_SetHost", f"{prefix}_Provider_SetHost")
    )

    prepared = list(source_lock()["files"])
    for header in sorted((ROOT / "include").glob("*.h")):
        relative = f"onnxruntime/core/providers/tensorrt/{header.name}"
        shutil.copyfile(header, source / relative)
        prepared.append(relative)
    return {relative: sha256(source / relative) for relative in prepared}


def prepare(source: Path, profile: str) -> dict:
    namespace(profile)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if commit != source_lock()["commit"]:
        raise ValueError("ORT checkout does not match the pinned source commit")
    manifest = source / ".kapsl-runtime-prepared.json"
    if manifest.exists():
        prepared = json.loads(manifest.read_text())
        if prepared["profile"] != profile or prepared["recipe"] != recipe_identity():
            raise ValueError(
                "prepared ORT checkout belongs to another profile or recipe"
            )
        # A local manifest is a receipt, not an authority. Recreate expected
        # hashes from the immutable Git objects and reviewed patch so editing
        # both a source file and its receipt cannot authorize an arbitrary build.
        with tempfile.TemporaryDirectory(prefix="kapsl-ort-source-check-") as temporary:
            pristine = Path(temporary)
            for relative in source_lock()["files"]:
                target = pristine / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    subprocess.check_output(
                        ["git", "show", f"HEAD:{relative}"], cwd=source
                    )
                )
            expected_files = apply_patches(pristine, profile)
        if (
            prepared.get("files") != expected_files
            or prepared.get("source_commit") != commit
        ):
            raise ValueError(
                "prepared ORT receipt differs from the reviewed source transformation"
            )
        for relative, digest in prepared["files"].items():
            path = source / relative
            if path.is_symlink() or not path.is_file() or sha256(path) != digest:
                raise ValueError(f"prepared ORT source was modified: {relative}")
        changed = set(
            subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], cwd=source, text=True
            ).splitlines()
        )
        if changed - set(prepared["files"]):
            raise ValueError("ORT checkout has changes outside the prepared patch set")
        untracked = set(
            subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=source,
                text=True,
            ).splitlines()
        )
        allowed = set(prepared["files"]) | {manifest.name}
        if untracked - allowed:
            raise ValueError(
                "ORT checkout has untracked source inputs outside the reviewed patch set"
            )
        return prepared
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    ).strip():
        raise ValueError("prepare ORT in a clean, dedicated source checkout")
    prepared = {
        "schema_version": 1,
        "profile": profile,
        "source_commit": commit,
        "recipe": recipe_identity(),
        "files": apply_patches(source, profile),
    }
    manifest.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source.resolve(), args.profile), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
