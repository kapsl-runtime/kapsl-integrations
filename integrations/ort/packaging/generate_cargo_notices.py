#!/usr/bin/env python3
"""Generate deterministic license notices for linked Rust dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


LICENSE_PREFIXES = ("license", "copying", "notice", "unlicense", "patents")
MAX_LICENSE_BYTES = 4 * 1024 * 1024
MAX_SUPPLEMENTAL_INDEX_BYTES = 1024 * 1024

SupplementalLicenseKey = tuple[str, str, str]
SupplementalLicenses = Mapping[SupplementalLicenseKey, Path]


class NoticeError(RuntimeError):
    """The dependency graph cannot produce complete deterministic notices."""


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_metadata(manifest_path: Path, target: str) -> dict[str, Any]:
    command = [
        "cargo",
        "metadata",
        "--locked",
        "--filter-platform",
        target,
        "--format-version",
        "1",
        "--manifest-path",
        str(manifest_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", "")
        raise NoticeError(f"load locked Cargo metadata: {error}: {detail}") from error
    if not isinstance(metadata, dict):
        raise NoticeError("Cargo metadata must be a JSON object")
    return metadata


def linked_package_ids(metadata: Mapping[str, Any], package_name: str) -> set[str]:
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    workspace_members = metadata.get("workspace_members")
    if (
        not isinstance(packages, list)
        or not isinstance(resolve, dict)
        or not isinstance(resolve.get("nodes"), list)
        or not isinstance(workspace_members, list)
    ):
        raise NoticeError(
            "Cargo metadata is missing packages, workspace members, or resolve nodes"
        )

    package_by_id = {
        str(package["id"]): package
        for package in packages
        if isinstance(package, dict) and "id" in package
    }
    roots = [
        package_id
        for package_id in workspace_members
        if package_id in package_by_id
        and package_by_id[package_id].get("name") == package_name
    ]
    if len(roots) != 1:
        raise NoticeError(
            f"expected one workspace package named {package_name}, found {len(roots)}"
        )

    nodes = {
        str(node["id"]): node
        for node in resolve["nodes"]
        if isinstance(node, dict) and "id" in node
    }
    linked: set[str] = set()
    pending = [str(roots[0])]
    while pending:
        package_id = pending.pop()
        if package_id in linked:
            continue
        linked.add(package_id)
        node = nodes.get(package_id)
        if node is None or not isinstance(node.get("deps"), list):
            raise NoticeError(f"Cargo resolve graph has no node for {package_id}")
        for dependency in node["deps"]:
            if not isinstance(dependency, dict):
                continue
            kinds = dependency.get("dep_kinds", [])
            if not isinstance(kinds, list):
                continue
            if any(
                isinstance(kind, dict) and kind.get("kind") is None for kind in kinds
            ):
                pending.append(str(dependency["pkg"]))
    linked.remove(str(roots[0]))
    return linked


def load_supplemental_licenses(index_path: Path) -> dict[SupplementalLicenseKey, Path]:
    try:
        payload = index_path.read_bytes()
    except OSError as error:
        raise NoticeError(
            f"read supplemental Rust license index {index_path}: {error}"
        ) from error
    if len(payload) > MAX_SUPPLEMENTAL_INDEX_BYTES:
        raise NoticeError(
            "supplemental Rust license index exceeds "
            f"{MAX_SUPPLEMENTAL_INDEX_BYTES} bytes"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NoticeError(
            f"decode supplemental Rust license index {index_path}: {error}"
        ) from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise NoticeError("supplemental Rust license index must use schema version 1")
    entries = document.get("licenses")
    if not isinstance(entries, list):
        raise NoticeError(
            "supplemental Rust license index must contain a licenses array"
        )

    root = index_path.resolve().parent
    supplements: dict[SupplementalLicenseKey, Path] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise NoticeError(
                f"supplemental Rust license entry {index} must be an object"
            )
        values = {
            field: entry.get(field)
            for field in ("name", "version", "license", "path", "sha256", "source")
        }
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise NoticeError(
                f"supplemental Rust license entry {index} has missing string fields"
            )
        relative = Path(values["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise NoticeError(
                f"supplemental Rust license entry {index} has an unsafe path"
            )
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise NoticeError(
                f"supplemental Rust license entry {index} escapes its index directory"
            )
        try:
            license_payload = path.read_bytes()
        except OSError as error:
            raise NoticeError(
                f"read supplemental Rust license {path}: {error}"
            ) from error
        if len(license_payload) > MAX_LICENSE_BYTES:
            raise NoticeError(
                f"supplemental Rust license exceeds {MAX_LICENSE_BYTES} bytes: {path}"
            )
        expected_digest = values["sha256"]
        actual_digest = hashlib.sha256(license_payload).hexdigest()
        if (
            len(expected_digest) != 64
            or expected_digest.lower() != expected_digest
            or actual_digest != expected_digest
        ):
            raise NoticeError(
                f"supplemental Rust license SHA-256 mismatch for {path}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        if not values["source"].startswith("https://"):
            raise NoticeError(
                f"supplemental Rust license entry {index} must pin an HTTPS source"
            )
        key = (values["name"], values["version"], values["license"])
        if key in supplements:
            raise NoticeError(
                "supplemental Rust license index repeats " + " ".join(key)
            )
        supplements[key] = path
    return supplements


def license_paths(
    package: Mapping[str, Any],
    workspace_license: Path,
    supplemental_licenses: SupplementalLicenses | None = None,
) -> list[Path]:
    manifest_path = Path(str(package.get("manifest_path", "")))
    if not manifest_path.is_file():
        raise NoticeError(f"dependency manifest is unavailable: {manifest_path}")
    package_root = manifest_path.parent
    candidates: set[Path] = set()
    license_file = package.get("license_file")
    if isinstance(license_file, str) and license_file:
        candidate = Path(license_file)
        if not candidate.is_absolute():
            candidate = package_root / candidate
        candidates.add(candidate)
    try:
        for candidate in package_root.iterdir():
            if candidate.is_file() and candidate.name.lower().startswith(
                LICENSE_PREFIXES
            ):
                candidates.add(candidate)
    except OSError as error:
        raise NoticeError(
            f"inspect dependency licenses in {package_root}: {error}"
        ) from error

    if not candidates:
        if str(package.get("name", "")).startswith("kapsl-"):
            candidates.add(workspace_license)
        else:
            key = (
                str(package.get("name", "")),
                str(package.get("version", "")),
                str(package.get("license", "")),
            )
            supplemental = (supplemental_licenses or {}).get(key)
            if supplemental is not None:
                candidates.add(supplemental)
    paths = sorted(candidates, key=lambda path: path.name.lower())
    if not paths:
        raise NoticeError(
            f"{package.get('name')} {package.get('version')} has no packaged license text"
        )
    for path in paths:
        if not path.is_file():
            raise NoticeError(f"dependency license is unavailable: {path}")
    return paths


def read_license(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise NoticeError(f"read dependency license {path}: {error}") from error
    if len(payload) > MAX_LICENSE_BYTES:
        raise NoticeError(
            f"dependency license exceeds {MAX_LICENSE_BYTES} bytes: {path}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoticeError(
            f"dependency license is not UTF-8: {path}: {error}"
        ) from error
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def render_notices(
    metadata: Mapping[str, Any],
    package_name: str,
    target: str,
    workspace_license: Path,
    supplemental_licenses: SupplementalLicenses | None = None,
) -> str:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise NoticeError("Cargo metadata packages must be an array")
    package_by_id = {
        str(package["id"]): package
        for package in packages
        if isinstance(package, dict) and "id" in package
    }
    linked = linked_package_ids(metadata, package_name)
    selected = sorted(
        (package_by_id[package_id] for package_id in linked),
        key=lambda package: (str(package.get("name")), str(package.get("version"))),
    )
    texts: dict[str, str] = {}
    text_labels: dict[str, set[str]] = {}
    inventory: list[dict[str, Any]] = []
    for package in selected:
        declared_license = package.get("license")
        if not isinstance(declared_license, str) or not declared_license.strip():
            raise NoticeError(
                f"{package.get('name')} {package.get('version')} has no declared license"
            )
        digests: list[str] = []
        for path in license_paths(package, workspace_license, supplemental_licenses):
            text = read_license(path)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            texts.setdefault(digest, text)
            text_labels.setdefault(digest, set()).add(path.name)
            digests.append(digest)
        inventory.append(
            {
                "name": str(package.get("name")),
                "version": str(package.get("version")),
                "license": declared_license,
                "source": str(package.get("source") or "workspace"),
                "repository": str(package.get("repository") or "unspecified"),
                "license_texts": sorted(set(digests)),
            }
        )

    lines = [
        "KAPSL ORT ADAPTER RUST DEPENDENCY NOTICES",
        "",
        f"Package: {package_name}",
        f"Target: {target}",
        "Dependency scope: normal linked dependencies from locked Cargo metadata",
        "",
        "DEPENDENCY INVENTORY",
        "====================",
        "",
    ]
    for package in inventory:
        lines.extend(
            [
                f"- {package['name']} {package['version']}",
                f"  Declared license: {package['license']}",
                f"  Source: {package['source']}",
                f"  Repository: {package['repository']}",
                "  License text SHA-256: " + ", ".join(package["license_texts"]),
            ]
        )
    lines.extend(["", "LICENSE TEXTS", "=============", ""])
    for digest in sorted(texts):
        labels = ", ".join(sorted(text_labels[digest]))
        lines.extend(
            [
                "-" * 80,
                f"SHA-256: {digest}",
                f"Source filenames: {labels}",
                "-" * 80,
                texts[digest].rstrip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest-path", type=Path, required=True)
    result.add_argument("--package", default="kapsl-backend-ort")
    result.add_argument("--target", default="x86_64-unknown-linux-gnu")
    result.add_argument("--workspace-license", type=Path, required=True)
    result.add_argument("--supplemental-license-index", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        metadata = load_metadata(args.manifest_path.resolve(), args.target)
        supplemental_licenses = load_supplemental_licenses(
            args.supplemental_license_index.resolve()
        )
        notices = render_notices(
            metadata,
            args.package,
            args.target,
            args.workspace_license.resolve(),
            supplemental_licenses,
        )
        atomic_write(args.output.resolve(), notices.encode("utf-8"))
        print(args.output.resolve())
        return 0
    except NoticeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
