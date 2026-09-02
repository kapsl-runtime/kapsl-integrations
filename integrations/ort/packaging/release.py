#!/usr/bin/env python3
"""Validate, split, and index immutable signed ORT pack releases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from package_cpu import (
    ARTIFACT_DOMAIN,
    PackageError,
    atomic_write,
    json_bytes,
    sign_artifact,
)


VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
STABLE_TAG = re.compile(
    rf"kapsl-ort-packs-v(?P<adapter>{VERSION})-kapsl-v(?P<kapsl>{VERSION})"
)
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
PROFILES = ("cpu", "cuda12", "tensorrt10")
MAX_RELEASE_PART_BYTES = 1_900_000_000
READ_BLOCK_BYTES = 8 * 1024 * 1024
PUBLIC_KEY_DER_PREFIX = bytes.fromhex("302a300506032b6570032100")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def parse_boolean(value: str, label: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise PackageError(f"{label} must be true or false")
    return value.lower() == "true"


def adapter_version(repository_root: Path) -> str:
    manifest = repository_root / "integrations/ort/Cargo.toml"
    with manifest.open("rb") as source:
        version = tomllib.load(source)["package"]["version"]
    if re.fullmatch(VERSION, version) is None:
        raise PackageError(
            "ORT adapter version must be an exact stable semantic version"
        )
    return version


def append_github_outputs(path: Path, values: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise PackageError(f"workflow output {name} contains a line break")
            output.write(f"{name}={value}\n")


def validate_release_ref(args: argparse.Namespace) -> None:
    version = adapter_version(args.repository_root.resolve())
    requested_publish = parse_boolean(args.requested_publish, "requested-publish")
    publish = args.event_name == "push" or requested_publish
    requested_kapsl = args.requested_kapsl_version
    match = STABLE_TAG.fullmatch(args.ref_name) if args.ref_type == "tag" else None

    if args.event_name == "push" and args.ref_type != "tag":
        raise PackageError("automatic ORT publication requires a tag push")
    if args.ref_type == "tag" and match is None:
        raise PackageError(
            f"release tag is not an exact stable ORT pack tag: {args.ref_name}"
        )
    if match is not None:
        if match.group("adapter") != version:
            raise PackageError(
                f"release tag adapter version is {match.group('adapter')}; expected {version}"
            )
        kapsl_version = match.group("kapsl")
        if requested_kapsl and requested_kapsl != kapsl_version:
            raise PackageError(
                f"requested Kapsl version {requested_kapsl} differs from tag {kapsl_version}"
            )
    else:
        kapsl_version = requested_kapsl
    if re.fullmatch(VERSION, kapsl_version or "") is None:
        raise PackageError(
            "Kapsl compatibility must be an exact stable semantic version"
        )

    expected_tag = f"kapsl-ort-packs-v{version}-kapsl-v{kapsl_version}"
    if publish and (args.ref_type != "tag" or args.ref_name != expected_tag):
        raise PackageError(
            f"publication requires selecting the exact existing tag {expected_tag}"
        )
    if args.requested_profile not in (*PROFILES, "all"):
        raise PackageError(f"unknown requested profile {args.requested_profile}")
    profiles = (
        PROFILES
        if publish or args.requested_profile == "all"
        else (args.requested_profile,)
    )
    append_github_outputs(
        args.github_output,
        {
            "adapter_version": version,
            "kapsl_version": kapsl_version,
            "publish": str(publish).lower(),
            "release_tag": expected_tag,
            "matrix": json.dumps({"profile": profiles}, separators=(",", ":")),
        },
    )


def parse_signature(path: Path) -> str:
    value = path.read_text(encoding="ascii").strip()
    if not value.startswith("ed25519:"):
        raise PackageError(f"artifact signature has an unsupported format: {path}")
    try:
        decoded = base64.b64decode(value.removeprefix("ed25519:"), validate=True)
    except ValueError as error:
        raise PackageError(f"artifact signature is invalid Base64: {path}") from error
    if len(decoded) != 64:
        raise PackageError(f"artifact signature is not 64 bytes: {path}")
    return value


def raw_public_key(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value.removeprefix("ed25519:"), validate=True)
    except ValueError as error:
        raise PackageError("expected signing public key is invalid Base64") from error
    if len(decoded) != 32:
        raise PackageError("expected signing public key is not 32 bytes")
    return decoded


def verify_signature(public_key: str, digest: str, signature: str) -> None:
    message = ARTIFACT_DOMAIN + f"sha256:{digest}".encode("ascii")
    with tempfile.TemporaryDirectory(prefix="kapsl-ort-signature-") as temporary:
        root = Path(temporary)
        key_path = root / "public.der"
        message_path = root / "message"
        signature_path = root / "signature"
        key_path.write_bytes(PUBLIC_KEY_DER_PREFIX + raw_public_key(public_key))
        message_path.write_bytes(message)
        signature_path.write_bytes(
            base64.b64decode(signature.removeprefix("ed25519:"), validate=True)
        )
        completed = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                str(key_path),
                "-rawin",
                "-in",
                str(message_path),
                "-sigfile",
                str(signature_path),
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise PackageError(
                "artifact signature does not match the release public key"
            )


def split_archive(
    path: Path, output_dir: Path, part_bytes: int
) -> list[dict[str, Any]]:
    if part_bytes <= 0 or part_bytes >= 2 * 1024 * 1024 * 1024:
        raise PackageError("release part size must be positive and below 2 GiB")
    result: list[dict[str, Any]] = []
    archive_size = path.stat().st_size
    remaining = archive_size
    with path.open("rb") as source:
        index = 0
        while remaining > 0:
            name = f"{path.name}.part-{index:03d}"
            destination = output_dir / name
            digest = hashlib.sha256()
            written = 0
            with destination.open("xb") as output:
                while written < part_bytes:
                    block = source.read(min(READ_BLOCK_BYTES, part_bytes - written))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                destination.unlink()
                raise PackageError(f"release archive ended early while writing {name}")
            result.append({"name": name, "sha256": digest.hexdigest(), "size": written})
            remaining -= written
            index += 1
    if not result or sum(item["size"] for item in result) != archive_size:
        raise PackageError(f"failed to split complete release archive {path}")
    return result


def release_url(repository: str, tag: str, name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise PackageError("release repository must be an owner/name pair")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise PackageError(f"release asset name is not portable: {name}")
    return f"https://github.com/{repository}/releases/download/{tag}/{name}"


def validate_release_identity(args: argparse.Namespace) -> None:
    if re.fullmatch(VERSION, args.adapter_version) is None:
        raise PackageError("release adapter version is not stable semantic versioning")
    if re.fullmatch(VERSION, args.kapsl_version) is None:
        raise PackageError("release Kapsl version is not stable semantic versioning")
    expected_tag = (
        f"kapsl-ort-packs-v{args.adapter_version}-kapsl-v{args.kapsl_version}"
    )
    if args.release_tag != expected_tag:
        raise PackageError(f"release identity requires exact tag {expected_tag}")
    if HEX_COMMIT.fullmatch(args.source_commit) is None:
        raise PackageError("release source commit is not exact lowercase hexadecimal")


def prepare_profile(args: argparse.Namespace) -> None:
    validate_release_identity(args)
    if args.profile not in PROFILES:
        raise PackageError(f"unknown ORT release profile {args.profile}")
    directory = args.directory.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise PackageError(f"profile release output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    filename = (
        f"kapsl-backend-onnx-{args.profile}-{args.kapsl_version}-linux-x86_64.tar.gz"
    )
    archive = directory / filename
    manifest_path = directory / f"{filename}.manifest.json"
    checksum_path = directory / f"{filename}.sha256"
    signature_path = directory / f"{filename}.sig"
    for path in (archive, manifest_path, checksum_path, signature_path):
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"release handoff is missing regular file {path}")

    archive_digest = sha256_file(archive)
    expected_checksum = f"{archive_digest}  {filename}\n"
    if checksum_path.read_text(encoding="ascii") != expected_checksum:
        raise PackageError("release archive checksum handoff does not match its bytes")
    signature = parse_signature(signature_path)
    verify_signature(args.expected_public_key, archive_digest, signature)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "backend": "onnx",
        "profile": args.profile,
        "pack_version": args.adapter_version,
        "compatible_kapsl": f"={args.kapsl_version}",
        "platform": "linux-x86_64",
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise PackageError(
                f"release manifest {field} is {manifest.get(field)!r}; expected {expected!r}"
            )

    parts = split_archive(archive, output_dir, args.part_bytes)
    for item in parts:
        item["url"] = release_url(args.repository, args.release_tag, item["name"])
    for source in (manifest_path, checksum_path, signature_path):
        shutil.copyfile(source, output_dir / source.name)
    catalog_name = f"{filename}.release.json"
    catalog_path = output_dir / catalog_name
    catalog = {
        "schema_version": 1,
        "release_tag": args.release_tag,
        "source_repository": f"https://github.com/{args.repository}",
        "source_commit": args.source_commit,
        "backend": "onnx",
        "profile": args.profile,
        "pack_version": args.adapter_version,
        "compatible_kapsl": f"={args.kapsl_version}",
        "platform": "linux-x86_64",
        "archive": {
            "name": filename,
            "sha256": archive_digest,
            "size": archive.stat().st_size,
            "signature": signature,
            "signature_asset": {
                "name": signature_path.name,
                "sha256": sha256_file(signature_path),
                "size": signature_path.stat().st_size,
            },
            "manifest": {
                "name": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "size": manifest_path.stat().st_size,
            },
            "checksum": {
                "name": checksum_path.name,
                "sha256": sha256_file(checksum_path),
                "size": checksum_path.stat().st_size,
            },
            "parts": parts,
        },
    }
    atomic_write(catalog_path, json_bytes(catalog))
    catalog_digest = sha256_file(catalog_path)
    catalog_signature = sign_artifact(
        args.signing_key, args.expected_public_key, catalog_digest
    )
    atomic_write(
        output_dir / f"{catalog_name}.sig", f"{catalog_signature}\n".encode("ascii")
    )
    if args.consume_archive:
        archive.unlink()


def validate_uploaded_assets(
    catalogs: Sequence[Path],
    catalog_payloads: Sequence[Mapping[str, Any]],
    github_assets_path: Path,
) -> None:
    expected: dict[str, tuple[int, str]] = {}

    def add(name: str, size: int, digest: str) -> None:
        if name in expected:
            raise PackageError(f"release catalogs repeat asset {name}")
        expected[name] = (size, digest)

    for path, payload in zip(catalogs, catalog_payloads, strict=True):
        signature_path = path.with_name(f"{path.name}.sig")
        add(path.name, path.stat().st_size, sha256_file(path))
        add(
            signature_path.name,
            signature_path.stat().st_size,
            sha256_file(signature_path),
        )
        archive = payload["archive"]
        for field in ("manifest", "checksum", "signature_asset"):
            item = archive[field]
            add(item["name"], item["size"], item["sha256"])
        for item in archive["parts"]:
            add(item["name"], item["size"], item["sha256"])

    observed_payload = json.loads(github_assets_path.read_text(encoding="utf-8"))
    observed_items = observed_payload.get("assets")
    if not isinstance(observed_items, list):
        raise PackageError("GitHub release asset inventory has no asset list")
    observed: dict[str, tuple[int, str]] = {}
    for item in observed_items:
        if not isinstance(item, dict) or item.get("state") != "uploaded":
            raise PackageError("GitHub release contains an incomplete asset")
        name = item.get("name")
        digest = item.get("digest")
        size = item.get("size")
        if not isinstance(name, str) or name in observed:
            raise PackageError("GitHub release contains an invalid duplicate asset")
        if not isinstance(size, int) or not isinstance(digest, str):
            raise PackageError(f"GitHub release asset metadata is incomplete: {name}")
        observed[name] = (size, digest.removeprefix("sha256:"))

    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
        raise PackageError(
            "GitHub release assets do not match signed catalogs: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )


def assemble_index(args: argparse.Namespace) -> None:
    validate_release_identity(args)
    input_dir = args.input_dir.resolve()
    catalogs = sorted(input_dir.glob("*.tar.gz.release.json"))
    if len(catalogs) != len(PROFILES):
        raise PackageError("release index requires exactly three profile catalogs")
    profiles: dict[str, Any] = {}
    catalog_payloads: list[Mapping[str, Any]] = []
    for path in catalogs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalog_payloads.append(payload)
        profile = payload.get("profile")
        signature_path = path.with_name(f"{path.name}.sig")
        signature = parse_signature(signature_path)
        digest = sha256_file(path)
        verify_signature(args.expected_public_key, digest, signature)
        expected = {
            "release_tag": args.release_tag,
            "source_commit": args.source_commit,
            "pack_version": args.adapter_version,
            "compatible_kapsl": f"={args.kapsl_version}",
        }
        if profile not in PROFILES or profile in profiles:
            raise PackageError(f"release index has invalid profile catalog {profile!r}")
        for field, value in expected.items():
            if payload.get(field) != value:
                raise PackageError(f"profile catalog {path.name} has invalid {field}")
        profiles[profile] = {
            "catalog": {
                "name": path.name,
                "url": release_url(args.repository, args.release_tag, path.name),
                "sha256": digest,
                "signature": signature,
            },
            "archive": payload["archive"],
        }
    if set(profiles) != set(PROFILES):
        raise PackageError("release index does not contain every required ORT profile")
    validate_uploaded_assets(catalogs, catalog_payloads, args.github_assets.resolve())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"kapsl-ort-packs-v{args.adapter_version}-kapsl-v{args.kapsl_version}"
        "-linux-x86_64.release.json"
    )
    output = output_dir / name
    payload = {
        "schema_version": 1,
        "release_tag": args.release_tag,
        "source_repository": f"https://github.com/{args.repository}",
        "source_commit": args.source_commit,
        "backend": "onnx",
        "pack_version": args.adapter_version,
        "compatible_kapsl": f"={args.kapsl_version}",
        "platform": "linux-x86_64",
        "profiles": {profile: profiles[profile] for profile in PROFILES},
    }
    atomic_write(output, json_bytes(payload))
    signature = sign_artifact(
        args.signing_key, args.expected_public_key, sha256_file(output)
    )
    atomic_write(output_dir / f"{name}.sig", f"{signature}\n".encode("ascii"))


def common_release_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--adapter-version", required=True)
    command.add_argument("--kapsl-version", required=True)
    command.add_argument("--release-tag", required=True)
    command.add_argument("--repository", required=True)
    command.add_argument("--source-commit", required=True)
    command.add_argument("--signing-key", type=Path, required=True)
    command.add_argument("--expected-public-key", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-ref")
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--event-name", required=True)
    validate.add_argument("--ref-type", required=True)
    validate.add_argument("--ref-name", required=True)
    validate.add_argument("--requested-kapsl-version", default="")
    validate.add_argument("--requested-profile", required=True)
    validate.add_argument("--requested-publish", required=True)
    validate.add_argument("--github-output", type=Path, required=True)
    validate.set_defaults(handler=validate_release_ref)

    profile = commands.add_parser("prepare-profile")
    common_release_arguments(profile)
    profile.add_argument("--profile", choices=PROFILES, required=True)
    profile.add_argument("--directory", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--part-bytes", type=int, default=MAX_RELEASE_PART_BYTES)
    profile.add_argument("--consume-archive", action="store_true")
    profile.set_defaults(handler=prepare_profile)

    index = commands.add_parser("assemble-index")
    common_release_arguments(index)
    index.add_argument("--input-dir", type=Path, required=True)
    index.add_argument("--github-assets", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.set_defaults(handler=assemble_index)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        args.handler(args)
        return 0
    except (OSError, KeyError, json.JSONDecodeError, PackageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
