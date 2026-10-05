"""Compute the optional signed BLAKE3 map from verified extract-pack bytes."""

import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from package_cpu import PackageError


def archive_blake3(archive_path: Path, manifest: dict[str, Any]) -> dict[str, str]:
    # The release caller first authenticates the archive and validates its
    # identity. This function binds both installed digest maps to those bytes.
    from blake3 import blake3

    if manifest.get("installer", {"kind": "extract"}) != {"kind": "extract"}:
        raise PackageError("BLAKE3 release metadata requires an extract pack")
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or manifest.get("entrypoint") not in files
    ):
        raise PackageError(
            "BLAKE3 release metadata requires the complete SHA-256 file map"
        )
    for path, digest in files.items():
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or any(part in ("", ".", "..") for part in path.split("/"))
            or "\\" in path
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            raise PackageError("invalid signed installed-file path or SHA-256 digest")
    result = {}
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                    raise PackageError("unsafe archive path in BLAKE3 input")
                name = path.as_posix()
                if member.isdir():
                    continue
                if not member.isfile() or name not in files or name in result:
                    raise PackageError(
                        "unexpected, duplicate or non-regular archive file"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageError("unreadable archive file")
                sha256 = hashlib.sha256()
                fast = blake3()
                while block := stream.read(1024 * 1024):
                    sha256.update(block)
                    fast.update(block)
                if sha256.hexdigest() != files[name].lower():
                    raise PackageError(f"archive SHA-256 differs from manifest: {name}")
                result[name] = fast.hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise PackageError(f"read archive for BLAKE3 metadata: {error}") from error
    if result.keys() != files.keys():
        raise PackageError("archive is missing signed installed files")
    existing = manifest.get("files_blake3")
    if existing is not None and (
        not isinstance(existing, dict)
        or existing.keys() != result.keys()
        or any(
            not isinstance(digest, str) or digest.lower() != result[path]
            for path, digest in existing.items()
        )
    ):
        raise PackageError("existing BLAKE3 map differs from verified archive")
    return dict(sorted(result.items()))
