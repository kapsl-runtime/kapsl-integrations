#!/usr/bin/env python3
"""Check pinned provider patches and the real TensorRT allocator interface without GPUs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import prepare_source
from prepare_source import ROOT, prepare, runtime_names, sha256, source_lock

sys.path.insert(0, str(ROOT.parent / "packaging"))
from ort_mirror import mirror_url  # noqa: E402

# Git mirrors of the ORT repository, used when no HTTP source serves the pinned
# files. github.com/microsoft/onnxruntime no longer resolves. A mirror is only
# a transport: every file is still checked against source-lock.json.
ORT_GIT_MIRRORS = ("https://gitcode.com/gh_mirrors/on/onnxruntime.git",)


def fetch(
    urls: Sequence[str], path: Path, digest: str, size: int | None = None
) -> None:
    """Write the first payload that matches the immutable source lock."""
    failures = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(
                    (size if size is not None else 2 * 1024 * 1024) + 1
                )
        except OSError as error:
            failures.append(f"{url}: {error}")
            continue
        if hashlib.sha256(data).hexdigest() == digest and (
            size is None or len(data) == size
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return
        failures.append(f"{url}: differs from its immutable source lock")
    raise ValueError(
        f"no source provided host fixture {path.name}: " + "; ".join(failures)
    )


def fetch_from_git_mirrors(lock: dict, files: dict[str, str], source: Path) -> None:
    """Read pinned ORT files from a mirror's release tag."""
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    failures = []
    for repository in ORT_GIT_MIRRORS:
        # A blob-filtered clone downloads under 1 MB and fetches the pinned
        # files on demand. Git clients too old to complete one fall back to a
        # plain shallow clone, which downloads the whole tree.
        for arguments, timeout in ((["--filter=blob:none"], 180), ([], 1800)):
            with tempfile.TemporaryDirectory(prefix="kapsl-ort-mirror-") as temporary:
                clone = Path(temporary) / "onnxruntime"
                try:
                    subprocess.run(
                        [
                            "git",
                            "clone",
                            "--quiet",
                            "--depth=1",
                            "--no-checkout",
                            f"--branch=v{lock['version']}",
                            *arguments,
                            repository,
                            str(clone),
                        ],
                        check=True,
                        env=environment,
                        timeout=timeout,
                    )
                    commit = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=clone, text=True
                    ).strip()
                    if commit != lock["commit"]:
                        raise ValueError("release tag is not the pinned source commit")
                    for name, digest in files.items():
                        data = subprocess.check_output(
                            ["git", "show", f"HEAD:{name}"],
                            cwd=clone,
                            env=environment,
                            timeout=timeout,
                        )
                        if hashlib.sha256(data).hexdigest() != digest:
                            raise ValueError(
                                f"{name} differs from its immutable source lock"
                            )
                        (source / name).parent.mkdir(parents=True, exist_ok=True)
                        (source / name).write_bytes(data)
                    return
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    failures.append(f"{repository} {arguments}: {error}")
    raise ValueError(
        "no source provided the pinned ORT host fixtures: " + "; ".join(failures)
    )


def fetch_ort_sources(lock: dict, source: Path) -> None:
    """Fetch every pinned ORT file: Kapsl mirror, upstream, then a Git mirror."""
    missing = {}
    for name, digest in lock["files"].items():
        mirrored = mirror_url(lock["version"], f"source/{lock['commit']}/{name}")
        upstream = (
            "https://raw.githubusercontent.com/microsoft/onnxruntime/"
            f"{lock['commit']}/{name}"
        )
        try:
            fetch([url for url in (mirrored, upstream) if url], source / name, digest)
        except ValueError:
            missing[name] = digest
    if missing:
        fetch_from_git_mirrors(lock, missing, source)


def test_allocator_adapter(source: Path, scratch: Path) -> None:
    """Compile the real prepared wrapper methods with a fault-injecting C table."""
    text = (source / "onnxruntime/core/session/allocator_adapters.cc").read_text()
    start = text.index("void* IAllocatorImplWrappingOrtAllocator::Alloc(size_t size)")
    end = text.index("void IAllocatorImplWrappingOrtAllocator::Free(void* p)", start)
    constants = re.findall(
        r"^constexpr uint32_t kOrtAllocator(?:Reserve|AllocOnStream)MinVersion = [0-9]+;$",
        text,
        re.MULTILINE,
    )
    if len(constants) != 2:
        raise ValueError("allocator fixture is missing the pinned ABI version guards")
    scratch.mkdir()
    (scratch / "allocator_adapter_methods.inc").write_text(
        "\n".join(constants) + "\n" + text[start:end]
    )
    executable = scratch / "allocator-adapter-test"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(scratch),
            str(ROOT / "tests/allocator_adapter_test.cc"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def main() -> None:
    lock = source_lock()
    with tempfile.TemporaryDirectory(prefix="kapsl-ort-provider-host-") as temporary:
        root = Path(temporary)
        headers = root / "headers"
        for name, item in lock["tensorrt_headers"]["files"].items():
            fetch(
                [
                    f"https://raw.githubusercontent.com/NVIDIA/TensorRT/{lock['tensorrt_headers']['commit']}/include/{name}"
                ],
                headers / name,
                item["sha256"],
                item["size"],
            )
        executable = root / "allocator-test"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-deprecated-declarations",
                "-I",
                str(ROOT / "include"),
                "-I",
                str(ROOT / "tests"),
                "-isystem",
                str(headers),
                str(ROOT / "tests/allocator_test.cc"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)
        print(
            "TensorRT 10.9 allocator table, lifecycle, async methods and failures: passed",
            flush=True,
        )
        pinned = root / "ort-source"
        fetch_ort_sources(lock, pinned)
        for profile in ("cuda12", "tensorrt10"):
            source = root / profile
            shutil.copytree(pinned, source)
            # A sparse fixture has a different Git commit from the full ORT
            # tree. Authenticate every file above, then exercise preparation
            # and tamper rejection using that fixture's own commit identity.
            subprocess.run(
                ["git", "init", "--quiet", "--initial-branch=fixture"],
                cwd=source,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=source, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Kapsl host test",
                    "-c",
                    "user.email=host-test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "Pinned source fixture",
                ],
                cwd=source,
                check=True,
            )
            fixture_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            fixture_lock = {**lock, "commit": fixture_commit}
            with patch.object(prepare_source, "source_lock", return_value=fixture_lock):
                prepared = prepare(source, profile)
                if prepare(source, profile) != prepared:
                    raise ValueError("resuming source preparation changed its identity")
                relative = next(iter(prepared["files"]))
                target = source / relative
                original = target.read_bytes()
                target.write_bytes(original + b"\n// unexpected source change\n")
                forged = {
                    **prepared,
                    "files": {**prepared["files"], relative: sha256(target)},
                }
                receipt = source / ".kapsl-runtime-prepared.json"
                receipt.write_text(json.dumps(forged))
                try:
                    prepare(source, profile)
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "editing source and its receipt bypassed source authentication"
                    )
                target.write_bytes(original)
                receipt.write_text(json.dumps(prepared))
            test_allocator_adapter(source, root / f"allocator-adapter-{profile}")
            print(
                f"{profile} actual allocator wrappers reject failed Alloc/Reserve/AllocOnStream, including legacy fallbacks: passed",
                flush=True,
            )
            names = runtime_names(profile)
            bridge = (
                source / "onnxruntime/core/session/provider_bridge_ort.cc"
            ).read_text()
            for name in names.values():
                if (
                    "providers_" in name
                    and name.removeprefix("lib").removesuffix(".so") not in bridge
                ):
                    raise ValueError(
                        f"provider loader does not select its isolated library: {name}"
                    )
            print(
                f"Pinned {profile} patches, resume/tamper rejection and provider namespaces: passed",
                flush=True,
            )
        if set(runtime_names("cuda12").values()) & set(
            runtime_names("tensorrt10").values()
        ):
            raise ValueError("accelerator profiles share a runtime SONAME")


if __name__ == "__main__":
    main()
