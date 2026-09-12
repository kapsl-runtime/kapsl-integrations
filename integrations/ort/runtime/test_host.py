#!/usr/bin/env python3
"""Check pinned provider patches and the real TensorRT allocator interface without GPUs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from unittest.mock import patch

import prepare_source
from prepare_source import ROOT, prepare, runtime_names, sha256, source_lock


def fetch(url: str, path: Path, digest: str, size: int | None = None) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read((size if size is not None else 2 * 1024 * 1024) + 1)
    if hashlib.sha256(data).hexdigest() != digest or (
        size is not None and len(data) != size
    ):
        raise ValueError(
            f"host fixture differs from its immutable source lock: {path.name}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


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
                f"https://raw.githubusercontent.com/NVIDIA/TensorRT/{lock['tensorrt_headers']['commit']}/include/{name}",
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
        for profile in ("cuda12", "tensorrt10"):
            source = root / profile
            for name, digest in lock["files"].items():
                fetch(
                    f"https://raw.githubusercontent.com/microsoft/onnxruntime/{lock['commit']}/{name}",
                    source / name,
                    digest,
                )
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
