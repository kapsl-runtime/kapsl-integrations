#!/usr/bin/env python3
"""Build governed ORT core/providers on a Linux CPU host with CUDA build tools.

This command does not execute GPU inference, provision a GPU or publish a pack.
The source checkout must already be at source-lock.json's immutable commit.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_source import (
    PROFILES,
    prepare,
    recipe_identity,
    runtime_names,
    sha256,
    source_lock,
)


def build_command(
    source: Path,
    build_dir: Path,
    profile: str,
    cuda: Path,
    cudnn: Path,
    tensorrt: Path | None,
    jobs: int,
) -> list[str]:
    if profile not in PROFILES or not 1 <= jobs <= 64:
        raise ValueError("invalid ORT build profile or parallelism")
    if profile == "tensorrt10" and tensorrt is None:
        raise ValueError("TensorRT source compilation requires the pinned TensorRT SDK")
    command = [
        sys.executable,
        str(source / "tools/ci_build/build.py"),
        "--build_dir",
        str(build_dir),
        "--config",
        "Release",
        "--update",
        "--build",
        "--build_shared_lib",
        "--skip_tests",
        "--parallel",
        str(jobs),
        "--allow_running_as_root",
        "--use_cuda",
        "--cuda_home",
        str(cuda),
        "--cudnn_home",
        str(cudnn),
    ]
    if profile == "tensorrt10":
        command += ["--use_tensorrt", "--tensorrt_home", str(tensorrt)]
    command += [
        "--cmake_extra_defines",
        "CMAKE_SHARED_LINKER_FLAGS=-Xlinker -Bsymbolic",
        "CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90",
        "onnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON",
    ]
    return command


def stage_outputs(
    build_dir: Path,
    output: Path,
    profile: str,
    prepared: dict,
    command: list[str],
    tools: dict,
) -> dict:
    if output.exists():
        raise ValueError("governed ORT output must be a new directory")
    output.mkdir(parents=True)
    mapping = runtime_names(profile)
    replacements = {
        **mapping,
        "libonnxruntime.so.1": mapping["libonnxruntime.so.1.23.2"],
    }
    files = {}
    for original, name in mapping.items():
        source = build_dir / "Release" / original
        if not source.is_file():
            raise ValueError(f"ORT source build did not produce {original}")
        library = output / name
        shutil.copyfile(source, library)
        library.chmod(0o755)
        subprocess.run(
            ["patchelf", "--set-soname", name, "--set-rpath", "$ORIGIN", str(library)],
            check=True,
        )
        needed = subprocess.check_output(
            ["patchelf", "--print-needed", str(library)], text=True
        ).splitlines()
        for dependency in needed:
            if dependency in replacements:
                subprocess.run(
                    [
                        "patchelf",
                        "--replace-needed",
                        dependency,
                        replacements[dependency],
                        str(library),
                    ],
                    check=True,
                )
            elif "onnxruntime" in dependency:
                raise ValueError(f"unresolved ORT runtime dependency: {dependency}")
        files[name] = {"sha256": sha256(library), "size": library.stat().st_size}
    provenance = {
        "schema_version": 1,
        "profile": profile,
        "source_commit": source_lock()["commit"],
        "source_repository": source_lock()["repository"],
        "recipe": recipe_identity(),
        "prepared_sources": prepared,
        "build_command": command,
        "toolchain": tools,
        "files": files,
    }
    (output / "governed-runtime.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cudnn-home", type=Path, required=True)
    parser.add_argument("--tensorrt-home", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise ValueError(
            "governed accelerator runtime compilation requires Linux x86-64"
        )
    source, build_dir = args.source.resolve(), args.build_dir.resolve()
    cuda, cudnn = args.cuda_home.resolve(), args.cudnn_home.resolve()
    tensorrt = args.tensorrt_home.resolve() if args.tensorrt_home else None
    if build_dir.is_relative_to(source):
        raise ValueError(
            "keep build products outside the authenticated ORT source checkout"
        )
    prepared = prepare(source, args.profile)
    if tensorrt is not None:
        for name, record in source_lock()["tensorrt_headers"]["files"].items():
            header = tensorrt / "include" / name
            if not header.is_file() or sha256(header) != record["sha256"]:
                raise ValueError(
                    f"TensorRT SDK header differs from the pinned 10.9 contract: {name}"
                )
    tools = {}
    for name, executable in [
        ("cuda", str(cuda / "bin/nvcc")),
        ("cmake", "cmake"),
        ("cxx", "c++"),
        ("patchelf", "patchelf"),
    ]:
        resolved = Path(shutil.which(executable) or executable).resolve(strict=True)
        tools[name] = {
            "path": str(resolved),
            "sha256": sha256(resolved),
            "version": subprocess.check_output([str(resolved), "--version"], text=True),
        }
    if "release 12.8," not in tools["cuda"]["version"]:
        raise ValueError(
            "governed ORT compilation requires the reviewed CUDA 12.8 toolchain"
        )
    command = build_command(
        source, build_dir, args.profile, cuda, cudnn, tensorrt, args.jobs
    )
    subprocess.run(command, cwd=source, check=True)
    prepare(source, args.profile)  # Reject source drift during build.
    stage_outputs(
        build_dir, args.output_dir.resolve(), args.profile, prepared, command, tools
    )


if __name__ == "__main__":
    main()
