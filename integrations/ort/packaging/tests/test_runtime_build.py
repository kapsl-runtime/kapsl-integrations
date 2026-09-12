from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
import build_runtime  # noqa: E402
from package_cpu import PackageError, inspect_glibc_contract  # noqa: E402


class BuildConfigurationTests(unittest.TestCase):
    def test_compiler_aliases_produce_the_same_build_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cuda = root / "cuda-12.8"
            cuda.mkdir()
            alias = root / "cuda"
            alias.symlink_to(cuda, target_is_directory=True)
            commands = [
                build_runtime.build_command(
                    root, root / "build", "cuda12", path, root, None, 4
                )
                for path in (cuda, alias)
            ]
            self.assertEqual(commands[0], commands[1])

    def test_rejects_provider_flags_lost_by_cmake_cache_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = root / "build/Release"
            release.mkdir(parents=True)
            values = {
                "CMAKE_HOME_DIRECTORY": str(root / "cmake"),
                "CMAKE_CUDA_COMPILER": str(root / "bin/nvcc"),
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CUDA_ARCHITECTURES": "75;80;86;89;90",
                "onnxruntime_USE_CUDA": "ON",
                "onnxruntime_USE_TENSORRT": "ON",
                "onnxruntime_BUILD_SHARED_LIB": "ON",
                "onnxruntime_BUILD_UNIT_TESTS": "OFF",
                "onnxruntime_DISABLE_RTTI": "OFF",
            }

            def verify(changes: dict[str, str]) -> dict[str, str]:
                (release / "CMakeCache.txt").write_text(
                    "\n".join(
                        f"{key}:STRING={value}"
                        for key, value in (values | changes).items()
                    )
                )
                return build_runtime.verify_build_configuration(
                    root, root / "build", "tensorrt10", root
                )

            self.assertEqual(verify({}), values)
            for key, value in (
                ("onnxruntime_USE_CUDA", "OFF"),
                ("onnxruntime_USE_TENSORRT", "OFF"),
                ("onnxruntime_BUILD_SHARED_LIB", "OFF"),
                ("onnxruntime_BUILD_UNIT_TESTS", "ON"),
                ("CMAKE_CUDA_ARCHITECTURES", "90"),
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    verify({key: value})


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and shutil.which("cc")
    and shutil.which("patchelf"),
    "ELF staging requires Linux, a C compiler and patchelf",
)
class RuntimeStagingTests(unittest.TestCase):
    def helper(self, root: Path, code: str) -> Path:
        source = root / "helper.c"
        source.write_text(code)
        library = root / "libhelper.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-nostdlib", str(source), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        return library

    def test_dependency_free_helper_needs_no_glibc_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = self.helper(
                Path(temporary),
                "extern void __gmon_start__(void) __attribute__((weak)); "
                "void helper(void) { if (__gmon_start__) __gmon_start__(); }",
            )
            self.assertIsNone(
                inspect_glibc_contract(library, "helper", allow_dependency_free=True)
            )
            with self.assertRaisesRegex(PackageError, "no versioned glibc"):
                inspect_glibc_contract(library, "entrypoint")

    def test_unversioned_required_and_unknown_weak_imports_are_rejected(self) -> None:
        for code in (
            "extern int required_import(void); int helper(void) { return required_import(); }",
            "extern int __isoc23_unknown(void) __attribute__((weak)); "
            "int helper(void) { return __isoc23_unknown ? __isoc23_unknown() : 0; }",
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                library = self.helper(Path(temporary), code)
                with self.assertRaisesRegex(PackageError, "no versioned glibc"):
                    inspect_glibc_contract(
                        library, "helper", allow_dependency_free=True
                    )

    def build_fixture(self, root: Path, profile: str) -> Path:
        release = root / "build/Release"
        release.mkdir(parents=True)
        mapping = build_runtime.runtime_names(profile)
        for original in mapping:
            provider = original.endswith(("_cuda.so", "_tensorrt.so"))
            code = (
                "extern int fixture_shared(void); int fixture_value(void) { return fixture_shared(); }"
                if provider
                else "int fixture_shared(void) { return 42; }"
            )
            # Build the dependency before providers, independent of mapping order.
            if original.endswith("_shared.so"):
                continue
            shared = release / "libonnxruntime_providers_shared.so"
            if not shared.exists():
                self.compile(
                    release, shared.name, "int fixture_shared(void) { return 42; }"
                )
            self.compile(release, original, code, provider)
        return root / "build"

    def compile(
        self, directory: Path, name: str, code: str, provider: bool = False
    ) -> None:
        source = directory / (name + ".c")
        source.write_text(code)
        command = [
            "cc",
            "-shared",
            "-fPIC",
            str(source),
            f"-Wl,-soname,{name}",
            "-o",
            str(directory / name),
        ]
        if provider:
            command += [f"-L{directory}", "-l:libonnxruntime_providers_shared.so"]
        subprocess.run(command, check=True, capture_output=True)

    def stage(self, build: Path, output: Path, profile: str) -> dict:
        return build_runtime.stage_outputs(
            build,
            output,
            profile,
            {},
            ["fixture-build"],
            {},
            configure_command=["fixture-configure"],
            configuration={},
        )

    def test_staged_libraries_load_with_private_sonames_and_dependencies(self) -> None:
        for profile in ("cuda12", "tensorrt10"):
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                build = self.build_fixture(root, profile)
                output = root / "staged"
                manifest = self.stage(build, output, profile)
                shutil.rmtree(build)
                self.assertEqual(
                    json.loads((output / "governed-runtime.json").read_text()), manifest
                )
                for name, record in manifest["files"].items():
                    library = output / name
                    self.assertEqual(build_runtime.sha256(library), record["sha256"])
                    self.assertEqual(
                        subprocess.check_output(
                            ["patchelf", "--print-soname", str(library)], text=True
                        ).strip(),
                        name,
                    )
                    self.assertEqual(
                        subprocess.check_output(
                            ["patchelf", "--print-rpath", str(library)], text=True
                        ).strip(),
                        "$ORIGIN",
                    )
                    if name.endswith(("_providers_cuda.so", "_providers_tensorrt.so")):
                        loaded = ctypes.CDLL(str(library))
                        self.assertEqual(loaded.fixture_value(), 42)

    def test_staging_rejects_a_tool_that_leaves_an_incorrect_soname(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self.build_fixture(root, "cuda12")
            run = subprocess.run

            def omit_soname(command: list[str], **kwargs):
                if "--set-soname" in command:
                    return subprocess.CompletedProcess(command, 0)
                return run(command, **kwargs)

            with patch.object(build_runtime.subprocess, "run", side_effect=omit_soname):
                with self.assertRaisesRegex(ValueError, "invalid dynamic contract"):
                    self.stage(build, root / "staged", "cuda12")
            self.assertFalse((root / "staged/governed-runtime.json").exists())


if __name__ == "__main__":
    unittest.main()
