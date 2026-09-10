from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import stat
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_ort_platform_runtime as platforms  # noqa: E402
import fetch_ort_runtime as legacy  # noqa: E402


def archive(entries: list[tuple[str, bytes]], kind: str, link: bool = False) -> bytes:
    output = io.BytesIO()
    if kind == "zip":
        with zipfile.ZipFile(output, "w") as handle:
            for name, value in entries:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (
                    (stat.S_IFLNK if link else stat.S_IFREG) | 0o755
                ) << 16
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    handle.writestr(info, value)
    else:
        with tarfile.open(fileobj=output, mode="w:gz") as handle:
            for name, value in entries:
                info = tarfile.TarInfo(name)
                if link:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "other"
                    handle.addfile(info)
                else:
                    info.size = len(value)
                    handle.addfile(info, io.BytesIO(value))
    return output.getvalue()


def fixture(payload: bytes, kind: str) -> platforms.PlatformRuntime:
    return platforms.PlatformRuntime(
        "fixture-cpu",
        "fixture-target",
        "adapter",
        f"fixture.{kind}",
        hashlib.sha256(payload).hexdigest(),
        1024 * 1024,
        (
            platforms.RuntimeFile(
                "lib/runtime", "runtime", 7, hashlib.sha256(b"runtime").hexdigest()
            ),
        ),
    )


class PlatformRuntimeTests(unittest.TestCase):
    def test_retained_platforms_have_explicit_distinct_runtime_locks(self):
        self.assertEqual(
            set(platforms.CPU_RUNTIMES),
            {
                "linux-x86_64",
                "linux-aarch64",
                "macos-x86_64",
                "macos-aarch64",
                "windows-x86_64",
            },
        )
        self.assertEqual(len({r.target for r in platforms.CPU_RUNTIMES.values()}), 5)
        self.assertEqual(
            len({r.archive_sha256 for r in platforms.CPU_RUNTIMES.values()}), 5
        )
        for name, runtime in platforms.CPU_RUNTIMES.items():
            self.assertEqual(runtime.platform, name)
            self.assertIn("/v1.23.2/", runtime.url)
            for file in runtime.files:
                self.assertEqual(Path(file.output_name).name, file.output_name)
                self.assertGreater(file.size, 0)
                self.assertRegex(file.sha256, r"^[0-9a-f]{64}$")
        linux = platforms.CPU_RUNTIMES["linux-x86_64"]
        self.assertEqual(linux.url, legacy.RUNTIME_ARCHIVE_URL)
        self.assertEqual(linux.files[0].sha256, legacy.RUNTIME_LIBRARY_SHA256)
        windows = platforms.CPU_RUNTIMES["windows-x86_64"]
        self.assertEqual(
            [f.output_name for f in windows.files if f.build_only], ["onnxruntime.lib"]
        )

    def test_tar_and_zip_extract_only_the_selected_regular_member(self):
        for kind in ("tgz", "zip"):
            with self.subTest(kind=kind):
                payload = archive(
                    [("lib/runtime", b"runtime"), ("../outside", b"untrusted")], kind
                )
                self.assertEqual(
                    platforms.extract_runtime(payload, fixture(payload, kind)),
                    {"runtime": b"runtime"},
                )

    def test_archive_member_hash_size_and_platform_mismatch_fail(self):
        for kind in ("tgz", "zip"):
            payload = archive([("lib/runtime", b"runtime")], kind)
            runtime = fixture(payload, kind)
            for candidate in (
                dataclasses.replace(runtime, archive_sha256="0" * 64),
                dataclasses.replace(runtime, maximum_archive_bytes=1),
                dataclasses.replace(
                    runtime,
                    files=(dataclasses.replace(runtime.files[0], sha256="0" * 64),),
                ),
                dataclasses.replace(
                    runtime, files=(dataclasses.replace(runtime.files[0], size=8),)
                ),
            ):
                with (
                    self.subTest(kind=kind),
                    self.assertRaises(platforms.PlatformRuntimeError),
                ):
                    platforms.extract_runtime(payload, candidate)
            with self.assertRaises(platforms.PlatformRuntimeError):
                platforms.extract_runtime(
                    payload, platforms.CPU_RUNTIMES["macos-aarch64"]
                )

    def test_duplicate_missing_and_link_members_are_rejected(self):
        for kind in ("tgz", "zip"):
            for entries, link in (
                ([("lib/runtime", b"runtime")] * 2, False),
                ([("lib/other", b"runtime")], False),
                ([("lib/runtime", b"runtime")], True),
            ):
                payload = archive(entries, kind, link)
                with (
                    self.subTest(kind=kind, entries=entries, link=link),
                    self.assertRaises(platforms.PlatformRuntimeError),
                ):
                    platforms.extract_runtime(payload, fixture(payload, kind))

    def test_all_inputs_are_authenticated_before_creating_output(self):
        payload = archive([("lib/runtime", b"runtime")], "tgz")
        runtime = fixture(payload, "tgz")
        bad = dataclasses.replace(
            runtime,
            files=runtime.files
            + (platforms.RuntimeFile("missing", "other", 1, "0" * 64),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "inputs"
            with self.assertRaises(platforms.PlatformRuntimeError):
                platforms.prepare(bad, payload, output)
            self.assertFalse(output.exists())
            platforms.prepare(runtime, payload, output)
            self.assertEqual((output / "runtime").read_bytes(), b"runtime")
            receipt = json.loads((output / "runtime-inputs.json").read_text())
            self.assertEqual(receipt["archive_sha256"], runtime.archive_sha256)
            self.assertEqual(receipt["platform"], "fixture-cpu")
            with self.assertRaisesRegex(platforms.PlatformRuntimeError, "empty"):
                platforms.prepare(runtime, payload, output)

    def test_windows_write_does_not_require_posix_fchmod(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.dll"
            with mock.patch.object(legacy, "os", wraps=legacy.os) as windows_os:
                windows_os.name = "nt"
                windows_os.fchmod = mock.Mock(
                    side_effect=AssertionError("fchmod is unavailable")
                )
                legacy.atomic_write(path, b"runtime")
            self.assertEqual(path.read_bytes(), b"runtime")


if __name__ == "__main__":
    unittest.main()
