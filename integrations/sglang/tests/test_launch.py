from __future__ import annotations

import contextlib
import io
import os
import types
import unittest
from pathlib import Path
from unittest import mock

from fake_sglang import ENVIRON, PINNED_VERSION, POOL_ENVIRON, POST_CAPTURE_ENV

from kapsl_sglang_adapter import launch
from kapsl_sglang_adapter.capacity import UnsupportedProfileError
from kapsl_sglang_adapter.config import ConfigError
from kapsl_sglang_adapter.hooks import (
    ENTRY_POINT_GROUP,
    PLUGIN_ENTRY_POINT,
    PLUGIN_NAME,
    HookInstallError,
)

PROJECT = Path(__file__).resolve().parents[1]


def entry(name: str = PLUGIN_NAME, value: str = PLUGIN_ENTRY_POINT):
    return types.SimpleNamespace(name=name, value=value)


def without_backing_flag() -> dict[str, str]:
    """A shared-pool environment as an operator writes it, before the launcher."""

    return {
        name: value for name, value in POOL_ENVIRON.items() if name != POST_CAPTURE_ENV
    }


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.steps: list[str] = []
        self.entries = [entry()]
        self.version = PINNED_VERSION

    def run_preflight(self, environ=ENVIRON, **overrides):
        def step(name: str, result=None):
            def call():
                self.steps.append(name)
                return result() if callable(result) else result

            return call

        collaborators = {
            "sglang_version": step("version", lambda: self.version),
            "entry_points": step("entry_points", lambda: self.entries),
            "load_plugins": step("load_plugins"),
            "verify_hooks": step("verify_hooks"),
        }
        collaborators.update(overrides)
        return launch.preflight(environ, **collaborators)

    def test_checks_run_cheapest_first_and_return_the_configuration(self) -> None:
        config = self.run_preflight()
        self.assertEqual(config.participant_id, "sglang:scheduler")
        self.assertEqual(
            self.steps, ["version", "entry_points", "load_plugins", "verify_hooks"]
        )

    def test_missing_configuration_stops_before_sglang_is_touched(self) -> None:
        with self.assertRaises(ConfigError):
            self.run_preflight(environ={})
        self.assertEqual(self.steps, [])

    def test_an_unpinned_sglang_stops_before_plugins_load(self) -> None:
        self.version = "0.5.22"
        with self.assertRaises(UnsupportedProfileError):
            self.run_preflight()
        self.assertEqual(self.steps, ["version"])

    def test_a_plugin_allowlist_that_omits_the_adapter_is_refused(self) -> None:
        for allowlist in ("xpu_hooks", "xpu_hooks, other", "kapsl"):
            with self.subTest(allowlist):
                self.steps.clear()
                with self.assertRaisesRegex(launch.PreflightError, "SGLANG_PLUGINS"):
                    self.run_preflight(
                        environ={**ENVIRON, "SGLANG_PLUGINS": allowlist}
                    )
                self.assertNotIn("load_plugins", self.steps)
        for allowlist in ("", " , ", "kapsl_kv", "xpu_hooks, kapsl_kv "):
            with self.subTest(allowlist):
                self.run_preflight(environ={**ENVIRON, "SGLANG_PLUGINS": allowlist})

    def test_the_entry_point_must_be_installed_exactly_once(self) -> None:
        for entries in (
            [],
            [entry(name="other")],
            [entry(value="somewhere.else:activate")],
            [entry(), entry()],
        ):
            with self.subTest(entries=entries):
                self.entries = entries
                self.steps.clear()
                with self.assertRaisesRegex(launch.PreflightError, PLUGIN_NAME):
                    self.run_preflight()
                self.assertNotIn("load_plugins", self.steps)
        self.entries = [entry(name="unrelated", value="x:y"), entry()]
        self.run_preflight()

    def test_unapplied_hooks_are_reported(self) -> None:
        def verify() -> None:
            raise HookInstallError("release_kv_cache cannot be resolved")

        with self.assertRaises(HookInstallError):
            self.run_preflight(verify_hooks=verify)

    def test_a_shared_pool_launch_may_leave_the_backing_to_the_launcher(self) -> None:
        unset = without_backing_flag()
        for environ in (unset, POOL_ENVIRON, {**unset, POST_CAPTURE_ENV: "true"}):
            self.assertTrue(self.run_preflight(environ=environ).shared_pool)

    def test_a_shared_pool_launch_with_the_backing_turned_off_is_refused(self) -> None:
        for value in ("0", "false", "No"):
            with self.subTest(value), self.assertRaisesRegex(
                launch.PreflightError, "turns off the VMM-backed KV pool"
            ):
                self.run_preflight(environ={**POOL_ENVIRON, POST_CAPTURE_ENV: value})
            self.assertEqual(self.steps, [])

    def test_the_opaque_tier_does_not_look_at_the_backing_variable(self) -> None:
        config = self.run_preflight(environ={**ENVIRON, POST_CAPTURE_ENV: "0"})
        self.assertFalse(config.shared_pool)


class MainTests(unittest.TestCase):
    def run_main(self, environ, **patches):
        stderr = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, environ, clear=True))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            stack.enter_context(mock.patch.object(launch.sys, "argv", ["x"]))
            run_module = stack.enter_context(
                mock.patch.object(launch.runpy, "run_module")
            )
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(launch, name, value))
            code = launch.main(["--model-path", "m", "--port", "30000"])
            argv = list(launch.sys.argv)
            self.environ_at_exit = dict(os.environ)
        return code, stderr.getvalue(), run_module, argv

    def verified(self, environ):
        return self.run_main(
            environ,
            installed_sglang_version=lambda: PINNED_VERSION,
            _load_sglang_plugins=lambda: None,
            verify_hooks_applied=lambda: None,
            metadata=types.SimpleNamespace(entry_points=lambda group: [entry()]),
        )

    def test_a_shared_pool_launch_turns_on_sglangs_vmm_backed_pool(self) -> None:
        code, stderr, run_module, _ = self.verified(without_backing_flag())
        self.assertEqual((code, stderr), (0, ""))
        run_module.assert_called_once()
        # Set before SGLang resolves its arguments, and inherited by the
        # scheduler subprocess.
        self.assertEqual(self.environ_at_exit[POST_CAPTURE_ENV], "1")

    def test_an_opaque_launch_leaves_the_environment_alone(self) -> None:
        code, _, run_module, _ = self.verified(ENVIRON)
        self.assertEqual(code, 0)
        run_module.assert_called_once()
        self.assertNotIn(POST_CAPTURE_ENV, self.environ_at_exit)

    def test_a_shared_pool_launch_with_the_backing_off_starts_nothing(self) -> None:
        code, stderr, run_module, _ = self.verified(
            {**POOL_ENVIRON, POST_CAPTURE_ENV: "false"}
        )
        self.assertEqual(code, launch.EXIT_CONFIG)
        self.assertIn("refusing to start SGLang ungoverned", stderr)
        run_module.assert_not_called()

    def test_unconfigured_launch_exits_without_starting_sglang(self) -> None:
        code, stderr, run_module, _ = self.run_main({})
        self.assertEqual(code, launch.EXIT_CONFIG)
        self.assertIn("refusing to start SGLang ungoverned", stderr)
        self.assertIn("KAPSL_KV_PARTICIPANT_CONFIG is required", stderr)
        run_module.assert_not_called()

    def test_missing_sglang_exits_without_starting_anything(self) -> None:
        def no_sglang() -> str:
            raise UnsupportedProfileError("SGLang is not installed")

        code, stderr, run_module, _ = self.run_main(
            ENVIRON, installed_sglang_version=no_sglang
        )
        self.assertEqual(code, launch.EXIT_CONFIG)
        self.assertIn("SGLang is not installed", stderr)
        run_module.assert_not_called()

    def test_an_import_failure_while_loading_plugins_is_refused(self) -> None:
        def broken() -> None:
            raise ImportError("No module named 'sglang.srt.plugins'")

        code, stderr, run_module, _ = self.run_main(
            ENVIRON,
            installed_sglang_version=lambda: PINNED_VERSION,
            _load_sglang_plugins=broken,
            metadata=types.SimpleNamespace(entry_points=lambda group: [entry()]),
        )
        self.assertEqual(code, launch.EXIT_CONFIG)
        self.assertIn("sglang.srt.plugins", stderr)
        run_module.assert_not_called()

    def test_a_verified_launch_hands_over_to_sglang_unchanged(self) -> None:
        groups: list[str] = []

        def entry_points(group: str):
            groups.append(group)
            return [entry()]

        code, stderr, run_module, argv = self.run_main(
            ENVIRON,
            installed_sglang_version=lambda: PINNED_VERSION,
            _load_sglang_plugins=lambda: None,
            verify_hooks_applied=lambda: None,
            metadata=types.SimpleNamespace(entry_points=entry_points),
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(groups, [ENTRY_POINT_GROUP])
        run_module.assert_called_once_with(
            "sglang.launch_server", run_name="__main__", alter_sys=True
        )
        self.assertEqual(
            argv, ["sglang.launch_server", "--model-path", "m", "--port", "30000"]
        )


class PackagingTests(unittest.TestCase):
    def test_pyproject_declares_the_plugin_and_launcher_the_code_expects(self) -> None:
        pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'[project.entry-points."{ENTRY_POINT_GROUP}"]', pyproject)
        self.assertIn(f'{PLUGIN_NAME} = "{PLUGIN_ENTRY_POINT}"', pyproject)
        self.assertIn(
            'kapsl-sglang-serve = "kapsl_sglang_adapter.launch:main"', pyproject
        )

    def test_package_and_project_versions_agree(self) -> None:
        from kapsl_sglang_adapter import ADAPTER_VERSION

        pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'version = "{ADAPTER_VERSION}"', pyproject)

    def test_license_files_match_the_repository_root(self) -> None:
        repository = PROJECT.parents[1]
        for name in ("LICENSE", "NOTICE"):
            canonical = repository / name
            if not canonical.exists():  # running from an sdist
                self.skipTest("repository root is not available")
            self.assertEqual(
                (PROJECT / name).read_text(encoding="utf-8").splitlines(),
                canonical.read_text(encoding="utf-8").splitlines(),
            )


if __name__ == "__main__":
    unittest.main()
