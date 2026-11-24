"""Launcher that refuses to start SGLang unless the adapter will govern it.

SGLang's plugin loader is fail-open by design: a plugin that is not installed,
is filtered out by ``SGLANG_PLUGINS``, raises, or whose hook target no longer
exists is logged and skipped, and the server starts anyway. ``kapsl-sglang-serve``
runs the same loader in the launcher process first and checks the outcome, so
those cases stop here, before the model loads, instead of producing an
unmanaged endpoint. It then hands over to ``sglang.launch_server`` unchanged.

The pre-flight covers the launcher process only. Each scheduler subprocess
re-verifies its own hooks in ``hooks.around_scheduler_init``.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata
from typing import Any

from .capacity import UnsupportedProfileError, require_pinned_version
from .config import AdapterConfig, ConfigError, load_config
from .hooks import (
    ENTRY_POINT_GROUP,
    PLUGIN_ENTRY_POINT,
    PLUGIN_NAME,
    HookInstallError,
    installed_sglang_version,
    verify_hooks_applied,
)

SGLANG_PLUGINS_ENV = "SGLANG_PLUGINS"
SGLANG_LAUNCH_MODULE = "sglang.launch_server"
EXIT_CONFIG = 78  # sysexits.h EX_CONFIG


class PreflightError(RuntimeError):
    """SGLang would start without the Kapsl adapter governing it."""


def check_plugin_selected(environ: Mapping[str, str]) -> None:
    allowed = {
        name.strip()
        for name in environ.get(SGLANG_PLUGINS_ENV, "").split(",")
        if name.strip()
    }
    if allowed and PLUGIN_NAME not in allowed:
        raise PreflightError(
            f"{SGLANG_PLUGINS_ENV} is set and does not include {PLUGIN_NAME!r}, "
            "so SGLang would skip the Kapsl adapter"
        )


def check_plugin_installed(entry_points: Iterable[Any]) -> None:
    values = [entry.value for entry in entry_points if entry.name == PLUGIN_NAME]
    if values != [PLUGIN_ENTRY_POINT]:
        raise PreflightError(
            f"entry point {PLUGIN_NAME!r} in group {ENTRY_POINT_GROUP!r} must "
            f"resolve to {PLUGIN_ENTRY_POINT!r} exactly once; found {values!r}"
        )


def preflight(
    environ: Mapping[str, str],
    *,
    sglang_version: Callable[[], str],
    entry_points: Callable[[], Iterable[Any]],
    load_plugins: Callable[[], None],
    verify_hooks: Callable[[], None],
) -> AdapterConfig:
    """Run every check that does not need a loaded model, cheapest first."""

    config = load_config(environ)
    require_pinned_version(sglang_version())
    check_plugin_selected(environ)
    check_plugin_installed(entry_points())
    load_plugins()
    verify_hooks()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        preflight(
            os.environ,
            sglang_version=installed_sglang_version,
            entry_points=lambda: metadata.entry_points(group=ENTRY_POINT_GROUP),
            load_plugins=_load_sglang_plugins,
            verify_hooks=verify_hooks_applied,
        )
    except (
        ConfigError,
        UnsupportedProfileError,
        PreflightError,
        HookInstallError,
        ImportError,
    ) as error:
        print(
            f"kapsl-sglang-serve: refusing to start SGLang ungoverned: {error}",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    sys.argv = [SGLANG_LAUNCH_MODULE, *arguments]
    runpy.run_module(SGLANG_LAUNCH_MODULE, run_name="__main__", alter_sys=True)
    return 0


def _load_sglang_plugins() -> None:
    from sglang.srt.plugins import load_plugins

    load_plugins()


if __name__ == "__main__":
    raise SystemExit(main())
