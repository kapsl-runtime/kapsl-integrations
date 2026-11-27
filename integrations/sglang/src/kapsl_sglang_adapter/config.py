"""Adapter configuration.

SGLang has no per-connector configuration channel comparable to vLLM's
``kv_connector_extra_config``, and its scheduler runs in a spawned subprocess
that shares nothing with the launcher except the environment. The participant
identity therefore arrives as one JSON document in one environment variable,
set once by whoever launches the server (an operator or a managing runtime).
The document is the participant configuration of the KV contract; see
:mod:`.participant_config`.

The document also selects the tier. ``kapsl_kv_mode: opaque`` is the default
tier and behaves as it always has. ``shared_pool`` together with
``kapsl_live_resize`` selects the elastic shared pool, in which the runtime
owns the KV memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .client import parse_unix_endpoint
from .participant_config import (
    FIELD_CONTROL_ENDPOINT,
    FIELD_LEASE_TTL_MS,
    FIELD_LIVE_RESIZE,
    FIELD_MEMORY_DOMAINS,
    FIELD_MODE,
    FIELD_RANK_DEVICE_MAP,
    FIELD_VMM_CONFORMANCE,
    MODE_OPAQUE,
    MODE_SHARED_POOL,
    PARTICIPANT_CONFIG_ENV,
    ParticipantConfig,
    ParticipantConfigError,
    load_participant_config,
)

# Adapter-local tuning; not part of the participant configuration.
ENV_DENIAL_BACKOFF_MS = "KAPSL_SGLANG_DENIAL_BACKOFF_MS"
ENV_RESIZE_POLL_MS = "KAPSL_SGLANG_RESIZE_POLL_MS"

DEFAULT_TIMEOUT_MS = 2_000
DEFAULT_DENIAL_BACKOFF_MS = 100
DEFAULT_RESIZE_POLL_MS = 100
MIN_LEASE_TTL_MS = 1_000
# The tier an installation gets unless its configuration asks for another.
SUPPORTED_MODE = MODE_OPAQUE
SUPPORTED_MODES = (MODE_OPAQUE, MODE_SHARED_POOL)

__all__ = [
    "DEFAULT_DENIAL_BACKOFF_MS",
    "DEFAULT_RESIZE_POLL_MS",
    "DEFAULT_TIMEOUT_MS",
    "ENV_DENIAL_BACKOFF_MS",
    "ENV_RESIZE_POLL_MS",
    "MIN_LEASE_TTL_MS",
    "PARTICIPANT_CONFIG_ENV",
    "SUPPORTED_MODE",
    "SUPPORTED_MODES",
    "AdapterConfig",
    "ConfigError",
    "load_config",
]


class ConfigError(ValueError):
    """The adapter is installed but not configured to reach a Kapsl runtime."""


@dataclass(frozen=True)
class AdapterConfig:
    control_endpoint: str
    model_fingerprint: str
    memory_domains: tuple[Mapping[str, Any], ...]
    participant_base: str
    lease_ttl_ms: int
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    denial_backoff_ms: int = DEFAULT_DENIAL_BACKOFF_MS
    mode: str = MODE_OPAQUE
    # Shared-pool tier only: the runtime's proof that it precharged the pool,
    # and how often the scheduler asks the runtime for a pending resize.
    provisioning_grant: Mapping[str, Any] | None = None
    resize_poll_ms: int = DEFAULT_RESIZE_POLL_MS

    @property
    def participant_id(self) -> str:
        """Registered identity: the runtime's base plus this process's role.

        The ``base:role`` shape matches the vLLM connector and the namespace a
        managing runtime uses to recognize the participants it launched.
        """

        return f"{self.participant_base}:scheduler"

    @property
    def shared_pool(self) -> bool:
        """Whether the elastic shared-pool tier was requested."""

        return self.mode == MODE_SHARED_POOL


def load_config(environ: Mapping[str, str]) -> AdapterConfig:
    try:
        document = load_participant_config(environ)
    except ParticipantConfigError as error:
        raise ConfigError(str(error)) from error
    if document.mode == MODE_SHARED_POOL:
        _require_elastic_shared_pool(document)
    elif document.mode != MODE_OPAQUE:
        raise ConfigError(
            f"{PARTICIPANT_CONFIG_ENV}: {FIELD_MODE}={document.mode!r} is not "
            f"implemented; this adapter supports {SUPPORTED_MODES}"
        )
    try:
        parse_unix_endpoint(document.control_endpoint)
    except ValueError as error:
        raise ConfigError(
            f"{PARTICIPANT_CONFIG_ENV}: {FIELD_CONTROL_ENDPOINT}: {error}"
        ) from error
    if document.lease_ttl_ms < MIN_LEASE_TTL_MS:
        raise ConfigError(
            f"{PARTICIPANT_CONFIG_ENV}: {FIELD_LEASE_TTL_MS} must be at least "
            f"{MIN_LEASE_TTL_MS}"
        )
    return AdapterConfig(
        control_endpoint=document.control_endpoint,
        model_fingerprint=document.model_fingerprint,
        memory_domains=document.memory_domains,
        participant_base=document.participant_id,
        lease_ttl_ms=document.lease_ttl_ms,
        timeout_ms=(
            DEFAULT_TIMEOUT_MS if document.timeout_ms is None else document.timeout_ms
        ),
        denial_backoff_ms=_milliseconds(
            environ, ENV_DENIAL_BACKOFF_MS, DEFAULT_DENIAL_BACKOFF_MS, minimum=0
        ),
        mode=document.mode,
        provisioning_grant=document.provisioning_grant,
        resize_poll_ms=_milliseconds(
            environ, ENV_RESIZE_POLL_MS, DEFAULT_RESIZE_POLL_MS, minimum=1
        ),
    )


def _require_elastic_shared_pool(document: ParticipantConfig) -> None:
    """Refuse every shared-pool configuration except the one implemented.

    That is an elastic CUDA VMM pool on one device. A fixed (CUDA IPC) pool
    and tensor-parallel placement are not implemented.
    """

    prefix = f"{PARTICIPANT_CONFIG_ENV}: {FIELD_MODE}={MODE_SHARED_POOL!r}"
    if not document.live_resize:
        raise ConfigError(
            f"{prefix} requires {FIELD_LIVE_RESIZE}: this adapter supports only "
            f"{MODE_OPAQUE!r} and the elastic shared pool, not a fixed one"
        )
    if document.vmm_conformance:
        raise ConfigError(
            f"{prefix}: {FIELD_VMM_CONFORMANCE} is not implemented by this adapter"
        )
    domains = document.memory_domains
    if len(domains) != 1 or domains[0].get("kind") != "cuda":
        raise ConfigError(
            f"{prefix} needs exactly one cuda entry in {FIELD_MEMORY_DOMAINS}; "
            "the pool of a single-rank server lives on one device"
        )
    ranks = document.rank_device_map
    if ranks and ranks != {"0": domains[0].get("device_id")}:
        raise ConfigError(
            f"{prefix}: {FIELD_RANK_DEVICE_MAP} may only map rank 0 to the "
            f"device in {FIELD_MEMORY_DOMAINS}"
        )


def _milliseconds(
    environ: Mapping[str, str], name: str, default: int, *, minimum: int
) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer") from error
    if value < 0:
        raise ConfigError(f"{name} cannot be negative")
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return value
