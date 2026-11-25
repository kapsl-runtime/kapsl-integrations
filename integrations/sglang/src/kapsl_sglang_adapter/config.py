"""Adapter configuration.

SGLang has no per-connector configuration channel comparable to vLLM's
``kv_connector_extra_config``, and its scheduler runs in a spawned subprocess
that shares nothing with the launcher except the environment. The participant
identity therefore arrives as one JSON document in one environment variable,
set once by whoever launches the server (an operator or a managing runtime).
The document is the participant configuration of the KV contract; see
:mod:`.participant_config`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .client import parse_unix_endpoint
from .participant_config import (
    FIELD_CONTROL_ENDPOINT,
    FIELD_LEASE_TTL_MS,
    FIELD_MODE,
    MODE_OPAQUE,
    PARTICIPANT_CONFIG_ENV,
    ParticipantConfigError,
    load_participant_config,
)

# Adapter-local tuning; not part of the participant configuration.
ENV_DENIAL_BACKOFF_MS = "KAPSL_SGLANG_DENIAL_BACKOFF_MS"

DEFAULT_TIMEOUT_MS = 2_000
DEFAULT_DENIAL_BACKOFF_MS = 100
MIN_LEASE_TTL_MS = 1_000
SUPPORTED_MODE = MODE_OPAQUE

__all__ = [
    "DEFAULT_DENIAL_BACKOFF_MS",
    "DEFAULT_TIMEOUT_MS",
    "ENV_DENIAL_BACKOFF_MS",
    "MIN_LEASE_TTL_MS",
    "PARTICIPANT_CONFIG_ENV",
    "SUPPORTED_MODE",
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

    @property
    def participant_id(self) -> str:
        """Registered identity: the runtime's base plus this process's role.

        The ``base:role`` shape matches the vLLM connector and the namespace a
        managing runtime uses to recognize the participants it launched.
        """

        return f"{self.participant_base}:scheduler"


def load_config(environ: Mapping[str, str]) -> AdapterConfig:
    try:
        document = load_participant_config(environ)
    except ParticipantConfigError as error:
        raise ConfigError(str(error)) from error
    if document.mode != SUPPORTED_MODE:
        raise ConfigError(
            f"{PARTICIPANT_CONFIG_ENV}: {FIELD_MODE}={document.mode!r} is not "
            f"implemented; this adapter supports only {SUPPORTED_MODE!r}"
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
        denial_backoff_ms=_denial_backoff_ms(environ),
    )


def _denial_backoff_ms(environ: Mapping[str, str]) -> int:
    raw = environ.get(ENV_DENIAL_BACKOFF_MS)
    if raw is None or not raw.strip():
        return DEFAULT_DENIAL_BACKOFF_MS
    try:
        value = int(raw.strip())
    except ValueError as error:
        raise ConfigError(f"{ENV_DENIAL_BACKOFF_MS} must be an integer") from error
    if value < 0:
        raise ConfigError(f"{ENV_DENIAL_BACKOFF_MS} cannot be negative")
    return value
