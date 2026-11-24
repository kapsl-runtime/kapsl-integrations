"""Adapter configuration.

SGLang has no per-connector configuration channel comparable to vLLM's
``kv_connector_extra_config``, and its scheduler runs in a spawned subprocess
that shares nothing with the launcher except the environment. The participant
identity is therefore read from environment variables, which whoever launches
the server (an operator or a managing runtime) sets once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .client import parse_unix_endpoint
from .contract import ContractValidationError, validate_memory_domains

# Participant identity handed down by the launcher.
ENV_CONTROL_ENDPOINT = "KAPSL_KV_CONTROL_ENDPOINT"
ENV_MODEL_FINGERPRINT = "KAPSL_KV_MODEL_FINGERPRINT"
ENV_MEMORY_DOMAINS = "KAPSL_KV_MEMORY_DOMAINS"
ENV_PARTICIPANT_ID = "KAPSL_KV_PARTICIPANT_ID"
ENV_MODE = "KAPSL_KV_MODE"
ENV_LEASE_TTL_MS = "KAPSL_KV_LEASE_TTL_MS"
ENV_TIMEOUT_MS = "KAPSL_KV_TIMEOUT_MS"
# Adapter-local tuning.
ENV_DENIAL_BACKOFF_MS = "KAPSL_SGLANG_DENIAL_BACKOFF_MS"

DEFAULT_PARTICIPANT_ID = "sglang"
DEFAULT_LEASE_TTL_MS = 30_000
DEFAULT_TIMEOUT_MS = 2_000
DEFAULT_DENIAL_BACKOFF_MS = 100
MIN_LEASE_TTL_MS = 1_000
SUPPORTED_MODE = "opaque"


class ConfigError(ValueError):
    """The adapter is installed but not configured to reach a Kapsl runtime."""


@dataclass(frozen=True)
class AdapterConfig:
    control_endpoint: str
    model_fingerprint: str
    memory_domains: tuple[Mapping[str, Any], ...]
    participant_base: str = DEFAULT_PARTICIPANT_ID
    lease_ttl_ms: int = DEFAULT_LEASE_TTL_MS
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    denial_backoff_ms: int = DEFAULT_DENIAL_BACKOFF_MS

    @property
    def participant_id(self) -> str:
        """Registered identity: the launcher's base plus this process's role.

        The ``base:role`` shape matches the vLLM connector and the namespace a
        managing runtime uses to recognize the participants it launched.
        """

        return f"{self.participant_base}:scheduler"


def load_config(environ: Mapping[str, str]) -> AdapterConfig:
    mode = environ.get(ENV_MODE, SUPPORTED_MODE).strip().lower()
    if mode != SUPPORTED_MODE:
        raise ConfigError(
            f"{ENV_MODE}={mode!r} is not implemented; this adapter supports only "
            f"{SUPPORTED_MODE!r}"
        )
    control_endpoint = _required(environ, ENV_CONTROL_ENDPOINT)
    try:
        parse_unix_endpoint(control_endpoint)
    except ValueError as error:
        raise ConfigError(f"{ENV_CONTROL_ENDPOINT}: {error}") from error
    participant_base = environ.get(ENV_PARTICIPANT_ID, DEFAULT_PARTICIPANT_ID).strip()
    if not participant_base:
        raise ConfigError(f"{ENV_PARTICIPANT_ID} must not be empty")
    lease_ttl_ms = _integer(environ, ENV_LEASE_TTL_MS, DEFAULT_LEASE_TTL_MS)
    if lease_ttl_ms < MIN_LEASE_TTL_MS:
        raise ConfigError(f"{ENV_LEASE_TTL_MS} must be at least {MIN_LEASE_TTL_MS}")
    timeout_ms = _integer(environ, ENV_TIMEOUT_MS, DEFAULT_TIMEOUT_MS)
    if timeout_ms <= 0:
        raise ConfigError(f"{ENV_TIMEOUT_MS} must be positive")
    denial_backoff_ms = _integer(
        environ, ENV_DENIAL_BACKOFF_MS, DEFAULT_DENIAL_BACKOFF_MS
    )
    if denial_backoff_ms < 0:
        raise ConfigError(f"{ENV_DENIAL_BACKOFF_MS} cannot be negative")
    return AdapterConfig(
        control_endpoint=control_endpoint,
        model_fingerprint=_required(environ, ENV_MODEL_FINGERPRINT),
        memory_domains=_memory_domains(_required(environ, ENV_MEMORY_DOMAINS)),
        participant_base=participant_base,
        lease_ttl_ms=lease_ttl_ms,
        timeout_ms=timeout_ms,
        denial_backoff_ms=denial_backoff_ms,
    )


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required for the Kapsl SGLang adapter")
    return value


def _integer(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer") from error


def _memory_domains(raw: str) -> tuple[Mapping[str, Any], ...]:
    try:
        domains = json.loads(raw)
        validate_memory_domains(domains)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{ENV_MEMORY_DOMAINS} must be a JSON list") from error
    except ContractValidationError as error:
        raise ConfigError(f"{ENV_MEMORY_DOMAINS}: {error}") from error
    return tuple(dict(domain) for domain in domains)
