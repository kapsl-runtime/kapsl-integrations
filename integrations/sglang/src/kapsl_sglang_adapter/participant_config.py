"""The participant configuration a managing runtime hands to an adapter.

One JSON object carries the identity and control-plane settings the runtime
assigned before the backend process existed. It is defined once, next to the KV
contract, as ``KvParticipantConfig`` in ``kapsl-kv-abi``; this module is its
Python reader. The field names are the ones the vLLM connector already reads
from ``kv_connector_extra_config``.

A backend with no configuration channel of its own receives the document in
the :data:`PARTICIPANT_CONFIG_ENV` environment variable. Fields this reader
does not know are ignored, so a runtime can add fields for a deeper tier
without breaking an opaque adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contract import ContractValidationError, validate_memory_domains

PARTICIPANT_CONFIG_ENV = "KAPSL_KV_PARTICIPANT_CONFIG"

MODE_OPAQUE = "opaque"
MODE_SHARED_POOL = "shared_pool"
MODES = (MODE_OPAQUE, MODE_SHARED_POOL)

FIELD_CONTROL_ENDPOINT = "kapsl_control_endpoint"
FIELD_PARTICIPANT_ID = "kapsl_participant_id"
FIELD_MODEL_FINGERPRINT = "kapsl_model_fingerprint"
FIELD_MODE = "kapsl_kv_mode"
FIELD_MEMORY_DOMAINS = "kapsl_memory_domains"
FIELD_RANK_DEVICE_MAP = "kapsl_rank_device_map"
FIELD_LEASE_TTL_MS = "kapsl_lease_ttl_ms"
FIELD_TIMEOUT_MS = "kapsl_timeout_ms"
FIELD_LIVE_RESIZE = "kapsl_live_resize"
FIELD_VMM_CONFORMANCE = "kapsl_vmm_conformance"
FIELD_PROVISIONING_GRANT = "kapsl_provisioning_grant"

_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_MAX_GRANT_TOKEN_LENGTH = 256
_HEX_DIGITS = frozenset("0123456789abcdef")


class ParticipantConfigError(ValueError):
    """The participant configuration document is missing or not acceptable."""


@dataclass(frozen=True)
class ParticipantConfig:
    control_endpoint: str
    # Base identity: a participant registers as this value or ``<base>:<suffix>``.
    participant_id: str
    model_fingerprint: str
    mode: str
    # Physical domains of the KV pool, in the runtime's device namespace.
    memory_domains: tuple[Mapping[str, Any], ...]
    lease_ttl_ms: int
    rank_device_map: Mapping[str, int] = field(default_factory=dict)
    timeout_ms: int | None = None
    live_resize: bool = False
    vmm_conformance: bool = False
    provisioning_grant: Mapping[str, Any] | None = None


def load_participant_config(environ: Mapping[str, str]) -> ParticipantConfig:
    """Read the document from :data:`PARTICIPANT_CONFIG_ENV`."""

    raw = environ.get(PARTICIPANT_CONFIG_ENV, "")
    if not raw.strip():
        raise ParticipantConfigError(
            f"{PARTICIPANT_CONFIG_ENV} is required: it carries the participant "
            "configuration the Kapsl runtime assigned to this server"
        )
    try:
        return parse_participant_config(raw)
    except ParticipantConfigError as error:
        raise ParticipantConfigError(f"{PARTICIPANT_CONFIG_ENV}: {error}") from error


def parse_participant_config(document: str | Mapping[str, Any]) -> ParticipantConfig:
    """Validate a participant configuration document."""

    if isinstance(document, str):
        try:
            document = json.loads(document)
        except json.JSONDecodeError as error:
            raise ParticipantConfigError(f"not valid JSON ({error})") from error
    if not isinstance(document, Mapping):
        raise ParticipantConfigError("the document must be a JSON object")

    mode = _string(document, FIELD_MODE)
    if mode not in MODES:
        raise ParticipantConfigError(
            f"{FIELD_MODE}={mode!r} is not a known mode; expected one of {MODES}"
        )
    memory_domains = document.get(FIELD_MEMORY_DOMAINS)
    try:
        validate_memory_domains(memory_domains)
    except ContractValidationError as error:
        raise ParticipantConfigError(f"{FIELD_MEMORY_DOMAINS}: {error}") from error
    lease_ttl_ms = _positive_int(document, FIELD_LEASE_TTL_MS)
    timeout_ms = (
        None
        if document.get(FIELD_TIMEOUT_MS) is None
        else _positive_int(document, FIELD_TIMEOUT_MS)
    )
    live_resize = _flag(document, FIELD_LIVE_RESIZE)
    vmm_conformance = _flag(document, FIELD_VMM_CONFORMANCE)
    if vmm_conformance and not live_resize:
        raise ParticipantConfigError(
            f"{FIELD_VMM_CONFORMANCE} requires {FIELD_LIVE_RESIZE}"
        )
    grant = document.get(FIELD_PROVISIONING_GRANT)
    if grant is not None:
        grant = _provisioning_grant(grant)
    if mode == MODE_OPAQUE and (live_resize or grant is not None):
        raise ParticipantConfigError(
            f"{FIELD_LIVE_RESIZE} and {FIELD_PROVISIONING_GRANT} require the "
            f"{MODE_SHARED_POOL} mode"
        )
    return ParticipantConfig(
        control_endpoint=_string(document, FIELD_CONTROL_ENDPOINT),
        participant_id=_string(document, FIELD_PARTICIPANT_ID),
        model_fingerprint=_string(document, FIELD_MODEL_FINGERPRINT),
        mode=mode,
        memory_domains=tuple(dict(domain) for domain in memory_domains),
        lease_ttl_ms=lease_ttl_ms,
        rank_device_map=_rank_device_map(document.get(FIELD_RANK_DEVICE_MAP)),
        timeout_ms=timeout_ms,
        live_resize=live_resize,
        vmm_conformance=vmm_conformance,
        provisioning_grant=grant,
    )


def _string(document: Mapping[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ParticipantConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(document: Mapping[str, Any], name: str) -> int:
    value = document.get(name)
    if not _is_int(value) or not 0 < value <= _UINT64_MAX:
        raise ParticipantConfigError(f"{name} must be a positive integer")
    return value


def _flag(document: Mapping[str, Any], name: str) -> bool:
    value = document.get(name, False)
    if not isinstance(value, bool):
        raise ParticipantConfigError(f"{name} must be a boolean")
    return value


def _rank_device_map(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ParticipantConfigError(f"{FIELD_RANK_DEVICE_MAP} must be an object")
    ranks: dict[str, int] = {}
    for rank, device_id in value.items():
        if not isinstance(rank, str) or not rank.isascii() or not rank.isdigit():
            raise ParticipantConfigError(
                f"{FIELD_RANK_DEVICE_MAP} keys must be decimal ranks"
            )
        if not _is_int(device_id) or not 0 <= device_id <= _UINT32_MAX:
            raise ParticipantConfigError(
                f"{FIELD_RANK_DEVICE_MAP} values must be device IDs"
            )
        ranks[rank] = device_id
    return ranks


def _provisioning_grant(value: Any) -> dict[str, Any]:
    problem = ParticipantConfigError(
        f"{FIELD_PROVISIONING_GRANT} requires a bounded token, a canonical "
        "sha256 geometry digest, a generation, and an expiry"
    )
    if not isinstance(value, Mapping):
        raise problem
    token = value.get("token")
    digest = value.get("geometry_digest")
    if (
        not isinstance(token, str)
        or not token.strip()
        or len(token) > _MAX_GRANT_TOKEN_LENGTH
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != len("sha256:") + 64
        or not set(digest[len("sha256:") :]) <= _HEX_DIGITS
    ):
        raise problem
    for name in ("authority_generation", "expires_at_unix_ms"):
        number = value.get(name)
        if not _is_int(number) or not 0 < number <= _UINT64_MAX:
            raise problem
    return dict(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
