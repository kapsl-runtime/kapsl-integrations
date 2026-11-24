"""Opaque-tier subset of the ``kapsl-kv-abi`` JSON control contract.

Keep this module dependency-free: it is imported by every SGLang process that
loads the adapter plugin, and its tests run on machines without SGLang, torch,
or CUDA.

Only the ``kv_connected`` / opaque tier is represented. Shared-pool topology,
attachment, resize, and detach messages are deliberately absent, so a
registration that names them is rejected here instead of being sent. See the
package README for why this is a reduced copy of the vLLM connector's contract
module rather than a shared dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

ABI_VERSION: dict[str, int] = {"major": 1, "minor": 5}
BACKEND = "sglang"

# Widths of the Rust wire types this JSON is deserialized into.
UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

_PROVIDER_DOMAIN_KINDS = frozenset({"host_pinned", "host_mapped", "provider"})
_RESPONSE_RESULTS = frozenset({"registered", "lease", "resize", "ack", "error"})
_SHARED_POOL_ONLY_FIELDS = ("topology", "adapter_profile", "provisioning_grant")


class ContractValidationError(ValueError):
    """A local or remote message violates the negotiated KV contract."""


def abi_version() -> dict[str, int]:
    return dict(ABI_VERSION)


def accepts_version(peer: Mapping[str, Any]) -> bool:
    """Whether this adapter can consume a message from ``peer``."""

    minor = peer.get("minor")
    return (
        peer.get("major") == ABI_VERSION["major"]
        and isinstance(minor, int)
        and not isinstance(minor, bool)
        and minor <= ABI_VERSION["minor"]
    )


def opaque_capabilities() -> dict[str, Any]:
    """Minimum official Kapsl tier with a backend-private KV layout."""

    return {
        "abi_version": abi_version(),
        "tier": "kv_connected",
        "metadata_mode": "opaque",
        "ownership": "backend",
        "features": ["capacity_leasing"],
        "transports": [{"kind": "backend_opaque"}],
    }


def opaque_registration(
    participant_id: str,
    model_fingerprint: str,
    capacity_groups: Sequence[Mapping[str, Any]],
    *,
    backend: str = BACKEND,
) -> dict[str, Any]:
    registration = {
        "participant_id": _nonempty(participant_id, "participant_id"),
        "backend": _nonempty(backend, "backend"),
        "model_fingerprint": _nonempty(model_fingerprint, "model_fingerprint"),
        "capabilities": opaque_capabilities(),
        "capacity_model": {"groups": [dict(group) for group in capacity_groups]},
    }
    validate_registration(registration)
    return registration


def validate_registration(registration: Mapping[str, Any]) -> None:
    _nonempty(registration.get("participant_id"), "participant_id")
    _nonempty(registration.get("backend"), "backend")
    _nonempty(registration.get("model_fingerprint"), "model_fingerprint")
    capabilities = _mapping(registration.get("capabilities"), "capabilities")
    if not accepts_version(_mapping(capabilities.get("abi_version"), "abi_version")):
        raise ContractValidationError("unsupported capability ABI version")
    features = capabilities.get("features")
    transports = capabilities.get("transports")
    if not isinstance(features, list) or not isinstance(transports, list):
        raise ContractValidationError(
            "capability features and transports must be lists"
        )
    if capabilities.get("tier") != "kv_connected":
        raise ContractValidationError(
            "this adapter implements only the kv_connected opaque tier"
        )
    if capabilities.get("metadata_mode") != "opaque":
        raise ContractValidationError("kv_connected SGLang must use opaque metadata")
    if capabilities.get("ownership") != "backend":
        raise ContractValidationError("opaque SGLang KV must remain backend owned")
    if features != ["capacity_leasing"]:
        raise ContractValidationError(
            "opaque SGLang advertises capacity_leasing and nothing else"
        )
    if transports != [{"kind": "backend_opaque"}]:
        raise ContractValidationError(
            "opaque SGLang advertises only the backend_opaque transport"
        )
    for field in _SHARED_POOL_ONLY_FIELDS:
        if field in registration:
            raise ContractValidationError(
                f"opaque registrations cannot include {field}"
            )

    capacity_model = _mapping(registration.get("capacity_model"), "capacity_model")
    groups = capacity_model.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("capacity_model.groups must be a non-empty list")
    seen: set[str] = set()
    pool_domains: dict[str, set[tuple[Any, ...]]] = {}
    for raw_group in groups:
        group = _mapping(raw_group, "capacity group")
        group_id = _nonempty(group.get("group_id"), "capacity group_id")
        pool_id = _nonempty(group.get("pool_id"), "capacity pool_id")
        if group_id in seen:
            raise ContractValidationError("capacity group IDs must be unique")
        seen.add(group_id)
        _positive_int(
            group.get("allocation_granularity_tokens"),
            "allocation_granularity_tokens",
            maximum=UINT32_MAX,
        )
        _positive_int(
            group.get("bytes_per_allocation"),
            "bytes_per_allocation",
            maximum=UINT64_MAX,
        )
        domains = validate_memory_domains(group.get("memory_domains"))
        previous_domains = pool_domains.setdefault(pool_id, domains)
        if previous_domains != domains:
            raise ContractValidationError(
                "capacity groups sharing a pool must name the same memory domains"
            )
        if group.get("max_allocations") is not None:
            _positive_int(
                group["max_allocations"], "max_allocations", maximum=UINT64_MAX
            )


def validate_memory_domains(memory_domains: Any) -> set[tuple[Any, ...]]:
    """Validate a domain list and return its canonical identities."""

    if not isinstance(memory_domains, list) or not memory_domains:
        raise ContractValidationError(
            "capacity group memory_domains must be a non-empty list"
        )
    identities: set[tuple[Any, ...]] = set()
    for raw_domain in memory_domains:
        domain = _mapping(raw_domain, "memory domain")
        kind = _nonempty(domain.get("kind"), "memory domain kind")
        identity: tuple[Any, ...]
        if kind == "host":
            identity = (kind,)
        elif kind == "cuda":
            identity = (
                kind,
                _nonnegative_int(
                    domain.get("device_id"), "device_id", maximum=UINT32_MAX
                ),
            )
        elif kind in _PROVIDER_DOMAIN_KINDS:
            provider = _nonempty(domain.get("provider"), "memory domain provider")
            device_id = domain.get("device_id")
            if device_id is not None:
                device_id = _nonnegative_int(
                    device_id, "device_id", maximum=UINT32_MAX
                )
            identity = (kind, provider, device_id)
        else:
            raise ContractValidationError(f"unknown KV memory domain kind {kind!r}")
        if identity in identities:
            raise ContractValidationError("capacity memory domains must be unique")
        identities.add(identity)
    return identities


def make_reserve_request(
    *,
    request_id: str,
    sequence_id: str,
    token_capacity: int,
    group_ids: Sequence[str],
    priority: int = 0,
    ttl_ms: int | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "sequence": {
            "request_id": _nonempty(request_id, "request_id"),
            "sequence_id": _nonempty(sequence_id, "sequence_id"),
        },
        "groups": [
            {
                "group_id": _nonempty(group_id, "group_id"),
                "token_capacity": token_capacity,
            }
            for group_id in group_ids
        ],
        "priority": priority,
    }
    if ttl_ms is not None:
        request["ttl_ms"] = ttl_ms
    validate_reserve_request(request)
    return request


def validate_reserve_request(request: Mapping[str, Any]) -> None:
    _validate_sequence(request.get("sequence"))
    groups = request.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("groups must be a non-empty list")
    seen: set[str] = set()
    for raw_group in groups:
        group = _mapping(raw_group, "group")
        group_id = _nonempty(group.get("group_id"), "group_id")
        if group_id in seen:
            raise ContractValidationError("group IDs must be unique")
        seen.add(group_id)
        _positive_int(
            group.get("token_capacity"),
            "token_capacity",
            maximum=UINT32_MAX,
        )
    priority = request.get("priority", 0)
    if (
        not isinstance(priority, int)
        or isinstance(priority, bool)
        or not INT32_MIN <= priority <= INT32_MAX
    ):
        raise ContractValidationError("priority must be a signed 32-bit integer")
    if request.get("ttl_ms") is not None:
        _positive_int(request["ttl_ms"], "ttl_ms", maximum=UINT64_MAX)
    if "prefix" in request:
        raise ContractValidationError(
            "opaque SGLang does not advertise prefix reservations"
        )


def validate_lease(
    lease: Mapping[str, Any], request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate an opaque capacity lease, optionally against its request.

    An opaque participant keeps block selection private, so a lease that
    carries block handles means the runtime believes it granted something else.
    """

    _nonempty(lease.get("lease_id"), "lease_id")
    _validate_sequence(lease.get("sequence"))
    groups = lease.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("lease groups must be a non-empty list")
    granted: dict[str, int] = {}
    for raw_group in groups:
        group = _mapping(raw_group, "lease group")
        group_id = _nonempty(group.get("group_id"), "group_id")
        if group_id in granted:
            raise ContractValidationError("lease group IDs must be unique")
        granted[group_id] = _positive_int(
            group.get("token_capacity"),
            "token_capacity",
            maximum=UINT32_MAX,
        )
        blocks = group.get("blocks", [])
        if not isinstance(blocks, list):
            raise ContractValidationError("lease blocks must be a list")
        if blocks:
            raise ContractValidationError(
                "an opaque capacity lease cannot carry block handles"
            )
    if lease.get("expires_at_unix_ms") is not None:
        _positive_int(
            lease["expires_at_unix_ms"], "expires_at_unix_ms", maximum=UINT64_MAX
        )
    if request is not None:
        if dict(lease["sequence"]) != dict(request["sequence"]):
            raise ContractValidationError(
                "lease sequence does not match the reservation"
            )
        requested = {
            group["group_id"]: group["token_capacity"] for group in request["groups"]
        }
        if set(granted) != set(requested):
            raise ContractValidationError(
                "lease groups do not match the reservation"
            )
        for group_id, token_capacity in requested.items():
            if granted[group_id] < token_capacity:
                raise ContractValidationError(
                    "lease grants less capacity than the reservation requested"
                )
    return deepcopy(dict(lease))


def validate_registration_receipt(
    receipt: Mapping[str, Any], participant_id: str
) -> dict[str, Any]:
    if receipt.get("participant_id") != _nonempty(participant_id, "participant_id"):
        raise ContractValidationError(
            "registration receipt participant_id does not match registration"
        )
    _positive_int(
        receipt.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    pools = receipt.get("shared_pools", [])
    if not isinstance(pools, list):
        raise ContractValidationError("shared_pools must be a list")
    if pools:
        raise ContractValidationError(
            "opaque SGLang registration cannot receive shared-pool bindings"
        )
    return deepcopy(dict(receipt))


def make_envelope(request_id: str, operation: str, **payload: Any) -> dict[str, Any]:
    envelope = {
        "abi_version": abi_version(),
        "request_id": _nonempty(request_id, "request_id"),
        "operation": _nonempty(operation, "operation"),
    }
    envelope.update(payload)
    return envelope


def validate_response(response: Mapping[str, Any], request_id: str) -> None:
    if not accepts_version(_mapping(response.get("abi_version"), "abi_version")):
        raise ContractValidationError("unsupported response ABI version")
    if response.get("request_id") != request_id:
        raise ContractValidationError("response request_id does not match request")
    if response.get("result") not in _RESPONSE_RESULTS:
        raise ContractValidationError("unknown control response result")


def _validate_sequence(value: Any) -> None:
    sequence = _mapping(value, "sequence")
    _nonempty(sequence.get("request_id"), "sequence.request_id")
    _nonempty(sequence.get("sequence_id"), "sequence.sequence_id")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractValidationError(f"{field} must be a positive integer")
    if value > maximum:
        raise ContractValidationError(f"{field} exceeds its wire width")
    return value


def _nonnegative_int(value: Any, field: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractValidationError(f"{field} must be a non-negative integer")
    if value > maximum:
        raise ContractValidationError(f"{field} exceeds its wire width")
    return value
