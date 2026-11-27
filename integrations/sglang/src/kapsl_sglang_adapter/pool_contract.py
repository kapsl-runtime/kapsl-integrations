"""Striped elastic shared-pool subset of the ``kapsl-kv-abi`` 1.6 contract.

ABI 1.6 lets one elastic ``cuda_vmm`` binding back several identically shaped
virtual ranges (stripes). SGLang keeps one K and one V buffer per layer, so a
striped binding backs those buffers as they are. This module mirrors the Rust
validation of the messages that tier adds: the ``shared_pool`` registration,
its receipt, the attachment report, and the resize poll, operation and
acknowledgement. It carries only the one shape this adapter uses: a single
capacity group in a single pool on a single CUDA device, participant-managed
block selection, and live resize.

Like :mod:`.contract`, it is dependency-free. The opaque tier stays at ABI 1.5
and never sends this module's messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .contract import (
    BACKEND,
    UINT32_MAX,
    UINT64_MAX,
    ContractValidationError,
    _mapping,
    _nonempty,
    _nonnegative_int,
    _positive_int,
    validate_memory_domains,
)

# First ABI version in which a participant may advertise striped pools.
STRIPED_POOL_ABI_VERSION: dict[str, int] = {"major": 1, "minor": 6}

# ``KV_MAX_POOL_STRIPES``: every handle of a response travels in one
# ``SCM_RIGHTS`` message, and Linux accepts at most 253 descriptors in one.
MAX_POOL_STRIPES = 253

STRIPED_POOL_FEATURES: tuple[str, ...] = (
    "capacity_leasing",
    "direct_attention_access",
    "external_pool_attachment",
    "participant_block_selection",
    "live_pool_resize",
    "striped_elastic_pool",
)
PROVISIONING_GRANT_FEATURE = "provisioning_grant"
CUDA_VMM_TRANSPORT: dict[str, str] = {"kind": "cuda_vmm"}

STAGE_MAP_WORKERS = "map_workers"
STAGE_ACTIVATE_SCHEDULER = "activate_scheduler"
STAGE_RETIRE_SCHEDULER = "retire_scheduler"
STAGE_UNMAP_WORKERS = "unmap_workers"
GROWTH_STAGES = frozenset({STAGE_MAP_WORKERS, STAGE_ACTIVATE_SCHEDULER})
SHRINK_STAGES = frozenset({STAGE_RETIRE_SCHEDULER, STAGE_UNMAP_WORKERS})
PHYSICAL_STAGES = frozenset({STAGE_MAP_WORKERS, STAGE_UNMAP_WORKERS})

SCHEDULER_ACTOR: dict[str, Any] = {"role": "scheduler"}

_ELEMENT_KINDS = frozenset({"f16", "bf16", "f32"})
_PROFILE_FIELDS = ("adapter_id", "adapter_version", "backend_version", "profile_id")
_SHARD_FIELDS = (
    "tensor_parallel_rank",
    "tensor_parallel_world_size",
    "pipeline_parallel_rank",
    "pipeline_parallel_world_size",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_GRANT_TOKEN_LENGTH = 256


def striped_pool_abi_version() -> dict[str, int]:
    return dict(STRIPED_POOL_ABI_VERSION)


def single_rank_shard() -> dict[str, int]:
    """The shard of a server that runs scheduler and model in one process."""

    return {
        "tensor_parallel_rank": 0,
        "tensor_parallel_world_size": 1,
        "pipeline_parallel_rank": 0,
        "pipeline_parallel_world_size": 1,
    }


def worker_actor(shard: Mapping[str, int]) -> dict[str, Any]:
    return {"role": "worker", "shard": dict(shard)}


def striped_pool_capabilities(*, provisioning_grant: bool = False) -> dict[str, Any]:
    """Runtime-owned CUDA VMM pool, striped, with block selection left to SGLang."""

    features = list(STRIPED_POOL_FEATURES)
    if provisioning_grant:
        features.append(PROVISIONING_GRANT_FEATURE)
    return {
        "abi_version": striped_pool_abi_version(),
        "tier": "shared_pool",
        "metadata_mode": "structured",
        "ownership": "kapsl_runtime",
        "features": features,
        "transports": [dict(CUDA_VMM_TRANSPORT)],
    }


def striped_pool_registration(
    participant_id: str,
    model_fingerprint: str,
    capacity_groups: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    provisioning_grant: Mapping[str, Any] | None = None,
    backend: str = BACKEND,
) -> dict[str, Any]:
    registration: dict[str, Any] = {
        "participant_id": _nonempty(participant_id, "participant_id"),
        "backend": _nonempty(backend, "backend"),
        "model_fingerprint": _nonempty(model_fingerprint, "model_fingerprint"),
        "capabilities": striped_pool_capabilities(
            provisioning_grant=provisioning_grant is not None
        ),
        "capacity_model": {"groups": [dict(group) for group in capacity_groups]},
        "adapter_profile": dict(profile),
        "topology": deepcopy(dict(topology)),
    }
    if provisioning_grant is not None:
        registration["provisioning_grant"] = dict(provisioning_grant)
    validate_striped_pool_registration(registration)
    return registration


def validate_striped_pool_registration(registration: Mapping[str, Any]) -> None:
    _nonempty(registration.get("participant_id"), "participant_id")
    _nonempty(registration.get("backend"), "backend")
    fingerprint = _nonempty(registration.get("model_fingerprint"), "model_fingerprint")
    capabilities = _mapping(registration.get("capabilities"), "capabilities")
    if capabilities.get("abi_version") != STRIPED_POOL_ABI_VERSION:
        raise ContractValidationError("striped elastic pools are registered at ABI 1.6")
    if (
        capabilities.get("tier") != "shared_pool"
        or capabilities.get("metadata_mode") != "structured"
        or capabilities.get("ownership") != "kapsl_runtime"
    ):
        raise ContractValidationError(
            "a striped pool is a structured, runtime-owned shared_pool registration"
        )
    grant = registration.get("provisioning_grant")
    expected_features = set(STRIPED_POOL_FEATURES)
    if grant is not None:
        expected_features.add(PROVISIONING_GRANT_FEATURE)
    features = capabilities.get("features")
    if (
        not isinstance(features, list)
        or len(features) != len(expected_features)
        or set(features) != expected_features
    ):
        raise ContractValidationError(
            "striped pool features must be exactly "
            + ", ".join(sorted(expected_features))
        )
    if capabilities.get("transports") != [CUDA_VMM_TRANSPORT]:
        raise ContractValidationError(
            "a striped elastic pool advertises only the cuda_vmm transport"
        )
    validate_adapter_profile(registration.get("adapter_profile"))
    if grant is not None:
        _validate_provisioning_grant(_mapping(grant, "provisioning_grant"))

    group = _single_capacity_group(registration)
    group_id = _nonempty(group.get("group_id"), "capacity group_id")
    _nonempty(group.get("pool_id"), "capacity pool_id")
    _positive_int(
        group.get("allocation_granularity_tokens"),
        "allocation_granularity_tokens",
        maximum=UINT32_MAX,
    )
    _positive_int(
        group.get("bytes_per_allocation"), "bytes_per_allocation", maximum=UINT64_MAX
    )
    _positive_int(group.get("max_allocations"), "max_allocations", maximum=UINT64_MAX)
    domains = group.get("memory_domains")
    validate_memory_domains(domains)
    if len(domains) != 1 or domains[0].get("kind") != "cuda":
        raise ContractValidationError(
            "a striped pool lives in exactly one cuda memory domain"
        )
    _validate_topology(
        _mapping(registration.get("topology"), "topology"), fingerprint, group_id
    )


def validate_adapter_profile(profile: Any) -> None:
    profile = _mapping(profile, "adapter_profile")
    for field in _PROFILE_FIELDS:
        _nonempty(profile.get(field), f"adapter_profile {field}")


def validate_shard(value: Any) -> dict[str, int]:
    shard = _mapping(value, "shard")
    numbers = {
        field: _nonnegative_int(shard.get(field), field, maximum=UINT32_MAX)
        for field in _SHARD_FIELDS
    }
    if (
        numbers["tensor_parallel_world_size"] == 0
        or numbers["pipeline_parallel_world_size"] == 0
        or numbers["tensor_parallel_rank"] >= numbers["tensor_parallel_world_size"]
        or numbers["pipeline_parallel_rank"] >= numbers["pipeline_parallel_world_size"]
    ):
        raise ContractValidationError(
            "parallel world sizes must be non-zero and ranks must be in range"
        )
    return numbers


# Receipt ----------------------------------------------------------------------


def validate_striped_pool_receipt(
    receipt: Mapping[str, Any], registration: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the runtime's receipt against the registration that caused it.

    Mirrors ``KvRegistrationReceipt::validate_for`` for a striped registration:
    one binding for the one group and domain, striped, elastic, over
    ``cuda_vmm``, with segments covering the mapped prefix of every stripe.
    """

    if receipt.get("participant_id") != registration.get("participant_id"):
        raise ContractValidationError(
            "registration receipt participant_id does not match registration"
        )
    _positive_int(
        receipt.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    pools = receipt.get("shared_pools")
    if not isinstance(pools, list) or len(pools) != 1:
        raise ContractValidationError(
            "a single-device striped registration receives exactly one binding"
        )
    group = _single_capacity_group(registration)
    binding = _mapping(pools[0], "shared pool binding")
    _nonempty(binding.get("binding_id"), "binding_id")
    _positive_int(binding.get("generation"), "binding generation", maximum=UINT64_MAX)
    _nonempty(binding.get("descriptor"), "binding descriptor")
    if binding.get("capacity_pool_id") != group["pool_id"]:
        raise ContractValidationError("binding names another capacity pool")
    if binding.get("group_ids") != [group["group_id"]]:
        raise ContractValidationError("binding does not bind this capacity group")
    if binding.get("memory_domain") != group["memory_domains"][0]:
        raise ContractValidationError("binding is placed in another memory domain")
    if binding.get("transport") != CUDA_VMM_TRANSPORT:
        raise ContractValidationError("binding uses an unadvertised transport")
    # ``allocation_mode`` defaults to runtime_leased when it is absent.
    if binding.get("allocation_mode") != "participant_managed":
        raise ContractValidationError(
            "binding allocation mode does not match participant block selection"
        )
    block_count = _positive_int(
        binding.get("block_count"), "block_count", maximum=UINT64_MAX
    )
    bytes_per_block = _positive_int(
        binding.get("bytes_per_block"), "bytes_per_block", maximum=UINT64_MAX
    )
    if bytes_per_block != group["bytes_per_allocation"]:
        raise ContractValidationError(
            "binding block size does not match the registered bytes_per_allocation"
        )
    if block_count > group["max_allocations"]:
        raise ContractValidationError(
            "binding provisions more blocks than the registration requested"
        )
    _validate_elastic(
        _mapping(binding.get("elastic"), "elastic pool geometry"),
        block_count=block_count,
        bytes_per_block=bytes_per_block,
    )
    return deepcopy(dict(receipt))


def stripe_count_of(document: Mapping[str, Any], bytes_per_block: int) -> int:
    """Stripe count of a striped binding or resize operation.

    A striped participant receives only striped documents, so the field must
    be present: between 2 and :data:`MAX_POOL_STRIPES`, dividing the block.
    """

    stripes = document.get("stripe_count")
    if stripes is None:
        raise ContractValidationError(
            "a striped participant must receive a stripe_count"
        )
    stripes = _positive_int(stripes, "stripe_count", maximum=UINT32_MAX)
    if not 2 <= stripes <= MAX_POOL_STRIPES:
        raise ContractValidationError(
            f"stripe_count must lie between 2 and {MAX_POOL_STRIPES}"
        )
    if bytes_per_block % stripes:
        raise ContractValidationError(
            "bytes_per_block must divide evenly across stripes"
        )
    return stripes


def check_segment_coverage(
    segments: Any, *, stripes: int, granularity: int, start: int, end: int
) -> list[dict[str, Any]]:
    """Check that ``segments`` cover ``[start, end)`` of every stripe exactly.

    Offsets are relative to the segment's stripe (omitted on the wire for
    stripe 0). Segment IDs and handle indices are unique across all stripes.
    Returns the segments ordered by stripe and offset, with ``stripe`` filled.
    """

    if not isinstance(segments, list) or not segments:
        raise ContractValidationError("segments must be a non-empty list")
    normalized = []
    for raw in segments:
        segment = dict(_mapping(raw, "VMM segment"))
        _nonempty(segment.get("segment_id"), "segment_id")
        segment["stripe"] = _nonnegative_int(
            segment.get("stripe", 0), "segment stripe", maximum=UINT32_MAX
        )
        offset = _nonnegative_int(
            segment.get("offset_bytes"), "offset_bytes", maximum=UINT64_MAX
        )
        length = _positive_int(
            segment.get("length_bytes"), "length_bytes", maximum=UINT64_MAX
        )
        _nonnegative_int(
            segment.get("handle_index"), "handle_index", maximum=UINT32_MAX
        )
        if offset % granularity or length % granularity or offset + length > UINT64_MAX:
            raise ContractValidationError(
                "segments require a bounded granularity-aligned byte range"
            )
        if segment["stripe"] >= stripes:
            raise ContractValidationError(
                "segments must name a stripe of their binding"
            )
        normalized.append(segment)
    normalized.sort(key=lambda segment: (segment["stripe"], segment["offset_bytes"]))
    segment_ids: set[str] = set()
    handle_indices: set[int] = set()
    stripe = 0
    expected_offset = start
    for segment in normalized:
        if segment["stripe"] != stripe:
            # The previous stripe must be complete and no stripe skipped.
            if expected_offset != end or segment["stripe"] != stripe + 1:
                raise ContractValidationError(
                    "segments must exactly cover the byte range in every stripe"
                )
            stripe = segment["stripe"]
            expected_offset = start
        if (
            segment["segment_id"] in segment_ids
            or segment["handle_index"] in handle_indices
            or segment["offset_bytes"] != expected_offset
        ):
            raise ContractValidationError(
                "segments must have unique IDs and handles and be dense in each stripe"
            )
        segment_ids.add(segment["segment_id"])
        handle_indices.add(segment["handle_index"])
        expected_offset += segment["length_bytes"]
    if stripe + 1 != stripes or expected_offset != end:
        raise ContractValidationError(
            "segments must exactly cover the byte range in every stripe"
        )
    return normalized


# Attachment -------------------------------------------------------------------


def make_attachment(
    *,
    participant_epoch: int,
    binding_id: str,
    shard: Mapping[str, int],
    profile: Mapping[str, Any],
    imported_bytes: int,
    mapped_bytes: int,
    views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attachment = {
        "participant_epoch": participant_epoch,
        "binding_id": binding_id,
        "shard": dict(shard),
        "profile": dict(profile),
        "imported_bytes": imported_bytes,
        "mapped_bytes": mapped_bytes,
        "views": [deepcopy(dict(view)) for view in views],
    }
    validate_attachment(attachment)
    return attachment


def validate_attachment(attachment: Mapping[str, Any]) -> None:
    _positive_int(
        attachment.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    _nonempty(attachment.get("binding_id"), "binding_id")
    imported = _positive_int(
        attachment.get("imported_bytes"), "imported_bytes", maximum=UINT64_MAX
    )
    mapped = _positive_int(
        attachment.get("mapped_bytes"), "mapped_bytes", maximum=UINT64_MAX
    )
    if mapped > imported:
        raise ContractValidationError("mapped_bytes exceeds imported_bytes")
    validate_shard(attachment.get("shard"))
    validate_adapter_profile(attachment.get("profile"))
    views = attachment.get("views")
    if not isinstance(views, list) or not views:
        raise ContractValidationError("attachment views must be a non-empty list")
    seen: set[tuple[str, int, int]] = set()
    for raw_view in views:
        view = _mapping(raw_view, "attachment view")
        group_id = _nonempty(view.get("group_id"), "attachment view group_id")
        layer = _mapping(view.get("layer"), "attachment view layer")
        index = _nonnegative_int(layer.get("index"), "layer index", maximum=UINT32_MAX)
        if layer.get("name") is not None:
            _nonempty(layer["name"], "layer name")
        stripe = _nonnegative_int(
            view.get("stripe", 0), "attachment view stripe", maximum=UINT32_MAX
        )
        offset = _nonnegative_int(
            view.get("offset_bytes"), "attachment view offset", maximum=UINT64_MAX
        )
        length = _positive_int(
            view.get("length_bytes"), "attachment view length", maximum=UINT64_MAX
        )
        if offset + length > imported:
            raise ContractValidationError(
                "attachment tensor view exceeds the imported binding"
            )
        if (group_id, index, stripe) in seen:
            raise ContractValidationError(
                "attachment views must identify unique group/layer/stripe triples"
            )
        seen.add((group_id, index, stripe))


# Resize -----------------------------------------------------------------------


def make_resize_poll_request(
    *, participant_epoch: int, actor: Mapping[str, Any], applied_generation: int
) -> dict[str, Any]:
    request = {
        "participant_epoch": participant_epoch,
        "actor": deepcopy(dict(actor)),
        "applied_generation": applied_generation,
    }
    validate_resize_poll_request(request)
    return request


def validate_resize_poll_request(request: Mapping[str, Any]) -> None:
    _positive_int(
        request.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    _validate_resize_actor(request.get("actor"))
    _nonnegative_int(
        request.get("applied_generation"), "applied_generation", maximum=UINT64_MAX
    )


def make_resize_ack_request(
    *,
    participant_epoch: int,
    actor: Mapping[str, Any],
    binding_id: str,
    resize_generation: int,
    stage: str,
    applied_block_count: int,
) -> dict[str, Any]:
    request = {
        "participant_epoch": participant_epoch,
        "actor": deepcopy(dict(actor)),
        "binding_id": binding_id,
        "resize_generation": resize_generation,
        "stage": stage,
        "applied_block_count": applied_block_count,
    }
    validate_resize_ack_request(request)
    return request


def validate_resize_ack_request(request: Mapping[str, Any]) -> None:
    _positive_int(
        request.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    _validate_resize_actor(request.get("actor"))
    _nonempty(request.get("binding_id"), "binding_id")
    _positive_int(
        request.get("resize_generation"), "resize_generation", maximum=UINT64_MAX
    )
    _resize_stage(request.get("stage"))
    _positive_int(
        request.get("applied_block_count"), "applied_block_count", maximum=UINT64_MAX
    )


def validate_resize_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one striped resize operation; mirrors the Rust ``validate``.

    A striped resize changes every stripe by the same tail, and the physical
    stages carry segments covering that tail in each stripe. The returned copy
    has its segments ordered by stripe and offset, each with ``stripe`` filled.
    """

    _positive_int(
        operation.get("participant_epoch"), "participant_epoch", maximum=UINT64_MAX
    )
    _positive_int(
        operation.get("resize_generation"), "resize_generation", maximum=UINT64_MAX
    )
    _nonempty(operation.get("binding_id"), "binding_id")
    stage = _resize_stage(operation.get("stage"))
    from_blocks = _positive_int(
        operation.get("from_block_count"), "from_block_count", maximum=UINT64_MAX
    )
    target_blocks = _positive_int(
        operation.get("target_block_count"), "target_block_count", maximum=UINT64_MAX
    )
    if from_blocks == target_blocks:
        raise ContractValidationError("resize block counts must differ")
    if stage not in (GROWTH_STAGES if target_blocks > from_blocks else SHRINK_STAGES):
        raise ContractValidationError("resize stage does not match its direction")
    bytes_per_block = _positive_int(
        operation.get("bytes_per_block"), "bytes_per_block", maximum=UINT64_MAX
    )
    granularity = _positive_int(
        operation.get("allocation_granularity_bytes"),
        "allocation_granularity_bytes",
        maximum=UINT64_MAX,
    )
    stripes = stripe_count_of(operation, bytes_per_block)
    from_bytes = _stripe_bytes(from_blocks, bytes_per_block, stripes)
    target_bytes = _stripe_bytes(target_blocks, bytes_per_block, stripes)
    if from_bytes % granularity or target_bytes % granularity:
        raise ContractValidationError(
            "resize endpoints must align to CUDA VMM allocation granularity"
        )
    segments = operation.get("segments", [])
    if not isinstance(segments, list):
        raise ContractValidationError("resize segments must be a list")
    validated = deepcopy(dict(operation))
    if stage not in PHYSICAL_STAGES:
        if segments:
            raise ContractValidationError("scheduler resize phases carry no segments")
        validated["segments"] = []
        return validated
    validated["segments"] = check_segment_coverage(
        segments,
        stripes=stripes,
        granularity=granularity,
        start=min(from_bytes, target_bytes),
        end=max(from_bytes, target_bytes),
    )
    return validated


# Internals --------------------------------------------------------------------


def _single_capacity_group(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    model = _mapping(registration.get("capacity_model"), "capacity_model")
    groups = model.get("groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ContractValidationError(
            "a striped SGLang registration has exactly one capacity group"
        )
    return _mapping(groups[0], "capacity group")


def _validate_topology(
    topology: Mapping[str, Any], fingerprint: str, group_id: str
) -> None:
    # The topology version must equal the capabilities version.
    if topology.get("abi_version") != STRIPED_POOL_ABI_VERSION:
        raise ContractValidationError("topology and capability ABI versions must match")
    if topology.get("model_fingerprint") != fingerprint:
        raise ContractValidationError(
            "topology model fingerprint must match registration"
        )
    validate_shard(topology.get("shard"))
    groups = topology.get("cache_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ContractValidationError("topology must describe exactly one cache group")
    group = _mapping(groups[0], "topology cache group")
    if group.get("group_id") != group_id:
        raise ContractValidationError(
            "topology and capacity model must describe the same group"
        )
    layers = group.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ContractValidationError("topology cache group layers must be non-empty")
    indices: set[int] = set()
    for raw_layer in layers:
        layer = _mapping(raw_layer, "topology layer")
        index = _nonnegative_int(layer.get("index"), "layer index", maximum=UINT32_MAX)
        if index in indices:
            raise ContractValidationError("topology layer indices must be unique")
        indices.add(index)
        if layer.get("name") is not None:
            _nonempty(layer["name"], "topology layer name")
    geometry = _mapping(group.get("geometry"), "topology geometry")
    if geometry.get("kind") != "paged_attention":
        raise ContractValidationError("SGLang KV is described as paged_attention")
    for field in ("block_size_tokens", "kv_heads", "key_head_dim", "value_head_dim"):
        _positive_int(geometry.get(field), f"geometry {field}", maximum=UINT32_MAX)
    element = _mapping(geometry.get("element_type"), "geometry element_type")
    if element.get("kind") not in _ELEMENT_KINDS:
        raise ContractValidationError(
            f"unsupported KV element type {element.get('kind')!r}"
        )
    layout = _mapping(geometry.get("layout"), "geometry layout")
    _nonempty(layout.get("kind"), "geometry layout kind")
    if _mapping(group.get("policy"), "topology policy").get("kind") != "full_attention":
        raise ContractValidationError("the striped profile covers full attention only")


def _validate_provisioning_grant(grant: Mapping[str, Any]) -> None:
    token = _nonempty(grant.get("token"), "provisioning grant token")
    digest = grant.get("geometry_digest")
    if (
        len(token) > _MAX_GRANT_TOKEN_LENGTH
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != len("sha256:") + 64
        or not set(digest[len("sha256:") :]) <= _HEX_DIGITS
    ):
        raise ContractValidationError(
            "provisioning grant requires a bounded token and a canonical sha256 "
            "geometry digest"
        )
    for field in ("authority_generation", "expires_at_unix_ms"):
        _positive_int(
            grant.get(field), f"provisioning grant {field}", maximum=UINT64_MAX
        )


def _validate_elastic(
    elastic: Mapping[str, Any], *, block_count: int, bytes_per_block: int
) -> None:
    minimum, mapped, maximum, granularity, alignment = (
        _positive_int(elastic.get(field), field, maximum=UINT64_MAX)
        for field in (
            "minimum_block_count",
            "mapped_block_count",
            "maximum_block_count",
            "allocation_granularity_bytes",
            "resize_alignment_blocks",
        )
    )
    if (
        maximum != block_count
        or minimum > mapped
        or mapped > maximum
        or minimum % alignment
        or mapped % alignment
    ):
        raise ContractValidationError(
            "elastic pool geometry requires an aligned minimum and mapped capacity "
            "within the maximum virtual capacity"
        )
    stripes = stripe_count_of(elastic, bytes_per_block)
    sizes = [
        _stripe_bytes(blocks, bytes_per_block, stripes)
        for blocks in (minimum, mapped, maximum)
    ]
    if any(size % granularity for size in sizes):
        raise ContractValidationError(
            "elastic pool minimum, mapped and maximum sizes (per stripe) must align "
            "to CUDA VMM granularity"
        )
    check_segment_coverage(
        elastic.get("segments"),
        stripes=stripes,
        granularity=granularity,
        start=0,
        end=sizes[1],
    )


def _stripe_bytes(block_count: int, bytes_per_block: int, stripes: int) -> int:
    total = block_count * bytes_per_block
    if total > UINT64_MAX:
        raise ContractValidationError("pool byte size overflows its wire width")
    return total // stripes


def _validate_resize_actor(value: Any) -> None:
    actor = _mapping(value, "resize actor")
    role = actor.get("role")
    if role == "scheduler":
        if set(actor) != {"role"}:
            raise ContractValidationError("scheduler resize actor has unknown fields")
        return
    if role != "worker" or set(actor) != {"role", "shard"}:
        raise ContractValidationError("resize actor role is unsupported")
    validate_shard(actor.get("shard"))


def _resize_stage(value: Any) -> str:
    stage = _nonempty(value, "resize stage")
    if stage not in GROWTH_STAGES | SHRINK_STAGES:
        raise ContractValidationError("resize stage is unsupported")
    return stage
