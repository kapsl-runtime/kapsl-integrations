"""Geometry and mapping state of one striped elastic KV pool.

Nothing here imports SGLang, torch, or CUDA. The SGLang-facing layer reduces
SGLang's VMM-backed KV buffers to :class:`PoolFacts`; this module decides
whether they fit the striped profile, checks the runtime's binding against
them, and keeps the table of runtime-owned segments mapped into each buffer.
Every CUDA call goes through the :class:`VmmBackend` seam.

One ABI block is one row of SGLang's buffers, so block 0 is SGLang's padding
slot and a pool of ``B`` mapped blocks serves ``B - 1`` tokens. Stripe ``s``
is buffer ``s`` in SGLang's own order: K of layer ``s`` for ``s < layers``,
V of layer ``s - layers`` after that.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .capacity import GROUP_ID, POOL_ID, UnsupportedProfileError
from .contract import ContractValidationError
from .pool_contract import (
    MAX_POOL_STRIPES,
    STRIPED_POOL_ABI_VERSION,
    check_segment_coverage,
    single_rank_shard,
)

logger = logging.getLogger(__name__)

ADAPTER_ID = "kapsl-sglang-adapter"
# Names the exact layout this adapter attaches: SGLang's own per-layer K and V
# buffers (NHD rows), VMM-backed, with the token allocator at page size 1.
SHARED_POOL_PROFILE_ID = "sglang-v1-striped-cuda-vmm/mha-nhd-token"

# SGLang keeps one extra page in front of the pool for padded-token writes.
PADDING_PAGES = 1

# torch dtype name -> (contract element kind, bytes per element).
_ELEMENT_TYPES: dict[str, tuple[str, int]] = {
    "float16": ("f16", 2),
    "bfloat16": ("bf16", 2),
    "float32": ("f32", 4),
}


class SharedPoolError(RuntimeError):
    """The shared pool cannot be attached or resized safely; serving must stop."""


class RetirementDeferred(Exception):
    """A tail retirement cannot be applied yet. Nothing was changed."""


class VmmBackend(Protocol):
    """The CUDA seam: map and unmap runtime-owned memory in one stripe.

    ``offset_bytes`` is relative to the start of the stripe. ``descriptor`` is
    an OS handle the caller keeps owning.
    """

    def map(
        self, stripe: int, offset_bytes: int, length_bytes: int, descriptor: int
    ) -> None: ...

    def unmap(self, stripe: int, offset_bytes: int, length_bytes: int) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class BufferFacts:
    """One of SGLang's VMM-backed KV buffers."""

    name: str
    reservation_offset: int
    reserved_bytes: int
    backed_bytes: int
    row_bytes: int
    tokens_per_row: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PoolFacts:
    """What the adapter reads from SGLang's VMM buffer owner before attaching."""

    device_id: int
    page_size: int
    reserved_tokens: int
    granularity_bytes: int
    reservation_bytes: int
    element_type: str
    buffers: tuple[BufferFacts, ...]


@dataclass(frozen=True)
class PoolGeometry:
    """SGLang's KV buffers described as stripes of equal blocks."""

    device_id: int
    page_size: int
    layer_count: int
    kv_heads: int
    head_dim: int
    element_kind: str
    stripe_block_bytes: int
    granularity_bytes: int
    maximum_blocks: int
    stripe_names: tuple[str, ...]

    @classmethod
    def from_facts(cls, facts: PoolFacts) -> PoolGeometry:
        reasons = _unsupported_layout_reasons(facts)
        if reasons:
            raise UnsupportedProfileError(
                "SGLang's KV pool is outside the Kapsl striped profile: "
                + "; ".join(reasons)
            )
        first = facts.buffers[0]
        return cls(
            device_id=facts.device_id,
            page_size=facts.page_size,
            layer_count=len(facts.buffers) // 2,
            kv_heads=first.shape[1],
            head_dim=first.shape[2],
            element_kind=_ELEMENT_TYPES[facts.element_type][0],
            stripe_block_bytes=first.row_bytes * facts.page_size,
            granularity_bytes=facts.granularity_bytes,
            maximum_blocks=facts.reserved_tokens // facts.page_size + PADDING_PAGES,
            stripe_names=tuple(buffer.name for buffer in facts.buffers),
        )

    @property
    def stripe_count(self) -> int:
        return len(self.stripe_names)

    @property
    def bytes_per_block(self) -> int:
        """Bytes of one block summed over all stripes, as the contract counts."""

        return self.stripe_block_bytes * self.stripe_count

    @property
    def resize_alignment_blocks(self) -> int:
        """Fewest blocks whose bytes in one stripe fill whole granules."""

        return self.granularity_bytes // math.gcd(
            self.granularity_bytes, self.stripe_block_bytes
        )

    def stripe_bytes(self, blocks: int) -> int:
        return blocks * self.stripe_block_bytes

    def tokens(self, blocks: int) -> int:
        """Tokens SGLang can allocate in a pool of ``blocks`` mapped blocks."""

        return (blocks - PADDING_PAGES) * self.page_size

    def capacity_groups(self, domain: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "group_id": GROUP_ID,
                "pool_id": POOL_ID,
                "allocation_granularity_tokens": self.page_size,
                "bytes_per_allocation": self.bytes_per_block,
                "memory_domains": [dict(domain)],
                "max_allocations": self.maximum_blocks,
            }
        ]

    def topology(self, model_fingerprint: str) -> dict[str, Any]:
        return {
            "abi_version": dict(STRIPED_POOL_ABI_VERSION),
            "model_fingerprint": model_fingerprint,
            "shard": single_rank_shard(),
            "cache_groups": [
                {
                    "group_id": GROUP_ID,
                    "layers": [{"index": layer} for layer in range(self.layer_count)],
                    "geometry": {
                        "kind": "paged_attention",
                        "block_size_tokens": self.page_size,
                        "kv_heads": self.kv_heads,
                        "key_head_dim": self.head_dim,
                        "value_head_dim": self.head_dim,
                        "element_type": {"kind": self.element_kind},
                        "layout": {"kind": "layer_sequence_head_dim"},
                    },
                    "policy": {"kind": "full_attention"},
                }
            ],
        }

    def attachment_views(self) -> list[dict[str, Any]]:
        """One view per K or V buffer: the whole stripe, named by its layer."""

        length = self.stripe_bytes(self.maximum_blocks)
        views = []
        for stripe in range(self.stripe_count):
            view: dict[str, Any] = {
                "group_id": GROUP_ID,
                "layer": {"index": stripe % self.layer_count},
                "offset_bytes": 0,
                "length_bytes": length,
            }
            if stripe:
                # Stripe 0 is the wire default and stays off the wire.
                view["stripe"] = stripe
            views.append(view)
        return views


def adapter_profile(adapter_version: str, sglang_version: str) -> dict[str, str]:
    """Exact adapter and backend build claiming the striped attachment."""

    return {
        "adapter_id": ADAPTER_ID,
        "adapter_version": adapter_version,
        "backend_version": sglang_version,
        "profile_id": SHARED_POOL_PROFILE_ID,
    }


@dataclass(frozen=True)
class ElasticBinding:
    """The runtime's binding after it was checked against SGLang's buffers."""

    binding_id: str
    minimum_blocks: int
    mapped_blocks: int
    maximum_blocks: int
    segments: tuple[Mapping[str, Any], ...]


def verify_binding(
    geometry: PoolGeometry, receipt: Mapping[str, Any]
) -> ElasticBinding:
    """Refuse a binding whose geometry is not exactly SGLang's.

    The receipt has already passed contract validation against the
    registration; this compares it with the buffers that will be mapped.
    """

    binding = receipt["shared_pools"][0]
    elastic = binding["elastic"]
    problems = []
    if elastic["stripe_count"] != geometry.stripe_count:
        problems.append(
            f"{elastic['stripe_count']} stripes for {geometry.stripe_count} K/V buffers"
        )
    elif binding["bytes_per_block"] // elastic["stripe_count"] != (
        geometry.stripe_block_bytes
    ):
        problems.append(
            f"{binding['bytes_per_block'] // elastic['stripe_count']} bytes per "
            f"block in a stripe, but a buffer row of {geometry.page_size} token(s) "
            f"is {geometry.stripe_block_bytes} bytes"
        )
    if binding["block_count"] != geometry.maximum_blocks:
        problems.append(
            f"{binding['block_count']} blocks for a reservation of "
            f"{geometry.maximum_blocks}"
        )
    if elastic["allocation_granularity_bytes"] != geometry.granularity_bytes:
        problems.append(
            f"allocation granularity {elastic['allocation_granularity_bytes']} "
            f"bytes, but SGLang's arena uses {geometry.granularity_bytes}"
        )
    if elastic["minimum_block_count"] <= PADDING_PAGES:
        problems.append("the minimum pool holds no token besides the padding page")
    if problems:
        raise SharedPoolError(
            "the runtime's binding does not match SGLang's KV buffers: "
            + "; ".join(problems)
        )
    segments = check_segment_coverage(
        elastic["segments"],
        stripes=geometry.stripe_count,
        granularity=geometry.granularity_bytes,
        start=0,
        end=geometry.stripe_bytes(elastic["mapped_block_count"]),
    )
    return ElasticBinding(
        binding_id=binding["binding_id"],
        minimum_blocks=elastic["minimum_block_count"],
        mapped_blocks=elastic["mapped_block_count"],
        maximum_blocks=elastic["maximum_block_count"],
        segments=tuple(segments),
    )


@dataclass(frozen=True)
class MappedSegment:
    segment_id: str
    stripe: int
    offset_bytes: int
    length_bytes: int

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.length_bytes


class StripedPool:
    """Runtime-owned segments mapped into SGLang's buffers, one prefix per stripe.

    Every stripe is always backed to the same block count. A failed mapping
    is rolled back before the error is raised, so the pool never reports a
    prefix that only some stripes have.
    """

    def __init__(
        self, geometry: PoolGeometry, binding: ElasticBinding, backend: VmmBackend
    ) -> None:
        self.geometry = geometry
        self.binding_id = binding.binding_id
        self.minimum_blocks = binding.minimum_blocks
        self.maximum_blocks = binding.maximum_blocks
        self._binding = binding
        self._backend = backend
        self._mapped_blocks = 0
        self._stripes: list[list[MappedSegment]] = [
            [] for _ in range(geometry.stripe_count)
        ]

    @property
    def mapped_blocks(self) -> int:
        return self._mapped_blocks

    def segment_count(self) -> int:
        return sum(len(stripe) for stripe in self._stripes)

    def map_initial(self, handles: Sequence[int]) -> None:
        """Map the prefix the runtime granted at registration."""

        if self._mapped_blocks:
            raise SharedPoolError("the pool's initial prefix is already mapped")
        self._map(self._binding.segments, handles, self._binding.mapped_blocks)

    def check_operation(self, operation: Mapping[str, Any]) -> None:
        """Refuse a resize operation that is not about this pool's geometry."""

        geometry = self.geometry
        if (
            operation["binding_id"] != self.binding_id
            or operation["bytes_per_block"] != geometry.bytes_per_block
            or operation["allocation_granularity_bytes"] != geometry.granularity_bytes
            or operation["stripe_count"] != geometry.stripe_count
        ):
            raise SharedPoolError(
                f"resize generation {operation['resize_generation']} does not "
                f"describe binding {self.binding_id!r} as it was attached"
            )
        if not (
            self.minimum_blocks
            <= operation["target_block_count"]
            <= self.maximum_blocks
        ):
            raise SharedPoolError(
                f"resize target {operation['target_block_count']} blocks is "
                f"outside [{self.minimum_blocks}, {self.maximum_blocks}]"
            )

    def map_growth(self, operation: Mapping[str, Any], handles: Sequence[int]) -> None:
        """Map the tail a ``map_workers`` operation adds to every stripe."""

        self.check_operation(operation)
        if operation["from_block_count"] != self._mapped_blocks:
            raise SharedPoolError(
                f"growth starts at {operation['from_block_count']} blocks, but "
                f"{self._mapped_blocks} are mapped"
            )
        self._map(operation["segments"], handles, operation["target_block_count"])

    def retirement_plan(self, target_blocks: int) -> list[MappedSegment]:
        """The mappings a shrink to ``target_blocks`` unmaps, highest first.

        CUDA unmaps whole mappings only, so a boundary that falls inside one
        is refused before anything is changed.
        """

        if not self.minimum_blocks <= target_blocks < self._mapped_blocks:
            raise SharedPoolError(
                f"cannot retire to {target_blocks} blocks: {self._mapped_blocks} "
                f"are mapped and the minimum is {self.minimum_blocks}"
            )
        boundary = self.geometry.stripe_bytes(target_blocks)
        plan = []
        for stripe in self._stripes:
            for segment in reversed(stripe):
                if segment.end_bytes <= boundary:
                    break
                if segment.offset_bytes < boundary:
                    raise SharedPoolError(
                        f"the boundary at {target_blocks} blocks falls inside "
                        f"mapping {segment.segment_id!r} of buffer "
                        f"{self.geometry.stripe_names[segment.stripe]}; CUDA "
                        "unmaps whole mappings only"
                    )
                plan.append(segment)
        return plan

    def unmap_tail(self, operation: Mapping[str, Any]) -> None:
        """Unmap the tail an ``unmap_workers`` operation releases."""

        self.check_operation(operation)
        if operation["from_block_count"] != self._mapped_blocks:
            raise SharedPoolError(
                f"shrink starts at {operation['from_block_count']} blocks, but "
                f"{self._mapped_blocks} are mapped"
            )
        target_blocks = operation["target_block_count"]
        plan = self.retirement_plan(target_blocks)
        named = {
            (
                segment["segment_id"],
                segment["stripe"],
                segment["offset_bytes"],
                segment["length_bytes"],
            )
            for segment in operation["segments"]
        }
        held = {
            (
                segment.segment_id,
                segment.stripe,
                segment.offset_bytes,
                segment.length_bytes,
            )
            for segment in plan
        }
        if named != held:
            raise SharedPoolError(
                "the runtime asked to unmap segments that are not the mappings "
                "this server holds above the boundary; CUDA unmaps whole "
                "mappings only"
            )
        # A retired slot must have no reader left on the device.
        self._backend.synchronize()
        for segment in plan:
            self._backend.unmap(
                segment.stripe, segment.offset_bytes, segment.length_bytes
            )
            self._stripes[segment.stripe].remove(segment)
        self._mapped_blocks = target_blocks

    def require_backed(self, span_bytes: Sequence[int]) -> None:
        """Refuse a request from SGLang to back more than the runtime mapped."""

        mapped = self.geometry.stripe_bytes(self._mapped_blocks)
        if len(span_bytes) != self.geometry.stripe_count:
            raise SharedPoolError(
                f"SGLang asked to back {len(span_bytes)} buffers of a pool with "
                f"{self.geometry.stripe_count}"
            )
        for name, span in zip(self.geometry.stripe_names, span_bytes, strict=True):
            if not 0 <= int(span) <= mapped:
                raise SharedPoolError(
                    f"SGLang asked to back {int(span)} bytes of buffer {name}, "
                    f"but the runtime mapped {mapped}"
                )

    def _map(
        self,
        segments: Sequence[Mapping[str, Any]],
        handles: Sequence[int],
        target_blocks: int,
    ) -> None:
        geometry = self.geometry
        try:
            ordered = check_segment_coverage(
                list(segments),
                stripes=geometry.stripe_count,
                granularity=geometry.granularity_bytes,
                start=geometry.stripe_bytes(self._mapped_blocks),
                end=geometry.stripe_bytes(target_blocks),
            )
        except ContractValidationError as error:
            raise SharedPoolError(
                f"segments do not extend every buffer from {self._mapped_blocks} "
                f"to {target_blocks} blocks: {error}"
            ) from error
        held = {segment.segment_id for stripe in self._stripes for segment in stripe}
        for segment in ordered:
            if segment["segment_id"] in held:
                raise SharedPoolError(
                    f"segment {segment['segment_id']!r} is already mapped"
                )
            if segment["handle_index"] >= len(handles):
                raise SharedPoolError(
                    f"segment {segment['segment_id']!r} names OS handle "
                    f"{segment['handle_index']} of {len(handles)}"
                )
        mapped: list[MappedSegment] = []
        try:
            for segment in ordered:
                self._backend.map(
                    segment["stripe"],
                    segment["offset_bytes"],
                    segment["length_bytes"],
                    handles[segment["handle_index"]],
                )
                mapped.append(
                    MappedSegment(
                        segment["segment_id"],
                        segment["stripe"],
                        segment["offset_bytes"],
                        segment["length_bytes"],
                    )
                )
        except BaseException as error:
            self._roll_back(mapped)
            raise SharedPoolError(
                f"mapping runtime memory into SGLang's KV buffers failed after "
                f"{len(mapped)} of {len(ordered)} segments: {error}"
            ) from error
        for segment in mapped:
            self._stripes[segment.stripe].append(segment)
        self._mapped_blocks = target_blocks

    def _roll_back(self, mapped: list[MappedSegment]) -> None:
        for segment in reversed(mapped):
            try:
                self._backend.unmap(
                    segment.stripe, segment.offset_bytes, segment.length_bytes
                )
            except Exception:  # the mapping error is the one to report
                logger.exception(
                    "could not unmap segment %s while rolling back a failed "
                    "KV pool mapping",
                    segment.segment_id,
                )


def _unsupported_layout_reasons(facts: PoolFacts) -> list[str]:
    """Every reason SGLang's buffers cannot be described as equal stripes."""

    reasons: list[str] = []
    buffers = facts.buffers
    count = len(buffers)
    if facts.page_size != 1:
        reasons.append(
            f"page size {facts.page_size}: only the token allocator with page "
            "size 1 is covered"
        )
    if count < 2 or count % 2:
        reasons.append(f"expected one K and one V buffer per layer, found {count}")
        return reasons
    if count > MAX_POOL_STRIPES:
        reasons.append(
            f"{count} KV buffers exceed the {MAX_POOL_STRIPES} stripes one "
            "binding can carry"
        )
    layers = count // 2
    expected_names = [f"k{layer}" for layer in range(layers)] + [
        f"v{layer}" for layer in range(layers)
    ]
    if [buffer.name for buffer in buffers] != expected_names:
        reasons.append("buffers are not ordered k0..k(L-1), v0..v(L-1)")
    if facts.element_type not in _ELEMENT_TYPES:
        reasons.append(
            f"KV element type {facts.element_type} is not covered (quantized and "
            "FP8 KV are outside the profile)"
        )
    if any(buffer.tokens_per_row != 1 for buffer in buffers):
        reasons.append("buffer rows are pages, not token slots (layout is not NHD)")
    first = buffers[0]
    rows = facts.reserved_tokens + facts.page_size * PADDING_PAGES
    if any(
        buffer.shape != first.shape or buffer.row_bytes != first.row_bytes
        for buffer in buffers
    ):
        reasons.append(
            "K and V buffers differ in shape (key and value head dimensions differ)"
        )
    elif len(first.shape) != 3 or first.shape[0] != rows or first.row_bytes <= 0:
        reasons.append(
            f"buffer shape {first.shape} is not ({rows} rows, heads, head dim)"
        )
    elif facts.element_type in _ELEMENT_TYPES and first.row_bytes != (
        first.shape[1] * first.shape[2] * _ELEMENT_TYPES[facts.element_type][1]
    ):
        reasons.append(
            f"row size {first.row_bytes} bytes does not match shape {first.shape}"
        )
    if any(buffer.reserved_bytes != rows * buffer.row_bytes for buffer in buffers):
        reasons.append("a buffer's reserved span is not rows times row size")
    if any(buffer.backed_bytes for buffer in buffers):
        reasons.append("SGLang already backed part of the pool with its own memory")
    granularity = facts.granularity_bytes
    if granularity <= 0:
        reasons.append(f"allocation granularity {granularity} is not positive")
        return reasons
    if first.row_bytes > 0 and (rows * first.row_bytes) % granularity:
        unit = granularity // math.gcd(granularity, first.row_bytes)
        reasons.append(
            f"a buffer of {rows} rows ({rows * first.row_bytes} bytes) does not "
            f"fill whole {granularity}-byte granules; --max-total-tokens must be "
            f"a multiple of {unit} rows less the padding page (k * {unit} - "
            f"{facts.page_size * PADDING_PAGES})"
        )
    end = 0
    for buffer in sorted(buffers, key=lambda buffer: buffer.reservation_offset):
        if buffer.reservation_offset % granularity or buffer.reservation_offset < end:
            reasons.append(
                f"buffer {buffer.name} is not placed on its own granule-aligned "
                "range of the reservation"
            )
            break
        end = buffer.reservation_offset + buffer.reserved_bytes
    else:
        if end > facts.reservation_bytes:
            reasons.append("the buffers extend past SGLang's address reservation")
    return reasons
