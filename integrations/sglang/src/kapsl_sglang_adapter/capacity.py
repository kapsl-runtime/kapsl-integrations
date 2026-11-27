"""Opaque byte accounting for SGLang's statically allocated KV pool.

Nothing here imports SGLang. The hook layer reduces a live scheduler to
:class:`SchedulerFacts`; this module decides whether that scheduler is inside
the profile the adapter was designed for and, if so, how its token pages map to
device bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import UINT32_MAX

# Exact SGLang releases whose hook targets and scheduler attributes this
# adapter was written against. Anything else is refused, not attempted.
PINNED_SGLANG_VERSIONS: tuple[str, ...] = ("0.5.23",)

# KV pool classes with one K/V (or fused KV) buffer set sized by
# ``max_total_num_tokens``. Hybrid, sliding-window, sparse, and recurrent-state
# pools need their own cache groups and are not modelled yet.
SUPPORTED_POOL_CLASSES: frozenset[str] = frozenset(
    {"MHATokenToKVPool", "MLATokenToKVPool"}
)

# SGLang device string -> the memory-domain kind the launcher must name.
_DEVICE_DOMAIN_KINDS: dict[str, str] = {"cuda": "cuda", "cpu": "host"}

GROUP_ID = "sglang.kv.0"
POOL_ID = "sglang.pool.0"


class UnsupportedProfileError(RuntimeError):
    """The running SGLang configuration is outside the governed profile."""


@dataclass(frozen=True)
class SchedulerFacts:
    """What the adapter needs to know about one SGLang scheduler process."""

    sglang_version: str
    device: str
    pool_class: str
    page_size: int
    max_total_num_tokens: int
    kv_pool_bytes: int
    post_capture_sizing: bool
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    disaggregation_mode: str
    speculative_decoding: bool
    # Names of enabled scheduler features that allocate KV outside the single
    # leased request row (see ``hooks.UNSUPPORTED_FEATURE_FLAGS``).
    unsupported_features: tuple[str, ...] = ()


def require_pinned_version(version: str) -> None:
    if version not in PINNED_SGLANG_VERSIONS:
        pinned = ", ".join(PINNED_SGLANG_VERSIONS)
        raise UnsupportedProfileError(
            f"SGLang {version} is not a pinned release for this adapter "
            f"(pinned: {pinned}); its scheduler hook targets are unverified"
        )


def unsupported_reasons(facts: SchedulerFacts) -> list[str]:
    """Every reason ``facts`` falls outside the governed profile."""

    reasons: list[str] = []
    if facts.sglang_version not in PINNED_SGLANG_VERSIONS:
        reasons.append(f"SGLang {facts.sglang_version} is not a pinned release")
    if facts.device not in _DEVICE_DOMAIN_KINDS:
        reasons.append(f"device {facts.device!r} has no Kapsl memory-domain mapping")
    if facts.pool_class not in SUPPORTED_POOL_CLASSES:
        reasons.append(f"KV pool class {facts.pool_class} is not modelled")
    if facts.post_capture_sizing:
        reasons.append("post-capture KV sizing reports a virtual upper bound")
    for name, size in (
        ("tensor", facts.tensor_parallel_size),
        ("pipeline", facts.pipeline_parallel_size),
        ("data", facts.data_parallel_size),
    ):
        if size != 1:
            reasons.append(
                f"{name}-parallel size {size} runs more than one scheduler rank"
            )
    if facts.disaggregation_mode != "null":
        reasons.append(
            f"disaggregation mode {facts.disaggregation_mode!r} admits requests "
            "outside the hooked prefill path"
        )
    if facts.speculative_decoding:
        reasons.append("speculative decoding adds an unmodelled draft KV pool")
    for feature in facts.unsupported_features:
        reasons.append(f"scheduler feature {feature} is not modelled")
    for name, value in (
        ("page_size", facts.page_size),
        ("max_total_num_tokens", facts.max_total_num_tokens),
        ("kv_pool_bytes", facts.kv_pool_bytes),
    ):
        if not _is_positive_int(value):
            reasons.append(f"{name} must be a positive integer, got {value!r}")
    if (
        _is_positive_int(facts.page_size)
        and _is_positive_int(facts.max_total_num_tokens)
        and facts.max_total_num_tokens < facts.page_size
    ):
        reasons.append("the KV pool is smaller than one page")
    return reasons


def require_supported(facts: SchedulerFacts) -> None:
    reasons = unsupported_reasons(facts)
    if reasons:
        raise UnsupportedProfileError(
            "SGLang configuration is outside the Kapsl opaque profile: "
            + "; ".join(reasons)
        )


def opaque_capacity_groups(
    facts: SchedulerFacts, memory_domains: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Describe the scheduler's KV pool as one opaque capacity group.

    One allocation is one SGLang page. Its byte cost is the pool's reported
    K/V bytes spread over the *schedulable* pages and rounded up, so a fully
    leased pool is charged at least what the pool physically occupies (SGLang
    keeps one extra padding page that no request can lease).
    """

    require_supported(facts)
    if len(memory_domains) != 1:
        raise UnsupportedProfileError(
            "a single-rank SGLang scheduler owns exactly one KV pool; "
            f"{len(memory_domains)} memory domains were configured"
        )
    domain = dict(memory_domains[0])
    expected_kind = _DEVICE_DOMAIN_KINDS[facts.device]
    if domain.get("kind") != expected_kind:
        raise UnsupportedProfileError(
            f"SGLang device {facts.device!r} needs a {expected_kind!r} memory "
            f"domain, but {domain.get('kind')!r} was configured"
        )
    pages = facts.max_total_num_tokens // facts.page_size
    return [
        {
            "group_id": GROUP_ID,
            "pool_id": POOL_ID,
            "allocation_granularity_tokens": facts.page_size,
            "bytes_per_allocation": -(-facts.kv_pool_bytes // pages),
            "memory_domains": [domain],
            "max_allocations": pages,
        }
    ]


def pool_token_capacity(groups: Sequence[Mapping[str, Any]]) -> int:
    """Largest token reservation the registered capacity model can express."""

    return min(
        min(
            int(group["allocation_granularity_tokens"]) * int(group["max_allocations"])
            for group in groups
        ),
        UINT32_MAX,
    )


def request_token_capacity(
    *, input_tokens: int, max_new_tokens: int, pool_tokens: int
) -> int:
    """Tokens of KV a request may come to hold: its prompt plus its output cap.

    A shared radix-cache prefix is still charged in full; the estimate is
    deliberately conservative, like SGLang's own admission budget. It is
    clamped to the pool, because SGLang never lets one request exceed it and a
    larger reservation could never be granted.
    """

    for name, value in (
        ("input_tokens", input_tokens),
        ("max_new_tokens", max_new_tokens),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not _is_positive_int(pool_tokens):
        raise ValueError("pool_tokens must be a positive integer")
    return max(1, min(input_tokens + max_new_tokens, pool_tokens))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
