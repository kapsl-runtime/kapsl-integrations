"""SGLang 0.5.23's VMM-backed KV pool, as the shared-pool tier touches it.

Everything the elastic tier reads or writes inside SGLang is in this module,
and every name it relies on is an SGLang internal that only the pinned release
is known to have. Objects are duck-typed and ``torch`` and SGLang's
``cuda_vmm_utils`` are passed in, so the host tests run this code against
stand-ins; nothing here has run against a real server or a GPU.

The pieces, each on one seam:

* :func:`pool_facts` reads the buffers of ``KvVmmBufferOwner`` (the arena
  SGLang builds when ``SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`` is on);
* :class:`SglangVmmBackend` is the CUDA seam: it maps a runtime-exported
  allocation into that arena where SGLang would have created its own, and
  unmaps whole mappings;
* :class:`SlotLists` edits the token allocator's free list;
* :class:`SglangPoolHost` changes the six places SGLang keeps the pool
  capacity in, together.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

from .capacity import SharedPoolDetails, UnsupportedProfileError
from .shared_pool import (
    BufferFacts,
    PoolFacts,
    RetirementDeferred,
    SharedPoolError,
)

# SGLang builds its KV pool on CUDA virtual memory only when this is set, and
# reads it in the launcher process while resolving server arguments.
POST_CAPTURE_ENV = "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING"
_TRUE_VALUES = frozenset({"true", "1", "yes", "y"})

# Scheduler attributes that, when truthy, mean the prefix cache keeps KV in a
# tier besides the pool. Evicting then copies data elsewhere asynchronously;
# the tail retirement here only knows slots that are free or cached in place.
HOST_CACHE_FLAGS: tuple[str, ...] = ("enable_hierarchical_cache", "enable_lmcache")

# SGLang sizes its KV pool from free GPU memory. The runtime owns this pool's
# memory, so its size is the runtime's decision: the byte budget is lifted and
# the token counts alone (``--max-total-tokens`` for the reservation, the
# runtime's mapped prefix after graph capture) decide.
UNBOUNDED_KV_BUDGET_BYTES = 1 << 60


def post_capture_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get(POST_CAPTURE_ENV, "").strip().lower() in _TRUE_VALUES


def shared_pool_launch_reasons(
    environ: Mapping[str, str], server_args: Any
) -> list[str]:
    """What rules out the shared pool before the model is loaded."""

    reasons: list[str] = []
    if not post_capture_enabled(environ):
        reasons.append(
            f"{POST_CAPTURE_ENV} is not set, so SGLang would allocate the KV pool "
            "itself (kapsl-sglang-serve sets it)"
        )
    try:
        for flag, name in (
            ("--tp-size", "tp_size"),
            ("--pp-size", "pp_size"),
            ("--dp-size", "dp_size"),
        ):
            if _index(getattr(server_args, name)) != 1:
                reasons.append(f"{flag} must be 1: one rank owns the pool")
        maximum = server_args.max_total_tokens
    except (AttributeError, TypeError) as error:
        raise UnsupportedProfileError(
            f"SGLang server arguments do not expose the expected fields ({error})"
        ) from error
    if maximum is None:
        reasons.append(
            "--max-total-tokens is required: it fixes the address range the pool "
            "can grow into"
        )
    else:
        try:
            positive = _index(maximum) > 0
        except TypeError:
            positive = False
        if not positive:
            reasons.append("--max-total-tokens must be a positive integer")
    return reasons


def lift_kv_budget(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], *, cap_tokens: int | None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Arguments for ``KVCacheConfigurator.config_from_budget`` under Kapsl.

    The byte budget is replaced with an unbounded one. ``cap_tokens`` is left
    alone for the call that sizes the reservation and set to the runtime's
    mapped prefix for the call that sizes the backed pool.
    """

    rewritten = list(args)
    options = dict(kwargs)
    if len(rewritten) >= 2:
        rewritten[1] = UNBOUNDED_KV_BUDGET_BYTES
    elif "budget_bytes" in options:
        options["budget_bytes"] = UNBOUNDED_KV_BUDGET_BYTES
    else:
        raise UnsupportedProfileError(
            "SGLang sized its KV pool without a byte budget argument"
        )
    if cap_tokens is not None:
        options["cap_tokens"] = cap_tokens
    return tuple(rewritten), options


def pool_facts(owner: Any) -> PoolFacts:
    """Reduce SGLang's ``KvVmmBufferOwner`` to the facts the pool is built on.

    Called before anything is backed. A missing attribute means SGLang's
    internals moved; that is reported, not guessed around.
    """

    try:
        arena = owner._arena
        if len(arena._allocation._mappings) or _index(arena.backed_bytes):
            raise UnsupportedProfileError(
                "SGLang already mapped memory of its own into the KV arena"
            )
        return PoolFacts(
            device_id=_index(owner.device_id),
            page_size=_index(owner.page_size),
            reserved_tokens=_index(owner._reserved_num_tokens),
            granularity_bytes=_index(arena.granularity),
            reservation_bytes=_index(arena.reserved),
            element_type=str(owner.store_dtype).removeprefix("torch."),
            buffers=tuple(
                BufferFacts(
                    name=str(spec.desc.name),
                    reservation_offset=_index(spec.offset),
                    reserved_bytes=_index(spec.reserved_span),
                    backed_bytes=_index(spec.backed_to),
                    row_bytes=_index(spec.desc.row_bytes),
                    tokens_per_row=_index(spec.desc.tokens_per_row),
                    shape=tuple(_index(size) for size in spec.desc.shape),
                )
                for spec in owner._specs
            ),
        )
    except (AttributeError, TypeError) as error:
        raise UnsupportedProfileError(
            f"SGLang's VMM KV backing does not expose the expected state ({error})"
        ) from error


class SglangVmmBackend:
    """The CUDA seam: runtime-owned allocations in SGLang's KV arena.

    ``vmm`` is ``sglang.srt.utils.cuda_vmm_utils`` and ``torch`` is torch.
    Mapping uses SGLang's own primitives for memory it did not create
    (``import_peer_handle`` and ``VmmReservation.map_existing``), and keeps
    the arena's bookkeeping (``backed_to``, the per-offset commit table and
    the backed total) as SGLang's own ``commit_range`` would.
    """

    def __init__(self, owner: Any, *, vmm: Any, torch: Any) -> None:
        self._owner = owner
        self._arena = owner._arena
        self._vmm = vmm
        self._torch = torch

    def map(
        self, stripe: int, offset_bytes: int, length_bytes: int, descriptor: int
    ) -> None:
        spec = self._owner._specs[stripe]
        arena = self._arena
        if _index(spec.backed_to) != offset_bytes:
            raise SharedPoolError(
                f"buffer {spec.desc.name} is backed to {spec.backed_to} bytes, "
                f"not {offset_bytes}"
            )
        # The descriptor is duplicated for the import; the caller keeps its own.
        handle = self._vmm.import_peer_handle(
            None, descriptor, use_fabric=False, peer_rank=0
        )
        try:
            with self._torch.cuda.device(arena.device_id):
                arena._allocation.map_existing(
                    spec.offset + offset_bytes, length_bytes, handle
                )
        finally:
            # The mapping, not the imported handle, keeps the memory alive.
            self._check(self._driver().cuMemRelease(handle), "cuMemRelease(kapsl)")
        spec.backed_to = offset_bytes + length_bytes
        arena._committed_by_offset[spec.offset] = spec.backed_to
        arena._range_backed += length_bytes

    def unmap(self, stripe: int, offset_bytes: int, length_bytes: int) -> None:
        spec = self._owner._specs[stripe]
        arena = self._arena
        mappings = arena._allocation._mappings
        mapping = (arena.base + spec.offset + offset_bytes, length_bytes, None)
        if _index(spec.backed_to) != offset_bytes + length_bytes or (
            mapping not in mappings
        ):
            raise SharedPoolError(
                f"buffer {spec.desc.name} has no mapping of {length_bytes} bytes "
                f"at {offset_bytes} that ends its backed prefix; CUDA unmaps "
                "whole mappings only"
            )
        self._check(
            self._driver().cuMemUnmap(mapping[0], length_bytes), "cuMemUnmap(kapsl)"
        )
        mappings.remove(mapping)
        spec.backed_to = offset_bytes
        arena._committed_by_offset[spec.offset] = offset_bytes
        arena._range_backed -= length_bytes

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def _driver(self) -> Any:
        return self._vmm._get_cuda_driver()

    def _check(self, result: Any, label: str) -> None:
        self._vmm.check_drv(result, label)


class SlotLists:
    """The free slot lists of SGLang's ``TokenToKVPoolAllocator``.

    Slots are numbered from 1; slot 0 is the padding slot. ``resize()`` on the
    allocator clears every allocation, so a live pool adds and removes free
    slots here instead.
    """

    def __init__(self, allocator: Any, torch: Any) -> None:
        self._allocator = allocator
        self._torch = torch

    def add(self, first: int, last: int) -> None:
        """Hand out slots ``first..last`` after the ones already free."""

        free = self._allocator.free_pages
        added = self._torch.arange(
            first, last + 1, dtype=self._torch.int64, device=free.device
        )
        self._allocator.free_pages = self._torch.cat((free, added))

    def free_above(self, bound: int) -> int:
        """How many slots above ``bound`` are free."""

        allocator = self._allocator
        return int((allocator.free_pages > bound).sum()) + int(
            (allocator.release_pages > bound).sum()
        )

    def drop_above(self, bound: int) -> None:
        """Stop handing out any slot above ``bound``."""

        allocator = self._allocator
        free, released = allocator.free_pages, allocator.release_pages
        allocator.free_pages = free[free <= bound]
        allocator.release_pages = released[released <= bound]


class SglangPoolHost:
    """SGLang's view of the pool's capacity, changed as one step.

    SGLang 0.5.23 keeps the capacity in six places: the pool's ``size``, the
    allocator's ``size`` (and ``num_pages`` where it has one), the model
    runner's and the scheduler's ``max_total_num_tokens``, and the copies in
    the scheduler's ``pool_stats_observer`` and ``invariant_checker``. Missing
    one makes SGLang's idle leak check stop the scheduler. The observer is a
    frozen, slotted dataclass, hence ``object.__setattr__``.
    """

    def __init__(
        self,
        scheduler: Any,
        owner: Any,
        *,
        torch: Any,
        evict_params: Callable[..., Any],
    ) -> None:
        try:
            self._scheduler = scheduler
            self._runner = scheduler.tp_worker.model_runner
            self._pool = self._runner.token_to_kv_pool
            self._allocator = scheduler.token_to_kv_pool_allocator
            self._tree_cache = scheduler.tree_cache
            self._observer = scheduler.pool_stats_observer
            self._checker = scheduler.invariant_checker
            self.capacity_fields()
        except (AttributeError, TypeError) as error:
            raise UnsupportedProfileError(
                f"SGLang scheduler does not expose the expected KV capacity state "
                f"({error})"
            ) from error
        self._owner = owner
        self._slots = SlotLists(self._allocator, torch)
        self._evict_params = evict_params

    def capacity_fields(self) -> dict[str, int]:
        return {
            "pool.size": _index(self._pool.size),
            "allocator.size": _index(self._allocator.size),
            "model_runner.max_total_num_tokens": _index(
                self._runner.max_total_num_tokens
            ),
            "scheduler.max_total_num_tokens": _index(
                self._scheduler.max_total_num_tokens
            ),
            "pool_stats_observer.max_total_num_tokens": _index(
                self._observer.max_total_num_tokens
            ),
            "invariant_checker.max_total_num_tokens": _index(
                self._checker.max_total_num_tokens
            ),
        }

    def tokens(self) -> int:
        fields = self.capacity_fields()
        if len(set(fields.values())) != 1:
            raise SharedPoolError(f"SGLang's KV capacity fields disagree: {fields}")
        return next(iter(fields.values()))

    def publish(self, tokens: int) -> None:
        """Growth. Running requests are untouched: nothing they use moves."""

        current = self.tokens()
        if tokens <= current:
            raise SharedPoolError(
                f"cannot grow SGLang's pool from {current} to {tokens} tokens"
            )
        self._set_capacity(tokens)
        self._slots.add(current + 1, tokens)

    def retire(self, tokens: int) -> None:
        """Shrink: evict what occupies the tail, then stop handing it out.

        Only on an idle server, and only once every slot above the boundary is
        free: a retired slot must have no reader. Eviction goes through the
        prefix cache's own least-recently-used order until the tail is free
        or nothing evictable is left.
        """

        current = self.tokens()
        if not 0 < tokens < current:
            raise SharedPoolError(
                f"cannot retire SGLang's pool from {current} to {tokens} tokens"
            )
        if not self.idle():
            raise RetirementDeferred("SGLang is not idle")
        busy = self._evict_tail(tokens, current - tokens)
        if busy:
            raise RetirementDeferred(
                f"{busy} KV slot(s) above token {tokens} are still referenced"
            )
        self._slots.drop_above(tokens)
        self._set_capacity(tokens)

    def idle(self) -> bool:
        """No batch is running or awaiting its result, and no queued request
        holds KV. Requests that only wait in the queue do not count: they own
        no slot, and the adapter does not admit them while a shrink waits."""

        return (
            bool(self._scheduler.is_fully_idle(ignore_waiting=True))
            and self._allocator.free_group is None
        )

    def _evict_tail(self, tokens: int, tail: int) -> int:
        """Evict cached prefixes until the tail is free. Returns slots still busy."""

        busy = tail - self._slots.free_above(tokens)
        while busy > 0:
            evictable = _index(self._tree_cache.evictable_size())
            if evictable <= 0:
                break
            self._tree_cache.evict(self._evict_params(num_tokens=busy))
            if _index(self._tree_cache.evictable_size()) >= evictable:
                break  # the cache evicted nothing; do not spin
            busy = tail - self._slots.free_above(tokens)
        if busy < 0:
            raise SharedPoolError(
                f"SGLang's free list holds {-busy} more slot(s) above token "
                f"{tokens} than the pool has"
            )
        return busy

    def _set_capacity(self, tokens: int) -> None:
        self._pool.size = tokens
        self._allocator.size = tokens
        if hasattr(self._allocator, "num_pages"):
            self._allocator.num_pages = tokens // _index(self._allocator.page_size)
        self._runner.max_total_num_tokens = tokens
        self._scheduler.max_total_num_tokens = tokens
        object.__setattr__(self._observer, "max_total_num_tokens", tokens)
        object.__setattr__(self._checker, "max_total_num_tokens", tokens)
        # SGLang's own record of what its VMM owner backs.
        self._owner._final_num_tokens = tokens


def shared_pool_details(scheduler: Any, owner: Any) -> SharedPoolDetails:
    """Read the scheduler facts only the shared-pool profile depends on."""

    try:
        allocator = scheduler.token_to_kv_pool_allocator
        pool = allocator.get_kvcache()
        return SharedPoolDetails(
            allocator_class=type(allocator).__name__,
            allocator_page_size=_index(allocator.page_size),
            quantized_kv=bool(pool.is_quantized_kv_cache),
            kv_shard_widening=_index(scheduler.kv_shard_widening),
            pool_is_attached=(
                pool is scheduler.tp_worker.model_runner.token_to_kv_pool
                and pool._post_capture_owner is owner
            ),
            context_length=_index(scheduler.model_config.context_len),
            host_cache_tiers=tuple(
                name for name in HOST_CACHE_FLAGS if getattr(scheduler, name)
            ),
        )
    except (AttributeError, TypeError) as error:
        raise UnsupportedProfileError(
            f"SGLang scheduler does not expose the expected KV pool state ({error})"
        ) from error


def _index(value: Any) -> int:
    """An exact Python integer. SGLang reports sizes as NumPy integers."""

    if isinstance(value, bool):
        raise TypeError(f"expected an integer, got {value!r}")
    return operator.index(value)
