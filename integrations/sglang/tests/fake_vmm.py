"""Stand-ins for the parts of SGLang, torch and CUDA the shared-pool tier touches.

Like ``fake_sglang``, these reproduce the attribute names and call shapes the
adapter was written against (SGLang 0.5.23 ``mem_cache/kv_vmm_backing.py``,
``mem_cache/allocator/token.py``, ``utils/cuda_vmm_utils.py`` and the
scheduler's capacity fields). They are written from reading that source. They
are not evidence that a real server, torch, or the CUDA driver behaves this
way; the README lists what a GPU run still has to confirm.

``FakePoolRuntime`` stands in for the Kapsl runtime's control listener. Its
resize sequencing follows the coordinator in ``kapsl-cli``
(``runtime/kv/control.rs``): a growth is ``map_workers`` then
``activate_scheduler``, a shrink is ``retire_scheduler`` then
``unmap_workers``, each acknowledged by the actor that applied it.
"""

from __future__ import annotations

import contextlib
import operator
import os
import types
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from kapsl_sglang_adapter.client import KapslKvControlError
from kapsl_sglang_adapter.pool_contract import (
    validate_attachment,
    validate_resize_ack_request,
    validate_resize_operation,
    validate_resize_poll_request,
    validate_striped_pool_receipt,
    validate_striped_pool_registration,
)

from fake_sglang import FakeClient

ARENA_BASE = 0x7F00_0000_0000


class NumpyInt:
    """An exact integer that is not an ``int``, like ``numpy.int64``."""

    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"np.int64({self._value})"


class FakeDtype:
    def __init__(self, name: str, itemsize: int) -> None:
        self.name = name
        self.itemsize = itemsize

    def __str__(self) -> str:
        return f"torch.{self.name}"


BFLOAT16 = FakeDtype("bfloat16", 2)
FLOAT32 = FakeDtype("float32", 4)
UINT8 = FakeDtype("uint8", 1)


class FakeTensor:
    """The few tensor operations the adapter applies to a free slot list."""

    def __init__(self, values: Any, device: str = "cuda:0") -> None:
        self.values = list(values)
        self.device = device

    def __len__(self) -> int:
        return len(self.values)

    def __gt__(self, bound: int) -> FakeTensor:
        return FakeTensor([value > bound for value in self.values], self.device)

    def __le__(self, bound: int) -> FakeTensor:
        return FakeTensor([value <= bound for value in self.values], self.device)

    def sum(self) -> int:
        return sum(self.values)

    def __getitem__(self, key: Any) -> FakeTensor:
        if isinstance(key, FakeTensor):
            kept = [
                value
                for value, keep in zip(self.values, key.values, strict=True)
                if keep
            ]
            return FakeTensor(kept, self.device)
        if isinstance(key, slice):
            return FakeTensor(self.values[key], self.device)
        raise TypeError(f"unsupported fake tensor index {key!r}")

    def tolist(self) -> list[int]:
        return list(self.values)


@dataclass
class PoolShape:
    """Geometry of the fake model's KV pool and of the fake device."""

    layers: int = 2
    kv_heads: int = 2
    head_dim: int = 8
    v_head_dim: int | None = None
    dtype: FakeDtype = BFLOAT16
    page_size: int = 1
    tokens_per_row: int = 1
    granularity: int = 1024
    context_len: int = 16
    # Free GPU memory SGLang would size its pool from if left to itself.
    free_bytes: int = 4096
    quantized: bool = False

    @property
    def row_bytes(self) -> int:
        return self.kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def cell_bytes(self) -> int:
        v_dim = self.head_dim if self.v_head_dim is None else self.v_head_dim
        return (
            self.layers * self.kv_heads * (self.head_dim + v_dim) * self.dtype.itemsize
        )


@dataclass
class CudaLog:
    """What the fake driver was asked to do, and where it should fail."""

    fail_map_at: int | None = None
    fail_unmap: bool = False
    fail_release: bool = False
    map_calls: int = 0
    imported: list[int] = field(default_factory=list)
    released: int = 0


@dataclass
class EvictParams:
    num_tokens: int = 0


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerPoolStatsObserver:
    max_total_num_tokens: int


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    max_total_num_tokens: int


class MemoryPoolConfig:
    def __init__(self, max_total_num_tokens: int) -> None:
        self.max_total_num_tokens = max_total_num_tokens


class KvBufferDesc:
    def __init__(
        self, name: str, shape: tuple, *, row_bytes: int, tokens_per_row: int
    ) -> None:
        self.name = name
        self.shape = tuple(shape)
        self.row_bytes = row_bytes
        self.tokens_per_row = tokens_per_row

    def _rows(self, num_tokens: int) -> int:
        return (max(num_tokens, 0) + self.tokens_per_row - 1) // self.tokens_per_row

    def _row(self) -> int:
        return operator.index(self.row_bytes)

    def reserved_span_bytes(self) -> int:
        return self.shape[0] * self._row()

    def prefix_span_bytes(self, num_tokens: int, page_size: int) -> int:
        return self._rows(num_tokens) * self._row()

    def final_span_bytes(self, num_tokens: int, page_size: int) -> int:
        return self._rows(max(num_tokens, 0) + page_size) * self._row()


class _BufferSpec:
    def __init__(self, desc: KvBufferDesc, offset: int, reserved_span: int) -> None:
        self.desc = desc
        self.offset = offset
        self.reserved_span = reserved_span
        self.backed_to = 0


class RadixCache:
    """Cached prefixes in least-recently-used order, each a list of KV slots."""

    def __init__(self, allocator: Any) -> None:
        self._allocator = allocator
        self.entries: list[list[int]] = []
        self.locked: list[list[int]] = []
        self.evict_calls: list[int] = []
        self.evicts = True

    def cache(self, slots: list[int], *, locked: bool = False) -> None:
        """Move ``slots`` out of the free list into a cached prefix."""

        taken = set(slots)
        free = self._allocator.free_pages
        assert taken <= set(free.values), "slots are not free"
        self._allocator.free_pages = FakeTensor(
            [slot for slot in free.values if slot not in taken], free.device
        )
        (self.locked if locked else self.entries).append(list(slots))

    def evictable_size(self) -> int:
        return sum(len(entry) for entry in self.entries)

    def evict(self, params: EvictParams) -> None:
        self.evict_calls.append(params.num_tokens)
        evicted = 0
        while self.evicts and evicted < params.num_tokens and self.entries:
            entry = self.entries.pop(0)
            self._allocator.free(FakeTensor(entry))
            evicted += len(entry)


class TokenToKVPoolAllocator:
    """SGLang's token allocator: slot 0 is padding, slots 1..size are handed out."""

    def __init__(self, size: int, kvcache: Any, *, need_sort: bool = False) -> None:
        self.size = size
        self.page_size = 1
        self.device = "cuda:0"
        self.need_sort = need_sort
        self._kvcache = kvcache
        self.free_group = None
        self.clear_calls = 0
        self.clear()

    def clear(self) -> None:
        self.clear_calls += 1
        self.free_pages = FakeTensor(range(1, self.size + 1), self.device)
        self.release_pages = FakeTensor([], self.device)

    def resize(self, config: MemoryPoolConfig) -> None:
        self.size = config.max_total_num_tokens
        self.clear()

    def get_kvcache(self) -> Any:
        return self._kvcache

    def available_size(self) -> int:
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int) -> FakeTensor | None:
        if need_size > len(self.free_pages):
            return None
        taken = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return taken

    def free(self, free_index: FakeTensor) -> None:
        if self.need_sort:
            self.release_pages = FakeTensor(
                self.release_pages.values + free_index.values, self.device
            )
        else:
            self.free_pages = FakeTensor(
                self.free_pages.values + free_index.values, self.device
            )


class PagedTokenToKVPoolAllocator(TokenToKVPoolAllocator):
    pass


def pool_facts_for(shape: PoolShape, max_total_tokens: int) -> Any:
    """The facts the adapter would read from a pool of this shape."""

    from kapsl_sglang_adapter.shared_pool import BufferFacts, PoolFacts

    rows = (max_total_tokens + shape.page_size) // shape.tokens_per_row
    v_head_dim = shape.head_dim if shape.v_head_dim is None else shape.v_head_dim
    stride = (
        _align_up(rows * shape.row_bytes, shape.granularity) + 3 * shape.granularity
    )
    buffers = []
    for prefix, dim in (("k", shape.head_dim), ("v", v_head_dim)):
        for layer in range(shape.layers):
            row_bytes = shape.kv_heads * dim * shape.dtype.itemsize
            buffers.append(
                BufferFacts(
                    name=f"{prefix}{layer}",
                    reservation_offset=len(buffers) * stride,
                    reserved_bytes=rows * row_bytes,
                    backed_bytes=0,
                    row_bytes=row_bytes,
                    tokens_per_row=shape.tokens_per_row,
                    shape=(rows, shape.kv_heads, dim),
                )
            )
    return PoolFacts(
        device_id=0,
        page_size=shape.page_size,
        reserved_tokens=max_total_tokens,
        granularity_bytes=shape.granularity,
        reservation_bytes=len(buffers) * stride + shape.granularity,
        element_type=shape.dtype.name,
        buffers=tuple(buffers),
    )


class RecordingBackend:
    """A ``VmmBackend`` that records calls instead of touching CUDA."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.mapped: set[tuple[int, int, int]] = set()
        self.fail_map_at: int | None = None
        self.fail_unmap = False
        self._maps = 0

    def map(
        self, stripe: int, offset_bytes: int, length_bytes: int, descriptor: int
    ) -> None:
        self._maps += 1
        if self.fail_map_at == self._maps:
            raise RuntimeError("cuMemMap(existing): CUDA_ERROR_INVALID_VALUE")
        os.fstat(descriptor)
        self.mapped.add((stripe, offset_bytes, length_bytes))
        self.events.append(f"map:{stripe}:{offset_bytes}:{length_bytes}")

    def unmap(self, stripe: int, offset_bytes: int, length_bytes: int) -> None:
        if self.fail_unmap:
            raise RuntimeError("cuMemUnmap: CUDA_ERROR_INVALID_VALUE")
        self.mapped.remove((stripe, offset_bytes, length_bytes))
        self.events.append(f"unmap:{stripe}:{offset_bytes}:{length_bytes}")

    def synchronize(self) -> None:
        self.events.append("synchronize")


class RecordingHost:
    """A ``PoolHost`` holding a token count, for participant tests."""

    def __init__(self, tokens: int, events: list[str] | None = None) -> None:
        self._tokens = tokens
        self.events = [] if events is None else events
        self.busy: str | None = None
        self.error: BaseException | None = None

    def tokens(self) -> int:
        return self._tokens

    def publish(self, tokens: int) -> None:
        if self.error is not None:
            raise self.error
        self.events.append(f"publish:{tokens}")
        self._tokens = tokens

    def retire(self, tokens: int) -> None:
        from kapsl_sglang_adapter.shared_pool import RetirementDeferred

        if self.error is not None:
            raise self.error
        if self.busy is not None:
            raise RetirementDeferred(self.busy)
        self.events.append(f"retire:{tokens}")
        self._tokens = tokens


def install_vmm_modules(fake: Any) -> None:
    """Add the VMM pool stand-ins to a ``FakeSglang`` module tree.

    The classes are created per fake, as in ``fake_sglang``, so a hook applied
    in one test never leaks into the next.
    """

    fake.cuda = CudaLog()

    class _Driver:
        def cuMemUnmap(self, address: int, size: int) -> tuple[int]:
            if fake.cuda.fail_unmap:
                return (1,)
            fake.events.append(f"cuda:unmap:{address - ARENA_BASE}:{size}")
            return (0,)

        def cuMemRelease(self, handle: Any) -> tuple[int]:
            if fake.cuda.fail_release:
                return (1,)
            fake.cuda.released += 1
            return (0,)

    driver = _Driver()

    def import_peer_handle(
        fabric_handle: Any, fd: int, *, use_fabric: bool, peer_rank: int
    ) -> Any:
        assert fabric_handle is None and use_fabric is False and peer_rank == 0
        os.fstat(fd)  # the descriptor must still be open when it is imported
        fake.cuda.imported.append(fd)
        return ("imported", fd)

    def check_drv(result: Any, label: str) -> None:
        if result[0] != 0:
            raise RuntimeError(f"{label}: CUDA_ERROR_{result[0]}")

    vmm = fake.modules["sglang.srt.utils.cuda_vmm_utils"]
    vmm.import_peer_handle = import_peer_handle
    vmm.check_drv = check_drv
    vmm._get_cuda_driver = lambda: driver

    @contextlib.contextmanager
    def device(index: int):
        yield

    torch = fake.modules["torch"]
    torch.int64 = "int64"
    torch.arange = lambda start, end, dtype=None, device=None: FakeTensor(
        range(start, end), device
    )
    torch.cat = lambda tensors: FakeTensor(
        [value for tensor in tensors for value in tensor.values], tensors[0].device
    )
    torch.cuda = types.SimpleNamespace(
        device=device, synchronize=lambda: fake.events.append("cuda:synchronize")
    )

    class VmmReservation:
        def __init__(self, size: int) -> None:
            self.size = size
            self.base = ARENA_BASE
            self._mappings: list[tuple[int, int, Any]] = []

        def map(self, offset: int, size: int, *, retain_handle: bool) -> None:
            # SGLang creating and mapping memory of its own.
            fake.events.append(f"sglang_created_memory:{offset}:{size}")
            self._mappings.append((self.base + offset, size, object()))

        def map_existing(self, offset: int, size: int, handle: Any) -> None:
            fake.cuda.map_calls += 1
            if fake.cuda.fail_map_at == fake.cuda.map_calls:
                raise RuntimeError("cuMemMap(existing): CUDA_ERROR_INVALID_VALUE")
            assert handle[0] == "imported"
            fake.events.append(f"cuda:map:{offset}:{size}")
            self._mappings.append((self.base + offset, size, None))

    class KvVmmArena:
        def __init__(self, device_id: int, reserve_bytes: int) -> None:
            self.device_id = device_id
            self.granularity = fake.shape.granularity
            self.reserved = reserve_bytes
            self._allocation = VmmReservation(reserve_bytes)
            self.base = self._allocation.base
            self._committed_by_offset: dict[int, int] = {}
            self._range_backed = 0

        def commit_range(self, offset: int, want_bytes: int) -> None:
            previous = self._committed_by_offset.get(offset, 0)
            if want_bytes <= previous:
                return
            self._allocation.map(
                offset + previous, want_bytes - previous, retain_handle=True
            )
            self._committed_by_offset[offset] = want_bytes
            self._range_backed += want_bytes - previous

        @property
        def backed_bytes(self) -> int:
            return self._range_backed

    kv_vmm_backing = fake.modules["sglang.srt.mem_cache.kv_vmm_backing"]

    class KvVmmBufferOwner:
        def __init__(
            self,
            *,
            device: str,
            device_id: int,
            store_dtype: FakeDtype,
            page_size: int,
            reserved_num_tokens: int,
            buffer_descs: list[KvBufferDesc],
        ) -> None:
            self.device = device
            self.device_id = device_id
            self.store_dtype = store_dtype
            self.page_size = page_size
            self._reserved_num_tokens = reserved_num_tokens
            self._final_num_tokens: int | None = None
            granularity = fake.shape.granularity
            # Each buffer gets a granule-aligned range with slack after it.
            stride = [
                _align_up(desc.reserved_span_bytes(), granularity) + 3 * granularity
                for desc in buffer_descs
            ]
            self._arena = kv_vmm_backing.KvVmmArena(
                device_id, reserve_bytes=sum(stride) + granularity
            )
            self._specs = []
            cursor = 0
            for desc, span in zip(buffer_descs, stride, strict=True):
                self._specs.append(
                    _BufferSpec(
                        desc, cursor + fake.misplace, desc.reserved_span_bytes()
                    )
                )
                cursor += span
            self.tensors = [object() for _ in buffer_descs]
            # Slot 0 must be resident before graph capture.
            self.ensure_prefix(self.page_size)

        def _back_spans(self, span_bytes: list[int]) -> None:
            granularity = self._arena.granularity
            for spec, span in zip(self._specs, span_bytes, strict=True):
                want = _align_up(span, granularity)
                if want > spec.backed_to:
                    self._arena.commit_range(spec.offset, want)
                    spec.backed_to = want

        def ensure_prefix(self, num_tokens: int) -> None:
            self._back_spans(
                [
                    spec.desc.prefix_span_bytes(num_tokens, self.page_size)
                    for spec in self._specs
                ]
            )

        def finalize(self, final_num_tokens: int) -> None:
            assert self.page_size <= final_num_tokens <= self._reserved_num_tokens
            self._back_spans(
                [
                    spec.desc.final_span_bytes(final_num_tokens, self.page_size)
                    for spec in self._specs
                ]
            )
            self._final_num_tokens = int(final_num_tokens)

        @property
        def backed_bytes(self) -> int:
            return self._arena.backed_bytes

    class KVCacheConfigurator:
        def __init__(self, cell_bytes: int, user_limit: int | None) -> None:
            self._cell_bytes = cell_bytes
            self._user_limit = user_limit

        def config_from_budget(
            self, budget_bytes: int, *, cap_tokens: int | None = None
        ) -> MemoryPoolConfig:
            fake.budget_calls.append((budget_bytes, cap_tokens))
            tokens = budget_bytes // self._cell_bytes
            if self._user_limit is not None:
                tokens = min(tokens, self._user_limit)
            if cap_tokens is not None:
                tokens = min(tokens, cap_tokens)
            return MemoryPoolConfig(tokens)

    kv_vmm_backing.KvVmmArena = KvVmmArena
    kv_vmm_backing.KvVmmBufferOwner = KvVmmBufferOwner
    fake.modules[
        "sglang.srt.mem_cache.kv_cache_configurator"
    ].KVCacheConfigurator = KVCacheConfigurator
    fake.modules["sglang.srt.mem_cache.base_prefix_cache"].EvictParams = EvictParams


def build_vmm_scheduler(fake: Any, scheduler: Any, server_args: Any) -> None:
    """Run the part of ``Scheduler.__init__`` that builds a VMM-backed KV pool.

    The order is SGLang 0.5.23's: size the reservation, create the pool (which
    backs one page), capture graphs, size and back the final pool, then copy
    the capacity into the scheduler and its observers.
    """

    from fake_sglang import MHATokenToKVPool

    shape: PoolShape = fake.shape
    backing = fake.modules["sglang.srt.mem_cache.kv_vmm_backing"]
    configurators = fake.modules["sglang.srt.mem_cache.kv_cache_configurator"]
    configurator = configurators.KVCacheConfigurator(
        shape.cell_bytes, server_args.max_total_tokens
    )
    reserved = configurator.config_from_budget(shape.free_bytes).max_total_num_tokens

    rows = (reserved + shape.page_size) // shape.tokens_per_row
    v_head_dim = shape.head_dim if shape.v_head_dim is None else shape.v_head_dim
    descs = [
        KvBufferDesc(
            f"{prefix}{layer}",
            (rows, shape.kv_heads, dim),
            row_bytes=NumpyInt(shape.kv_heads * dim * shape.dtype.itemsize),
            tokens_per_row=shape.tokens_per_row,
        )
        for prefix, dim in (("k", shape.head_dim), ("v", v_head_dim))
        for layer in range(shape.layers)
    ]
    owners = [
        backing.KvVmmBufferOwner(
            device="cuda",
            device_id=0,
            store_dtype=shape.dtype,
            page_size=shape.page_size,
            reserved_num_tokens=reserved,
            buffer_descs=descs,
        )
        for _ in range(fake.vmm_pools)
    ]
    owner = owners[0]
    pool = MHATokenToKVPool((rows * shape.row_bytes, rows * shape.row_bytes))
    pool.__class__ = fake.pool_class
    pool.post_capture_active = True
    pool._post_capture_owner = owner
    pool.is_quantized_kv_cache = shape.quantized
    pool.size = reserved
    pool.page_size = shape.page_size
    allocator = fake.allocator_class(reserved, pool)
    allocator.page_size = shape.page_size
    runner = types.SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_kv_pool_allocator=allocator,
        kv_cache_configurator=configurator,
        max_total_num_tokens=reserved,
    )
    fake.events.append("graphs_captured")

    # post_capture_resize_kv_pool
    reserved_cap = runner.max_total_num_tokens
    config = configurator.config_from_budget(shape.free_bytes, cap_tokens=reserved_cap)
    owner.finalize(config.max_total_num_tokens)
    pool.size = config.max_total_num_tokens
    allocator.resize(config)
    runner.max_total_num_tokens = NumpyInt(config.max_total_num_tokens)

    final = config.max_total_num_tokens
    scheduler.tp_worker = types.SimpleNamespace(model_runner=runner)
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.tree_cache = RadixCache(allocator)
    scheduler.max_total_num_tokens = final
    scheduler.page_size = shape.page_size
    scheduler.pool_stats_observer = SchedulerPoolStatsObserver(
        max_total_num_tokens=final
    )
    scheduler.invariant_checker = SchedulerInvariantChecker(max_total_num_tokens=final)
    scheduler.model_config = types.SimpleNamespace(context_len=shape.context_len)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class FakePoolRuntime(FakeClient):
    """Answers control operations like the runtime's shared-pool coordinator."""

    def __init__(
        self,
        events: list[str] | None = None,
        *,
        granularity: int = 1024,
        initial_blocks: int = 64,
        minimum_blocks: int = 32,
    ) -> None:
        super().__init__()
        self.events = [] if events is None else events
        self.granularity = granularity
        self.initial_blocks = initial_blocks
        self.minimum_blocks = minimum_blocks
        self.epoch = 7
        self.binding_id = "cuda-vmm:sglang:scheduler:7:sglang.pool.0:0"
        self.registration: dict[str, Any] | None = None
        self.mapped_blocks = 0
        self.generation = 0
        self.resize: dict[str, Any] | None = None
        # Segments the runtime believes are mapped, one list per resize step.
        self.mapped_segments: list[list[dict]] = []
        # (read end, write end) of one pipe per descriptor handed out.
        self._pipes: list[tuple[int, int]] = []
        self.completed: list[tuple[int, int]] = []
        # Test hooks: change the receipt or an operation before it is returned,
        # and observe the server at the moment a stage is acknowledged.
        self.mutate_receipt: Callable[[dict[str, Any]], None] | None = None
        self.mutate_operation: Callable[[dict[str, Any]], None] | None = None
        self.on_ack: Callable[[dict[str, Any]], None] | None = None
        self.fail_polls = 0
        self.lose_ack_replies = 0

    # Geometry of the registered pool ------------------------------------------

    @property
    def stripes(self) -> int:
        group = self.registration["topology"]["cache_groups"][0]
        return 2 * len(group["layers"])

    @property
    def bytes_per_block(self) -> int:
        return self.registration["capacity_model"]["groups"][0]["bytes_per_allocation"]

    def _stripe_bytes(self, blocks: int) -> int:
        return blocks * self.bytes_per_block // self.stripes

    def _segments(
        self, label: str, start_blocks: int, end_blocks: int, first_handle: int = 0
    ) -> list[dict]:
        """One segment per stripe covering the same block range."""

        start, end = self._stripe_bytes(start_blocks), self._stripe_bytes(end_blocks)
        segments = []
        for stripe in range(self.stripes):
            segment = {
                "segment_id": f"{self.binding_id}:{label}:{stripe}",
                "offset_bytes": start,
                "length_bytes": end - start,
                "handle_index": first_handle + stripe,
            }
            if stripe:
                segment["stripe"] = stripe
            segments.append(segment)
        return segments

    def _open_descriptors(self, count: int) -> list[int]:
        """Hand out ``count`` descriptors, each the read end of its own pipe.

        Descriptor numbers are reused by the OS, so a number says nothing
        about whether the adapter closed what it was given. A pipe does: its
        write end reports a broken pipe once every read end is closed.
        """

        pipes = [os.pipe() for _ in range(count)]
        self._pipes.extend(pipes)
        return [read_end for read_end, _ in pipes]

    def open_descriptors(self) -> list[int]:
        """Descriptors handed to the adapter that it has not closed."""

        still_open = []
        for read_end, write_end in list(self._pipes):
            try:
                os.write(write_end, b"x")
            except BrokenPipeError:
                os.close(write_end)
                self._pipes.remove((read_end, write_end))
                continue
            still_open.append(read_end)
        return still_open

    def close(self) -> None:
        """Release every descriptor still open. For test cleanup."""

        for read_end in self.open_descriptors():
            os.close(read_end)
        for _, write_end in self._pipes:
            os.close(write_end)
        self._pipes.clear()

    # Registration -------------------------------------------------------------

    def register_shared_pool(
        self, registration: dict[str, Any]
    ) -> tuple[dict[str, Any], list[int]]:
        self._record("register", registration)
        validate_striped_pool_registration(registration)
        self.registration = registration
        group = registration["capacity_model"]["groups"][0]
        self.mapped_blocks = self.initial_blocks
        # As the runtime provisions it: the minimum, then the headroom above it,
        # so the pool can later shrink to its minimum on a mapping boundary.
        self.mapped_segments = [self._segments("minimum", 0, self.minimum_blocks)]
        if self.initial_blocks > self.minimum_blocks:
            self.mapped_segments.append(
                self._segments(
                    "headroom",
                    self.minimum_blocks,
                    self.initial_blocks,
                    first_handle=self.stripes,
                )
            )
        receipt = {
            "participant_id": registration["participant_id"],
            "participant_epoch": self.epoch,
            "shared_pools": [
                {
                    "binding_id": self.binding_id,
                    "capacity_pool_id": group["pool_id"],
                    "generation": 1,
                    "group_ids": [group["group_id"]],
                    "memory_domain": dict(group["memory_domains"][0]),
                    "block_count": group["max_allocations"],
                    "bytes_per_block": group["bytes_per_allocation"],
                    "allocation_mode": "participant_managed",
                    "transport": {"kind": "cuda_vmm"},
                    "descriptor": "scm_rights:cuda-vmm-v1",
                    "elastic": {
                        "minimum_block_count": self.minimum_blocks,
                        "mapped_block_count": self.initial_blocks,
                        "maximum_block_count": group["max_allocations"],
                        "allocation_granularity_bytes": self.granularity,
                        "resize_alignment_blocks": self.minimum_blocks,
                        "stripe_count": self.stripes,
                        "segments": [
                            dict(segment)
                            for step in self.mapped_segments
                            for segment in step
                        ],
                    },
                }
            ],
        }
        if self.mutate_receipt is not None:
            self.mutate_receipt(receipt)
        handles = self._open_descriptors(self.stripes * len(self.mapped_segments))
        return validate_striped_pool_receipt(receipt, registration), handles

    def attach(self, attachment: dict[str, Any]) -> None:
        self._record("attach", attachment)
        validate_attachment(attachment)

    def activate(self, participant_epoch: int) -> None:
        self._record("activate", participant_epoch)

    # Resize -------------------------------------------------------------------

    def request_resize(self, target_blocks: int) -> int:
        """Start a resize, as the runtime's supervisor does."""

        assert self.resize is None, "a resize is already in progress"
        self.generation += 1
        growing = target_blocks > self.mapped_blocks
        if growing:
            segments = self._segments(
                f"resize:{self.generation}", self.mapped_blocks, target_blocks
            )
        else:
            # Whole segments above the boundary, as the runtime recorded them.
            boundary = self._stripe_bytes(target_blocks)
            segments = [
                dict(segment)
                for step in self.mapped_segments
                for segment in step
                if segment["offset_bytes"] >= boundary
            ]
            for index, segment in enumerate(segments):
                segment["handle_index"] = index
        self.resize = {
            "generation": self.generation,
            "from": self.mapped_blocks,
            "target": target_blocks,
            "stage": "map_workers" if growing else "retire_scheduler",
            "segments": segments,
        }
        return self.generation

    def poll_resize(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[int], bool]:
        self._record("resize_poll", request)
        validate_resize_poll_request(request)
        if self.fail_polls:
            self.fail_polls -= 1
            raise KapslKvControlError("coordinator busy", kind="transport")
        resize = self.resize
        if resize is None:
            return [], [], False
        if request["applied_generation"] >= resize["generation"]:
            return [], [], True
        role = request["actor"]["role"]
        stage = resize["stage"]
        physical = stage in ("map_workers", "unmap_workers")
        if (role == "worker") != physical:
            return [], [], True
        operation = {
            "participant_epoch": self.epoch,
            "resize_generation": resize["generation"],
            "binding_id": self.binding_id,
            "stage": stage,
            "from_block_count": resize["from"],
            "target_block_count": resize["target"],
            "bytes_per_block": self.bytes_per_block,
            "allocation_granularity_bytes": self.granularity,
            "stripe_count": self.stripes,
            "segments": [dict(s) for s in resize["segments"]] if physical else [],
        }
        if self.mutate_operation is not None:
            self.mutate_operation(operation)
        try:
            # The control client validates what it receives.
            validated = validate_resize_operation(operation)
        except ValueError as error:
            raise KapslKvControlError(
                f"invalid resize operation: {error}", kind="protocol"
            ) from error
        handles = self._open_descriptors(self.stripes) if stage == "map_workers" else []
        return [validated], handles, True

    def ack_resize(self, request: dict[str, Any]) -> None:
        self._record("resize_ack", request)
        validate_resize_ack_request(request)
        resize = self.resize
        role = request["actor"]["role"]
        if (
            resize is None
            or request["resize_generation"] != resize["generation"]
            or request["stage"] != resize["stage"]
            or request["applied_block_count"] != resize["target"]
            or request["binding_id"] != self.binding_id
            or (role == "worker")
            != (resize["stage"] in ("map_workers", "unmap_workers"))
        ):
            raise KapslKvControlError(
                "resize acknowledgement does not exactly match the pending operation",
                kind="invalid_request",
            )
        self.events.append(f"control:ack:{resize['stage']}")
        if self.on_ack is not None:
            self.on_ack(request)
        if resize["stage"] == "map_workers":
            resize["stage"] = "activate_scheduler"
        elif resize["stage"] == "retire_scheduler":
            resize["stage"] = "unmap_workers"
        else:
            if resize["target"] > resize["from"]:
                self.mapped_segments.append(resize["segments"])
            else:
                boundary = self._stripe_bytes(resize["target"])
                self.mapped_segments = [
                    step
                    for step in self.mapped_segments
                    if step[0]["offset_bytes"] < boundary
                ]
            self.mapped_blocks = resize["target"]
            self.completed.append((resize["from"], resize["target"]))
            self.resize = None
        if self.lose_ack_replies:
            self.lose_ack_replies -= 1
            raise KapslKvControlError("reply lost", kind="transport")
