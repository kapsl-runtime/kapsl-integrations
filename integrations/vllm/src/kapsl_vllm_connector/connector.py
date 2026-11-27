"""Thin compatibility layer for vLLM's experimental V1 connector API."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable

from .client import KapslKvControlClient, KapslKvControlError
from .contract import (
    ABI_VERSION,
    make_resize_ack_request,
    make_resize_poll_request,
    make_shared_pool_attachment,
    make_reserve_request,
    opaque_registration,
    shared_pool_registration,
)
from .planning import (
    UINT64_MAX,
    extract_cache_group_topologies,
    shared_pool_block_stride,
    validate_certified_shared_pool_execution,
)
from .shared_pool import (
    SharedPoolImportError,
    SharedPoolTailBusyError,
    VllmElasticBlockPool,
    VllmSharedPoolHook,
    select_cuda_binding,
    vllm_backing_geometry,
    vllm_distributed_rank,
)

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        SupportsHMA,
    )
except ImportError as import_error:  # Keep protocol/client usable without vLLM.
    _VLLM_IMPORT_ERROR: ImportError | None = import_error

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        pass

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorRole:  # type: ignore[no-redef]
        pass

    class SupportsHMA:  # type: ignore[no-redef]
        pass
else:
    _VLLM_IMPORT_ERROR = None

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
except ImportError:  # No vLLM, or a build without worker-to-scheduler notes.
    _WORKER_NOTES_IMPORTED = False

    class KVConnectorWorkerMetadata:  # type: ignore[no-redef]
        pass
else:
    _WORKER_NOTES_IMPORTED = True


# vLLM installs its output handler on the ``vllm`` logger rather than the
# process root logger. Keep connector records below that namespace so worker
# evidence and lifecycle diagnostics reach the managed child log.
logger = logging.getLogger(f"vllm.{__name__}")
ADAPTER_VERSION = "0.7.2"
ADAPTER_PROFILE_ID = "vllm-v1-packed-cuda-ipc/flash-attn"
ELASTIC_ADAPTER_PROFILE_ID = "vllm-v1-packed-cuda-vmm/flash-attn-blnhc"
# Kapsl reads this before it launches vLLM. True means live resize stays safe
# while vLLM overlaps two model batches (async scheduling), so Kapsl may leave
# async scheduling on. A connector without the name gets --no-async-scheduling.
LIVE_RESIZE_ASYNC_SCHEDULING = True

# vLLM steps an idle engine about once per millisecond while a connector
# reports pending push work. An elastic scheduler reports it for as long as
# Kapsl may still shrink the pool, so pace that loop: one zero-token step per
# interval keeps a shrink deliverable without spending a core on it. A request
# that arrives during the pause waits at most this long.
_IDLE_KEEPALIVE_INTERVAL_SECONDS = 0.01
# Kapsl decides to grow from a utilization sample it takes every two seconds,
# and the scheduler learns of it from a poll every quarter second. A grow
# requested as the last request finishes would otherwise reach a scheduler
# that has already gone to sleep, so stay awake this long after going idle.
_POST_LOAD_KEEPALIVE_SECONDS = 5.0
# Pauses between attempts to settle a resize acknowledgement whose reply was
# lost, and the least time between repeated warnings about one stalled poll.
_RESIZE_ACK_RETRY_DELAYS_SECONDS = (0.05, 0.25, 1.0)
_STALL_WARNING_INTERVAL_SECONDS = 10.0
# Pause after a reservation the pool could not fit. One failed attempt per
# pause, however many requests are waiting: each attempt is also what tells
# Kapsl that demand exceeds the mapped pool, so it must keep arriving, but a
# scheduler that steps every millisecond must not turn it into a flood.
_CAPACITY_RETRY_INTERVAL_SECONDS = 0.02
# With async scheduling at most two steps await their worker note. Far more
# means the notes are not coming back, and no tail retirement can be fenced.
_MAX_OUTSTANDING_STEPS = 64
# Longest time new requests wait for a tail retirement that only finished
# requests still block. Those blocks come back within the two steps vLLM keeps
# in flight; past this limit something else holds them, and the requests run.
_RETIRE_HOLD_LIMIT_SECONDS = 2.0


@dataclass
class KapslConnectorMetadata(KVConnectorMetadata):
    """Per-step note from the scheduler to its workers.

    No KV bytes travel with a step. ``resize_pending`` tells an elastic worker
    whether Kapsl has resize work for it. The default is the safe answer: a
    worker that polls without need spends a round trip, one that never polls
    lets a resize run into its acknowledgement deadline.

    ``step`` numbers the step when vLLM overlaps model batches (async
    scheduling). Each worker returns the number after it has run the step, and
    the scheduler acknowledges a tail retirement only once the numbers show
    that no batch scheduled before the retirement is still outstanding.
    ``None`` means batches do not overlap, and nothing is returned.
    """

    resize_pending: bool = True
    step: int | None = None


@dataclass
class KapslWorkerStepMetadata(KVConnectorWorkerMetadata):
    """A worker's note that it has run one numbered scheduler step.

    ``workers`` counts the workers a merged note speaks for. vLLM merges the
    notes of all tensor-parallel workers into one and leaves out a worker that
    returned none, so the scheduler counts them instead of trusting the merge.
    """

    step: int
    workers: int = 1

    def aggregate(self, other: Any) -> "KapslWorkerStepMetadata":
        other_step = getattr(other, "step", None)
        other_workers = getattr(other, "workers", None)
        if not _is_step_number(other_step) or not _is_step_number(other_workers):
            # A note without a number confirms nothing: steps start at 1.
            return KapslWorkerStepMetadata(step=0, workers=self.workers)
        return KapslWorkerStepMetadata(
            step=min(self.step, other_step), workers=self.workers + other_workers
        )


class KapslConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """KV-connected vLLM participant with optional Kapsl-owned backing.

    This connector lets Kapsl admit and lease logical KV capacity before vLLM
    allocates request blocks. It intentionally reports zero external prefix
    hits and performs no worker-side copies, so vLLM never consumes KV bytes
    that Kapsl cannot actually deliver.
    """

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        if transfer is None:
            return None
        raw = _extra(transfer, "kapsl_live_resize", False)
        if not isinstance(raw, bool):
            raise ValueError("kapsl_live_resize must be a boolean")
        return "BLNHC" if raw else None

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "KapslConnectorV1 must be installed in an environment with vLLM V1"
            ) from _VLLM_IMPORT_ERROR
        if not hasattr(KVConnectorBase_V1, "on_new_request"):
            raise RuntimeError(
                "this vLLM build lacks the on_new_request KV hook required for "
                "Kapsl pre-allocation admission"
            )

        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        endpoint = _required_extra(self._kv_transfer_config, "kapsl_control_endpoint")
        model_fingerprint = _required_extra(
            self._kv_transfer_config, "kapsl_model_fingerprint"
        )
        participant_base = str(
            _extra(self._kv_transfer_config, "kapsl_participant_id", "vllm")
        ).strip()
        if not participant_base:
            raise ValueError("kapsl_participant_id must not be empty")
        role_name = getattr(role, "name", str(role)).lower()
        self._is_scheduler = role_name == "scheduler"
        engine_id = str(getattr(self._kv_transfer_config, "engine_id", "engine"))
        self._mode = str(
            _extra(self._kv_transfer_config, "kapsl_kv_mode", "opaque")
        ).strip().lower()
        if self._mode not in {"opaque", "shared_pool"}:
            raise ValueError("kapsl_kv_mode must be 'opaque' or 'shared_pool'")
        raw_live_resize = _extra(
            self._kv_transfer_config, "kapsl_live_resize", False
        )
        if not isinstance(raw_live_resize, bool):
            raise ValueError("kapsl_live_resize must be a boolean")
        self._live_resize = raw_live_resize
        if self._live_resize and self._mode != "shared_pool":
            raise ValueError("kapsl_live_resize requires shared_pool mode")
        self._vmm_conformance = _validated_vmm_conformance(
            self._kv_transfer_config, self._live_resize
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tensor_parallel_size = int(
            getattr(parallel_config, "tensor_parallel_size", 1) or 1
        )
        if self._mode == "shared_pool":
            if self._live_resize:
                _validate_elastic_shared_pool_execution(vllm_config, kv_cache_config)
            else:
                _validate_shared_pool_execution(vllm_config)
        self._participant_id = (
            f"{participant_base}:{engine_id}"
            if self._mode == "shared_pool"
            else f"{participant_base}:{engine_id}:{role_name}"
        )
        memory_domains = _required_memory_domains(self._kv_transfer_config)
        if self._mode == "shared_pool" and any(
            domain.get("kind") != "cuda" for domain in memory_domains
        ):
            raise ValueError("vLLM shared_pool currently supports CUDA domains only")
        rank_device_map = (
            _validated_shared_rank_device_map(
                self._kv_transfer_config,
                memory_domains,
                tensor_parallel_size,
            )
            if self._mode == "shared_pool"
            else None
        )
        self._capacity_groups = _vllm_capacity_groups(
            kv_cache_config,
            memory_domains,
            shared_pool=self._mode == "shared_pool",
        )
        self._group_ids = [group["group_id"] for group in self._capacity_groups]
        shared_topology = (
            _vllm_topology(
                kv_cache_config,
                model_fingerprint,
                tensor_parallel_world_size=tensor_parallel_size,
            )
            if self._mode == "shared_pool"
            else None
        )
        shared_profile = (
            _vllm_adapter_profile(vllm_config, live_resize=self._live_resize)
            if self._mode == "shared_pool"
            else None
        )
        provisioning_grant = (
            _provisioning_grant(self._kv_transfer_config)
            if self._mode == "shared_pool"
            else None
        )
        registration = (
            shared_pool_registration(
                self._participant_id,
                model_fingerprint,
                self._capacity_groups,
                shared_topology,
                shared_profile,
                backend="vllm",
                provisioning_grant=provisioning_grant,
                live_resize=self._live_resize,
            )
            if self._mode == "shared_pool"
            else opaque_registration(
                self._participant_id,
                model_fingerprint,
                self._capacity_groups,
                backend="vllm",
            )
        )
        timeout_ms = int(_extra(self._kv_transfer_config, "kapsl_timeout_ms", 2000))
        self._lease_ttl_ms = int(
            _extra(self._kv_transfer_config, "kapsl_lease_ttl_ms", 30_000)
        )
        if self._lease_ttl_ms < 1000:
            raise ValueError("kapsl_lease_ttl_ms must be at least 1000")
        self._client: KapslKvControlClient | None = None
        self._leases: dict[str, dict[str, Any]] = {}
        self._lease_lock = threading.RLock()
        self._control_failure: KapslKvControlError | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._shared_pool_hook: VllmSharedPoolHook | None = None
        self._participant_epoch: int | None = None
        self._shared_shard: dict[str, int] | None = None
        self._shared_profile: dict[str, str] | None = shared_profile
        self._shared_attached = False
        self._shared_active = False
        self._activation_lock = threading.Lock()
        self._resize_lock = threading.RLock()
        self._resize_pending = False
        self._resize_applied_generation = 0
        self._pending_scheduler_operations: list[dict[str, Any]] = []
        self._deferred_retire_generation = 0
        # Overlapping batches (async scheduling): steps are numbered, workers
        # return the numbers, and a tail retirement is acknowledged only after
        # its fence step. See _apply_scheduler_resizes.
        self._overlapping_batches = self._live_resize and _overlapping_model_batches(
            vllm_config
        )
        self._steps_issued = 0
        self._steps_outstanding: set[int] = set()
        self._step_notes_failed = False
        # Every tensor-parallel rank maps the pool and must return each step.
        self._worker_count = tensor_parallel_size
        self._retire_fences: dict[int, int] = {}
        self._retire_hold: tuple[int, float, int] | None = None
        self._retire_hold_expired_generation = 0
        # Requests that hold a lease and own blocks.
        self._block_owners: set[str] = set()
        self._worker_step: int | None = None
        self._idle_step_completed = False
        self._last_lease_released_at = float("-inf")
        self._stall_warned_at = float("-inf")
        # Reservations the coordinator could not fit yet, by request. vLLM
        # keeps these requests in its waiting queue until one succeeds.
        self._waiting_reservations: dict[str, dict[str, Any]] = {}
        self._capacity_retry_at = float("-inf")
        self._capacity_warned_at = float("-inf")
        self._elastic_block_pool: VllmElasticBlockPool | None = None
        self._worker_forward_lock = threading.Lock()
        self._worker_forward_active = False
        if self._is_scheduler or self._mode == "shared_pool":
            self._client = KapslKvControlClient(
                endpoint,
                self._participant_id,
                timeout_seconds=timeout_ms / 1000.0,
            )
            received_handles: list[int] = []
            if self._live_resize and not self._is_scheduler:
                receipt, received_handles = self._client.register_with_handles(
                    registration
                )
            else:
                receipt = self._client.register(registration)
            try:
                self._participant_epoch = int(receipt["participant_epoch"])
                if self._live_resize:
                    (
                        minimum_blocks,
                        initial_blocks,
                        maximum_blocks,
                    ) = _elastic_receipt_block_counts(receipt)
                    if self._is_scheduler:
                        self._elastic_block_pool = VllmElasticBlockPool(
                            initial_blocks,
                            maximum_blocks,
                            minimum_blocks=minimum_blocks,
                        )
                if self._mode == "shared_pool" and not self._is_scheduler:
                    global_rank = (
                        0 if tensor_parallel_size == 1 else vllm_distributed_rank()
                    )
                    binding = select_cuda_binding(
                        receipt,
                        kv_cache_config,
                        rank_device_map,
                        global_rank=global_rank,
                        live_resize=self._live_resize,
                    )
                    self._shared_pool_hook = VllmSharedPoolHook(
                        binding,
                        kv_cache_config,
                        handles=received_handles if self._live_resize else None,
                        conformance=self._vmm_conformance,
                    )
                    if shared_topology is None:
                        raise AssertionError("shared topology was not constructed")
                    self._shared_shard = dict(shared_topology["shard"])
                    self._shared_shard["tensor_parallel_rank"] = global_rank
            finally:
                _close_os_handles(received_handles)
            logger.info(
                "registered Kapsl KV participant %s in %s mode (%s role)",
                self._participant_id,
                self._mode,
                role_name,
            )
        self._start_scheduler_control(engine_id)

    # Scheduler-side lifecycle -------------------------------------------------

    def _start_scheduler_control(self, engine_id: str) -> None:
        if not self._is_scheduler:
            return
        # The pinned vLLM build constructs its scheduler connector only after
        # every worker has initialized and registered its KV tensors. Activate
        # here so Kapsl can publish the fully attached participant as Routable;
        # deferring activation until on_new_request creates a readiness cycle
        # because Kapsl deliberately will not route that first request yet.
        # Coordinator activation still verifies the complete binding set and
        # therefore fails startup closed if vLLM's construction order drifts.
        if self._mode == "shared_pool":
            self._ensure_shared_active()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"kapsl-kv-heartbeat-{engine_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def on_new_request(self, request: "Request") -> None:
        if not self._is_scheduler:
            return
        self._raise_if_control_failed()
        self._ensure_shared_active()
        request_id = _request_id(request)
        with self._lease_lock:
            if request_id in self._leases or request_id in self._waiting_reservations:
                return
        reserve_request = make_reserve_request(
            request_id=request_id,
            sequence_id=request_id,
            token_capacity=_request_token_capacity(request),
            group_ids=self._group_ids,
            priority=_request_priority(request),
            ttl_ms=self._lease_ttl_ms,
        )
        if not self._reserve_or_wait(request_id, reserve_request):
            with self._lease_lock:
                self._waiting_reservations.setdefault(request_id, reserve_request)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        del num_computed_tokens
        self._raise_if_control_failed()
        if self._is_scheduler and not self._reservation_ready(_request_id(request)):
            # No answer yet: vLLM leaves the request in its waiting queue and
            # asks again on its next step, so the request owns no block until
            # Kapsl has leased it the capacity.
            return None, False
        if self._is_scheduler and self._tail_retirement_holds_admission():
            # Same answer for another reason: a shrink admitted while this
            # replica was idle must take its tail out of the free queue before
            # a new request can be given blocks there.
            return None, False
        # Opaque phase 1 does not have a worker data plane, so claiming a cache
        # hit here would make vLLM skip computation without restoring KV bytes.
        return 0, False

    def _reservation_ready(self, request_id: str) -> bool:
        """Whether the request holds its lease, repeating a waiting reservation."""

        with self._lease_lock:
            reserve_request = self._waiting_reservations.get(request_id)
            if reserve_request is None:
                return True
            if time.monotonic() < self._capacity_retry_at:
                return False
        if not self._reserve_or_wait(request_id, reserve_request):
            return False
        with self._lease_lock:
            self._waiting_reservations.pop(request_id, None)
        return True

    def _reserve_or_wait(self, request_id: str, reserve_request: dict[str, Any]) -> bool:
        """Reserve the request's capacity. False means the pool cannot fit it yet.

        A full pool is not a failure. A live-resize pool that sat idle has
        been shrunk to one sequence and grows again on demand, and a fixed
        pool frees capacity as running requests finish. Raising here would
        end vLLM's engine core and every request on it; waiting costs this
        request some queueing time.
        """

        client = self._control_client()
        try:
            lease = client.reserve(reserve_request)
        except KapslKvControlError as error:
            if error.kind != "capacity_exhausted":
                raise
            now = time.monotonic()
            self._capacity_retry_at = now + _CAPACITY_RETRY_INTERVAL_SECONDS
            if now - self._capacity_warned_at >= _STALL_WARNING_INTERVAL_SECONDS:
                self._capacity_warned_at = now
                logger.warning(
                    "Kapsl KV pool cannot fit request %s yet; it waits for capacity: %s",
                    request_id,
                    error,
                )
            if self._live_resize:
                # Growth arrives as a resize operation. Fetch it now instead of
                # at the next background poll so the wait is one or two steps.
                try:
                    self._poll_scheduler_resizes()
                except KapslKvControlError as poll_error:
                    if poll_error.kind != "transport":
                        raise
                    self._warn_control_stall("resize poll", poll_error)
            return False
        with self._lease_lock:
            previous = self._leases.setdefault(request_id, lease)
            only_lease = previous is lease and len(self._leases) == 1
        if previous is not lease:
            # A duplicate callback raced with this one. Return the redundant
            # lease immediately instead of double-accounting the request.
            client.release(str(lease["lease_id"]))
        elif self._live_resize and only_lease:
            self._settle_resizes_before_allocation()
        return True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del blocks, num_external_tokens
        self._raise_if_control_failed()
        if self._is_scheduler and self._overlapping_batches:
            with self._lease_lock:
                self._block_owners.add(_request_id(request))

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KapslConnectorMetadata:
        del scheduler_output
        self._raise_if_control_failed()
        # Apply first, number afterwards: a tail retired here is fenced on the
        # step that is being built.
        self._apply_scheduler_resizes()
        self._idle_step_completed = True
        return KapslConnectorMetadata(
            resize_pending=self._workers_have_resize_work(),
            step=self._issue_step(),
        )

    def _issue_step(self) -> int | None:
        """Number the step being built, or None if batches never overlap."""

        if not self._overlapping_batches:
            return None
        with self._resize_lock:
            self._steps_issued += 1
            step = self._steps_issued
            if self._step_notes_failed:
                return step
            self._steps_outstanding.add(step)
            if len(self._steps_outstanding) <= _MAX_OUTSTANDING_STEPS:
                return step
            self._step_notes_failed = True
            self._steps_outstanding.clear()
        logger.error(
            "vLLM returned no worker note for %d steps in a row. A KV tail "
            "retirement can no longer be fenced against batches in flight, so "
            "none will be acknowledged and Kapsl will not shrink this pool.",
            _MAX_OUTSTANDING_STEPS,
        )
        return step

    def update_connector_output(self, connector_output: Any) -> None:
        """Record that every worker has run one numbered step."""

        if not self._is_scheduler or not self._overlapping_batches:
            return
        note = getattr(connector_output, "kv_connector_worker_meta", None)
        step = getattr(note, "step", None)
        if (
            not _is_step_number(step)
            or getattr(note, "workers", None) != self._worker_count
        ):
            # No number, or not from every worker: the step stays outstanding.
            return
        with self._resize_lock:
            self._steps_outstanding.discard(step)
            waiting = bool(self._retire_fences)
        if waiting:
            # This may have been the step a retired tail was waiting for.
            self._apply_scheduler_resizes()

    def _steps_settled_through(self, fence: int) -> bool:
        """Whether every worker has run every step numbered up to ``fence``."""

        with self._resize_lock:
            return (
                not self._step_notes_failed
                and fence <= self._steps_issued
                and all(step > fence for step in self._steps_outstanding)
            )

    def _unapplied_retirement(self) -> int | None:
        """Generation of a queued shrink whose tail is still in the free queue."""

        with self._resize_lock:
            for operation in self._pending_scheduler_operations:
                generation = int(operation["resize_generation"])
                if (
                    str(operation["stage"]) == "retire_scheduler"
                    and generation not in self._retire_fences
                ):
                    return generation
        return None

    def _tail_retirement_holds_admission(self) -> bool:
        """Whether waiting requests must stay unallocated for a queued shrink.

        Kapsl admits a shrink while this replica holds no lease, and expects
        the tail to be free then. With overlapping batches it may not be yet:
        vLLM keeps the blocks of a request that just finished until the last
        step that could write them has returned. If the next request were
        allocated now it could be given tail blocks, and the shrink would wait
        for that request to end, possibly past Kapsl's acknowledgement
        deadline. So while nothing that owns blocks is running, new requests
        wait the step or two until the tail is out of the free queue.
        """

        if not self._overlapping_batches:
            return False
        generation = self._unapplied_retirement()
        if generation is None:
            return False
        with self._lease_lock:
            if self._block_owners:
                # A running request may own tail blocks for as long as it
                # runs. Holding everyone else that long would starve them.
                return False
        now = time.monotonic()
        hold = self._retire_hold
        if hold is None or hold[0] != generation:
            hold = (generation, now, -1)
        if hold[2] != self._steps_issued:
            # Once per scheduler step: the request asking owns no block yet,
            # so this is as good a moment as the first lease after idle.
            self._retire_hold = (generation, hold[1], self._steps_issued)
            self._apply_scheduler_resizes()
            if self._unapplied_retirement() != generation:
                return False
        if now - hold[1] < _RETIRE_HOLD_LIMIT_SECONDS:
            return True
        if self._retire_hold_expired_generation != generation:
            self._retire_hold_expired_generation = generation
            logger.warning(
                "Kapsl KV shrink generation %d still finds its tail referenced "
                "%.1f s after the replica went idle; new requests proceed and "
                "the shrink waits for them",
                generation,
                _RETIRE_HOLD_LIMIT_SECONDS,
            )
        return False

    def _workers_have_resize_work(self) -> bool:
        """Whether workers should ask Kapsl for resize work on this step.

        The scheduler already learns about a resize from its control thread
        (four polls a second) and from the poll that follows a refused
        reservation. Passing that on spares every worker a control round trip
        per model step, which cost about 1.5 ms per generated token. A worker
        that hears a quarter of a second late only maps or unmaps later, and
        both orders stay safe: blocks are activated after every worker mapped
        them, and retired blocks are never read again.
        """

        if not self._live_resize:
            return False
        with self._resize_lock:
            return self._resize_pending or bool(self._pending_scheduler_operations)

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        if not self._is_scheduler or not self._live_resize:
            return
        elastic = self._elastic_block_pool
        if elastic is None:
            raise SharedPoolImportError(
                "elastic scheduler has no negotiated block-pool geometry"
            )
        elastic.bind(gpu_block_pool)

    def has_pending_push_work(self) -> bool:
        with self._resize_lock:
            pending = self._resize_pending or bool(
                self._pending_scheduler_operations
            )
            elastic = self._elastic_block_pool
            # The pinned vLLM core blocks on its request queue once nothing
            # reports work, and nothing can wake it from there: the background
            # control poll can discover retire_scheduler work but cannot make
            # the sleeping scheduler thread apply and acknowledge it, so the
            # coordinator fences the participant when the acknowledgement
            # deadline passes. Kapsl shrinks an idle pool step by step down to
            # its certified minimum, which lies below the startup capacity, so
            # keep the supported zero-token push-work loop alive for as long
            # as a shrink can still arrive.
            shrinkable = (
                elastic is not None
                and elastic.current_blocks > elastic.minimum_blocks
            )
        if pending:
            return True
        settling = (
            elastic is not None
            and time.monotonic() - self._last_lease_released_at
            < _POST_LOAD_KEEPALIVE_SECONDS
        )
        if not (shrinkable or settling):
            return False
        self._pace_idle_keepalive()
        return True

    def _pace_idle_keepalive(self) -> None:
        """Hold the idle keep-alive loop to one zero-token step per interval.

        vLLM asks for pending push work several times per loop iteration, so
        pause only once after each completed step, and never while a request
        holds a lease: vLLM is stepping for that request anyway.
        """

        if not self._idle_step_completed:
            return
        with self._lease_lock:
            if self._leases:
                return
        self._idle_step_completed = False
        time.sleep(_IDLE_KEEPALIVE_INTERVAL_SECONDS)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = _request_id(request)
        with self._lease_lock:
            # An aborted request may still be waiting for its reservation.
            self._waiting_reservations.pop(request_id, None)
            self._block_owners.discard(request_id)
            lease = self._leases.pop(request_id, None)
            if lease is not None and not self._leases:
                self._last_lease_released_at = time.monotonic()
        if lease is None:
            return False, None

        lease_id = str(lease["lease_id"])
        first_error: Exception | None = None
        try:
            self._control_client().commit(lease_id, _request_computed_tokens(request))
        except Exception as error:  # release must still be attempted
            first_error = error
        try:
            self._control_client().release(lease_id)
        except Exception as error:
            if first_error is None:
                first_error = error
            else:
                logger.error("release also failed for KV lease %s: %s", lease_id, error)
        if first_error is not None:
            raise first_error
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        return self.request_finished(request, [])

    # Worker-side interface ----------------------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del kwargs
        if self._is_scheduler or not self._live_resize:
            return
        has_model_forward = getattr(forward_context, "attn_metadata", None) is not None
        self._worker_forward_lock.acquire()
        self._worker_forward_active = True
        try:
            # Ask only when the scheduler said so. Without a readable note the
            # worker asks anyway.
            metadata = getattr(self, "_connector_metadata", None)
            # Returned to the scheduler once this step has run.
            self._worker_step = getattr(metadata, "step", None)
            if getattr(metadata, "resize_pending", True):
                self._apply_worker_resizes()
            if not has_model_forward:
                # Pinned vLLM uses a zero-token connector step to drain
                # has_pending_push_work() while idle. There is no attention
                # forward to fence in that path and wait_for_save is skipped.
                self._worker_forward_active = False
                self._worker_forward_lock.release()
        except Exception:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
            raise

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        if self._worker_forward_active:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
        return None

    def build_connector_worker_meta(self) -> KapslWorkerStepMetadata | None:
        """Tell the scheduler which numbered step this worker has just run.

        vLLM calls this after the step's forward pass has been issued and
        hands the note to the scheduler together with the step's output, so a
        note for step N means this worker starts nothing older than N+1.
        """

        step = self._worker_step
        self._worker_step = None
        if not _is_step_number(step):
            return None
        return KapslWorkerStepMetadata(step=step)

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self._mode != "shared_pool" or self._is_scheduler:
            return
        if self._shared_attached:
            return
        hook = self._shared_pool_hook
        if hook is None or not hook.used:
            raise SharedPoolImportError(
                "vLLM registered KV tensors without consuming the Kapsl CUDA IPC pool"
            )
        if (
            self._participant_epoch is None
            or self._shared_shard is None
            or self._shared_profile is None
        ):
            raise SharedPoolImportError(
                "shared-pool registration state is incomplete before attachment"
            )
        attachment = make_shared_pool_attachment(
            participant_epoch=self._participant_epoch,
            binding_id=hook.binding_id,
            shard=self._shared_shard,
            profile=self._shared_profile,
            imported_bytes=hook.imported_bytes,
            mapped_bytes=hook.mapped_bytes if self._live_resize else None,
            views=hook.attachment_views(kv_caches),
        )
        self._control_client().attach(attachment)
        self._shared_attached = True
        logger.info(
            "attached Kapsl shared KV binding %s for participant %s",
            hook.binding_id,
            self._participant_id,
        )

    def shutdown(self) -> None:
        if self._worker_forward_active:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
        if self._shared_pool_hook is not None:
            self._shared_pool_hook.shutdown()
            self._shared_pool_hook = None
        if not self._is_scheduler:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self._lease_ttl_ms / 1000.0))
        with self._lease_lock:
            leases = list(self._leases.values())
            self._leases.clear()
            self._waiting_reservations.clear()
        for lease in leases:
            lease_id = str(lease["lease_id"])
            try:
                self._control_client().release(lease_id)
            except KapslKvControlError as error:
                logger.warning("failed to release KV lease %s at shutdown: %s", lease_id, error)

    def _heartbeat_loop(self) -> None:
        interval_seconds = (
            0.25 if self._live_resize else max(0.25, self._lease_ttl_ms / 3000.0)
        )
        # A coordinator that is briefly busy has not revoked anything: a lease
        # stays valid for its full TTL after the last heartbeat that reached
        # it. Keep trying through transport failures for two thirds of the TTL
        # and fail closed while the remaining third still covers every lease.
        # Any answer the coordinator does give is authoritative immediately.
        grace_seconds = self._lease_ttl_ms * 2.0 / 3000.0
        retry_seconds = min(interval_seconds, 1.0)
        wait_seconds = interval_seconds
        now = time.monotonic()
        last_resize_contact = now
        last_lease_contact = now
        while not self._heartbeat_stop.wait(wait_seconds):
            wait_seconds = interval_seconds
            if self._live_resize:
                try:
                    self._poll_scheduler_resizes()
                except KapslKvControlError as error:
                    if (
                        error.kind == "transport"
                        and time.monotonic() - last_resize_contact < grace_seconds
                    ):
                        self._warn_control_stall("resize poll", error)
                        continue
                    with self._lease_lock:
                        self._control_failure = error
                    logger.error(
                        "lost Kapsl live-resize authority; scheduler will fail closed: %s",
                        error,
                    )
                    return
                last_resize_contact = time.monotonic()
            with self._lease_lock:
                has_live_leases = bool(self._leases)
            if not has_live_leases:
                # Nothing can expire, so the next lease starts a fresh window.
                last_lease_contact = time.monotonic()
                continue
            try:
                self._control_client().heartbeat()
            except KapslKvControlError as error:
                transient = (
                    error.kind == "transport"
                    and time.monotonic() - last_lease_contact < grace_seconds
                )
                with self._lease_lock:
                    if not self._leases:
                        continue
                    if not transient:
                        self._control_failure = error
                if transient:
                    self._warn_control_stall("lease heartbeat", error)
                    wait_seconds = retry_seconds
                    continue
                logger.error(
                    "lost Kapsl authority for active KV leases; scheduler will fail closed: %s",
                    error,
                )
                return
            last_lease_contact = time.monotonic()

    def _warn_control_stall(self, operation: str, error: Exception) -> None:
        now = time.monotonic()
        if now - self._stall_warned_at < _STALL_WARNING_INTERVAL_SECONDS:
            return
        self._stall_warned_at = now
        logger.warning(
            "Kapsl KV control %s did not answer and will be retried: %s",
            operation,
            error,
        )

    def _settle_resizes_before_allocation(self) -> None:
        """Apply a pending tail retirement before the first request owns blocks.

        Kapsl admits a shrink only while this participant holds no lease, and
        vLLM hands out block IDs from a rotating free queue, so a request that
        is already running can own blocks anywhere in the pool. The first
        request after an idle moment has reserved its lease but owns no blocks
        yet: this is the last point at which a shrink admitted during that
        idle moment still finds its tail unreferenced. Fetch and apply it here
        instead of waiting for the background poll to deliver it after
        allocation.

        With overlapping batches the tail can still be referenced here: vLLM
        keeps a finished request's blocks until its last step has returned.
        The retirement then stays queued, and _tail_retirement_holds_admission
        keeps the request unallocated until it has been applied.
        """

        try:
            self._poll_scheduler_resizes()
        except KapslKvControlError as error:
            if error.kind != "transport":
                raise
            # The background poll delivers it later. A retirement that then
            # finds a referenced tail is deferred, not applied.
            self._warn_control_stall("resize poll", error)
            return
        self._apply_scheduler_resizes()

    def _poll_scheduler_resizes(self) -> None:
        if self._participant_epoch is None:
            raise SharedPoolImportError("elastic scheduler has no participant epoch")
        operations, handles, pending = self._control_client().poll_resize_state_with_handles(
            make_resize_poll_request(
                participant_epoch=self._participant_epoch,
                actor={"role": "scheduler"},
                applied_generation=self._resize_applied_generation,
            )
        )
        try:
            if handles:
                raise SharedPoolImportError(
                    "scheduler resize phase received CUDA allocation handles"
                )
            with self._resize_lock:
                known = {
                    int(operation["resize_generation"])
                    for operation in self._pending_scheduler_operations
                }
                self._pending_scheduler_operations.extend(
                    operation
                    for operation in operations
                    if int(operation["resize_generation"])
                    > self._resize_applied_generation
                    and int(operation["resize_generation"]) not in known
                )
                self._resize_pending = pending
        finally:
            _close_os_handles(handles)

    def _apply_scheduler_resizes(self) -> None:
        if not self._live_resize:
            return
        elastic = self._elastic_block_pool
        if elastic is None:
            raise SharedPoolImportError("elastic scheduler block pool is unavailable")
        # Operations stay queued until they are acknowledged. The background
        # poll skips generations that are queued or already applied, so it
        # cannot hand this thread a second copy of an operation whose
        # acknowledgement is in flight.
        #
        # Growth is applied and acknowledged at once in every mode: Kapsl
        # offers it only after every worker mapped the new blocks, and only
        # steps scheduled after this point can be given them.
        #
        # A tail retirement is different when vLLM overlaps model batches
        # (async scheduling). The acknowledgement is what lets Kapsl tell the
        # workers to unmap the tail, and a batch scheduled earlier may not
        # have run yet. Such a batch can still write blocks whose reference
        # count is already zero: a request can end, and vLLM free its blocks,
        # before the last batch it was scheduled in has returned. (The pinned
        # vLLM defers that free for a connector with a consumer role. The
        # fence does not depend on it.) So the retirement is split in two:
        #
        #   1. Take the tail out of vLLM's free queue and prefix cache. From
        #      here on no new step can be given a tail block.
        #   2. Acknowledge only after every worker has returned the number of
        #      every step up to the fence: the step being built now, or the
        #      next one if none is. By then each batch scheduled before (1)
        #      has run, and each later one cannot touch the tail.
        #
        # With one batch at a time there is no earlier batch, and the
        # acknowledgement follows at once.
        with self._resize_lock:
            operations = list(self._pending_scheduler_operations)
        actor = {"role": "scheduler"}
        for operation in operations:
            stage = str(operation["stage"])
            if stage not in {"activate_scheduler", "retire_scheduler"}:
                raise SharedPoolImportError("scheduler received a worker resize stage")
            target = int(operation["target_block_count"])
            generation = int(operation["resize_generation"])
            fenced = stage == "retire_scheduler" and self._overlapping_batches
            fence = self._retire_fences.get(generation) if fenced else None
            if fenced and fence is None and self._step_notes_failed:
                # Already reported by _issue_step. This retirement could never
                # be acknowledged, so the pool is left as it is.
                return
            if fence is None:
                try:
                    elastic.apply(target)
                except SharedPoolTailBusyError as error:
                    # Nothing was changed. A request still owns part of the
                    # tail; retry on later steps and leave the deadline to the
                    # coordinator instead of ending every request in flight.
                    if self._deferred_retire_generation != generation:
                        self._deferred_retire_generation = generation
                        logger.warning(
                            "deferred Kapsl KV shrink generation %d to %d blocks: %s",
                            generation,
                            target,
                            error,
                        )
                    return
                if fenced:
                    with self._resize_lock:
                        fence = self._steps_issued + 1
                        self._retire_fences[generation] = fence
                    logger.info(
                        "retired the KV tail for shrink generation %d to %d "
                        "blocks; acknowledging once step %d has run on every "
                        "worker",
                        generation,
                        target,
                        fence,
                    )
            if fence is not None and not self._steps_settled_through(fence):
                # An earlier batch is still outstanding. The operation stays
                # queued, which also keeps the scheduler stepping.
                return
            self._acknowledge_resize(
                make_resize_ack_request(
                    participant_epoch=int(operation["participant_epoch"]),
                    actor=actor,
                    binding_id=str(operation["binding_id"]),
                    resize_generation=generation,
                    stage=stage,
                    applied_block_count=target,
                ),
                actor,
            )
            with self._resize_lock:
                self._resize_applied_generation = max(
                    self._resize_applied_generation, generation
                )
                self._pending_scheduler_operations = [
                    queued
                    for queued in self._pending_scheduler_operations
                    if int(queued["resize_generation"]) != generation
                    or str(queued["binding_id"]) != str(operation["binding_id"])
                ]
                if fence is not None and not any(
                    int(queued["resize_generation"]) == generation
                    for queued in self._pending_scheduler_operations
                ):
                    del self._retire_fences[generation]

    def _acknowledge_resize(
        self, request: Mapping[str, Any], actor: Mapping[str, Any]
    ) -> None:
        """Acknowledge one applied resize stage through a busy coordinator.

        An acknowledgement cannot simply be repeated: once it lands the stage
        advances and a second copy no longer matches. After a transport
        failure, ask the coordinator whether this binding is still waiting for
        the acknowledgement, and repeat it only when it is.
        """

        client = self._control_client()
        delays = iter(_RESIZE_ACK_RETRY_DELAYS_SECONDS)
        outcome_known = True
        while True:
            try:
                if outcome_known:
                    client.ack_resize(request)
                    return
                if not self._resize_acknowledgement_awaited(request, actor):
                    return
                outcome_known = True
                continue
            except KapslKvControlError as error:
                if error.kind != "transport":
                    raise
                delay = next(delays, None)
                if delay is None:
                    raise
                outcome_known = False
            time.sleep(delay)

    def _resize_acknowledgement_awaited(
        self, request: Mapping[str, Any], actor: Mapping[str, Any]
    ) -> bool:
        generation = int(request["resize_generation"])
        operations, handles, _ = self._control_client().poll_resize_state_with_handles(
            make_resize_poll_request(
                participant_epoch=int(request["participant_epoch"]),
                actor=actor,
                applied_generation=generation - 1,
            )
        )
        _close_os_handles(handles)
        return any(
            int(operation["resize_generation"]) == generation
            and str(operation["binding_id"]) == str(request["binding_id"])
            and str(operation["stage"]) == str(request["stage"])
            for operation in operations
        )

    def _apply_worker_resizes(self) -> None:
        if (
            not self._live_resize
            or self._participant_epoch is None
            or self._shared_shard is None
        ):
            return
        hook = self._shared_pool_hook
        if hook is None:
            raise SharedPoolImportError("elastic worker has no shared-pool hook")
        actor = {"role": "worker", "shard": self._shared_shard}
        try:
            operations, handles, _ = (
                self._control_client().poll_resize_state_with_handles(
                    make_resize_poll_request(
                        participant_epoch=self._participant_epoch,
                        actor=actor,
                        applied_generation=self._resize_applied_generation,
                    )
                )
            )
        except KapslKvControlError as error:
            if error.kind != "transport":
                raise
            # A map or unmap that waits one more forward is safe: the
            # scheduler activates blocks only after every worker acknowledged
            # the mapping, and retired blocks are never read again. The
            # scheduler's control thread fails closed if authority stays lost.
            self._warn_control_stall("worker resize poll", error)
            return
        try:
            for operation in operations:
                hook.apply_worker_resize(operation, handles)
                target = int(operation["target_block_count"])
                generation = int(operation["resize_generation"])
                self._acknowledge_resize(
                    make_resize_ack_request(
                        participant_epoch=int(operation["participant_epoch"]),
                        actor=actor,
                        binding_id=str(operation["binding_id"]),
                        resize_generation=generation,
                        stage=str(operation["stage"]),
                        applied_block_count=target,
                    ),
                    actor,
                )
                self._resize_applied_generation = max(
                    self._resize_applied_generation, generation
                )
        finally:
            _close_os_handles(handles)

    def _ensure_shared_active(self) -> None:
        if self._mode != "shared_pool" or self._shared_active:
            return
        with self._activation_lock:
            if self._shared_active:
                return
            if self._participant_epoch is None:
                raise SharedPoolImportError(
                    "shared-pool participant has no registration epoch"
                )
            self._control_client().activate(self._participant_epoch)
            self._shared_active = True
            logger.info(
                "activated Kapsl shared KV participant %s epoch=%s",
                self._participant_id,
                self._participant_epoch,
            )

    def _raise_if_control_failed(self) -> None:
        with self._lease_lock:
            failure = self._control_failure
        if failure is not None:
            raise KapslKvControlError(
                f"Kapsl KV authority heartbeat failed: {failure}",
                kind=failure.kind,
            ) from failure

    def _control_client(self) -> KapslKvControlClient:
        if self._client is None:
            raise RuntimeError("KV control client is not initialized for this connector role")
        return self._client


def _extra(config: Any, key: str, default: Any) -> Any:
    getter = getattr(config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    values = getattr(config, "kv_connector_extra_config", None) or {}
    return values.get(key, default)


def _required_extra(config: Any, key: str) -> str:
    value = str(_extra(config, key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required for KapslConnectorV1")
    return value


def _validated_vmm_conformance(config: Any, live_resize: bool) -> bool:
    value = _extra(config, "kapsl_vmm_conformance", False)
    if not isinstance(value, bool):
        raise ValueError("kapsl_vmm_conformance must be a boolean")
    if value and not live_resize:
        raise ValueError("kapsl_vmm_conformance requires live resize")
    return value


def _validate_shared_pool_execution(vllm_config: Any) -> None:
    validate_certified_shared_pool_execution(vllm_config)


def _validate_elastic_shared_pool_execution(
    vllm_config: Any, kv_cache_config: Any
) -> None:
    _validate_shared_pool_execution(vllm_config)
    if str(getattr(kv_cache_config, "kv_cache_layout", "")).strip().upper() != "BLNHC":
        raise SharedPoolImportError(
            "live-resize shared_pool requires vLLM's block-outermost BLNHC layout"
        )
    concurrent_batches = _max_concurrent_batches(vllm_config)
    if concurrent_batches == 1:
        return
    # More than one batch in flight: a tail retirement must wait for every
    # batch scheduled before it. That is implemented for the two batches
    # async scheduling overlaps and for nothing else.
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if (
        concurrent_batches != 2
        or getattr(scheduler_config, "async_scheduling", None) is not True
    ):
        raise SharedPoolImportError(
            "live-resize shared_pool requires one in-flight model batch per "
            "worker, or the two that vLLM async scheduling overlaps; this "
            f"configuration keeps {concurrent_batches} in flight"
        )
    if not _worker_step_notes_supported():
        raise SharedPoolImportError(
            "live-resize shared_pool cannot run with vLLM async scheduling on "
            "this vLLM build: workers cannot tell the scheduler which steps "
            "they have run, so a tail retirement could not wait for the "
            "batches in flight. Start vLLM with --no-async-scheduling."
        )


def _max_concurrent_batches(vllm_config: Any) -> int:
    concurrent_batches = getattr(vllm_config, "max_concurrent_batches", 1)
    if not isinstance(concurrent_batches, int) or isinstance(concurrent_batches, bool):
        raise SharedPoolImportError("vLLM max_concurrent_batches must be an integer")
    if concurrent_batches < 1:
        raise SharedPoolImportError("vLLM max_concurrent_batches must be positive")
    return concurrent_batches


def _overlapping_model_batches(vllm_config: Any) -> bool:
    """Whether vLLM schedules a batch before the previous one has returned."""

    return _max_concurrent_batches(vllm_config) > 1


def _worker_step_notes_supported() -> bool:
    """Whether this vLLM build hands worker notes back to the scheduler."""

    return (
        _WORKER_NOTES_IMPORTED
        and callable(getattr(KVConnectorBase_V1, "build_connector_worker_meta", None))
        and callable(getattr(KVConnectorBase_V1, "update_connector_output", None))
    )


def _is_step_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _vllm_adapter_profile(
    vllm_config: Any, *, live_resize: bool = False
) -> dict[str, str]:
    # Keep the production profile coupled to the same constraints exercised by
    # the hardware probe.  A generic "vLLM" profile would accidentally allow a
    # build to switch from FlashAttention to another reader at startup.
    _validate_shared_pool_execution(vllm_config)
    try:
        backend_version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        try:
            import vllm

            backend_version = str(getattr(vllm, "__version__", "")).strip()
        except ImportError as error:
            raise SharedPoolImportError(
                "cannot identify the vLLM build for shared-pool attachment"
            ) from error
    if not backend_version:
        raise SharedPoolImportError(
            "cannot identify the vLLM build for shared-pool attachment"
        )
    return {
        "adapter_id": "kapsl-vllm-connector",
        "adapter_version": ADAPTER_VERSION,
        "backend_version": backend_version,
        "profile_id": (
            ELASTIC_ADAPTER_PROFILE_ID if live_resize else ADAPTER_PROFILE_ID
        ),
    }


def _request_id(request: Any) -> str:
    value = str(getattr(request, "request_id", "")).strip()
    if not value:
        raise ValueError("vLLM request has no request_id")
    return value


def _request_token_capacity(request: Any) -> int:
    prompt = getattr(request, "prompt_token_ids", None) or []
    all_tokens = getattr(request, "all_token_ids", None) or prompt
    sampling = getattr(request, "sampling_params", None)
    max_tokens = getattr(sampling, "max_tokens", 0) if sampling is not None else 0
    return max(1, len(all_tokens) + max(0, int(max_tokens or 0)))


def _request_computed_tokens(request: Any) -> int:
    explicit = getattr(request, "num_computed_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    return len(getattr(request, "all_token_ids", None) or [])


def _request_priority(request: Any) -> int:
    return int(getattr(request, "priority", 0) or 0)


def _required_memory_domains(config: Any) -> list[dict[str, Any]]:
    raw_domains = _extra(config, "kapsl_memory_domains", None)
    if (
        not isinstance(raw_domains, Sequence)
        or isinstance(raw_domains, (str, bytes))
        or not raw_domains
    ):
        raise ValueError(
            "kapsl_memory_domains must be a non-empty list of runtime-visible domains"
        )
    domains: list[dict[str, Any]] = []
    for index, raw_domain in enumerate(raw_domains):
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"kapsl_memory_domains[{index}] must be an object")
        domains.append(dict(raw_domain))
    return domains


def _provisioning_grant(config: Any) -> dict[str, Any] | None:
    raw_grant = _extra(config, "kapsl_provisioning_grant", None)
    if raw_grant is None:
        return None
    if not isinstance(raw_grant, Mapping):
        raise ValueError("kapsl_provisioning_grant must be an object")
    return dict(raw_grant)


def _elastic_receipt_block_counts(
    receipt: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Return the certified minimum, mapped, and maximum block counts."""

    pools = receipt.get("shared_pools")
    if not isinstance(pools, list) or not pools:
        raise SharedPoolImportError("elastic registration returned no shared pools")
    shapes: set[tuple[int, int, int]] = set()
    for pool in pools:
        if not isinstance(pool, Mapping):
            raise SharedPoolImportError("elastic shared pool must be an object")
        elastic = pool.get("elastic")
        if not isinstance(elastic, Mapping):
            raise SharedPoolImportError("elastic shared pool metadata is missing")
        shapes.add(
            (
                _positive_integer(
                    elastic.get("minimum_block_count"), "minimum_block_count"
                ),
                _positive_integer(
                    elastic.get("mapped_block_count"), "mapped_block_count"
                ),
                _positive_integer(
                    elastic.get("maximum_block_count"), "maximum_block_count"
                ),
            )
        )
    if len(shapes) != 1:
        raise SharedPoolImportError(
            "tensor-parallel elastic bindings must share one block-count state"
        )
    minimum, initial, maximum = shapes.pop()
    if minimum > initial or initial > maximum:
        raise SharedPoolImportError(
            "elastic minimum/mapped capacity exceeds its next capacity bound"
        )
    return minimum, initial, maximum


def _close_os_handles(handles: Sequence[int]) -> None:
    for descriptor in handles:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _vllm_capacity_groups(
    kv_cache_config: Any,
    memory_domains: Sequence[Mapping[str, Any]],
    *,
    shared_pool: bool = False,
    spec_kind_classifier: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract byte-accounting hints without exposing vLLM block handles."""

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    raw_max_allocations = getattr(kv_cache_config, "num_blocks", None)
    if not groups or raw_max_allocations is None:
        raise ValueError(
            "vLLM KVCacheConfig must expose non-empty kv_cache_groups and num_blocks"
        )
    max_allocations = _positive_integer(raw_max_allocations, "vLLM num_blocks")

    # Modern packed layouts publish the exact shared allocation size. Opaque
    # compatibility keeps the page-derived fallback for older vLLM releases.
    if shared_pool:
        topologies = extract_cache_group_topologies(
            kv_cache_config,
            spec_kind_classifier=spec_kind_classifier,
        )
        group_shapes = [
            (group.block_size_tokens, group.bytes_per_group_block)
            for group in topologies
        ]
        certified_pool_stride = shared_pool_block_stride(topologies)
        _, configured_blocks, pool_stride = vllm_backing_geometry(kv_cache_config)
        if configured_blocks != max_allocations:
            raise AssertionError("vLLM backing geometry changed during extraction")
        if pool_stride != certified_pool_stride:
            raise ValueError(
                "resolved packed allocation block stride disagrees with "
                "certified cache-group page accounting"
            )
    else:
        group_shapes = []
        for index, group in enumerate(groups):
            spec = getattr(group, "kv_cache_spec", None)
            layer_names = getattr(group, "layer_names", None)
            if spec is None or not layer_names:
                raise ValueError(
                    f"vLLM KV cache group {index} has no spec or layers"
                )
            block_size = _positive_integer(
                getattr(spec, "block_size", None),
                f"vLLM KV cache group {index} block_size",
            )
            page_size = _positive_integer(
                getattr(spec, "page_size_bytes", None),
                f"vLLM KV cache group {index} page_size_bytes",
            )
            # UniformTypeKVCacheSpecs.page_size_bytes already sums its per-layer
            # specs; an ordinary merged spec describes one layer and must be
            # scaled by the number of layers sharing the block table.
            group_bytes = (
                page_size
                if getattr(spec, "kv_cache_specs", None) is not None
                else page_size * len(layer_names)
            )
            group_shapes.append((block_size, group_bytes))
        # vLLM HMA groups alias one backing block pool; a shared pool ID tells
        # Kapsl to charge the maximum reservation rather than summing groups.
        pool_stride = max(group_bytes for _, group_bytes in group_shapes)
    capacity_groups: list[dict[str, Any]] = []
    for index, (block_size, _) in enumerate(group_shapes):
        capacity_groups.append(
            {
                "group_id": f"vllm.group.{index}",
                "pool_id": "vllm.pool.0",
                "allocation_granularity_tokens": block_size,
                "bytes_per_allocation": pool_stride,
                "memory_domains": [dict(domain) for domain in memory_domains],
                "max_allocations": max_allocations,
            }
        )
    return capacity_groups


def _vllm_topology(
    kv_cache_config: Any,
    model_fingerprint: str,
    *,
    tensor_parallel_world_size: int = 1,
    spec_kind_classifier: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Translate supported vLLM attention specs into ABI topology metadata."""

    raw_layout_id = getattr(kv_cache_config, "kv_cache_layout", None)
    if not isinstance(raw_layout_id, str) or not raw_layout_id.strip():
        raise ValueError("resolved vLLM KVCacheConfig must expose kv_cache_layout")
    layout_id = raw_layout_id.strip()
    resolved_groups = extract_cache_group_topologies(
        kv_cache_config,
        spec_kind_classifier=spec_kind_classifier,
    )
    cache_groups: list[dict[str, Any]] = []
    next_layer_index = 0
    for group in resolved_groups:
        if group.policy_kind == "sliding_window":
            policy = {
                "kind": "sliding_window",
                "window_tokens": group.window_tokens,
            }
        else:
            policy = {"kind": "full_attention"}
        layers = [
            {"index": next_layer_index + offset, "name": str(layer_name)}
            for offset, layer_name in enumerate(group.layers)
        ]
        next_layer_index += len(layers)
        cache_groups.append(
            {
                "group_id": group.group_id,
                "layers": layers,
                "geometry": {
                    "kind": "paged_attention",
                    "block_size_tokens": group.block_size_tokens,
                    "kv_heads": group.kv_heads,
                    "key_head_dim": group.key_head_dim,
                    "value_head_dim": group.value_head_dim,
                    "element_type": _element_type(group.element_type.name),
                    "layout": {
                        "kind": "backend_native",
                        "layout_id": f"vllm:{layout_id}",
                    },
                },
                "policy": policy,
            }
        )
    return {
        "abi_version": dict(ABI_VERSION),
        "model_fingerprint": model_fingerprint,
        # One registration describes the collective vLLM engine. Physical TP
        # replicas are enumerated by capacity_model.memory_domains.
        "shard": {
            "tensor_parallel_rank": 0,
            "tensor_parallel_world_size": tensor_parallel_world_size,
            "pipeline_parallel_rank": 0,
            "pipeline_parallel_world_size": 1,
        },
        "cache_groups": cache_groups,
    }


def _element_type(dtype: Any) -> dict[str, Any]:
    name = str(dtype).lower().removeprefix("torch.")
    known = {
        "float16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float": "f32",
        "int8": "i8",
        "float8_e4m3fn": "fp8_e4m3",
        "float8_e4m3fnuz": "fp8_e4m3",
    }
    if name in known:
        return {"kind": known[name]}
    if not name or name == "none":
        raise ValueError("vLLM shared_pool KV dtype is unavailable")
    return {"kind": "custom", "name": name}


def _strict_nonnegative_integer(value: Any, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > UINT64_MAX
    ):
        raise ValueError(
            f"{field} must be a non-negative unsigned 64-bit integer"
        )
    return value


def _canonical_rank(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _strict_nonnegative_integer(value, "kapsl_rank_device_map rank")
    if isinstance(value, str) and (
        value == "0"
        or (
            value
            and value[0] in "123456789"
            and all(character in "0123456789" for character in value[1:])
        )
    ):
        return _strict_nonnegative_integer(
            int(value), "kapsl_rank_device_map rank"
        )
    raise ValueError(
        "kapsl_rank_device_map ranks must be canonical non-negative integers"
    )


def _rank_device_map(config: Any) -> dict[int, int] | None:
    raw_map = _extra(config, "kapsl_rank_device_map", None)
    if raw_map is None:
        return None
    if not isinstance(raw_map, Mapping):
        raise ValueError("kapsl_rank_device_map must be an object")
    result: dict[int, int] = {}
    for raw_rank, raw_device_id in raw_map.items():
        rank = _canonical_rank(raw_rank)
        device_id = _strict_nonnegative_integer(
            raw_device_id, "kapsl_rank_device_map device ID"
        )
        if rank in result:
            raise ValueError(
                "kapsl_rank_device_map contains duplicate or colliding ranks"
            )
        result[rank] = device_id
    if not result:
        raise ValueError("kapsl_rank_device_map must not be empty")
    return result


def _validated_shared_rank_device_map(
    config: Any,
    memory_domains: Sequence[Mapping[str, Any]],
    tensor_parallel_size: int,
) -> dict[int, int] | None:
    tensor_parallel_size = _positive_integer(
        tensor_parallel_size, "vLLM tensor_parallel_size"
    )
    device_ids = [
        _strict_nonnegative_integer(
            domain.get("device_id"),
            f"kapsl_memory_domains[{index}].device_id",
        )
        for index, domain in enumerate(memory_domains)
    ]
    if (
        len(device_ids) != tensor_parallel_size
        or len(set(device_ids)) != len(device_ids)
    ):
        raise ValueError(
            "vLLM shared_pool requires exactly one distinct CUDA domain per tensor-parallel rank"
        )
    rank_device_map = _rank_device_map(config)
    if rank_device_map is None:
        if tensor_parallel_size == 1:
            return None
        raise ValueError(
            "tensor-parallel shared_pool requires kapsl_rank_device_map"
        )
    expected_ranks = set(range(tensor_parallel_size))
    if set(rank_device_map) != expected_ranks:
        raise ValueError(
            "kapsl_rank_device_map must contain every tensor-parallel global rank exactly once"
        )
    if set(rank_device_map.values()) != set(device_ids):
        raise ValueError(
            "kapsl_rank_device_map devices must exactly match kapsl_memory_domains"
        )
    return rank_device_map
