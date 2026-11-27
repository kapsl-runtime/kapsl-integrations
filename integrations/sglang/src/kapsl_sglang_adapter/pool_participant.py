"""Lifecycle of one Kapsl KV participant whose pool the runtime owns.

Leases work exactly as in the opaque tier (:class:`.participant.KvParticipant`).
What this adds is the pool itself:

* at startup, :meth:`SharedPoolParticipant.attach_pool` registers SGLang's KV
  buffers as a striped elastic pool, checks the runtime's binding against
  them, and maps the granted prefix; :meth:`SharedPoolParticipant.activate`
  then reports the attachment, after which the runtime grants leases;
* while serving, :meth:`SharedPoolParticipant.service_resize` is called on the
  scheduler thread between steps and applies whatever resize the runtime has
  pending.

The control protocol sequences a resize through a worker stage and a scheduler
stage. An SGLang server at tensor-parallel size 1 runs both roles in one
process, so this participant answers as both, in order:

* growth: map every stripe (``map_workers``), then publish the new slots and
  capacity (``activate_scheduler``);
* shrink: retire the tail slots (``retire_scheduler``), then unmap
  (``unmap_workers``).

A shrink is applied only while no request holds a lease and SGLang is idle;
until then it is deferred and new admissions wait. Anything else that goes
wrong latches a failure, exactly as a lost lease authority does, and the
server stops serving.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .client import KapslKvControlError, close_handles
from .config import AdapterConfig
from .participant import ControlClient, KvParticipant
from .pool_contract import (
    SCHEDULER_ACTOR,
    STAGE_ACTIVATE_SCHEDULER,
    STAGE_MAP_WORKERS,
    STAGE_RETIRE_SCHEDULER,
    STAGE_UNMAP_WORKERS,
    make_attachment,
    make_resize_ack_request,
    make_resize_poll_request,
    single_rank_shard,
    striped_pool_registration,
    worker_actor,
)
from .shared_pool import (
    PoolFacts,
    PoolGeometry,
    RetirementDeferred,
    SharedPoolError,
    StripedPool,
    VmmBackend,
    adapter_profile,
    verify_binding,
)

logger = logging.getLogger(__name__)

# Pauses between attempts to settle an acknowledgement whose reply was lost.
_ACK_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.05, 0.25, 1.0)
_WARNING_INTERVAL_SECONDS = 5.0


class SharedPoolControlClient(ControlClient, Protocol):
    def register_shared_pool(
        self, registration: dict[str, Any]
    ) -> tuple[dict[str, Any], list[int]]: ...

    def attach(self, attachment: dict[str, Any]) -> None: ...

    def activate(self, participant_epoch: int) -> None: ...

    def poll_resize(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[int], bool]: ...

    def ack_resize(self, request: dict[str, Any]) -> None: ...


class PoolHost(Protocol):
    """The server's side of a resize: SGLang's capacity fields and free slots."""

    def tokens(self) -> int:
        """Tokens the server currently serves from the pool."""

    def publish(self, tokens: int) -> None:
        """Growth: raise capacity to ``tokens`` and hand out the new slots."""

    def retire(self, tokens: int) -> None:
        """Shrink: free and remove every slot above ``tokens``.

        Raises :class:`RetirementDeferred`, having changed no capacity, while
        the server is busy or a slot above the boundary is still referenced.
        """


class SharedPoolParticipant(KvParticipant):
    def __init__(
        self,
        config: AdapterConfig,
        client: SharedPoolControlClient,
        *,
        adapter_version: str,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_interval_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            config,
            client,
            clock=clock,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        self._pool_client = client
        self._adapter_version = adapter_version
        self._sleep = sleep
        self._pool: StripedPool | None = None
        self._host: PoolHost | None = None
        self._epoch = 0
        self._profile: dict[str, str] = {}
        self._shard = single_rank_shard()
        # Blocks SGLang serves from. Equal to the mapped blocks except between
        # the two stages of a resize.
        self._active_blocks = 0
        self._worker_generation = 0
        self._scheduler_generation = 0
        self._deferred_retire: dict[str, Any] | None = None
        self._next_poll = float("-inf")
        self._poll_interval = config.resize_poll_ms / 1000.0
        # The runtime keeps a lease valid for its full TTL, so an unanswered
        # poll is tolerated for two thirds of it before authority is lost.
        self._poll_grace = config.lease_ttl_ms * 2.0 / 3000.0
        self._last_poll_contact = clock()
        self._warned_at = float("-inf")

    # Startup -------------------------------------------------------------------

    def attach_pool(
        self, facts: PoolFacts, backend: VmmBackend, *, sglang_version: str
    ) -> StripedPool:
        """Register SGLang's KV buffers and map the prefix the runtime granted.

        Called while SGLang creates its pool, before any of it is backed. Any
        failure aborts startup; nothing SGLang allocated itself backs the pool.
        """

        if self._pool is not None:
            raise SharedPoolError(
                "SGLang created a second VMM-backed KV pool; the striped profile "
                "covers exactly one"
            )
        geometry = PoolGeometry.from_facts(facts)
        domain = dict(self._config.memory_domains[0])
        groups = geometry.capacity_groups(domain)
        profile = adapter_profile(self._adapter_version, sglang_version)
        registration = striped_pool_registration(
            self._config.participant_id,
            self._config.model_fingerprint,
            groups,
            geometry.topology(self._config.model_fingerprint),
            profile,
            provisioning_grant=self._config.provisioning_grant,
        )
        receipt, handles = self._pool_client.register_shared_pool(registration)
        try:
            pool = StripedPool(geometry, verify_binding(geometry, receipt), backend)
            pool.map_initial(handles)
        finally:
            # The mappings keep the memory alive; the descriptors are not needed.
            close_handles(handles)
        self._pool = pool
        self._epoch = receipt["participant_epoch"]
        self._profile = profile
        self._active_blocks = pool.mapped_blocks
        self._group_ids = [group["group_id"] for group in groups]
        # A request may reserve up to the maximum the pool can grow to.
        self._pool_tokens = geometry.tokens(pool.maximum_blocks)
        logger.info(
            "attached Kapsl KV pool %s for %s (epoch %s): %s stripes, %s of %s "
            "blocks mapped, minimum %s",
            pool.binding_id,
            self._config.participant_id,
            self._epoch,
            geometry.stripe_count,
            pool.mapped_blocks,
            pool.maximum_blocks,
            pool.minimum_blocks,
        )
        return pool

    def activate(self, host: PoolHost) -> None:
        """Report the attachment and let the runtime start granting leases."""

        pool = self._require_pool()
        if self._registered:
            raise RuntimeError("Kapsl KV participant is already active")
        geometry = pool.geometry
        expected = geometry.tokens(pool.mapped_blocks)
        if host.tokens() != expected:
            raise SharedPoolError(
                f"SGLang finalized its KV pool at {host.tokens()} tokens, but the "
                f"runtime mapped {expected}"
            )
        attachment = make_attachment(
            participant_epoch=self._epoch,
            binding_id=pool.binding_id,
            shard=self._shard,
            profile=self._profile,
            imported_bytes=pool.maximum_blocks * geometry.bytes_per_block,
            mapped_bytes=pool.mapped_blocks * geometry.bytes_per_block,
            views=geometry.attachment_views(),
        )
        self._pool_client.attach(attachment)
        self._pool_client.activate(self._epoch)
        self._host = host
        self._last_poll_contact = self._clock()
        self._registered = True

    @property
    def pool(self) -> StripedPool | None:
        return self._pool

    def minimum_tokens(self) -> int:
        pool = self._require_pool()
        return pool.geometry.tokens(pool.minimum_blocks)

    def startup_tokens(self) -> int:
        """Tokens SGLang must finalize its pool at: what the runtime mapped."""

        pool = self._require_pool()
        return pool.geometry.tokens(pool.mapped_blocks)

    def register(self, facts: Any) -> dict[str, Any]:
        raise RuntimeError(
            "a shared-pool participant registers while SGLang creates its KV "
            "pool (attach_pool), not after the scheduler is built"
        )

    # Admission -----------------------------------------------------------------

    def admit(
        self,
        request_id: str,
        *,
        input_tokens: int,
        max_new_tokens: int,
        priority: int = 0,
    ) -> bool:
        if self._deferred_retire is not None and not self.holds(request_id):
            # The runtime is waiting for memory back. A new request would keep
            # the server busy past the runtime's deadline, so it waits in
            # SGLang's queue until the tail is retired.
            self._require_authority()
            return False
        denied_until = self._denied_until
        admitted = super().admit(
            request_id,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        if not admitted and self._denied_until != denied_until:
            # A fresh capacity denial is what makes the runtime grow the pool.
            # Fetch that resize at the next step, not at the next interval.
            self._next_poll = float("-inf")
        return admitted

    def _finish(self, request_id: str, *, computed_tokens: int | None) -> None:
        super()._finish(request_id, computed_tokens=computed_tokens)
        if self._deferred_retire is not None and not self.live_leases():
            # The last lease is back: retire the tail at the next step.
            self._next_poll = float("-inf")

    # Resize --------------------------------------------------------------------

    def service_resize(self) -> None:
        """Apply the runtime's pending resize, if any.

        Runs on the scheduler thread between steps. It returns without a
        control round trip unless a poll is due. Any failure is latched and
        raised: the scheduler must stop rather than serve from a pool whose
        state the runtime and this server no longer agree on.
        """

        self._require_authority()
        now = self._clock()
        if now < self._next_poll:
            return
        self._next_poll = now + self._poll_interval
        try:
            self._drive_resize()
        except BaseException as error:
            self.fail(error)
            raise

    def _drive_resize(self) -> None:
        """Take one resize as far as it can go in this step."""

        operation = self._deferred_retire
        pending = True
        if operation is None:
            operation, pending = self._poll_scheduler()
        if operation is not None:
            # Shrink starts here; growth ends here.
            if (
                self._apply_scheduler_stage(operation)
                and operation["stage"] == STAGE_RETIRE_SCHEDULER
            ):
                self._worker_turn()
            return
        if pending and self._worker_turn() == STAGE_MAP_WORKERS:
            operation, _ = self._poll_scheduler()
            if operation is not None:
                self._apply_scheduler_stage(operation)

    def _poll_scheduler(self) -> tuple[dict[str, Any] | None, bool]:
        """The scheduler stage on offer, and whether any resize is in progress."""

        polled = self._poll(SCHEDULER_ACTOR, self._scheduler_generation)
        if polled is None:
            return None, False
        operations, handles, pending = polled
        carried_handles = bool(handles)
        close_handles(handles)
        if carried_handles or len(operations) > 1:
            raise SharedPoolError(
                "the runtime answered a scheduler resize poll with OS handles or "
                "more than one operation"
            )
        return (operations[0] if operations else None), pending

    def _worker_turn(self) -> str | None:
        """Apply the worker stage on offer, if any, and return which it was."""

        polled = self._poll(worker_actor(self._shard), self._worker_generation)
        if polled is None:
            return None
        operations, handles, _ = polled
        try:
            if len(operations) > 1:
                raise SharedPoolError(
                    "the runtime sent more than one worker resize operation for "
                    "a single binding"
                )
            if not operations:
                return None
            self._apply_worker_stage(operations[0], handles)
            return operations[0]["stage"]
        finally:
            close_handles(handles)

    def _apply_worker_stage(
        self, operation: Mapping[str, Any], handles: list[int]
    ) -> None:
        pool = self._require_pool()
        self._require_epoch(operation)
        stage = operation["stage"]
        target = operation["target_block_count"]
        started = time.perf_counter()
        if stage == STAGE_MAP_WORKERS:
            pool.map_growth(operation, handles)
        elif stage == STAGE_UNMAP_WORKERS:
            if self._active_blocks != target:
                raise SharedPoolError(
                    f"asked to unmap down to {target} blocks while "
                    f"{self._active_blocks} are still in service"
                )
            pool.unmap_tail(operation)
        else:
            raise SharedPoolError(f"worker poll returned scheduler stage {stage!r}")
        logger.info(
            "Kapsl KV pool %s: %s generation %s, %s blocks mapped (%.1f ms)",
            pool.binding_id,
            stage,
            operation["resize_generation"],
            pool.mapped_blocks,
            (time.perf_counter() - started) * 1000.0,
        )
        self._acknowledge(operation, worker_actor(self._shard))
        self._worker_generation = operation["resize_generation"]

    def _apply_scheduler_stage(self, operation: Mapping[str, Any]) -> bool:
        """Apply and acknowledge a scheduler stage. ``False`` when it is deferred."""

        pool = self._require_pool()
        host = self._require_host()
        self._require_epoch(operation)
        pool.check_operation(operation)
        stage = operation["stage"]
        target = operation["target_block_count"]
        if operation["from_block_count"] != self._active_blocks:
            raise SharedPoolError(
                f"{stage} starts at {operation['from_block_count']} blocks, but "
                f"{self._active_blocks} are in service"
            )
        if stage == STAGE_ACTIVATE_SCHEDULER:
            if pool.mapped_blocks != target:
                raise SharedPoolError(
                    f"asked to activate {target} blocks while {pool.mapped_blocks} "
                    "are mapped"
                )
            host.publish(pool.geometry.tokens(target))
            # The capacity a queued request was denied may have arrived.
            self._denied_until = float("-inf")
        elif stage == STAGE_RETIRE_SCHEDULER:
            # Refused before anything changes if the boundary is inside a mapping.
            pool.retirement_plan(target)
            if not self._retire(host, operation):
                return False
        else:
            raise SharedPoolError(f"scheduler poll returned worker stage {stage!r}")
        self._active_blocks = target
        logger.info(
            "Kapsl KV pool %s: %s generation %s, %s tokens in service",
            pool.binding_id,
            stage,
            operation["resize_generation"],
            pool.geometry.tokens(target),
        )
        self._acknowledge(operation, SCHEDULER_ACTOR)
        self._scheduler_generation = operation["resize_generation"]
        return True

    def _retire(self, host: PoolHost, operation: Mapping[str, Any]) -> bool:
        """Retire the tail now, or keep the operation for a later step."""

        pool = self._require_pool()
        reason = None
        if self.live_leases():
            reason = f"{self.live_leases()} request(s) hold a lease"
        else:
            try:
                host.retire(pool.geometry.tokens(operation["target_block_count"]))
            except RetirementDeferred as deferred:
                reason = str(deferred)
        if reason is None:
            self._deferred_retire = None
            return True
        if self._deferred_retire is None:
            logger.warning(
                "deferred Kapsl KV shrink generation %s to %s blocks: %s",
                operation["resize_generation"],
                operation["target_block_count"],
                reason,
            )
        self._deferred_retire = dict(operation)
        return False

    def _poll(
        self, actor: Mapping[str, Any], applied_generation: int
    ) -> tuple[list[dict[str, Any]], list[int], bool] | None:
        """One resize poll. ``None`` when the runtime did not answer in time."""

        request = make_resize_poll_request(
            participant_epoch=self._epoch,
            actor=actor,
            applied_generation=applied_generation,
        )
        try:
            polled = self._pool_client.poll_resize(request)
        except KapslKvControlError as error:
            now = self._clock()
            if (
                error.kind != "transport"
                or now - self._last_poll_contact >= self._poll_grace
            ):
                raise
            self._warn(now, "resize poll did not answer and will be retried", error)
            return None
        self._last_poll_contact = self._clock()
        return polled

    def _acknowledge(
        self, operation: Mapping[str, Any], actor: Mapping[str, Any]
    ) -> None:
        """Acknowledge one applied stage through a busy coordinator.

        An acknowledgement cannot simply be repeated: once it lands the stage
        advances and a second copy no longer matches. After a transport
        failure, ask whether the runtime still offers this stage, and repeat
        the acknowledgement only if it does.
        """

        request = make_resize_ack_request(
            participant_epoch=self._epoch,
            actor=actor,
            binding_id=operation["binding_id"],
            resize_generation=operation["resize_generation"],
            stage=operation["stage"],
            applied_block_count=operation["target_block_count"],
        )
        delays = iter(_ACK_RETRY_DELAYS_SECONDS)
        outcome_known = True
        while True:
            try:
                if outcome_known:
                    self._pool_client.ack_resize(request)
                    return
                if not self._still_offered(operation, actor):
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
            self._sleep(delay)

    def _still_offered(
        self, operation: Mapping[str, Any], actor: Mapping[str, Any]
    ) -> bool:
        operations, handles, _ = self._pool_client.poll_resize(
            make_resize_poll_request(
                participant_epoch=self._epoch,
                actor=actor,
                applied_generation=operation["resize_generation"] - 1,
            )
        )
        close_handles(handles)
        return any(
            offered["resize_generation"] == operation["resize_generation"]
            and offered["stage"] == operation["stage"]
            and offered["binding_id"] == operation["binding_id"]
            for offered in operations
        )

    # Internals -----------------------------------------------------------------

    def _require_pool(self) -> StripedPool:
        if self._pool is None:
            raise SharedPoolError(
                "SGLang has not created a VMM-backed KV pool for the Kapsl "
                "runtime to back"
            )
        return self._pool

    def _require_host(self) -> PoolHost:
        if self._host is None:
            raise SharedPoolError("the Kapsl KV pool is attached but not active")
        return self._host

    def _require_epoch(self, operation: Mapping[str, Any]) -> None:
        if operation["participant_epoch"] != self._epoch:
            raise SharedPoolError(
                f"resize generation {operation['resize_generation']} is addressed "
                f"to epoch {operation['participant_epoch']}, not {self._epoch}"
            )

    def _warn(self, now: float, message: str, error: Exception) -> None:
        if now - self._warned_at < _WARNING_INTERVAL_SECONDS:
            return
        self._warned_at = now
        logger.warning("Kapsl KV control %s: %s", message, error)
