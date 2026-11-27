"""Lifecycle of one opaque Kapsl KV participant.

This is the narrow interface the SGLang-facing hooks drive. It knows the
control protocol and the adapter's failure policy, and nothing about SGLang:

* allocation paths (:meth:`admit`, :meth:`require_lease`) raise once Kapsl
  authority is lost, so the scheduler stops instead of serving ungoverned;
* release paths (:meth:`release`, :meth:`cancel`) never raise, because they run
  inside SGLang's own cleanup. A failure there is latched and surfaces at the
  next allocation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from .capacity import (
    SchedulerFacts,
    opaque_capacity_groups,
    pool_token_capacity,
    request_token_capacity,
)
from .client import KapslKvControlError
from .config import AdapterConfig
from .contract import (
    INT32_MAX,
    INT32_MIN,
    make_reserve_request,
    opaque_registration,
)

logger = logging.getLogger(__name__)

CAPACITY_EXHAUSTED = "capacity_exhausted"


class ControlClient(Protocol):
    def register(self, registration: dict[str, Any]) -> dict[str, Any]: ...

    def reserve(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def commit(self, lease_id: str, computed_tokens: int) -> None: ...

    def heartbeat(self) -> None: ...

    def release(self, lease_id: str) -> None: ...


class KvParticipant:
    def __init__(
        self,
        config: AdapterConfig,
        client: ControlClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._clock = clock
        self._heartbeat_interval = (
            max(0.25, config.lease_ttl_ms / 3000.0)
            if heartbeat_interval_seconds is None
            else heartbeat_interval_seconds
        )
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, Any]] = {}
        self._failure: KapslKvControlError | None = None
        self._registered = False
        self._closed = False
        self._group_ids: list[str] = []
        self._pool_tokens = 0
        self._denied_until = float("-inf")
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    # Startup -----------------------------------------------------------------

    def register(self, facts: SchedulerFacts) -> dict[str, Any]:
        """Register the scheduler's KV pool. Any failure aborts startup."""

        if self._registered:
            raise RuntimeError("Kapsl KV participant is already registered")
        groups = opaque_capacity_groups(facts, self._config.memory_domains)
        registration = opaque_registration(
            self._config.participant_id, self._config.model_fingerprint, groups
        )
        receipt = self._client.register(registration)
        self._group_ids = [group["group_id"] for group in groups]
        self._pool_tokens = pool_token_capacity(groups)
        self._registered = True
        logger.info(
            "registered Kapsl KV participant %s (opaque, epoch %s, "
            "%s pages of %s tokens)",
            self._config.participant_id,
            receipt["participant_epoch"],
            groups[0]["max_allocations"],
            groups[0]["allocation_granularity_tokens"],
        )
        return registration

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="kapsl-kv-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    # Allocation paths: fail closed ---------------------------------------------

    def admit(
        self,
        request_id: str,
        *,
        input_tokens: int,
        max_new_tokens: int,
        priority: int = 0,
    ) -> bool:
        """Obtain a capacity lease before SGLang may allocate KV for a request.

        Returns ``False`` when Kapsl has no capacity; the request must then
        stay unadmitted. Raises when authority is lost or the runtime answers
        with anything other than a lease or a capacity denial.
        """

        self._require_authority()
        with self._lock:
            if request_id in self._leases:
                return True
        now = self._clock()
        if now < self._denied_until:
            return False
        request = make_reserve_request(
            request_id=request_id,
            sequence_id=request_id,
            token_capacity=request_token_capacity(
                input_tokens=input_tokens,
                max_new_tokens=max_new_tokens,
                pool_tokens=self._pool_tokens,
            ),
            group_ids=self._group_ids,
            priority=max(INT32_MIN, min(INT32_MAX, priority)),
            ttl_ms=self._config.lease_ttl_ms,
        )
        try:
            lease = self._client.reserve(request)
        except KapslKvControlError as error:
            if error.kind == CAPACITY_EXHAUSTED:
                self._denied_until = now + self._config.denial_backoff_ms / 1000.0
                return False
            self.fail(error)
            raise
        with self._lock:
            self._leases[request_id] = lease
        return True

    def holds(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._leases

    def require_lease(self, request_id: str) -> None:
        """Refuse a KV allocation for a request that holds no Kapsl lease."""

        self._require_authority()
        if not self.holds(request_id):
            error = KapslKvControlError(
                f"SGLang request {request_id!r} reached KV allocation without a "
                "Kapsl capacity lease",
                kind="ungoverned_allocation",
            )
            self.fail(error)
            raise error

    # Release paths: never raise --------------------------------------------------

    def cancel(self, request_id: str) -> None:
        """Return a lease SGLang did not use (it declined to admit the request)."""

        self._finish(request_id, computed_tokens=None)

    def release(self, request_id: str, *, computed_tokens: int) -> None:
        """Report completion and return the lease once SGLang frees the KV row."""

        self._finish(request_id, computed_tokens=max(0, computed_tokens))

    def fail(self, error: BaseException) -> None:
        """Latch loss of authority; the next allocation path raises."""

        if not isinstance(error, KapslKvControlError):
            error = KapslKvControlError(str(error), kind="adapter")
        with self._lock:
            if self._failure is None:
                self._failure = error
                logger.error(
                    "Kapsl KV authority lost; SGLang admission will fail closed: %s",
                    error,
                )

    @property
    def failure(self) -> KapslKvControlError | None:
        with self._lock:
            return self._failure

    def live_leases(self) -> int:
        with self._lock:
            return len(self._leases)

    def shutdown(self) -> None:
        """Stop renewing and hand every live lease back. Safe to call twice."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            leases = list(self._leases.values())
            self._leases.clear()
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._heartbeat_interval + 1.0)
        for lease in leases:
            try:
                self._client.release(str(lease["lease_id"]))
            except KapslKvControlError as error:
                logger.warning(
                    "failed to release KV lease %s at shutdown: %s",
                    lease["lease_id"],
                    error,
                )

    # Heartbeat ---------------------------------------------------------------------

    def heartbeat_once(self) -> bool:
        """Renew every live lease. ``False`` means authority was lost."""

        with self._lock:
            if self._failure is not None:
                return False
            if not self._leases:
                return True
        try:
            self._client.heartbeat()
        except KapslKvControlError as error:
            with self._lock:
                if not self._leases:
                    # The last lease was returned while this renewal was in
                    # flight; there is nothing left for Kapsl to reclaim.
                    return True
            self.fail(error)
            return False
        return True

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            if not self.heartbeat_once():
                return

    # Internals -----------------------------------------------------------------------

    def _require_authority(self) -> None:
        with self._lock:
            failure = self._failure
            usable = self._registered and not self._closed
        if failure is not None:
            raise KapslKvControlError(
                f"Kapsl KV authority was lost: {failure}", kind=failure.kind
            ) from failure
        if not usable:
            raise KapslKvControlError(
                "Kapsl KV participant is not registered or has shut down",
                kind="not_registered",
            )

    def _finish(self, request_id: str, *, computed_tokens: int | None) -> None:
        with self._lock:
            lease = self._leases.pop(request_id, None)
        if lease is None:
            return
        # Capacity this participant just returned may satisfy its own backlog.
        self._denied_until = float("-inf")
        lease_id = str(lease["lease_id"])
        try:
            if computed_tokens is not None:
                self._client.commit(lease_id, computed_tokens)
        except KapslKvControlError as error:
            self.fail(error)
        try:
            self._client.release(lease_id)
        except KapslKvControlError as error:
            self.fail(error)
