"""Live resize while vLLM overlaps two model batches (async scheduling).

The first half scripts single interleavings against the scripted coordinator
used by ``test_connector_resize``. The second half runs the connector inside a
small model of vLLM's engine loop, block pool and worker, and of Kapsl's
resize stages, and checks one property over many random interleavings: no
model batch ever runs against a block the worker has unmapped.
"""

from __future__ import annotations

import random
import unittest
from collections import deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from kapsl_vllm_connector import LIVE_RESIZE_ASYNC_SCHEDULING
from kapsl_vllm_connector import connector as connector_module
from kapsl_vllm_connector.connector import (
    KapslConnectorMetadata,
    KapslWorkerStepMetadata,
    _validate_elastic_shared_pool_execution,
)
from kapsl_vllm_connector.shared_pool import (
    SharedPoolImportError,
    VllmElasticBlockPool,
)
from test_connector_resize import (
    FakeElastic,
    _operation,
    _request,
    _scheduler,
    _worker,
)

_LOGGER = "vllm.kapsl_vllm_connector.connector"


def _step() -> SimpleNamespace:
    """A scheduler output. The connector reads nothing from it."""

    return SimpleNamespace()


def _returned(step: int | None) -> SimpleNamespace:
    """What vLLM hands the scheduler once every worker returned ``step``."""

    note = None if step is None else KapslWorkerStepMetadata(step=step)
    return SimpleNamespace(kv_connector_worker_meta=note)


class OverlappingBatchShrinkTests(unittest.TestCase):
    """A tail retirement waits for every batch scheduled before it."""

    def test_steps_are_numbered_only_when_batches_overlap(self) -> None:
        overlapping = _scheduler([], overlapping=True)
        one_at_a_time = _scheduler([])

        self.assertEqual(
            [overlapping.build_connector_meta(_step()).step for _ in range(3)],
            [1, 2, 3],
        )
        self.assertIsNone(one_at_a_time.build_connector_meta(_step()).step)

    def test_shrink_is_not_acknowledged_until_both_outstanding_batches_return(
        self,
    ) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        # Two batches are with the workers when Kapsl asks for the shrink.
        connector.build_connector_meta(_step())
        connector.build_connector_meta(_step())
        operation = _operation(3, "retire_scheduler", 4)
        connector._pending_scheduler_operations = [operation]

        with self.assertLogs(_LOGGER, "INFO"):
            metadata = connector.build_connector_meta(_step())

        # The tail left the free queue, so the step being built and every
        # later one cannot be given a tail block. Nothing is acknowledged:
        # the acknowledgement is what lets Kapsl unmap the tail on workers.
        self.assertEqual(events, [("apply", 4)])
        self.assertEqual(metadata.step, 3)
        self.assertTrue(metadata.resize_pending)
        self.assertEqual(connector._pending_scheduler_operations, [operation])
        self.assertTrue(connector.has_pending_push_work())

        # Whatever order the notes arrive in, both earlier batches count.
        connector.update_connector_output(_returned(3))
        self.assertEqual(events, [("apply", 4)])
        connector.update_connector_output(_returned(1))
        self.assertEqual(events, [("apply", 4)])
        connector.update_connector_output(_returned(2))

        self.assertEqual(events, [("apply", 4), ("ack", "retire_scheduler", 3)])
        self.assertEqual(connector._pending_scheduler_operations, [])
        self.assertEqual(connector._resize_applied_generation, 3)
        self.assertEqual(connector._retire_fences, {})

    def test_shrink_waits_for_the_step_that_was_being_built(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]

        with self.assertLogs(_LOGGER, "INFO"):
            connector.build_connector_meta(_step())
        # Nothing was in flight, yet the step that was being built is: its
        # block tables were filled before the tail left the free queue.
        self.assertEqual(events, [("apply", 4)])

        connector.build_connector_meta(_step())
        self.assertEqual(events, [("apply", 4)])
        connector.update_connector_output(_returned(1))

        self.assertEqual(events, [("apply", 4), ("ack", "retire_scheduler", 3)])

    def test_the_tail_is_retired_once_however_often_the_step_is_retried(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]

        with self.assertLogs(_LOGGER, "INFO") as logs:
            for _ in range(5):
                connector.build_connector_meta(_step())

        self.assertEqual(events, [("apply", 4)])
        self.assertEqual(len(logs.records), 1)

    def test_resize_that_arrives_between_two_overlapping_batches(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector.build_connector_meta(_step())
        # The next batch is scheduled before the previous one has returned.
        connector.build_connector_meta(_step())
        connector._client.polls = [([_operation(3, "retire_scheduler", 4)], [], True)]
        connector._poll_scheduler_resizes()
        events.clear()

        # The previous batch returns. The second one is still outstanding and
        # was scheduled against the old capacity.
        connector.update_connector_output(_returned(1))
        self.assertEqual(events, [])

        with self.assertLogs(_LOGGER, "INFO"):
            connector.build_connector_meta(_step())
        self.assertEqual(events, [("apply", 4)])
        connector.update_connector_output(_returned(2))
        self.assertEqual(events, [("apply", 4)])
        connector.update_connector_output(_returned(3))

        self.assertEqual(events, [("apply", 4), ("ack", "retire_scheduler", 3)])

    def test_growth_is_applied_at_once_with_two_batches_outstanding(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector.build_connector_meta(_step())
        connector.build_connector_meta(_step())
        connector._pending_scheduler_operations = [
            _operation(4, "activate_scheduler", 16)
        ]

        connector.build_connector_meta(_step())

        # Every worker mapped the new blocks before Kapsl offered them, and
        # only steps scheduled from here on can be given them.
        self.assertEqual(events, [("apply", 16), ("ack", "activate_scheduler", 4)])
        self.assertEqual(connector._retire_fences, {})

    def test_every_binding_of_one_shrink_shares_its_fence(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        second = dict(_operation(3, "retire_scheduler", 4), binding_id="binding-1")
        connector._pending_scheduler_operations = [
            _operation(3, "retire_scheduler", 4),
            second,
        ]

        with self.assertLogs(_LOGGER, "INFO"):
            connector.build_connector_meta(_step())
        connector.update_connector_output(_returned(1))

        # Tensor-parallel ranks share one block pool: retired once, then one
        # acknowledgement per binding.
        self.assertEqual(
            events,
            [
                ("apply", 4),
                ("ack", "retire_scheduler", 3),
                ("ack", "retire_scheduler", 3),
            ],
        )
        self.assertEqual(connector._pending_scheduler_operations, [])
        self.assertEqual(connector._retire_fences, {})

    def test_one_batch_at_a_time_still_acknowledges_at_once(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]

        connector.build_connector_meta(_step())

        self.assertEqual(events, [("apply", 4), ("ack", "retire_scheduler", 3)])


class MissingWorkerNoteTests(unittest.TestCase):
    """Without the workers' notes nothing proves a batch has run."""

    def test_step_without_a_note_never_releases_a_retired_tail(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]

        with self.assertLogs(_LOGGER, "INFO"):
            connector.build_connector_meta(_step())
        connector.update_connector_output(_returned(None))
        connector.update_connector_output(SimpleNamespace())
        # A note that some worker did not confirm merges to step 0.
        connector.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=KapslWorkerStepMetadata(step=1).aggregate(
                    object()
                )
            )
        )
        connector.build_connector_meta(_step())
        connector.update_connector_output(_returned(2))

        self.assertEqual(events, [("apply", 4)])
        self.assertTrue(connector.has_pending_push_work())

    def test_notes_that_stop_coming_are_reported_and_fail_closed(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        limit = connector_module._MAX_OUTSTANDING_STEPS

        with self.assertLogs(_LOGGER, "ERROR") as logs:
            for _ in range(limit + 5):
                connector.build_connector_meta(_step())
        self.assertEqual(len(logs.records), 1)
        self.assertIn("will not shrink", logs.output[0])
        # Bounded: the scheduler does not remember every unanswered step.
        self.assertEqual(connector._steps_outstanding, set())

        operation = _operation(3, "retire_scheduler", 4)
        connector._pending_scheduler_operations = [operation]
        metadata = connector.build_connector_meta(_step())
        connector.update_connector_output(_returned(metadata.step))

        # The pool is left alone and the tail stays mapped on every worker.
        # Kapsl's acknowledgement deadline, not an unfenced unmap, ends this.
        self.assertEqual(events, [])
        self.assertEqual(connector._pending_scheduler_operations, [operation])
        self.assertTrue(connector.has_pending_push_work())

    def test_retirement_already_under_way_is_not_acknowledged_once_notes_stop(
        self,
    ) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]

        with self.assertLogs(_LOGGER, "INFO"):
            fence = connector.build_connector_meta(_step()).step
        with self.assertLogs(_LOGGER, "ERROR"):
            for _ in range(connector_module._MAX_OUTSTANDING_STEPS + 1):
                connector.build_connector_meta(_step())
        connector.update_connector_output(_returned(fence))

        self.assertEqual(events, [("apply", 4)])


class IdleShrinkBeforeAllocationTests(unittest.TestCase):
    """The first request after an idle moment never lands in a retiring tail."""

    def _clock(self, start: float = 100.0):
        clock = [start]
        return clock, patch(
            "kapsl_vllm_connector.connector.time.monotonic",
            side_effect=lambda: clock[0],
        )

    def test_first_request_after_idle_takes_the_tail_out_before_allocation(
        self,
    ) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._client.polls = [([_operation(3, "retire_scheduler", 4)], [], True)]
        request = _request("request-1")

        with self.assertLogs(_LOGGER, "INFO"):
            connector.on_new_request(request)

        # on_new_request runs before vLLM allocates any block for the request.
        self.assertEqual(events, ["reserve", ("poll", "scheduler"), ("apply", 4)])
        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))

        metadata = connector.build_connector_meta(_step())
        self.assertEqual(events, ["reserve", ("poll", "scheduler"), ("apply", 4)])
        connector.update_connector_output(_returned(metadata.step))

        self.assertEqual(events[-1], ("ack", "retire_scheduler", 3))

    def test_new_requests_wait_while_a_finished_request_still_holds_the_tail(
        self,
    ) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        # vLLM keeps a finished request's blocks until its last batch has
        # returned, so the tail is still referenced for a step or two.
        connector._elastic_block_pool = FakeElastic(events, busy=3)
        connector._client.polls = [([_operation(3, "retire_scheduler", 4)], [], True)]
        first, second = _request("request-1"), _request("request-2")
        clock, patched = self._clock()

        with patched, self.assertLogs(_LOGGER, "INFO"):
            connector.on_new_request(first)
            connector.on_new_request(second)
            self.assertEqual(
                events, ["reserve", ("poll", "scheduler"), "reserve"]
            )

            # Both hold a lease and neither may be given blocks yet. One
            # attempt per scheduler step, however many requests wait.
            self.assertEqual(
                connector.get_num_new_matched_tokens(first, 0), (None, False)
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(second, 0), (None, False)
            )
            self.assertEqual(connector._elastic_block_pool.busy, 1)

            # The scheduling step ends with the tail still referenced.
            connector.build_connector_meta(_step())
            self.assertEqual(connector._elastic_block_pool.busy, 0)

            # The finished request's last batch returned: the tail is free,
            # leaves the queue, and the requests are admitted behind it.
            self.assertEqual(
                connector.get_num_new_matched_tokens(first, 0), (0, False)
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(second, 0), (0, False)
            )

        self.assertEqual(
            events, ["reserve", ("poll", "scheduler"), "reserve", ("apply", 4)]
        )

    def test_running_request_is_no_reason_to_hold_new_ones(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._elastic_block_pool = FakeElastic(events, busy=10)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]
        running, late = _request("request-1"), _request("request-2")
        with self.assertLogs(_LOGGER, "WARNING"):
            connector.on_new_request(running)
        connector.update_state_after_alloc(running, None, 0)
        connector.on_new_request(late)
        attempts = connector._elastic_block_pool.busy

        # The running request may own tail blocks until it ends. Holding the
        # next one that long would starve it, and gains nothing.
        self.assertEqual(connector.get_num_new_matched_tokens(late, 0), (0, False))
        self.assertEqual(connector._elastic_block_pool.busy, attempts)

    def test_request_end_gives_up_block_ownership(self) -> None:
        connector = _scheduler([], overlapping=True)
        request = _request("request-1")
        connector.on_new_request(request)
        connector.update_state_after_alloc(request, None, 0)
        self.assertEqual(connector._block_owners, {"request-1"})

        connector.request_finished(request, [])

        self.assertEqual(connector._block_owners, set())

    def test_hold_ends_after_its_limit(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, overlapping=True)
        connector._elastic_block_pool = FakeElastic(events, busy=1000)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]
        request = _request("request-1")
        clock, patched = self._clock()

        with patched, self.assertLogs(_LOGGER, "WARNING") as logs:
            connector.on_new_request(request)
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (None, False)
            )
            clock[0] += connector_module._RETIRE_HOLD_LIMIT_SECONDS - 0.1
            connector.build_connector_meta(_step())
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (None, False)
            )
            clock[0] += 0.2
            connector.build_connector_meta(_step())
            # Something other than a finished request holds the tail. The
            # request runs; the shrink is left to Kapsl's deadline as before.
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (0, False)
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (0, False)
            )

        self.assertEqual(
            len([line for line in logs.output if "new requests proceed" in line]), 1
        )

    def test_one_batch_at_a_time_never_holds_a_request(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._elastic_block_pool = FakeElastic(events, busy=10)
        connector._pending_scheduler_operations = [_operation(3, "retire_scheduler", 4)]
        request = _request("request-1")

        with self.assertLogs(_LOGGER, "WARNING"):
            connector.on_new_request(request)

        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))


class WorkerStepNoteTests(unittest.TestCase):
    def _run(self, worker: Any, metadata: Any, *, forward: bool) -> Any:
        worker._connector_metadata = metadata
        worker.start_load_kv(
            SimpleNamespace(attn_metadata=object() if forward else None)
        )
        worker.wait_for_save()
        return worker.build_connector_worker_meta()

    def test_worker_returns_the_number_of_the_step_it_ran(self) -> None:
        worker = _worker([])

        note = self._run(
            worker, KapslConnectorMetadata(resize_pending=False, step=7), forward=True
        )

        self.assertEqual(note, KapslWorkerStepMetadata(step=7))
        # One note per step: nothing is left over for the next one.
        self.assertIsNone(worker.build_connector_worker_meta())

    def test_zero_token_step_is_returned_too(self) -> None:
        worker = _worker([])

        note = self._run(
            worker, KapslConnectorMetadata(resize_pending=False, step=8), forward=False
        )

        # An idle engine steps without a forward pass, and an idle shrink is
        # fenced on exactly such steps.
        self.assertEqual(note, KapslWorkerStepMetadata(step=8))

    def test_unnumbered_step_returns_nothing(self) -> None:
        worker = _worker([])

        self.assertIsNone(
            self._run(worker, KapslConnectorMetadata(resize_pending=False), forward=True)
        )
        self.assertIsNone(self._run(worker, None, forward=True))

    def test_notes_of_tensor_parallel_workers_merge_to_the_slowest(self) -> None:
        merged = KapslWorkerStepMetadata(step=9).aggregate(
            KapslWorkerStepMetadata(step=8)
        )

        self.assertEqual(merged, KapslWorkerStepMetadata(step=8))
        self.assertEqual(
            KapslWorkerStepMetadata(step=9).aggregate(SimpleNamespace()).step, 0
        )

    def test_growth_is_mapped_on_the_step_that_announces_it(self) -> None:
        events: list[Any] = []
        worker = _worker(events)
        worker._client.polls = [([_operation(4, "map_workers", 16)], [], True)]

        self._run(
            worker, KapslConnectorMetadata(resize_pending=True, step=3), forward=True
        )

        # Mapping never waits for batches in flight: it only adds memory.
        self.assertEqual(
            events, [("poll", "worker"), ("map", 16), ("ack", "map_workers", 4)]
        )


def _vllm_config(
    *, concurrent_batches: Any, async_scheduling: Any
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(enable_sleep_mode=False),
        attention_config=SimpleNamespace(backend="FLASH_ATTN", backend_per_kind={}),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        max_concurrent_batches=concurrent_batches,
    )


class ExecutionModeTests(unittest.TestCase):
    """Which batch overlap a live-resize pool accepts at startup."""

    _CACHE = SimpleNamespace(kv_cache_layout="BLNHC")

    def _validate(self, **config: Any) -> None:
        _validate_elastic_shared_pool_execution(_vllm_config(**config), self._CACHE)

    def test_connector_declares_that_it_fences_async_scheduling(self) -> None:
        # Kapsl leaves async scheduling on only for a connector that says so.
        self.assertIs(LIVE_RESIZE_ASYNC_SCHEDULING, True)

    def test_one_batch_at_a_time_is_accepted(self) -> None:
        self._validate(concurrent_batches=1, async_scheduling=False)

    def test_async_scheduling_is_accepted_when_workers_can_return_notes(self) -> None:
        with patch.object(
            connector_module, "_worker_step_notes_supported", return_value=True
        ):
            self._validate(concurrent_batches=2, async_scheduling=True)

    def test_async_scheduling_is_refused_without_worker_notes(self) -> None:
        with patch.object(
            connector_module, "_worker_step_notes_supported", return_value=False
        ):
            with self.assertRaisesRegex(
                SharedPoolImportError, "--no-async-scheduling"
            ):
                self._validate(concurrent_batches=2, async_scheduling=True)

    def test_host_without_vllm_cannot_return_worker_notes(self) -> None:
        # These tests run without vLLM, where the hooks do not exist.
        self.assertFalse(connector_module._worker_step_notes_supported())
        with self.assertRaisesRegex(SharedPoolImportError, "--no-async-scheduling"):
            self._validate(concurrent_batches=2, async_scheduling=True)

    def test_overlap_from_anything_but_async_scheduling_is_refused(self) -> None:
        with patch.object(
            connector_module, "_worker_step_notes_supported", return_value=True
        ):
            # Pipeline parallelism also overlaps batches, in a different way.
            for async_scheduling in (False, None, "true", 1):
                with self.assertRaisesRegex(SharedPoolImportError, "keeps 2 in flight"):
                    self._validate(
                        concurrent_batches=2, async_scheduling=async_scheduling
                    )

    def test_more_than_two_batches_in_flight_are_refused(self) -> None:
        with patch.object(
            connector_module, "_worker_step_notes_supported", return_value=True
        ):
            with self.assertRaisesRegex(SharedPoolImportError, "keeps 3 in flight"):
                self._validate(concurrent_batches=3, async_scheduling=True)

    def test_malformed_batch_count_is_refused(self) -> None:
        for value in (True, "2", 2.0, None):
            with self.assertRaisesRegex(SharedPoolImportError, "must be an integer"):
                self._validate(concurrent_batches=value, async_scheduling=True)
        with self.assertRaisesRegex(SharedPoolImportError, "must be positive"):
            self._validate(concurrent_batches=0, async_scheduling=False)


# ---------------------------------------------------------------------------
# A model of the parts of vLLM and Kapsl that decide when a block is touched.
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False


class _FreeQueue:
    def __init__(self, blocks: list[_Block]) -> None:
        self.values = list(blocks)

    def remove(self, block: _Block) -> None:
        self.values.remove(block)

    def append_n(self, blocks: list[_Block]) -> None:
        self.values.extend(blocks)

    def popleft(self) -> _Block:
        return self.values.pop(0)


class _BlockPool:
    """vLLM's BlockPool: reference counts and a free queue."""

    def __init__(self, maximum_blocks: int) -> None:
        self.blocks = [_Block(index) for index in range(maximum_blocks)]
        # vLLM takes block 0 for its null block and never frees it.
        self.blocks[0].is_null = True
        self.free_block_queue = _FreeQueue(self.blocks[1:])
        self.num_gpu_blocks = maximum_blocks

    def _maybe_evict_cached_block(self, block: _Block) -> bool:
        return False

    def allocate(self, count: int) -> list[_Block] | None:
        if len(self.free_block_queue.values) < count:
            return None
        blocks = [self.free_block_queue.popleft() for _ in range(count)]
        for block in blocks:
            block.ref_cnt += 1
        return blocks

    def free(self, blocks: list[_Block]) -> None:
        released = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                released.append(block)
        # vLLM reuses blocks without a prefix-cache hash first.
        self.free_block_queue.values[:0] = released


class _Coordinator:
    """Kapsl's resize stages, and the control client both roles talk to."""

    def __init__(self, blocks: int) -> None:
        self.blocks = blocks
        self.leases: set[str] = set()
        self.resize: dict[str, Any] | None = None
        self.generation = 0
        self.finished: list[tuple[str, int]] = []
        # Engine steps the longest resize stayed open. Kapsl fences a replica
        # whose resize misses its acknowledgement deadline.
        self.clock = 0
        self.longest_resize = 0

    # Supervisor ------------------------------------------------------------

    def request_resize(self, target: int) -> bool:
        if self.resize is not None or target == self.blocks:
            return False
        shrinking = target < self.blocks
        if shrinking and self.leases:
            # Kapsl admits a shrink only while the replica holds no lease.
            return False
        self.generation += 1
        self.resize = {
            "generation": self.generation,
            "from": self.blocks,
            "target": target,
            "stage": "retire_scheduler" if shrinking else "map_workers",
            "since": self.clock,
        }
        return True

    # Control client ---------------------------------------------------------

    def reserve(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = f"lease-{request['sequence']['request_id']}"
        self.leases.add(lease_id)
        return {"lease_id": lease_id}

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        pass

    def release(self, lease_id: str) -> None:
        self.leases.discard(lease_id)

    def poll_resize_state_with_handles(self, request: dict[str, Any]):
        resize = self.resize
        if resize is None:
            return [], [], False
        role = request["actor"]["role"]
        scheduler_stage = resize["stage"] in {"retire_scheduler", "activate_scheduler"}
        if (role == "scheduler") != scheduler_stage or resize["generation"] <= int(
            request["applied_generation"]
        ):
            return [], [], True
        operation = {
            "participant_epoch": 7,
            "resize_generation": resize["generation"],
            "binding_id": "binding-0",
            "stage": resize["stage"],
            "from_block_count": resize["from"],
            "target_block_count": resize["target"],
        }
        return [operation], [], True

    def ack_resize(self, request: dict[str, Any]) -> None:
        resize = self.resize
        if (
            resize is None
            or request["resize_generation"] != resize["generation"]
            or request["stage"] != resize["stage"]
        ):
            raise AssertionError(f"unexpected acknowledgement {request}")
        following = {
            "retire_scheduler": "unmap_workers",
            "map_workers": "activate_scheduler",
        }.get(resize["stage"])
        if following is not None:
            resize["stage"] = following
            return
        self.finished.append(
            ("shrink" if resize["target"] < resize["from"] else "grow", resize["target"])
        )
        self.longest_resize = max(self.longest_resize, self.clock - resize["since"])
        self.blocks = resize["target"]
        self.resize = None


class _WorkerMemory:
    """The worker's mapped prefix of the KV pool."""

    def __init__(self, blocks: int) -> None:
        self.mapped_blocks = blocks

    def apply_worker_resize(self, operation: dict[str, Any], handles: list[int]) -> None:
        self.mapped_blocks = int(operation["target_block_count"])


@dataclass
class _SimRequest:
    request_id: str
    length: int
    prompt_token_ids: list[int] = field(default_factory=lambda: [1, 2, 3])
    sampling_params: Any = field(default_factory=lambda: SimpleNamespace(max_tokens=64))
    priority: int = 0
    num_computed_tokens: int = 0
    blocks: list[_Block] = field(default_factory=list)
    scheduled: int = 0
    produced: int = 0
    last_step: int = 0
    finished: bool = False


class _Engine:
    """vLLM's engine loop with a batch queue of two, as far as blocks go.

    ``fenced`` is the connector as it ships. Without it the connector runs as
    it does for one batch at a time, which is what async scheduling would get
    if the startup check were simply removed.
    """

    _BLOCK_TOKENS = 3
    _MAXIMUM = 40
    _MINIMUM = 6

    def __init__(
        self,
        seed: int,
        *,
        fenced: bool,
        defer_block_free: bool,
        worker: str,
        workers_always_poll: bool,
    ) -> None:
        self.rng = random.Random(seed)
        self.defer_block_free = defer_block_free
        self.worker_mode = worker
        self.workers_always_poll = workers_always_poll
        initial = 24
        self.pool = _BlockPool(self._MAXIMUM)
        self.elastic = VllmElasticBlockPool(
            initial, self._MAXIMUM, minimum_blocks=self._MINIMUM
        )
        self.elastic.bind(self.pool)
        self.coordinator = _Coordinator(initial)
        self.memory = _WorkerMemory(initial)
        self.scheduler = _scheduler([], overlapping=fenced)
        self.scheduler._client = self.coordinator
        self.scheduler._elastic_block_pool = self.elastic
        self.worker = _worker([], client=self.coordinator)
        self.worker._shared_pool_hook = self.memory
        self.batch_queue: deque[Any] = deque()
        self.inbox: deque[Any] = deque()
        self.waiting: list[_SimRequest] = []
        self.running: list[_SimRequest] = []
        self.deferred: list[tuple[int, list[_Block]]] = []
        self.step_index = 0
        self.processed = 0
        self.arrivals = 0
        self.violations: list[str] = []
        self.completed_requests = 0
        self.held_admissions = 0

    # Scheduler ---------------------------------------------------------------

    def add_request(self, length: int) -> None:
        self.arrivals += 1
        request = _SimRequest(f"request-{self.arrivals}", length)
        self.scheduler.on_new_request(request)
        self.waiting.append(request)

    def _has_requests(self) -> bool:
        return bool(
            self.running or self.waiting or self.scheduler.has_pending_push_work()
        )

    def _schedule(self) -> Any:
        self.step_index += 1
        scheduled: list[_SimRequest] = []
        for request in self.running:
            if request.scheduled % self._BLOCK_TOKENS == 0:
                blocks = self.pool.allocate(1)
                if blocks is None:
                    continue
                request.blocks.extend(blocks)
            scheduled.append(request)
        still_waiting: list[_SimRequest] = []
        for request in self.waiting:
            matched, _ = self.scheduler.get_num_new_matched_tokens(request, 0)
            # The coordinator here never runs out of capacity, so a request
            # is only ever kept waiting for a tail retirement.
            self.held_admissions += matched is None
            blocks = None if matched is None else self.pool.allocate(2)
            if blocks is None:
                still_waiting.append(request)
                continue
            request.blocks = blocks
            self.scheduler.update_state_after_alloc(request, None, 0)
            self.running.append(request)
            scheduled.append(request)
        self.waiting = still_waiting
        touched: set[int] = set()
        for request in scheduled:
            request.scheduled += 1
            request.last_step = self.step_index
            touched.update(block.block_id for block in request.blocks)
        step = SimpleNamespace(
            index=self.step_index,
            requests=scheduled,
            touched=frozenset(touched),
            note=None,
        )
        step.metadata = self.scheduler.build_connector_meta(step)
        return step

    def _update_from_output(self, step: Any) -> None:
        self.processed = step.index
        ready = [entry for entry in self.deferred if entry[0] <= self.processed]
        self.deferred = [entry for entry in self.deferred if entry[0] > self.processed]
        for _, blocks in ready:
            self.pool.free(blocks)
        for request in step.requests:
            if request.finished:
                # The request ended on an earlier step's output. This step
                # was already scheduled, and wrote its blocks all the same.
                continue
            request.produced += 1
            request.num_computed_tokens += 1
            if request.produced < request.length:
                continue
            request.finished = True
            self.completed_requests += 1
            self.running.remove(request)
            self.scheduler.request_finished(request, [])
            blocks, request.blocks = request.blocks, []
            if self.defer_block_free and request.last_step > self.processed:
                self.deferred.append((self.step_index, blocks))
            else:
                self.pool.free(blocks)
        if step.note is not None:
            self.scheduler.update_connector_output(
                SimpleNamespace(kv_connector_worker_meta=step.note)
            )

    # Worker ------------------------------------------------------------------

    def _run_on_worker(self, step: Any) -> None:
        metadata = step.metadata
        if self.workers_always_poll:
            metadata = replace(metadata, resize_pending=True)
        self.worker._connector_metadata = metadata
        self.worker.start_load_kv(
            SimpleNamespace(attn_metadata=object() if step.requests else None)
        )
        unmapped = sorted(
            block for block in step.touched if block >= self.memory.mapped_blocks
        )
        if unmapped:
            self.violations.append(
                f"step {step.index} wrote blocks {unmapped} with "
                f"{self.memory.mapped_blocks} blocks mapped"
            )
        self.worker.wait_for_save()
        step.note = self.worker.build_connector_worker_meta()
        self.worker._connector_metadata = None

    def _worker_runs(self, *, through: Any | None = None, limit: int | None = None) -> None:
        while self.inbox and limit != 0:
            step = self.inbox.popleft()
            self._run_on_worker(step)
            if step is through:
                return
            if limit is not None:
                limit -= 1

    # Engine loop ---------------------------------------------------------------

    def step(self) -> None:
        self.coordinator.clock += 1
        if self.worker_mode == "random":
            # The worker gets ahead by as much as it happens to.
            self._worker_runs(limit=self.rng.randint(0, 2))
        if self._has_requests():
            step = self._schedule()
            self.inbox.append(step)
            if self.worker_mode == "inline":
                # One process: the step runs before the scheduler goes on.
                self._worker_runs()
            self.batch_queue.appendleft(step)
            if len(self.batch_queue) < 2 and (step.requests or self._has_requests()):
                return
        elif not self.batch_queue:
            return
        step = self.batch_queue.pop()
        # Block on the oldest step. A worker in its own process may only now
        # get to it.
        self._worker_runs(through=step)
        self._update_from_output(step)

    def settled(self) -> bool:
        """No request and no resize is left. Keep-alive steps may go on."""

        return not (self.running or self.waiting or self.coordinator.resize)

    def flush(self) -> None:
        """Return the steps still in the batch queue, as an engine stop does."""

        while self.batch_queue:
            step = self.batch_queue.pop()
            self._worker_runs(through=step)
            self._update_from_output(step)


def _drive(engine: _Engine, ticks: int) -> None:
    """Random load, a supervisor that resizes, and the control thread's poll."""

    rng = engine.rng

    def arrive() -> None:
        if rng.random() < 0.25 and len(engine.waiting) < 3:
            engine.add_request(rng.randint(1, 9))

    def supervise() -> None:
        if engine.coordinator.resize is not None:
            return
        current = engine.coordinator.blocks
        roll = rng.random()
        if roll < 0.5 and current > engine._MINIMUM:
            engine.coordinator.request_resize(
                max(engine._MINIMUM, current - rng.randint(2, 8))
            )
        elif roll < 0.6 and current < engine._MAXIMUM:
            engine.coordinator.request_resize(
                min(engine._MAXIMUM, current + rng.randint(2, 8))
            )

    def control_poll() -> None:
        if rng.random() < 0.5:
            engine.scheduler._poll_scheduler_resizes()

    for tick in range(ticks):
        # Requests, Kapsl's supervisor and the connector's control thread do
        # not wait for each other: any order between two engine steps.
        actions = [supervise, control_poll]
        if tick < ticks * 3 // 4:
            actions.append(arrive)
        rng.shuffle(actions)
        for action in actions:
            action()
        engine.step()
    for _ in range(400):
        if engine.settled():
            break
        engine.scheduler._poll_scheduler_resizes()
        engine.step()
    engine.flush()


class OverlappingBatchSimulationTests(unittest.TestCase):
    """No batch may ever run against memory its worker has unmapped."""

    _SEEDS = range(300)
    _TICKS = 160

    def _run(self, seed: int, **options: Any) -> _Engine:
        engine = _Engine(seed, **options)
        clock = [0.0]

        def monotonic() -> float:
            # A millisecond per look at the clock keeps every run repeatable.
            clock[0] += 0.001
            return clock[0]

        with patch("kapsl_vllm_connector.connector.time.sleep"), patch(
            "kapsl_vllm_connector.connector.time.monotonic", side_effect=monotonic
        ), patch.object(connector_module.logger, "info"), patch.object(
            connector_module.logger, "warning"
        ):
            _drive(engine, self._TICKS)
        return engine

    # Engine steps a resize may stay open. Kapsl's deadline is 30 seconds; in
    # this model a resize that only waits for batches in flight and for the
    # control poll takes well under twenty steps.
    _LONGEST_RESIZE = 40

    def _assert_safe_and_live(self, *, expect_held: bool = False, **options: Any) -> None:
        shrinks = grows = completed = held = 0
        for seed in self._SEEDS:
            engine = self._run(seed, fenced=True, **options)
            self.assertEqual(engine.violations, [], f"seed {seed}")
            # Nothing is left waiting on a resize that cannot finish: no
            # request, and no retirement that would run into Kapsl's deadline.
            self.assertTrue(engine.settled(), f"seed {seed} did not settle")
            self.assertEqual(engine.scheduler._retire_fences, {}, f"seed {seed}")
            self.assertEqual(engine.scheduler._steps_outstanding, set(), f"seed {seed}")
            self.assertFalse(engine.scheduler._step_notes_failed, f"seed {seed}")
            self.assertEqual(
                engine.memory.mapped_blocks, engine.coordinator.blocks, f"seed {seed}"
            )
            self.assertEqual(
                engine.elastic.current_blocks, engine.coordinator.blocks, f"seed {seed}"
            )
            self.assertEqual(engine.completed_requests, engine.arrivals, f"seed {seed}")
            # No shrink is left behind a request that was given tail blocks.
            self.assertLessEqual(
                engine.coordinator.longest_resize, self._LONGEST_RESIZE, f"seed {seed}"
            )
            shrinks += sum(1 for kind, _ in engine.coordinator.finished if kind == "shrink")
            grows += sum(1 for kind, _ in engine.coordinator.finished if kind == "grow")
            completed += engine.completed_requests
            held += engine.held_admissions
        # The properties above are vacuous unless the model really resized
        # under load.
        self.assertGreater(shrinks, 200)
        self.assertGreater(grows, 200)
        self.assertGreater(completed, 1000)
        if expect_held:
            # Some request did arrive while a finished one still held the tail.
            self.assertGreater(held, 0)
        else:
            self.assertEqual(held, 0)

    def test_worker_in_its_own_process_that_starts_each_step_as_late_as_it_can(
        self,
    ) -> None:
        # Blocks are freed as soon as a request ends, the hardest case: the
        # tail looks free while a scheduled batch has yet to write it.
        self._assert_safe_and_live(
            defer_block_free=False, worker="lazy", workers_always_poll=False
        )

    def test_worker_that_asks_for_resize_work_on_every_step(self) -> None:
        self._assert_safe_and_live(
            defer_block_free=False, worker="lazy", workers_always_poll=True
        )

    def test_worker_that_runs_ahead_by_a_random_amount(self) -> None:
        self._assert_safe_and_live(
            defer_block_free=False, worker="random", workers_always_poll=True
        )

    def test_worker_in_the_scheduler_process(self) -> None:
        self._assert_safe_and_live(
            defer_block_free=False, worker="inline", workers_always_poll=True
        )

    def test_vllm_that_defers_freeing_blocks_of_batches_in_flight(self) -> None:
        # Pinned vLLM does this for a connector with a consumer role. The
        # tail then stays referenced until the batch returns, and requests
        # that arrive meanwhile wait instead of landing in it.
        self._assert_safe_and_live(
            expect_held=True,
            defer_block_free=True,
            worker="lazy",
            workers_always_poll=True,
        )
        self._assert_safe_and_live(
            expect_held=True,
            defer_block_free=True,
            worker="random",
            workers_always_poll=False,
        )

    def test_the_model_catches_an_unfenced_retirement(self) -> None:
        # The same runs with the connector acknowledging at once, as it does
        # for one batch at a time: this is what dropping the startup check
        # without the fence would do. The model must see the damage,
        # otherwise the tests above prove nothing.
        failing = [
            seed
            for seed in self._SEEDS
            if self._run(
                seed,
                fenced=False,
                defer_block_free=False,
                worker="lazy",
                workers_always_poll=True,
            ).violations
        ]

        self.assertGreater(len(failing), 10)


if __name__ == "__main__":
    unittest.main()
