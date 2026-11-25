from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from kapsl_vllm_connector.client import KapslKvControlError
from kapsl_vllm_connector.connector import KapslConnectorV1
from kapsl_vllm_connector.shared_pool import SharedPoolTailBusyError


def _transport() -> KapslKvControlError:
    return KapslKvControlError("timed out", kind="transport")


def _operation(generation: int, stage: str, target: int) -> dict[str, Any]:
    return {
        "participant_epoch": 7,
        "resize_generation": generation,
        "binding_id": "binding-0",
        "stage": stage,
        "from_block_count": 8,
        "target_block_count": target,
    }


class FakeElastic:
    def __init__(self, events: list[Any], *, busy: int = 0) -> None:
        self.events = events
        self.busy = busy
        self.minimum_blocks = 2
        self.initial_blocks = 8
        self.current_blocks = 8

    def apply(self, target: int) -> None:
        if self.busy:
            self.busy -= 1
            raise SharedPoolTailBusyError("1 of 4 tail blocks are still referenced")
        self.events.append(("apply", target))
        self.current_blocks = target


class FakeClient:
    """Scripted coordinator: each list holds one reply per expected call."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.polls: list[Any] = []
        self.acks: list[Any] = []
        self.heartbeats: list[Any] = []

    def reserve(self, request: dict[str, Any]) -> dict[str, Any]:
        self.events.append("reserve")
        return {"lease_id": f"lease-{request['sequence']['request_id']}"}

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        self.events.append(("commit", lease_id))

    def release(self, lease_id: str) -> None:
        self.events.append(("release", lease_id))

    def poll_resize_state_with_handles(self, request: dict[str, Any]):
        self.events.append(("poll", request["actor"]["role"]))
        reply = self.polls.pop(0) if self.polls else ([], [], False)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def ack_resize(self, request: dict[str, Any]) -> None:
        self.events.append(
            ("ack", request["stage"], request["resize_generation"])
        )
        reply = self.acks.pop(0) if self.acks else None
        if isinstance(reply, Exception):
            raise reply

    def heartbeat(self) -> None:
        self.events.append("heartbeat")
        reply = self.heartbeats.pop(0) if self.heartbeats else None
        if isinstance(reply, Exception):
            raise reply


class ScriptedStop:
    """Stands in for the heartbeat stop event: runs a fixed number of ticks."""

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.waits: list[float] = []

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.ticks -= 1
        return self.ticks < 0


def _scheduler(events: list[Any], *, live_resize: bool = True) -> KapslConnectorV1:
    connector = KapslConnectorV1.__new__(KapslConnectorV1)
    connector._is_scheduler = True
    connector._live_resize = live_resize
    connector._mode = "shared_pool"
    connector._shared_active = True
    connector._participant_epoch = 7
    connector._participant_id = "vllm:test"
    connector._group_ids = ["vllm.group.0"]
    connector._lease_ttl_ms = 30_000
    connector._leases = {}
    connector._lease_lock = threading.RLock()
    connector._control_failure = None
    connector._resize_lock = threading.RLock()
    connector._resize_pending = False
    connector._resize_applied_generation = 0
    connector._pending_scheduler_operations = []
    connector._deferred_retire_generation = 0
    connector._idle_step_completed = False
    connector._last_lease_released_at = float("-inf")
    connector._stall_warned_at = float("-inf")
    connector._elastic_block_pool = FakeElastic(events) if live_resize else None
    connector._client = FakeClient(events)
    return connector


def _request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_tokens=16),
        priority=0,
    )


class IdleKeepAliveTests(unittest.TestCase):
    def test_scheduler_stays_awake_while_a_shrink_can_still_arrive(self) -> None:
        connector = _scheduler([])
        elastic = connector._elastic_block_pool

        with patch("kapsl_vllm_connector.connector.time.sleep") as sleep:
            # Startup capacity lies above the certified minimum, and Kapsl
            # shrinks an idle pool below startup: the loop must stay alive.
            self.assertTrue(connector.has_pending_push_work())
            elastic.current_blocks = 12
            self.assertTrue(connector.has_pending_push_work())

            elastic.current_blocks = elastic.minimum_blocks
            self.assertFalse(connector.has_pending_push_work())

            connector._resize_pending = True
            self.assertTrue(connector.has_pending_push_work())
            connector._resize_pending = False
            connector._pending_scheduler_operations = [_operation(2, "retire_scheduler", 4)]
            self.assertTrue(connector.has_pending_push_work())
        sleep.assert_not_called()

    def test_idle_loop_pauses_once_per_completed_step(self) -> None:
        connector = _scheduler([])

        with patch("kapsl_vllm_connector.connector.time.sleep") as sleep:
            # vLLM asks three times per loop iteration; only the first call
            # after a completed step may pause.
            connector.build_connector_meta(SimpleNamespace())
            self.assertTrue(connector.has_pending_push_work())
            self.assertTrue(connector.has_pending_push_work())
            self.assertTrue(connector.has_pending_push_work())
            self.assertEqual(sleep.call_count, 1)

            connector.build_connector_meta(SimpleNamespace())
            self.assertTrue(connector.has_pending_push_work())
            self.assertEqual(sleep.call_count, 2)

    def test_idle_loop_never_pauses_while_a_request_holds_a_lease(self) -> None:
        connector = _scheduler([])
        connector._leases = {"request-1": {"lease_id": "lease-1"}}

        with patch("kapsl_vllm_connector.connector.time.sleep") as sleep:
            connector.build_connector_meta(SimpleNamespace())
            self.assertTrue(connector.has_pending_push_work())
        sleep.assert_not_called()

    def test_pending_work_is_never_paced(self) -> None:
        connector = _scheduler([])
        connector._resize_pending = True

        with patch("kapsl_vllm_connector.connector.time.sleep") as sleep:
            connector.build_connector_meta(SimpleNamespace())
            self.assertTrue(connector.has_pending_push_work())
        sleep.assert_not_called()

    def test_scheduler_stays_awake_briefly_after_its_last_request(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        elastic = connector._elastic_block_pool
        # At the certified minimum nothing can shrink, so only a grow that
        # Kapsl requests as the load ends can still need this scheduler.
        elastic.current_blocks = elastic.minimum_blocks
        connector.on_new_request(_request("request-1"))
        clock = [100.0]

        with patch("kapsl_vllm_connector.connector.time.sleep"), patch(
            "kapsl_vllm_connector.connector.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.assertFalse(connector.has_pending_push_work())
            connector.request_finished(_request("request-1"), [])
            self.assertTrue(connector.has_pending_push_work())
            clock[0] = 104.9
            self.assertTrue(connector.has_pending_push_work())
            clock[0] = 105.1
            self.assertFalse(connector.has_pending_push_work())

    def test_fixed_pool_scheduler_reports_no_push_work(self) -> None:
        connector = _scheduler([], live_resize=False)
        connector._last_lease_released_at = float("inf")
        self.assertFalse(connector.has_pending_push_work())


class ShrinkOrderingTests(unittest.TestCase):
    def test_first_request_after_idle_retires_the_tail_before_allocation(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.polls = [
            ([_operation(3, "retire_scheduler", 4)], [], True)
        ]

        connector.on_new_request(_request("request-1"))

        # on_new_request runs before vLLM allocates any block for the request.
        self.assertEqual(
            events,
            [
                "reserve",
                ("poll", "scheduler"),
                ("apply", 4),
                ("ack", "retire_scheduler", 3),
            ],
        )
        self.assertEqual(connector._resize_applied_generation, 3)
        self.assertEqual(connector._pending_scheduler_operations, [])

    def test_overlapping_requests_do_not_poll_again(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)

        connector.on_new_request(_request("request-1"))
        events.clear()
        # A shrink is admitted only while no lease is held, so a request that
        # arrives while another one runs has nothing to settle.
        connector.on_new_request(_request("request-2"))

        self.assertEqual(events, ["reserve"])

    def test_fixed_pool_requests_never_poll_for_resizes(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, live_resize=False)

        connector.on_new_request(_request("request-1"))

        self.assertEqual(events, ["reserve"])

    def test_unanswered_poll_does_not_fail_the_request(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.polls = [_transport()]

        connector.on_new_request(_request("request-1"))

        self.assertEqual(events, ["reserve", ("poll", "scheduler")])
        self.assertIn("request-1", connector._leases)

    def test_fenced_resize_still_fails_closed(self) -> None:
        connector = _scheduler([])
        connector._client.polls = [
            KapslKvControlError("generation 3 is fenced", kind="invalid_request")
        ]

        with self.assertRaisesRegex(KapslKvControlError, "fenced"):
            connector.on_new_request(_request("request-1"))

    def test_referenced_tail_defers_the_shrink_without_acknowledging(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._elastic_block_pool.busy = 2
        operation = _operation(3, "retire_scheduler", 4)
        connector._pending_scheduler_operations = [operation]

        with self.assertLogs("vllm.kapsl_vllm_connector.connector", "WARNING") as logs:
            connector._apply_scheduler_resizes()
            connector._apply_scheduler_resizes()

        # One warning per generation, no acknowledgement, operation kept.
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(events, [])
        self.assertEqual(connector._pending_scheduler_operations, [operation])
        self.assertEqual(connector._resize_applied_generation, 0)
        self.assertTrue(connector.has_pending_push_work())

        connector._apply_scheduler_resizes()

        self.assertEqual(events, [("apply", 4), ("ack", "retire_scheduler", 3)])
        self.assertEqual(connector._pending_scheduler_operations, [])
        self.assertEqual(connector._resize_applied_generation, 3)

    def test_operation_stays_queued_until_it_is_acknowledged(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        operation = _operation(3, "activate_scheduler", 12)
        connector._pending_scheduler_operations = [operation]
        queued_during_ack: list[list[dict[str, Any]]] = []

        def ack(request: dict[str, Any]) -> None:
            # The background poll dedupes against this queue. Had the
            # operation been removed already, a poll reply computed before
            # the acknowledgement would queue a second copy.
            queued_during_ack.append(list(connector._pending_scheduler_operations))

        connector._client.ack_resize = ack
        connector._apply_scheduler_resizes()

        self.assertEqual(queued_during_ack, [[operation]])
        self.assertEqual(connector._pending_scheduler_operations, [])


class ResizeAcknowledgementTests(unittest.TestCase):
    def _acknowledge(self, connector: KapslConnectorV1) -> None:
        connector._acknowledge_resize(
            {
                "participant_epoch": 7,
                "resize_generation": 3,
                "binding_id": "binding-0",
                "stage": "retire_scheduler",
                "applied_block_count": 4,
            },
            {"role": "scheduler"},
        )

    def test_lost_reply_is_not_repeated_once_the_stage_advanced(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.acks = [_transport()]
        # The coordinator no longer offers the operation: the first
        # acknowledgement landed and only its reply was lost.
        connector._client.polls = [([], [], True)]

        with patch("kapsl_vllm_connector.connector.time.sleep"):
            self._acknowledge(connector)

        self.assertEqual(
            events, [("ack", "retire_scheduler", 3), ("poll", "scheduler")]
        )

    def test_acknowledgement_is_repeated_while_the_coordinator_still_waits(
        self,
    ) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.acks = [_transport(), None]
        connector._client.polls = [
            ([_operation(3, "retire_scheduler", 4)], [], True)
        ]

        with patch("kapsl_vllm_connector.connector.time.sleep"):
            self._acknowledge(connector)

        self.assertEqual(
            events,
            [
                ("ack", "retire_scheduler", 3),
                ("poll", "scheduler"),
                ("ack", "retire_scheduler", 3),
            ],
        )

    def test_unknown_outcome_is_never_acknowledged_blindly(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.acks = [_transport()]
        connector._client.polls = [_transport(), _transport(), _transport()]

        with patch("kapsl_vllm_connector.connector.time.sleep"):
            with self.assertRaises(KapslKvControlError) as caught:
                self._acknowledge(connector)

        self.assertEqual(caught.exception.kind, "transport")
        self.assertEqual(
            [event for event in events if event[0] == "ack"],
            [("ack", "retire_scheduler", 3)],
        )

    def test_coordinator_rejection_is_reported_immediately(self) -> None:
        connector = _scheduler([])
        connector._client.acks = [
            KapslKvControlError("does not match", kind="invalid_request")
        ]

        with self.assertRaisesRegex(KapslKvControlError, "does not match"):
            self._acknowledge(connector)


class HeartbeatToleranceTests(unittest.TestCase):
    def _run(self, connector: KapslConnectorV1, ticks: int) -> ScriptedStop:
        stop = ScriptedStop(ticks)
        connector._heartbeat_stop = stop
        connector._heartbeat_loop()
        return stop

    def test_unanswered_heartbeat_is_retried_within_the_lease_ttl(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, live_resize=False)
        connector._leases = {"request-1": {"lease_id": "lease-1"}}
        connector._client.heartbeats = [_transport(), None]

        stop = self._run(connector, 2)

        self.assertIsNone(connector._control_failure)
        self.assertEqual(events, ["heartbeat", "heartbeat"])
        # The repeat comes sooner than the ordinary ten-second interval.
        self.assertEqual(stop.waits[:2], [10.0, 1.0])

    def test_unanswered_heartbeat_fails_closed_before_leases_expire(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, live_resize=False)
        connector._leases = {"request-1": {"lease_id": "lease-1"}}
        connector._client.heartbeats = [_transport(), _transport()]
        clock = iter([0.0, 12.0, 12.0, 21.0, 21.0])

        with patch(
            "kapsl_vllm_connector.connector.time.monotonic",
            side_effect=lambda: next(clock),
        ):
            self._run(connector, 5)

        self.assertEqual(events, ["heartbeat", "heartbeat"])
        self.assertIsNotNone(connector._control_failure)
        with self.assertRaisesRegex(KapslKvControlError, "heartbeat failed"):
            connector._raise_if_control_failed()

    def test_coordinator_rejection_fails_closed_at_once(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events, live_resize=False)
        connector._leases = {"request-1": {"lease_id": "lease-1"}}
        connector._client.heartbeats = [
            KapslKvControlError("participant is not registered", kind="not_found")
        ]

        self._run(connector, 3)

        self.assertEqual(events, ["heartbeat"])
        self.assertEqual(connector._control_failure.kind, "not_found")

    def test_unanswered_resize_poll_is_retried(self) -> None:
        events: list[Any] = []
        connector = _scheduler(events)
        connector._client.polls = [
            _transport(),
            ([_operation(3, "retire_scheduler", 4)], [], True),
        ]

        self._run(connector, 2)

        self.assertIsNone(connector._control_failure)
        self.assertEqual(
            [int(op["resize_generation"]) for op in connector._pending_scheduler_operations],
            [3],
        )

    def test_fenced_resize_poll_fails_closed_at_once(self) -> None:
        connector = _scheduler([])
        connector._client.polls = [
            KapslKvControlError("generation 3 is fenced", kind="invalid_request")
        ]

        self._run(connector, 3)

        self.assertEqual(connector._control_failure.kind, "invalid_request")


class WorkerResizePollTests(unittest.TestCase):
    def _worker(self, events: list[Any]) -> KapslConnectorV1:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = False
        connector._live_resize = True
        connector._participant_epoch = 7
        connector._shared_shard = {
            "tensor_parallel_rank": 0,
            "tensor_parallel_world_size": 1,
            "pipeline_parallel_rank": 0,
            "pipeline_parallel_world_size": 1,
        }
        connector._resize_applied_generation = 0
        connector._stall_warned_at = float("-inf")
        connector._client = FakeClient(events)
        connector._shared_pool_hook = SimpleNamespace(
            apply_worker_resize=lambda operation, handles: events.append(
                ("map", operation["target_block_count"])
            )
        )
        return connector

    def test_unanswered_poll_lets_the_forward_proceed(self) -> None:
        events: list[Any] = []
        connector = self._worker(events)
        connector._client.polls = [_transport()]

        with self.assertLogs("vllm.kapsl_vllm_connector.connector", "WARNING"):
            connector._apply_worker_resizes()

        self.assertEqual(events, [("poll", "worker")])
        self.assertEqual(connector._resize_applied_generation, 0)

    def test_worker_applies_then_acknowledges(self) -> None:
        events: list[Any] = []
        connector = self._worker(events)
        connector._client.polls = [([_operation(3, "map_workers", 12)], [], True)]

        connector._apply_worker_resizes()

        self.assertEqual(
            events, [("poll", "worker"), ("map", 12), ("ack", "map_workers", 3)]
        )
        self.assertEqual(connector._resize_applied_generation, 3)

    def test_coordinator_rejection_stops_the_forward(self) -> None:
        connector = self._worker([])
        connector._client.polls = [
            KapslKvControlError("generation 3 is fenced", kind="invalid_request")
        ]

        with self.assertRaisesRegex(KapslKvControlError, "fenced"):
            connector._apply_worker_resizes()


if __name__ == "__main__":
    unittest.main()
