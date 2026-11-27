"""The shared-pool participant against a fake runtime, host and CUDA seam."""

from __future__ import annotations

import unittest
from typing import Any

from fake_sglang import pool_environ
from fake_vmm import (
    FakePoolRuntime,
    PoolShape,
    RecordingBackend,
    RecordingHost,
    pool_facts_for,
)
from pool_case import silence_logging

from kapsl_sglang_adapter.client import KapslKvControlError
from kapsl_sglang_adapter.config import load_config
from kapsl_sglang_adapter.pool_participant import SharedPoolParticipant
from kapsl_sglang_adapter.shared_pool import SharedPoolError

VERSION = "0.3.0"
SGLANG = "0.5.23"


class ParticipantTestCase(unittest.TestCase):
    def setUp(self) -> None:
        silence_logging(self)
        self.events: list[str] = []
        self.runtime = FakePoolRuntime(self.events)
        self.addCleanup(self.runtime.close)
        self.backend = RecordingBackend(self.events)
        self.now = 0.0
        self.sleeps: list[float] = []
        self.participant = SharedPoolParticipant(
            load_config(pool_environ()),
            self.runtime,
            adapter_version=VERSION,
            clock=lambda: self.now,
            heartbeat_interval_seconds=3600,
            sleep=self.sleeps.append,
        )
        self.addCleanup(self.participant.shutdown)

    def attach(self) -> None:
        self.participant.attach_pool(
            pool_facts_for(PoolShape(), 255), self.backend, sglang_version=SGLANG
        )

    def serve(self) -> RecordingHost:
        """An attached, activated participant with the startup traffic cleared."""

        self.attach()
        host = RecordingHost(63, self.events)
        self.participant.activate(host)
        self.runtime.calls.clear()
        self.events.clear()
        return host

    def tick(self, seconds: float = 1.0) -> None:
        self.now += seconds
        self.participant.service_resize()

    def admit(self, request_id: str) -> bool:
        return self.participant.admit(request_id, input_tokens=8, max_new_tokens=8)


class StartupTests(ParticipantTestCase):
    def test_attach_registers_then_maps_then_closes_the_descriptors(self) -> None:
        self.attach()
        self.assertEqual(self.runtime.operations(), ["register"])
        self.assertEqual(len(self.backend.mapped), 8)
        self.assertEqual(self.runtime.open_descriptors(), [])
        self.assertEqual(self.participant.startup_tokens(), 63)
        self.assertEqual(self.participant.minimum_tokens(), 31)
        # Not serving yet: the runtime has not been told the pool is attached.
        with self.assertRaises(KapslKvControlError) as raised:
            self.admit("rid-1")
        self.assertEqual(raised.exception.kind, "not_registered")

    def test_activation_reports_the_attachment_then_activates(self) -> None:
        self.attach()
        self.participant.activate(RecordingHost(63))
        self.assertEqual(self.runtime.operations(), ["register", "attach", "activate"])
        self.assertTrue(self.admit("rid-1"))
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.participant.activate(RecordingHost(63))

    def test_a_pool_sglang_finalized_at_another_size_is_not_activated(self) -> None:
        self.attach()
        with self.assertRaisesRegex(
            SharedPoolError, "finalized its KV pool at 255 tokens.*mapped 63"
        ):
            self.participant.activate(RecordingHost(255))
        self.assertEqual(self.runtime.operations(), ["register"])

    def test_a_refused_attachment_or_activation_stops_the_start(self) -> None:
        for operation in ("attach", "activate"):
            with self.subTest(operation):
                self.setUp()
                self.attach()
                self.runtime.errors[operation] = KapslKvControlError(
                    "not allowlisted", kind="invalid_capabilities"
                )
                with self.assertRaises(KapslKvControlError):
                    self.participant.activate(RecordingHost(63))
                with self.assertRaises(KapslKvControlError):
                    self.admit("rid-1")

    def test_the_pool_must_be_attached_before_anything_else(self) -> None:
        with self.assertRaisesRegex(SharedPoolError, "has not created"):
            self.participant.startup_tokens()
        with self.assertRaisesRegex(SharedPoolError, "has not created"):
            self.participant.activate(RecordingHost(63))
        with self.assertRaisesRegex(RuntimeError, "attach_pool"):
            self.participant.register(object())

    def test_a_second_pool_is_refused(self) -> None:
        self.attach()
        with self.assertRaisesRegex(SharedPoolError, "second VMM-backed KV pool"):
            self.attach()
        self.assertEqual(self.runtime.operations(), ["register"])


class ResizeTests(ParticipantTestCase):
    def test_growth_is_two_acknowledged_stages_in_one_step(self) -> None:
        host = self.serve()
        self.runtime.request_resize(128)

        self.tick()

        self.assertEqual(
            self.events,
            [f"map:{stripe}:2048:2048" for stripe in range(4)]
            + [
                "control:ack:map_workers",
                "publish:127",
                "control:ack:activate_scheduler",
            ],
        )
        self.assertEqual(host.tokens(), 127)
        # Scheduler poll, worker poll and acknowledgement, scheduler poll and
        # acknowledgement: five round trips for a growth.
        self.assertEqual(
            self.runtime.operations(),
            ["resize_poll", "resize_poll", "resize_ack", "resize_poll", "resize_ack"],
        )
        polls = [call[1] for call in self.runtime.calls if call[0] == "resize_poll"]
        self.assertEqual(
            [poll["actor"]["role"] for poll in polls],
            ["scheduler", "worker", "scheduler"],
        )
        self.assertEqual(self.runtime.open_descriptors(), [])

    def test_shrink_is_retired_then_unmapped(self) -> None:
        host = self.serve()
        self.runtime.request_resize(32)

        self.tick()

        self.assertEqual(
            self.events,
            ["retire:31", "control:ack:retire_scheduler", "synchronize"]
            + [f"unmap:{stripe}:1024:1024" for stripe in range(4)]
            + ["control:ack:unmap_workers"],
        )
        self.assertEqual(host.tokens(), 31)
        self.assertEqual(
            self.runtime.operations(),
            ["resize_poll", "resize_ack", "resize_poll", "resize_ack"],
        )

    def test_each_actor_reports_the_generation_it_applied(self) -> None:
        self.serve()
        self.runtime.request_resize(128)
        self.tick()
        self.runtime.request_resize(64)
        self.tick()
        self.runtime.calls.clear()
        self.tick()
        self.assertEqual(
            self.runtime.calls,
            [
                (
                    "resize_poll",
                    {
                        "participant_epoch": 7,
                        "actor": {"role": "scheduler"},
                        "applied_generation": 2,
                    },
                )
            ],
        )

    def test_a_deferred_shrink_holds_new_admissions_until_it_is_applied(self) -> None:
        host = self.serve()
        host.busy = "SGLang is not idle"
        self.runtime.request_resize(32)
        self.tick()
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        self.assertFalse(self.admit("rid-1"))
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        # The operation is kept, not polled for again.
        self.tick()
        self.assertEqual(self.runtime.operations(), ["resize_poll"])

        host.busy = None
        self.tick()
        self.assertEqual(host.tokens(), 31)
        self.assertTrue(self.admit("rid-1"))

    def test_an_interrupted_resize_resumes_at_its_next_stage(self) -> None:
        host = self.serve()
        self.runtime.request_resize(128)
        self.participant._worker_turn()
        self.assertEqual(self.participant.pool.mapped_blocks, 128)
        self.assertEqual(host.tokens(), 63)

        self.tick()
        self.assertEqual(host.tokens(), 127)

        self.runtime.request_resize(64)
        operation, _ = self.participant._poll_scheduler()
        self.participant._apply_scheduler_stage(operation)
        self.assertEqual(self.participant.pool.mapped_blocks, 128)
        self.tick()
        self.assertEqual(self.participant.pool.mapped_blocks, 64)
        self.assertEqual(self.runtime.completed, [(64, 128), (128, 64)])


class ControlFailureTests(ParticipantTestCase):
    def test_an_unanswered_poll_is_tolerated_within_the_lease_grace(self) -> None:
        self.serve()
        # Lease TTL 30 s: authority survives 20 s without an answer.
        self.runtime.fail_polls = 3
        for _ in range(3):
            self.tick(seconds=6.0)
        self.assertIsNone(self.participant.failure)
        self.tick(seconds=6.0)
        self.assertIsNone(self.participant.failure)

        self.runtime.fail_polls = 10
        for _ in range(3):
            self.tick(seconds=6.0)
        with self.assertRaises(KapslKvControlError) as raised:
            self.tick(seconds=6.0)
        self.assertEqual(raised.exception.kind, "transport")
        self.assertIsNotNone(self.participant.failure)
        with self.assertRaises(KapslKvControlError):
            self.admit("rid-1")

    def test_a_runtime_rejection_of_a_poll_is_fatal_at_once(self) -> None:
        self.serve()
        self.runtime.errors["resize_poll"] = KapslKvControlError(
            "live KV resize generation 3 is fenced", kind="invalid_request"
        )
        with self.assertRaises(KapslKvControlError):
            self.tick()
        self.assertIsNotNone(self.participant.failure)

    def test_a_lost_acknowledgement_reply_is_not_repeated(self) -> None:
        host = self.serve()
        self.runtime.lose_ack_replies = 1
        self.runtime.request_resize(128)

        self.tick()

        # The acknowledgement landed; only its reply was lost. The runtime no
        # longer offers the stage, so it is not acknowledged twice.
        self.assertEqual(host.tokens(), 127)
        self.assertEqual(self.sleeps, [0.05])
        self.assertEqual(self.runtime.operations().count("resize_ack"), 2)
        self.assertIsNone(self.participant.failure)
        self.assertEqual(self.runtime.open_descriptors(), [])

    def test_an_acknowledgement_that_never_arrived_is_repeated(self) -> None:
        host = self.serve()
        failures = [KapslKvControlError("connection refused", kind="transport")]
        original = self.runtime.ack_resize

        def flaky(request: dict[str, Any]) -> None:
            if failures:
                raise failures.pop()
            original(request)

        self.runtime.ack_resize = flaky
        self.runtime.request_resize(128)
        self.tick()
        self.assertEqual(host.tokens(), 127)
        self.assertEqual(self.runtime.completed, [(64, 128)])

    def test_an_acknowledgement_that_cannot_be_settled_is_fatal(self) -> None:
        self.serve()
        self.runtime.errors["resize_ack"] = KapslKvControlError(
            "connection refused", kind="transport"
        )
        self.runtime.request_resize(128)
        with self.assertRaises(KapslKvControlError):
            self.tick()
        self.assertEqual(self.sleeps, [0.05, 0.25, 1.0])
        self.assertIsNotNone(self.participant.failure)

    def test_operations_for_another_epoch_or_stage_are_refused(self) -> None:
        def another_epoch(operation: dict[str, Any]) -> None:
            operation["participant_epoch"] = 8

        cases = {
            "another epoch": (another_epoch, 128, "addressed to epoch 8, not 7"),
        }
        for name, (mutate, target, needle) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.serve()
                self.runtime.mutate_operation = mutate
                self.runtime.request_resize(target)
                with self.assertRaisesRegex(SharedPoolError, needle):
                    self.tick()
                self.assertEqual(self.backend.events, [])
                self.assertIsNotNone(self.participant.failure)

    def test_stages_out_of_order_are_refused(self) -> None:
        host = self.serve()
        self.runtime.request_resize(128)
        # The runtime offers the activation although nothing was mapped.
        self.runtime.resize["stage"] = "activate_scheduler"
        with self.assertRaisesRegex(SharedPoolError, "activate 128 blocks while 64"):
            self.tick()
        self.assertEqual(host.tokens(), 63)

        self.setUp()
        host = self.serve()
        self.runtime.request_resize(32)
        # The runtime offers the unmap although the tail is still in service.
        self.runtime.resize["stage"] = "unmap_workers"
        with self.assertRaisesRegex(
            SharedPoolError, "unmap down to 32 blocks while 64"
        ):
            self.tick()
        self.assertEqual(self.backend.events, [])

    def test_a_host_failure_while_publishing_is_latched(self) -> None:
        host = self.serve()
        host.error = RuntimeError("capacity fields disagree")
        self.runtime.request_resize(128)
        with self.assertRaisesRegex(RuntimeError, "capacity fields disagree"):
            self.tick()
        self.assertNotIn("control:ack:activate_scheduler", self.events)
        self.assertEqual(self.participant.failure.kind, "adapter")

    def test_nothing_is_polled_after_authority_is_lost(self) -> None:
        self.serve()
        self.participant.fail(KapslKvControlError("heartbeat lost", kind="transport"))
        with self.assertRaises(KapslKvControlError):
            self.tick()
        self.assertEqual(self.runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
