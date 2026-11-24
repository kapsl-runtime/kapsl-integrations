from __future__ import annotations

import json
import logging
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from fake_sglang import ENVIRON, FakeClient

from kapsl_sglang_adapter.capacity import SchedulerFacts, UnsupportedProfileError
from kapsl_sglang_adapter.client import KapslKvControlError
from kapsl_sglang_adapter.config import load_config
from kapsl_sglang_adapter.participant import KvParticipant

FIXTURES = Path(__file__).with_name("fixtures")

FACTS = SchedulerFacts(
    sglang_version="0.5.23",
    device="cuda",
    pool_class="MHATokenToKVPool",
    page_size=1,
    max_total_num_tokens=8192,
    kv_pool_bytes=8192 * 131072,
    post_capture_sizing=False,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    disaggregation_mode="null",
    speculative_decoding=False,
)

DENIED = KapslKvControlError("device KV budget exhausted", kind="capacity_exhausted")
UNREACHABLE = KapslKvControlError("connection refused", kind="transport")


def setUpModule() -> None:
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class ParticipantTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.clock = Clock()
        self.participant = KvParticipant(
            load_config({**ENVIRON, "KAPSL_SGLANG_DENIAL_BACKOFF_MS": "100"}),
            self.client,
            clock=self.clock,
        )

    def register(self) -> None:
        self.participant.register(FACTS)
        self.client.calls.clear()

    def admit(self, request_id: str = "rid-1", **overrides) -> bool:
        arguments = {"input_tokens": 100, "max_new_tokens": 28, **overrides}
        return self.participant.admit(request_id, **arguments)


class RegistrationTests(ParticipantTestCase):
    def test_registers_the_documented_opaque_registration(self) -> None:
        self.participant.register(FACTS)
        fixture = json.loads(
            (FIXTURES / "sglang_opaque_registration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.client.calls, [("register", fixture)])

    def test_an_unsupported_scheduler_never_reaches_the_runtime(self) -> None:
        with self.assertRaises(UnsupportedProfileError):
            self.participant.register(replace(FACTS, tensor_parallel_size=2))
        self.assertEqual(self.client.calls, [])
        with self.assertRaises(KapslKvControlError) as raised:
            self.admit()
        self.assertEqual(raised.exception.kind, "not_registered")

    def test_a_rejected_registration_aborts_startup(self) -> None:
        self.client.errors["register"] = KapslKvControlError(
            "runtime has no bounded external-lease authority",
            kind="invalid_capabilities",
        )
        with self.assertRaises(KapslKvControlError):
            self.participant.register(FACTS)
        with self.assertRaises(KapslKvControlError):
            self.admit()

    def test_registering_twice_is_a_programming_error(self) -> None:
        self.register()
        with self.assertRaises(RuntimeError):
            self.participant.register(FACTS)


class AdmissionTests(ParticipantTestCase):
    def test_admission_reserves_prompt_plus_output_cap(self) -> None:
        self.register()
        self.assertTrue(self.admit(priority=3))
        self.assertEqual(
            self.client.calls,
            [
                (
                    "reserve",
                    {
                        "sequence": {"request_id": "rid-1", "sequence_id": "rid-1"},
                        "groups": [{"group_id": "sglang.kv.0", "token_capacity": 128}],
                        "priority": 3,
                        "ttl_ms": 30_000,
                    },
                )
            ],
        )
        self.assertTrue(self.participant.holds("rid-1"))
        self.assertEqual(self.participant.live_leases(), 1)

    def test_a_held_lease_is_not_reserved_again(self) -> None:
        self.register()
        self.assertTrue(self.admit())
        self.assertTrue(self.admit())
        self.assertEqual(self.client.operations(), ["reserve"])

    def test_reservation_is_clamped_to_the_pool_and_wire_priority(self) -> None:
        self.register()
        self.admit(max_new_tokens=1 << 30, priority=1 << 62)
        self.admit("rid-2", priority=-(1 << 62))
        first, second = (call[1] for call in self.client.calls)
        self.assertEqual(first["groups"][0]["token_capacity"], 8192)
        self.assertEqual(first["priority"], 2**31 - 1)
        self.assertEqual(second["priority"], -(2**31))

    def test_capacity_denial_is_not_an_error_and_backs_off(self) -> None:
        self.register()
        self.client.errors["reserve"] = DENIED
        self.assertFalse(self.admit())
        self.assertFalse(self.participant.holds("rid-1"))
        self.assertIsNone(self.participant.failure)

        # Within the backoff no request asks the runtime again.
        self.clock.now += 0.099
        self.assertFalse(self.admit())
        self.assertFalse(self.admit("rid-2"))
        self.assertEqual(self.client.operations(), ["reserve"])

        self.clock.now += 0.002
        del self.client.errors["reserve"]
        self.assertTrue(self.admit())
        self.assertEqual(self.client.operations(), ["reserve", "reserve"])

    def test_returning_a_lease_ends_the_denial_backoff(self) -> None:
        self.register()
        self.assertTrue(self.admit("rid-1"))
        self.client.errors["reserve"] = DENIED
        self.assertFalse(self.admit("rid-2"))
        del self.client.errors["reserve"]

        self.participant.release("rid-1", computed_tokens=110)
        self.assertTrue(self.admit("rid-2"))

    def test_any_other_reserve_failure_fails_closed(self) -> None:
        for kind in ("transport", "protocol", "not_found", "invalid_request", None):
            with self.subTest(kind):
                self.setUp()
                self.register()
                self.client.errors["reserve"] = KapslKvControlError("boom", kind=kind)
                with self.assertRaises(KapslKvControlError):
                    self.admit()
                self.assertIsNotNone(self.participant.failure)

                # Authority is latched: nothing is admitted and the runtime is
                # not asked again, even if it has recovered.
                del self.client.errors["reserve"]
                with self.assertRaises(KapslKvControlError) as raised:
                    self.admit("rid-2")
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(self.client.operations(), ["reserve"])


class ReleaseTests(ParticipantTestCase):
    def test_release_commits_computed_tokens_then_returns_the_lease(self) -> None:
        self.register()
        self.admit()
        self.client.calls.clear()
        self.participant.release("rid-1", computed_tokens=117)
        self.assertEqual(
            self.client.calls,
            [
                ("commit", "kapsl-kv-0000000000000001", 117),
                ("release", "kapsl-kv-0000000000000001"),
            ],
        )
        self.assertEqual(self.participant.live_leases(), 0)

    def test_cancel_returns_an_unused_lease_without_a_commit(self) -> None:
        self.register()
        self.admit()
        self.client.calls.clear()
        self.participant.cancel("rid-1")
        self.assertEqual(self.client.calls, [("release", "kapsl-kv-0000000000000001")])

    def test_releasing_an_unknown_request_is_a_no_op(self) -> None:
        self.register()
        self.participant.release("never-admitted", computed_tokens=5)
        self.participant.cancel("never-admitted")
        self.assertEqual(self.client.calls, [])

    def test_release_never_raises_but_latches_the_failure(self) -> None:
        self.register()
        self.admit()
        self.client.calls.clear()
        self.client.errors["commit"] = UNREACHABLE
        self.client.errors["release"] = UNREACHABLE

        self.participant.release("rid-1", computed_tokens=117)

        # The release is still attempted after the commit failed.
        self.assertEqual(self.client.operations(), ["commit", "release"])
        self.assertIs(self.participant.failure, UNREACHABLE)
        with self.assertRaises(KapslKvControlError):
            self.admit("rid-2")

    def test_an_allocation_without_a_lease_is_refused_and_latched(self) -> None:
        self.register()
        self.admit("rid-1")
        self.participant.require_lease("rid-1")
        with self.assertRaises(KapslKvControlError) as raised:
            self.participant.require_lease("rid-2")
        self.assertEqual(raised.exception.kind, "ungoverned_allocation")
        with self.assertRaises(KapslKvControlError):
            self.participant.require_lease("rid-1")

    def test_shutdown_returns_every_live_lease_and_is_idempotent(self) -> None:
        self.register()
        self.admit("rid-1")
        self.admit("rid-2")
        self.client.calls.clear()
        self.client.errors["release"] = UNREACHABLE

        self.participant.shutdown()
        self.participant.shutdown()

        self.assertEqual(
            self.client.calls,
            [
                ("release", "kapsl-kv-0000000000000001"),
                ("release", "kapsl-kv-0000000000000002"),
            ],
        )
        with self.assertRaises(KapslKvControlError):
            self.admit("rid-3")


class HeartbeatTests(ParticipantTestCase):
    def test_idle_participants_send_no_heartbeat(self) -> None:
        self.register()
        self.assertTrue(self.participant.heartbeat_once())
        self.assertEqual(self.client.calls, [])

    def test_one_heartbeat_renews_all_live_leases(self) -> None:
        self.register()
        self.admit("rid-1")
        self.admit("rid-2")
        self.client.calls.clear()
        self.assertTrue(self.participant.heartbeat_once())
        self.assertEqual(self.client.calls, [("heartbeat",)])

    def test_losing_the_runtime_with_live_leases_fails_closed(self) -> None:
        self.register()
        self.admit()
        self.client.errors["heartbeat"] = UNREACHABLE
        self.assertFalse(self.participant.heartbeat_once())
        self.assertIs(self.participant.failure, UNREACHABLE)
        with self.assertRaises(KapslKvControlError) as raised:
            self.admit("rid-2")
        self.assertEqual(raised.exception.kind, "transport")
        self.assertIs(raised.exception.__cause__, UNREACHABLE)
        # A latched participant stops renewing.
        self.client.calls.clear()
        self.assertFalse(self.participant.heartbeat_once())
        self.assertEqual(self.client.calls, [])

    def test_a_failed_heartbeat_with_nothing_left_to_renew_is_ignored(self) -> None:
        self.register()
        self.admit()
        participant = self.participant

        class DrainingClient(FakeClient):
            def heartbeat(self) -> None:
                # The scheduler returns the last lease while a renewal is in flight.
                participant.release("rid-1", computed_tokens=1)
                raise UNREACHABLE

        draining = DrainingClient()
        participant._client = draining
        self.assertTrue(participant.heartbeat_once())
        self.assertIsNone(participant.failure)

    def test_background_thread_renews_until_shutdown(self) -> None:
        renewed = threading.Event()

        class SignallingClient(FakeClient):
            def heartbeat(self) -> None:
                super().heartbeat()
                renewed.set()

        client = SignallingClient()
        participant = KvParticipant(
            load_config(ENVIRON), client, heartbeat_interval_seconds=0.01
        )
        participant.register(FACTS)
        participant.admit("rid-1", input_tokens=1, max_new_tokens=1)
        participant.start_heartbeat()
        participant.start_heartbeat()
        try:
            self.assertTrue(renewed.wait(timeout=5))
        finally:
            participant.shutdown()
        self.assertEqual(client.operations()[-1], "release")
        self.assertFalse(participant._heartbeat_thread.is_alive())

    def test_default_heartbeat_interval_is_a_third_of_the_lease_ttl(self) -> None:
        self.assertEqual(self.participant._heartbeat_interval, 10.0)
        short = KvParticipant(
            load_config({**ENVIRON, "KAPSL_KV_LEASE_TTL_MS": "1000"}), FakeClient()
        )
        self.assertAlmostEqual(short._heartbeat_interval, 1 / 3)


if __name__ == "__main__":
    unittest.main()
