from __future__ import annotations

import json
import logging
import os
import types
import unittest
from pathlib import Path
from unittest import mock

from fake_sglang import (
    ENVIRON,
    PINNED_VERSION,
    AddReqResult,
    Allocator,
    DisaggregationMode,
    FakeClient,
    FakeSglang,
    HookRegistry,
    HookType,
    HybridLinearKVPool,
    MHATokenToKVPool,
    Req,
    ServerArgs,
    SpecAlgorithm,
    configure_scheduler,
)

from kapsl_sglang_adapter import hooks
from kapsl_sglang_adapter.capacity import SchedulerFacts, UnsupportedProfileError
from kapsl_sglang_adapter.client import KapslKvControlError
from kapsl_sglang_adapter.config import ConfigError
from kapsl_sglang_adapter.participant import KvParticipant

FIXTURES = Path(__file__).with_name("fixtures")
LEASE_1 = "kapsl-kv-0000000000000001"


def setUpModule() -> None:
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class HookTestCase(unittest.TestCase):
    def setUp(self) -> None:
        hooks.reset_for_tests()
        self.addCleanup(hooks.reset_for_tests)
        self.sglang = FakeSglang().install()
        self.addCleanup(self.sglang.uninstall)
        self.client = FakeClient()
        for patcher in (
            mock.patch.dict(os.environ, ENVIRON, clear=True),
            mock.patch.object(
                hooks, "installed_sglang_version", return_value=PINNED_VERSION
            ),
            mock.patch.object(hooks, "build_participant", self.build_participant),
            mock.patch.object(hooks.atexit, "register"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def build_participant(self, config) -> KvParticipant:
        participant = KvParticipant(
            config, self.client, heartbeat_interval_seconds=3600
        )
        self.addCleanup(participant.shutdown)
        return participant

    def install_hooks(self, skip: tuple[str, ...] = ()) -> None:
        hooks.activate()
        HookRegistry.apply_hooks(skip=skip)

    def start(self, server_args: ServerArgs | None = None):
        self.install_hooks()
        scheduler = self.sglang.scheduler.Scheduler(server_args or ServerArgs(), None)
        self.client.calls.clear()
        self.sglang.events.clear()
        return scheduler

    def add_one_req(self, adder, req):
        # Keyword arguments, as SGLang's scheduler passes them.
        return adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    def assert_not_governing(self) -> None:
        adder = self.sglang.schedule_policy.PrefillAdder()
        with self.assertRaises(hooks.AdapterStateError):
            self.add_one_req(adder, Req("rid-1"))
        with self.assertRaises(hooks.AdapterStateError):
            self.sglang.memory_pool.ReqToTokenPool().alloc([Req("rid-1")])
        self.assertEqual(self.sglang.events.count("sglang_admission:rid-1"), 0)
        self.assertNotIn("row:rid-1", self.sglang.events)


class ActivationTests(HookTestCase):
    def test_plugin_registers_exactly_the_pinned_targets(self) -> None:
        hooks.activate()
        self.assertEqual(
            HookRegistry.registered,
            [
                (
                    "sglang.srt.managers.scheduler.Scheduler.__init__",
                    hooks.around_scheduler_init,
                    HookType.AROUND,
                ),
                (
                    "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
                    hooks.around_add_one_req,
                    HookType.AROUND,
                ),
                (
                    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.alloc",
                    hooks.before_alloc_request_rows,
                    HookType.BEFORE,
                ),
                (
                    "sglang.srt.mem_cache.common.release_kv_cache",
                    hooks.after_release_kv_cache,
                    HookType.AFTER,
                ),
            ],
        )

    def test_verification_passes_only_when_every_target_is_wrapped(self) -> None:
        hooks.activate()
        with self.assertRaisesRegex(hooks.HookInstallError, "not wrapped"):
            hooks.verify_hooks_applied()
        HookRegistry.apply_hooks()
        hooks.verify_hooks_applied()

    def test_verification_names_each_unhooked_target(self) -> None:
        self.install_hooks(skip=(hooks.ROW_ALLOC_TARGET, hooks.ROW_RELEASE_TARGET))
        with self.assertRaises(hooks.HookInstallError) as raised:
            hooks.verify_hooks_applied()
        message = str(raised.exception)
        self.assertIn(hooks.ROW_ALLOC_TARGET, message)
        self.assertIn(hooks.ROW_RELEASE_TARGET, message)
        self.assertNotIn(hooks.ADMISSION_TARGET, message)

    def test_verification_reports_a_target_that_no_longer_exists(self) -> None:
        self.install_hooks()
        del self.sglang.common.release_kv_cache
        with self.assertRaisesRegex(hooks.HookInstallError, "cannot be resolved"):
            hooks.verify_hooks_applied()

    def test_another_plugins_wrapper_does_not_count_as_ours(self) -> None:
        self.install_hooks(skip=(hooks.ADMISSION_TARGET,))
        HookRegistry.reset()
        HookRegistry.register(
            hooks.ADMISSION_TARGET,
            lambda original, *args, **kwargs: original(*args, **kwargs),
            HookType.AROUND,
        )
        HookRegistry.apply_hooks()
        with self.assertRaisesRegex(hooks.HookInstallError, "add_one_req"):
            hooks.verify_hooks_applied()

    def test_our_hook_is_found_beneath_another_plugins_wrapper(self) -> None:
        self.install_hooks()
        HookRegistry.reset()
        HookRegistry.register(
            hooks.ADMISSION_TARGET,
            lambda original, *args, **kwargs: original(*args, **kwargs),
            HookType.AROUND,
        )
        HookRegistry.apply_hooks()
        hooks.verify_hooks_applied()


class StartupTests(HookTestCase):
    def test_participant_is_registered_once_the_kv_pool_exists(self) -> None:
        self.install_hooks()
        self.sglang.scheduler.Scheduler(ServerArgs(), None)

        fixture = json.loads(
            (FIXTURES / "sglang_opaque_registration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.sglang.events, ["model_loaded"])
        self.assertEqual(self.client.calls, [("register", fixture)])
        hooks.atexit.register.assert_called_once()

    def test_missing_configuration_fails_before_the_model_loads(self) -> None:
        del os.environ["KAPSL_KV_PARTICIPANT_CONFIG"]
        self.install_hooks()
        with self.assertRaises(ConfigError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, [])
        self.assertEqual(self.client.calls, [])
        self.assert_not_governing()

    def test_an_unpinned_sglang_fails_before_the_model_loads(self) -> None:
        hooks.installed_sglang_version.return_value = "0.5.24"
        self.install_hooks()
        with self.assertRaises(UnsupportedProfileError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, [])
        self.assert_not_governing()

    def test_partially_applied_hooks_fail_before_the_model_loads(self) -> None:
        self.install_hooks(skip=(hooks.ROW_RELEASE_TARGET,))
        with self.assertRaises(hooks.HookInstallError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, [])
        self.assertEqual(self.client.calls, [])

    def test_a_missing_admission_result_enum_fails_startup(self) -> None:
        self.install_hooks()
        del self.sglang.schedule_policy.AddReqResult
        with self.assertRaises(hooks.HookInstallError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, [])

    def test_a_scheduler_outside_the_profile_is_never_registered(self) -> None:
        self.install_hooks()
        with self.assertRaisesRegex(UnsupportedProfileError, "pipeline-parallel"):
            self.sglang.scheduler.Scheduler(ServerArgs(pp_size=2), None)
        self.assertEqual(self.client.calls, [])
        self.assert_not_governing()

    def test_a_rejected_registration_stops_the_scheduler(self) -> None:
        self.client.errors["register"] = KapslKvControlError(
            "connection refused", kind="transport"
        )
        self.install_hooks()
        with self.assertRaises(KapslKvControlError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        hooks.atexit.register.assert_not_called()
        self.assert_not_governing()

    def test_a_failed_start_is_terminal_for_the_process(self) -> None:
        document = os.environ.pop("KAPSL_KV_PARTICIPANT_CONFIG")
        self.install_hooks()
        with self.assertRaises(ConfigError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)

        # Repairing the cause does not let the same process try again.
        os.environ["KAPSL_KV_PARTICIPANT_CONFIG"] = document
        with self.assertRaises(hooks.AdapterStateError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, [])
        self.assertEqual(self.client.calls, [])

    def test_hooks_without_a_constructed_scheduler_refuse_to_allocate(self) -> None:
        self.install_hooks()
        self.assert_not_governing()

    def test_only_one_scheduler_per_process_is_governed(self) -> None:
        self.start()
        with self.assertRaises(hooks.AdapterStateError):
            self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assert_not_governing()

    def test_warm_up_rows_during_construction_are_not_client_requests(self) -> None:
        self.install_hooks()
        self.sglang.scheduler.Scheduler.warm_up_request = Req("warm-up")
        self.sglang.scheduler.Scheduler(ServerArgs(), None)
        self.assertEqual(self.sglang.events, ["model_loaded", "row:warm-up"])
        self.assertEqual(self.client.operations(), ["register"])


class AdmissionTests(HookTestCase):
    def test_lease_spans_admission_to_kv_row_release(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        req = Req("rid-1", input_tokens=100, max_new_tokens=28, priority=2)

        result = self.add_one_req(adder, req)
        self.assertIs(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(self.client.operations(), ["reserve"])
        reserve = self.client.calls[0][1]
        self.assertEqual(reserve["sequence"]["request_id"], "rid-1")
        self.assertEqual(reserve["groups"][0]["token_capacity"], 128)
        self.assertEqual(reserve["priority"], 2)

        self.sglang.memory_pool.ReqToTokenPool().alloc([req])
        self.assertTrue(req.kv.holds_kv)

        req.output_ids = [1] * 17
        self.sglang.common.release_kv_cache(req, None, checkpoint=True)
        self.assertEqual(
            self.client.calls[1:], [("commit", LEASE_1, 117), ("release", LEASE_1)]
        )
        self.assertEqual(
            self.sglang.events,
            ["sglang_admission:rid-1", "row:rid-1", "row_released:rid-1"],
        )
        self.assertEqual(hooks._STATE.participant.live_leases(), 0)

    def test_kapsl_denial_keeps_the_request_out_of_sglang_admission(self) -> None:
        self.start()
        self.client.errors["reserve"] = KapslKvControlError(
            "device KV budget exhausted", kind="capacity_exhausted"
        )
        adder = self.sglang.schedule_policy.PrefillAdder()

        result = self.add_one_req(adder, Req("rid-1"))

        self.assertIs(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(self.sglang.events, [])
        self.assertEqual(hooks._STATE.participant.live_leases(), 0)

    def test_lease_is_returned_when_sglang_declines_the_request(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        adder.accept = False

        result = self.add_one_req(adder, Req("rid-1"))

        self.assertIs(result, AddReqResult.NO_TOKEN)
        self.assertEqual(self.client.calls[1:], [("release", LEASE_1)])
        self.assertEqual(hooks._STATE.participant.live_leases(), 0)

    def test_lease_is_returned_when_sglang_admission_raises(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        adder.error = RuntimeError("prefill budget invariant")

        with self.assertRaisesRegex(RuntimeError, "prefill budget invariant"):
            self.add_one_req(adder, Req("rid-1"))

        self.assertEqual(self.client.operations(), ["reserve", "release"])

    def test_a_request_that_already_holds_a_lease_is_not_reserved_again(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        req = Req("rid-1")
        self.add_one_req(adder, req)

        # SGLang revisits the request and declines this time; the lease that
        # covers its existing KV row must survive.
        adder.accept = False
        self.assertIs(self.add_one_req(adder, req), AddReqResult.NO_TOKEN)
        self.assertEqual(self.client.operations(), ["reserve"])
        self.assertEqual(hooks._STATE.participant.live_leases(), 1)

    def test_retraction_returns_the_lease_and_readmission_takes_a_new_one(self) -> None:
        self.start()
        req = Req("rid-1")
        self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), req)
        self.sglang.memory_pool.ReqToTokenPool().alloc([req])

        # SGLang retracts the unfinished request under KV pressure, requeues
        # it, and later admits it again.
        req.output_ids = [1, 2, 3]
        self.sglang.common.release_kv_cache(req, None, checkpoint=False)
        self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), req)

        self.assertEqual(
            self.client.operations(), ["reserve", "commit", "release", "reserve"]
        )
        self.assertEqual(hooks._STATE.participant.live_leases(), 1)

    def test_lost_authority_stops_admission(self) -> None:
        self.start()
        hooks._STATE.participant.fail(
            KapslKvControlError("connection refused", kind="transport")
        )
        adder = self.sglang.schedule_policy.PrefillAdder()
        with self.assertRaises(KapslKvControlError):
            self.add_one_req(adder, Req("rid-1"))
        self.assertEqual(self.sglang.events, [])
        self.assertEqual(self.client.calls, [])

    def test_a_request_without_a_rid_is_refused(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        with self.assertRaises(KapslKvControlError):
            self.add_one_req(adder, Req(None))
        self.assertEqual(self.sglang.events, [])

    def test_missing_output_cap_and_priority_default_to_zero(self) -> None:
        self.start()
        adder = self.sglang.schedule_policy.PrefillAdder()
        self.add_one_req(adder, Req("rid-1", input_tokens=5, max_new_tokens=None))
        reserve = self.client.calls[0][1]
        self.assertEqual(reserve["groups"][0]["token_capacity"], 5)
        self.assertEqual(reserve["priority"], 0)


class RowGuardTests(HookTestCase):
    def test_a_request_cannot_get_a_kv_row_without_a_lease(self) -> None:
        self.start()
        req = Req("rid-unleased")
        with self.assertRaises(KapslKvControlError) as raised:
            self.sglang.memory_pool.ReqToTokenPool().alloc([req])
        self.assertEqual(raised.exception.kind, "ungoverned_allocation")
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(self.sglang.events, [])
        # The breach is latched: the scheduler cannot keep admitting.
        with self.assertRaises(KapslKvControlError):
            self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), Req("rid-2"))

    def test_a_request_keeping_its_row_needs_no_new_lease(self) -> None:
        self.start()
        continuing = Req("rid-chunked")
        continuing.kv.req_pool_idx = 4
        rows = self.sglang.memory_pool.ReqToTokenPool().alloc([continuing])
        self.assertEqual(rows, [4])

    def test_every_new_row_in_a_batch_is_checked(self) -> None:
        self.start()
        leased, unleased = Req("rid-1"), Req("rid-2")
        self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), leased)
        with self.assertRaises(KapslKvControlError):
            self.sglang.memory_pool.ReqToTokenPool().alloc([leased, unleased])
        self.assertFalse(leased.kv.holds_kv)


class ReleasePathTests(HookTestCase):
    def test_release_before_the_adapter_governs_is_a_no_op(self) -> None:
        self.install_hooks()
        req = Req("rid-1")
        req.kv.req_pool_idx = 0
        self.sglang.common.release_kv_cache(req, None, checkpoint=False)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(self.client.calls, [])

    def test_a_release_path_failure_is_latched_not_raised(self) -> None:
        self.start()
        self.sglang.common.release_kv_cache(Req(None), None, checkpoint=False)
        self.assertEqual(self.sglang.events, ["row_released:None"])
        self.assertIsNotNone(hooks._STATE.participant.failure)
        with self.assertRaises(KapslKvControlError):
            self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), Req("rid-1"))

    def test_a_failed_release_rpc_does_not_interrupt_sglang_cleanup(self) -> None:
        self.start()
        req = Req("rid-1")
        self.add_one_req(self.sglang.schedule_policy.PrefillAdder(), req)
        self.client.errors["release"] = KapslKvControlError("gone", kind="transport")

        self.sglang.common.release_kv_cache(req, None, checkpoint=True)

        self.assertIn("row_released:rid-1", self.sglang.events)
        self.assertIsNotNone(hooks._STATE.participant.failure)

    def test_the_original_return_value_is_preserved(self) -> None:
        self.start()
        self.assertIsNone(
            self.sglang.common.release_kv_cache(Req("rid-9"), None, checkpoint=False)
        )


class SchedulerFactsTests(unittest.TestCase):
    def scheduler(self):
        return configure_scheduler(types.SimpleNamespace())

    def facts(self, scheduler) -> SchedulerFacts:
        return hooks.scheduler_facts(scheduler, sglang_version=PINNED_VERSION)

    def test_facts_of_a_baseline_scheduler(self) -> None:
        self.assertEqual(
            self.facts(self.scheduler()),
            SchedulerFacts(
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
                unsupported_features=(),
            ),
        )

    def test_a_pool_may_report_one_fused_size(self) -> None:
        scheduler = self.scheduler()
        scheduler.token_to_kv_pool_allocator = Allocator(MHATokenToKVPool(4096))
        self.assertEqual(self.facts(scheduler).kv_pool_bytes, 4096)

    def test_departures_from_the_profile_are_captured_not_hidden(self) -> None:
        scheduler = self.scheduler()
        scheduler.server_args = ServerArgs(pp_size=2, dp_size=4)
        scheduler.tp_size = 8
        scheduler.device = "xpu"
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.spec_algorithm = SpecAlgorithm(none=False)
        scheduler.is_hybrid_swa = True
        scheduler.dllm_config = object()
        pool = HybridLinearKVPool((1, 1))
        pool.post_capture_active = True
        scheduler.token_to_kv_pool_allocator = Allocator(pool)

        facts = self.facts(scheduler)

        self.assertEqual(facts.device, "xpu")
        self.assertEqual(facts.pool_class, "HybridLinearKVPool")
        self.assertTrue(facts.post_capture_sizing)
        self.assertEqual(
            (
                facts.tensor_parallel_size,
                facts.pipeline_parallel_size,
                facts.data_parallel_size,
            ),
            (8, 2, 4),
        )
        self.assertEqual(facts.disaggregation_mode, "decode")
        self.assertTrue(facts.speculative_decoding)
        self.assertEqual(facts.unsupported_features, ("is_hybrid_swa", "dllm_config"))

    def test_a_moved_scheduler_attribute_is_an_unsupported_profile(self) -> None:
        for attribute in (
            "token_to_kv_pool_allocator",
            "server_args",
            "page_size",
            "max_total_num_tokens",
            "tp_size",
            "disaggregation_mode",
            "spec_algorithm",
            "device",
            *hooks.UNSUPPORTED_FEATURE_FLAGS,
        ):
            with self.subTest(attribute):
                scheduler = self.scheduler()
                delattr(scheduler, attribute)
                with self.assertRaises(UnsupportedProfileError):
                    self.facts(scheduler)

    def test_a_pool_may_report_numpy_integers(self) -> None:
        # SGLang 0.5.23 returns (np.int64, np.int64); found on the first GPU
        # run under Kapsl, where the adapter refused the pool.
        class Int64:
            def __init__(self, value: int) -> None:
                self.value = value

            def __index__(self) -> int:
                return self.value

        scheduler = self.scheduler()
        scheduler.token_to_kv_pool_allocator = Allocator(
            MHATokenToKVPool((Int64(100669440), Int64(100669440)))
        )
        self.assertEqual(self.facts(scheduler).kv_pool_bytes, 2 * 100669440)

    def test_an_unreadable_pool_size_is_an_unsupported_profile(self) -> None:
        for report in (None, 1.5, (1, 2.0), (), True, (1, True), -1, (4, -4)):
            with self.subTest(report=report):
                scheduler = self.scheduler()
                scheduler.token_to_kv_pool_allocator = Allocator(
                    MHATokenToKVPool(report)
                )
                with self.assertRaises(UnsupportedProfileError):
                    self.facts(scheduler)


if __name__ == "__main__":
    unittest.main()
