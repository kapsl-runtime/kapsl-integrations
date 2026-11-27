"""The shared-pool tier driven through the hooks, against SGLang stand-ins."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any

from fake_sglang import (
    ENVIRON,
    POST_CAPTURE_ENV,
    AddReqResult,
    HookRegistry,
    HookType,
    MLATokenToKVPool,
    Req,
    pool_environ,
)
from fake_vmm import PagedTokenToKVPoolAllocator
from pool_case import GRANULE, MAX_TOKENS, ROW, STRIPES, PoolHookTestCase

from kapsl_sglang_adapter import hooks
from kapsl_sglang_adapter.capacity import UnsupportedProfileError
from kapsl_sglang_adapter.client import KapslKvControlError
from kapsl_sglang_adapter.participant import KvParticipant
from kapsl_sglang_adapter.sglang_pool import UNBOUNDED_KV_BUDGET_BYTES
from kapsl_sglang_adapter.shared_pool import SharedPoolError

FIXTURES = Path(__file__).with_name("fixtures")


def maps(events: list[str]) -> list[str]:
    return [event for event in events if event.startswith("cuda:map")]


class ActivationTests(PoolHookTestCase):
    def test_the_shared_pool_adds_four_hooks_to_the_opaque_four(self) -> None:
        hooks.activate()
        self.assertEqual(
            HookRegistry.registered[:4],
            [(spec.target, spec.function, HookType[spec.kind]) for spec in hooks.HOOKS],
        )
        self.assertEqual(
            HookRegistry.registered[4:],
            [
                (
                    "sglang.srt.mem_cache.kv_cache_configurator"
                    ".KVCacheConfigurator.config_from_budget",
                    hooks.before_config_from_budget,
                    HookType.BEFORE,
                ),
                (
                    "sglang.srt.mem_cache.kv_vmm_backing.KvVmmBufferOwner._back_spans",
                    hooks.around_back_spans,
                    HookType.AROUND,
                ),
                (
                    "sglang.srt.mem_cache.kv_vmm_backing.KvVmmArena.commit_range",
                    hooks.around_arena_commit_range,
                    HookType.AROUND,
                ),
                (
                    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
                    hooks.before_scheduler_step,
                    HookType.BEFORE,
                ),
            ],
        )

    def test_every_pool_hook_must_be_applied_before_the_model_loads(self) -> None:
        for target in (spec.target for spec in hooks.SHARED_POOL_HOOKS):
            with self.subTest(target):
                hooks.reset_for_tests()
                self.sglang.uninstall()
                self.sglang = type(self.sglang)().install()
                self.install_hooks(skip=(target,))
                with self.assertRaisesRegex(hooks.HookInstallError, "not wrapped"):
                    self.construct()
                self.assertEqual(self.sglang.events, [])
                self.assertEqual(self.runtime.calls, [])


class StartupTests(PoolHookTestCase):
    def test_the_pool_is_registered_and_mapped_while_sglang_builds_it(self) -> None:
        self.install_hooks()
        scheduler = self.construct()

        # The runtime's memory is mapped where SGLang would have created its
        # own: after the model loads, before graph capture touches slot 0.
        events = self.sglang.events
        self.assertEqual(events[0], "model_loaded")
        self.assertEqual(events[-1], "graphs_captured")
        self.assertEqual(len(maps(events)), len(events) - 2)
        self.assertEqual(
            self.mappings(scheduler),
            [
                (stripe, offset, GRANULE)
                for stripe in range(STRIPES)
                for offset in (0, GRANULE)
            ],
        )
        self.assertEqual(self.runtime.operations(), ["register", "attach", "activate"])
        fixture = json.loads(
            (FIXTURES / "sglang_striped_registration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.runtime.calls[0][1], fixture)
        self.assertEqual(self.runtime.calls[2], ("activate", self.runtime.epoch))
        hooks.atexit.register.assert_called_once()

        # SGLang serves exactly what the runtime mapped: 64 blocks, one of
        # them the padding slot.
        self.assert_capacity(scheduler, 63)
        self.assertEqual(self.free_slots(scheduler), list(range(1, 64)))
        # The descriptors were only needed to map.
        self.assertEqual(self.runtime.open_descriptors(), [])

    def test_sglang_never_creates_kv_memory(self) -> None:
        self.install_hooks()
        scheduler = self.construct()
        self.assertFalse(
            [event for event in self.sglang.events if "sglang_created" in event]
        )
        owner = scheduler.tp_worker.model_runner.token_to_kv_pool._post_capture_owner
        # SGLang's own bookkeeping agrees with what was mapped for it.
        self.assertEqual(owner.backed_bytes, STRIPES * 64 * ROW)
        self.assertEqual({spec.backed_to for spec in owner._specs}, {64 * ROW})
        self.assertEqual(owner._final_num_tokens, 63)
        with self.assertRaisesRegex(hooks.AdapterStateError, "only the Kapsl runtime"):
            owner._arena.commit_range(0, 4 * GRANULE)
        with self.assertRaisesRegex(SharedPoolError, "the runtime mapped 2048"):
            owner.ensure_prefix(100)

    def test_the_pool_is_sized_by_the_runtime_not_by_free_gpu_memory(self) -> None:
        # Left alone, the fake SGLang would size a 32-token pool from free
        # memory. The reservation follows --max-total-tokens instead, and the
        # backed pool follows the runtime's mapped prefix.
        self.assertEqual(
            self.sglang.shape.free_bytes // self.sglang.shape.cell_bytes, 32
        )
        self.install_hooks()
        scheduler = self.construct()
        self.assertEqual(
            self.sglang.budget_calls,
            [(UNBOUNDED_KV_BUDGET_BYTES, None), (UNBOUNDED_KV_BUDGET_BYTES, 63)],
        )
        owner = scheduler.tp_worker.model_runner.token_to_kv_pool._post_capture_owner
        self.assertEqual(owner._reserved_num_tokens, MAX_TOKENS)

    def test_the_attachment_reports_one_view_per_buffer(self) -> None:
        self.install_hooks()
        self.construct()
        attachment = self.runtime.calls[1][1]
        self.assertEqual(attachment["participant_epoch"], self.runtime.epoch)
        self.assertEqual(attachment["binding_id"], self.runtime.binding_id)
        self.assertEqual(
            attachment["profile"]["adapter_version"], hooks.ADAPTER_VERSION
        )
        self.assertEqual(attachment["imported_bytes"], 256 * STRIPES * ROW)
        self.assertEqual(attachment["mapped_bytes"], 64 * STRIPES * ROW)
        self.assertEqual(
            [view.get("stripe", 0) for view in attachment["views"]], [0, 1, 2, 3]
        )

    def test_a_provisioning_grant_is_presented_at_registration(self) -> None:
        grant = {
            "token": "kvg1_test",
            "geometry_digest": "sha256:" + "ab" * 32,
            "authority_generation": 7,
            "expires_at_unix_ms": 1_900_000_000_000,
        }
        os.environ.update(pool_environ(kapsl_provisioning_grant=grant))
        self.install_hooks()
        self.construct()
        registration = self.runtime.calls[0][1]
        self.assertEqual(registration["provisioning_grant"], grant)
        self.assertIn("provisioning_grant", registration["capabilities"]["features"])

    def assert_not_serving(self) -> None:
        self.assertEqual(hooks._STATE.phase, "failed")
        self.assertNotIn("activate", self.runtime.operations())
        adder = self.sglang.schedule_policy.PrefillAdder()
        with self.assertRaises(hooks.AdapterStateError):
            adder.add_one_req(
                Req("rid-1"), has_chunked_req=False, truncation_align_size=None
            )

    def test_launches_outside_the_profile_fail_before_the_model_loads(self) -> None:
        cases: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
            "pool backing off": ({POST_CAPTURE_ENV: None}, {}, "is not set"),
            "pool backing false": ({POST_CAPTURE_ENV: "false"}, {}, "is not set"),
            "tensor parallel": ({}, {"tp_size": 2}, "--tp-size must be 1"),
            "pipeline parallel": ({}, {"pp_size": 2}, "--pp-size must be 1"),
            "data parallel": ({}, {"dp_size": 2}, "--dp-size must be 1"),
            "no maximum": (
                {},
                {"max_total_tokens": None},
                "--max-total-tokens is required",
            ),
            "zero maximum": ({}, {"max_total_tokens": 0}, "positive integer"),
        }
        for name, (environment, arguments, needle) in cases.items():
            with self.subTest(name):
                hooks.reset_for_tests()
                HookRegistry.reset()
                self.sglang.uninstall()
                self.sglang = type(self.sglang)().install()
                for key, value in environment.items():
                    os.environ.pop(key, None)
                    if value is not None:
                        os.environ[key] = value
                self.install_hooks()
                with self.assertRaisesRegex(UnsupportedProfileError, needle):
                    self.construct(**arguments)
                self.assertEqual(self.sglang.events, [])
                self.assertEqual(self.runtime.calls, [])
                os.environ[POST_CAPTURE_ENV] = "1"

    def test_a_pool_layout_outside_the_profile_is_never_registered(self) -> None:
        self.sglang.shape.page_size = 16
        self.sglang.shape.tokens_per_row = 16
        self.install_hooks()
        with self.assertRaisesRegex(UnsupportedProfileError, "page size 16"):
            self.construct()
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(maps(self.sglang.events), [])
        self.assert_not_serving()

    def test_a_maximum_that_is_not_whole_granules_is_never_registered(self) -> None:
        self.install_hooks()
        with self.assertRaisesRegex(UnsupportedProfileError, r"k \* 32 - 1"):
            self.construct(max_total_tokens=256)
        self.assertEqual(self.runtime.calls, [])

    def test_schedulers_outside_the_profile_are_never_activated(self) -> None:
        def mla(fake):
            fake.pool_class = MLATokenToKVPool

        def paged(fake):
            fake.allocator_class = PagedTokenToKVPoolAllocator

        def quantized(fake):
            fake.shape.quantized = True

        def long_context(fake):
            fake.shape.context_len = 32

        cases = {
            "MLA pool": (mla, "MLATokenToKVPool is not MHATokenToKVPool"),
            "paged allocator": (paged, "PagedTokenToKVPoolAllocator is not"),
            "quantized KV": (quantized, "quantized KV"),
            "context longer than the minimum pool": (
                long_context,
                "minimum pool holds 31 tokens, fewer than the context length 32",
            ),
        }
        for name, (configure, needle) in cases.items():
            with self.subTest(name):
                hooks.reset_for_tests()
                HookRegistry.reset()
                self.close_descriptors()
                self.sglang.uninstall()
                self.sglang = type(self.sglang)().install()
                self.runtime.calls.clear()
                configure(self.sglang)
                self.install_hooks()
                with self.assertRaisesRegex(UnsupportedProfileError, needle):
                    self.construct()
                # The pool was provisioned, but the runtime never grants a
                # lease to a participant that did not activate.
                self.assertEqual(self.runtime.operations(), ["register"])
                self.assert_not_serving()

    def test_scheduler_flags_outside_the_profile_are_never_activated(self) -> None:
        self.install_hooks()
        original = self.sglang.scheduler.Scheduler.__init__.__wrapped__

        def widened(scheduler, *args, **kwargs):
            original(scheduler, *args, **kwargs)
            scheduler.kv_shard_widening = 2
            scheduler.is_hybrid_swa = True
            scheduler.enable_hierarchical_cache = True

        HookRegistry.reset()
        self.sglang.scheduler.Scheduler.__init__ = widened
        hooks.activate()
        HookRegistry.apply_hooks(
            skip=tuple(spec.target for spec in hooks.SHARED_POOL_HOOKS)
        )
        with self.assertRaises(UnsupportedProfileError) as raised:
            self.construct()
        self.assertIn("KV shard widening 2", str(raised.exception))
        self.assertIn("is_hybrid_swa", str(raised.exception))
        self.assertIn("enable_hierarchical_cache offloads", str(raised.exception))
        self.assert_not_serving()

    def test_a_second_vmm_backed_pool_stops_the_start(self) -> None:
        self.sglang.vmm_pools = 2
        self.install_hooks()
        with self.assertRaisesRegex(SharedPoolError, "second VMM-backed KV pool"):
            self.construct()
        self.assert_not_serving()

    def test_a_scheduler_without_a_vmm_backed_pool_is_not_governed(self) -> None:
        self.install_hooks()
        with self.assertRaisesRegex(UnsupportedProfileError, "without a VMM-backed"):
            self.construct(vmm_pool=False)
        self.assertEqual(self.runtime.calls, [])
        self.assert_not_serving()

    def test_a_binding_with_another_geometry_is_not_mapped(self) -> None:
        def half(receipt: dict[str, Any]) -> None:
            binding = receipt["shared_pools"][0]
            binding["block_count"] = 128
            binding["elastic"]["maximum_block_count"] = 128

        self.runtime.mutate_receipt = half
        self.install_hooks()
        with self.assertRaisesRegex(SharedPoolError, "128 blocks for a reservation"):
            self.construct()
        self.assertEqual(maps(self.sglang.events), [])
        self.assertEqual(self.runtime.open_descriptors(), [])
        self.assert_not_serving()

    def test_a_failed_mapping_at_startup_is_rolled_back(self) -> None:
        self.sglang.cuda.fail_map_at = 6
        self.install_hooks()
        with self.assertRaisesRegex(SharedPoolError, "after 5 of 8 segments"):
            self.construct()
        events = self.sglang.events
        self.assertEqual(len(maps(events)), 5)
        self.assertEqual(len([e for e in events if e.startswith("cuda:unmap")]), 5)
        self.assertEqual(self.runtime.open_descriptors(), [])
        self.assert_not_serving()

    def test_a_refused_registration_stops_the_start(self) -> None:
        self.runtime.errors["register"] = KapslKvControlError(
            "shared-pool adapter profile is not allowlisted",
            kind="invalid_capabilities",
        )
        self.install_hooks()
        with self.assertRaises(KapslKvControlError):
            self.construct()
        self.assertEqual(maps(self.sglang.events), [])
        self.assert_not_serving()

    def test_a_vmm_pool_outside_a_scheduler_start_is_refused(self) -> None:
        self.install_hooks()
        backing = self.sglang.modules["sglang.srt.mem_cache.kv_vmm_backing"]
        with self.assertRaisesRegex(
            hooks.AdapterStateError, "outside a Kapsl-governed"
        ):
            backing.KvVmmBufferOwner(
                device="cuda",
                device_id=0,
                store_dtype=self.sglang.shape.dtype,
                page_size=1,
                reserved_num_tokens=MAX_TOKENS,
                buffer_descs=[],
            )


class ServingTests(PoolHookTestCase):
    def add_one_req(self, req: Req) -> Any:
        adder = self.sglang.schedule_policy.PrefillAdder()
        return adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    def finish(self, req: Req) -> None:
        self.sglang.common.release_kv_cache(req, None, checkpoint=True)

    def grow(self, scheduler: Any, blocks: int = 128) -> None:
        self.runtime.request_resize(blocks)
        self.step(scheduler)
        self.runtime.calls.clear()
        self.sglang.events.clear()

    def test_leases_work_as_in_the_opaque_tier(self) -> None:
        self.start()
        req = Req("rid-1", input_tokens=40, max_new_tokens=10)
        self.assertIs(self.add_one_req(req), AddReqResult.CONTINUE)
        self.assertEqual(self.runtime.operations(), ["reserve"])
        self.assertEqual(self.runtime.calls[0][1]["groups"][0]["token_capacity"], 50)
        self.sglang.memory_pool.ReqToTokenPool().alloc([req])
        self.finish(req)
        self.assertEqual(self.runtime.operations(), ["reserve", "commit", "release"])

    def test_a_request_may_reserve_up_to_the_pool_maximum(self) -> None:
        # More than the 63 tokens mapped now: the runtime decides whether to
        # grow. Never more than the reservation can hold.
        self.start()
        self.add_one_req(Req("rid-1", input_tokens=300, max_new_tokens=300))
        self.assertEqual(
            self.runtime.calls[0][1]["groups"][0]["token_capacity"], MAX_TOKENS
        )

    def test_an_idle_step_costs_one_poll_per_interval(self) -> None:
        scheduler = self.start()
        self.step(scheduler)
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        self.assertEqual(self.runtime.calls[0][1]["actor"], {"role": "scheduler"})
        # Further steps inside the 100 ms interval make no control round trip.
        for _ in range(5):
            self.step(scheduler, seconds=0.01)
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        self.step(scheduler, seconds=0.1)
        self.assertEqual(self.runtime.operations(), ["resize_poll"] * 2)
        self.assertEqual(self.sglang.events, ["step"] * 7)

    def test_growth_maps_every_stripe_then_publishes_then_acknowledges(self) -> None:
        scheduler = self.start()
        seen: list[tuple[str, int, int, int]] = []
        self.runtime.on_ack = lambda request: seen.append(
            (
                request["stage"],
                len(self.mappings(scheduler)),
                scheduler.max_total_num_tokens,
                len(self.free_slots(scheduler)),
            )
        )
        self.runtime.request_resize(128)

        self.step(scheduler)

        self.assertEqual(
            self.sglang.events,
            [
                f"cuda:map:{stripe * 11 * GRANULE + 2 * GRANULE}:{2 * GRANULE}"
                for stripe in range(STRIPES)
            ]
            + ["control:ack:map_workers", "control:ack:activate_scheduler", "step"],
        )
        # When the mapping is acknowledged every stripe is mapped and nothing
        # is published yet; when the activation is acknowledged, the slots and
        # the capacity already are.
        self.assertEqual(
            seen,
            [
                ("map_workers", 3 * STRIPES, 63, 63),
                ("activate_scheduler", 3 * STRIPES, 127, 127),
            ],
        )
        self.assertEqual(self.runtime.completed, [(64, 128)])
        self.assertEqual(self.runtime.open_descriptors(), [])
        acks = [call[1] for call in self.runtime.calls if call[0] == "resize_ack"]
        self.assertEqual(
            [(ack["actor"]["role"], ack["applied_block_count"]) for ack in acks],
            [("worker", 128), ("scheduler", 128)],
        )

    def test_growth_changes_all_six_capacity_fields_together(self) -> None:
        scheduler = self.start()
        self.assert_capacity(scheduler, 63)
        self.grow(scheduler)
        self.assertEqual(
            self.capacity(scheduler),
            {
                "pool.size": 127,
                "allocator.size": 127,
                "runner": 127,
                "scheduler": 127,
                "observer": 127,
                "checker": 127,
            },
        )
        # The new slots follow the ones that were already free, and the
        # allocator was never cleared: resize() would drop live allocations.
        allocator = scheduler.token_to_kv_pool_allocator
        self.assertEqual(allocator.free_pages.tolist(), list(range(1, 128)))
        self.assertEqual(allocator.clear_calls, 2)  # construction and startup
        owner = scheduler.tp_worker.model_runner.token_to_kv_pool._post_capture_owner
        self.assertEqual(owner._final_num_tokens, 127)
        self.assertEqual(owner.backed_bytes, STRIPES * 128 * ROW)

    def test_growth_does_not_wait_for_running_requests(self) -> None:
        scheduler = self.start()
        req = Req("rid-1")
        self.add_one_req(req)
        allocator = scheduler.token_to_kv_pool_allocator
        held = allocator.alloc(50)
        self.runtime.request_resize(128)

        self.step(scheduler)

        self.assert_capacity(scheduler, 127)
        # Slots in use are untouched; the new ones are simply added.
        self.assertEqual(allocator.free_pages.tolist(), list(range(51, 128)))
        self.assertEqual(held.tolist(), list(range(1, 51)))
        self.assertEqual(self.participant.live_leases(), 1)

    def test_a_denied_request_fetches_the_growth_at_the_next_step(self) -> None:
        scheduler = self.start()
        self.step(scheduler)
        self.runtime.calls.clear()
        self.runtime.errors["reserve"] = KapslKvControlError(
            "KV pool cannot fit the reservation", kind="capacity_exhausted"
        )
        req = Req("rid-1", input_tokens=90, max_new_tokens=10)
        self.assertIs(self.add_one_req(req), AddReqResult.OTHER)
        # The denial is what makes the runtime grow the pool.
        self.runtime.request_resize(128)
        del self.runtime.errors["reserve"]

        # No waiting for the poll interval, and no denial back-off left after.
        self.step(scheduler, seconds=0.001)
        self.assert_capacity(scheduler, 127)
        self.assertIs(self.add_one_req(req), AddReqResult.CONTINUE)

    def test_an_idle_pool_retires_its_tail_then_unmaps_it(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        self.runtime.request_resize(64)

        self.step(scheduler)

        self.assertEqual(
            self.sglang.events,
            ["control:ack:retire_scheduler", "cuda:synchronize"]
            + [
                f"cuda:unmap:{stripe * 11 * GRANULE + 2 * GRANULE}:{2 * GRANULE}"
                for stripe in range(STRIPES)
            ]
            + ["control:ack:unmap_workers", "step"],
        )
        self.assert_capacity(scheduler, 63)
        self.assertEqual(self.free_slots(scheduler), list(range(1, 64)))
        self.assertEqual(len(self.mappings(scheduler)), 2 * STRIPES)
        self.assertEqual(self.runtime.completed[-1], (128, 64))
        # SGLang's idle test is asked with waiting requests ignored: they own
        # no KV, and the adapter holds them back while a shrink waits.
        self.assertEqual(self.sglang.idle_queries, [True])
        owner = scheduler.tp_worker.model_runner.token_to_kv_pool._post_capture_owner
        self.assertEqual(owner.backed_bytes, STRIPES * 64 * ROW)

    def test_the_pool_shrinks_to_its_minimum_and_grows_back(self) -> None:
        scheduler = self.start()
        self.runtime.request_resize(32)
        self.step(scheduler)
        self.assert_capacity(scheduler, 31)
        self.assertEqual(len(self.mappings(scheduler)), STRIPES)
        self.runtime.request_resize(256)
        self.step(scheduler)
        self.assert_capacity(scheduler, MAX_TOKENS)
        self.assertEqual(self.free_slots(scheduler), list(range(1, 256)))

    def test_retirement_waits_while_a_request_holds_a_lease(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        running = Req("rid-1")
        self.add_one_req(running)
        self.runtime.calls.clear()
        self.runtime.request_resize(64)

        self.step(scheduler)

        # Refused for now: nothing retired, unmapped, or acknowledged.
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        self.assert_capacity(scheduler, 127)
        self.assertEqual(len(self.mappings(scheduler)), 3 * STRIPES)
        self.assertIsNone(self.participant.failure)

        # New requests wait in SGLang's queue instead of keeping the server
        # busy past the runtime's deadline. No reservation is attempted.
        waiting = Req("rid-2")
        self.assertIs(self.add_one_req(waiting), AddReqResult.OTHER)
        self.assertEqual(self.runtime.operations(), ["resize_poll"])
        for _ in range(3):
            self.step(scheduler)
        self.assertEqual(self.runtime.operations(), ["resize_poll"])

        # The last lease comes back: the tail is retired at the very next
        # step, and admission resumes.
        self.finish(running)
        self.step(scheduler, seconds=0.001)
        self.assert_capacity(scheduler, 63)
        self.assertEqual(self.runtime.completed[-1], (128, 64))
        self.assertIs(self.add_one_req(waiting), AddReqResult.CONTINUE)

    def test_retirement_waits_while_sglang_is_not_idle(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        scheduler.idle = False
        self.runtime.request_resize(64)
        self.step(scheduler)
        self.assert_capacity(scheduler, 127)
        self.assertNotIn("resize_ack", self.runtime.operations())

        scheduler.idle = True
        self.step(scheduler)
        self.assert_capacity(scheduler, 63)

    def test_retirement_evicts_only_what_the_tail_needs(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        cache = scheduler.tree_cache
        # Least recently used first. Only the first two prefixes sit above the
        # new boundary of 63 tokens.
        cache.cache(list(range(100, 110)))
        cache.cache(list(range(64, 70)))
        cache.cache(list(range(10, 20)))
        self.runtime.request_resize(64)

        self.step(scheduler)

        self.assert_capacity(scheduler, 63)
        # 16 tail slots were busy; evicting in the cache's own order freed
        # them after two prefixes, and the third prefix survived.
        self.assertEqual(cache.evict_calls, [16])
        self.assertEqual(cache.entries, [list(range(10, 20))])
        self.assertEqual(
            self.free_slots(scheduler),
            [slot for slot in range(1, 64) if slot not in range(10, 20)],
        )

    def test_an_unused_tail_is_retired_without_touching_the_prefix_cache(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        cache = scheduler.tree_cache
        cache.cache(list(range(1, 40)))
        self.runtime.request_resize(64)
        self.step(scheduler)
        self.assert_capacity(scheduler, 63)
        self.assertEqual(cache.evict_calls, [])
        self.assertEqual(cache.entries, [list(range(1, 40))])

    def test_retirement_evicts_through_a_sorting_allocator(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        allocator = scheduler.token_to_kv_pool_allocator
        allocator.need_sort = True
        scheduler.tree_cache.cache(list(range(90, 100)))
        self.runtime.request_resize(64)
        self.step(scheduler)
        # Evicted slots land on the release list; both lists lose the tail.
        self.assert_capacity(scheduler, 63)
        self.assertEqual(allocator.release_pages.tolist(), [])
        self.assertEqual(allocator.free_pages.tolist(), list(range(1, 64)))

    def test_retirement_waits_while_a_tail_slot_is_still_referenced(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        cache = scheduler.tree_cache
        cache.cache(list(range(80, 90)), locked=True)
        cache.cache(list(range(1, 9)))
        self.runtime.request_resize(64)

        self.step(scheduler)

        # Everything evictable went, the locked prefix did not, so the tail
        # stays: nothing is removed from the free list or the capacity.
        self.assertEqual(cache.entries, [])
        self.assert_capacity(scheduler, 127)
        self.assertEqual(len(self.free_slots(scheduler)), 127 - 10)
        self.assertNotIn("resize_ack", self.runtime.operations())

        scheduler.token_to_kv_pool_allocator.free(
            type(scheduler.token_to_kv_pool_allocator.free_pages)(cache.locked.pop())
        )
        self.step(scheduler)
        self.assert_capacity(scheduler, 63)

    def test_a_boundary_inside_a_mapping_is_refused_and_stops_serving(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        # 96 blocks is a whole number of granules, but it is inside the
        # two-granule mappings growth added. CUDA cannot unmap half of one.
        self.runtime.request_resize(96)
        self.runtime.resize["segments"] = []

        with self.assertRaisesRegex(SharedPoolError, "CUDA unmaps whole mappings only"):
            self.step(scheduler)

        # Refused before anything changed, and latched.
        self.assert_capacity(scheduler, 127)
        self.assertEqual(self.free_slots(scheduler), list(range(1, 128)))
        self.assertEqual(len(self.mappings(scheduler)), 3 * STRIPES)
        self.assertNotIn("resize_ack", self.runtime.operations())
        self.assert_latched()

    def assert_latched(self) -> None:
        self.assertIsNotNone(self.participant.failure)
        with self.assertRaises(KapslKvControlError):
            self.add_one_req(Req("after-failure"))
        self.assertEqual(self.participant.live_leases(), 0)

    def test_a_failed_mapping_during_growth_stops_serving(self) -> None:
        scheduler = self.start()
        self.sglang.cuda.fail_map_at = self.sglang.cuda.map_calls + 3
        self.runtime.request_resize(128)

        with self.assertRaisesRegex(SharedPoolError, "after 2 of 4 segments"):
            self.step(scheduler)

        # The two stripes that were mapped are unmapped again, nothing is
        # published or acknowledged, and the descriptors are closed.
        self.assertEqual(len(self.mappings(scheduler)), 2 * STRIPES)
        self.assert_capacity(scheduler, 63)
        self.assertNotIn("resize_ack", self.runtime.operations())
        self.assertEqual(self.runtime.open_descriptors(), [])
        self.assert_latched()
        # Every later step fails as well: the scheduler cannot keep serving.
        with self.assertRaises(KapslKvControlError):
            self.step(scheduler)

    def test_a_failed_unmap_stops_serving(self) -> None:
        scheduler = self.start()
        self.grow(scheduler)
        self.sglang.cuda.fail_unmap = True
        self.runtime.request_resize(64)
        with self.assertRaisesRegex(RuntimeError, "cuMemUnmap"):
            self.step(scheduler)
        self.assert_latched()

    def test_a_resize_the_runtime_describes_differently_stops_serving(self) -> None:
        def another_binding(operation: dict[str, Any]) -> None:
            operation["binding_id"] = "cuda-vmm:someone-else"

        scheduler = self.start()
        self.runtime.mutate_operation = another_binding
        self.runtime.request_resize(128)
        with self.assertRaisesRegex(SharedPoolError, "does not describe binding"):
            self.step(scheduler)
        self.assertEqual(maps(self.sglang.events), [])
        self.assert_latched()

    def test_a_rejected_acknowledgement_stops_serving(self) -> None:
        scheduler = self.start()
        self.runtime.errors["resize_ack"] = KapslKvControlError(
            "live KV resize generation 1 is fenced", kind="invalid_request"
        )
        self.runtime.request_resize(128)
        with self.assertRaises(KapslKvControlError):
            self.step(scheduler)
        self.assert_latched()

    def test_capacity_fields_that_disagree_stop_serving(self) -> None:
        scheduler = self.start()
        # A seventh copy SGLang might grow in a later release would show up
        # as disagreement; so does anything else editing one field.
        scheduler.invariant_checker.max_total_num_tokens = 62
        self.runtime.request_resize(128)
        with self.assertRaisesRegex(SharedPoolError, "capacity fields disagree"):
            self.step(scheduler)
        self.assert_latched()

    def test_a_step_without_a_serving_participant_is_refused(self) -> None:
        self.install_hooks()
        scheduler = object.__new__(self.sglang.scheduler.Scheduler)
        with self.assertRaises(hooks.AdapterStateError):
            scheduler.get_next_batch_to_run(running_batch=None, last_batch=None)
        self.assertEqual(self.sglang.events, [])


class OpaqueTierTests(PoolHookTestCase):
    """The default tier installs nothing new and never touches the VMM pool."""

    environ = ENVIRON

    def build_participant(self, config: Any) -> KvParticipant:
        participant = KvParticipant(
            config, self.runtime, heartbeat_interval_seconds=3600
        )
        self.addCleanup(participant.shutdown)
        return participant

    def test_opaque_mode_registers_and_verifies_only_its_four_hooks(self) -> None:
        hooks.activate()
        self.assertEqual(
            [target for target, _, _ in HookRegistry.registered],
            [spec.target for spec in hooks.HOOKS],
        )
        HookRegistry.apply_hooks()
        # The pool targets may not even exist: the opaque tier does not look.
        backing = self.sglang.modules["sglang.srt.mem_cache.kv_vmm_backing"]
        del backing.KvVmmBufferOwner, backing.KvVmmArena
        del self.sglang.scheduler.Scheduler.get_next_batch_to_run
        hooks.verify_hooks_applied()

        self.sglang.scheduler.Scheduler(self.server_args(vmm_pool=False), None)
        registration = self.runtime.calls[0][1]
        fixture = json.loads(
            (FIXTURES / "sglang_opaque_registration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registration, fixture)
        self.assertEqual(self.runtime.operations(), ["register"])
        self.assertIsNone(hooks._STATE.pool_setup)

    def test_opaque_mode_leaves_sglangs_vmm_pool_alone(self) -> None:
        self.install_hooks()
        backing = self.sglang.modules["sglang.srt.mem_cache.kv_vmm_backing"]
        for function in (
            backing.KvVmmBufferOwner._back_spans,
            backing.KvVmmArena.commit_range,
            self.sglang.scheduler.Scheduler.get_next_batch_to_run,
        ):
            self.assertFalse(hasattr(function, "__wrapped__"), function)
        # A server with post-capture sizing on is still outside the opaque
        # profile, as before.
        with self.assertRaisesRegex(UnsupportedProfileError, "post-capture KV sizing"):
            self.construct()
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(self.sglang.budget_calls, [(4096, None), (4096, 32)])


if __name__ == "__main__":
    unittest.main()
