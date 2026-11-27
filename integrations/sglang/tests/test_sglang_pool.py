"""The SGLang-facing seams of the shared-pool tier, against stand-ins.

These run the real seam code (``SglangVmmBackend``, ``SlotLists``,
``SglangPoolHost``) with a fake ``torch`` and a fake ``cuda_vmm_utils``. They
show the code is consistent with the SGLang source it was written from, not
that SGLang or CUDA behave that way.
"""

from __future__ import annotations

import os
import types
import unittest
from typing import Any

from fake_sglang import POOL_ENVIRON, ServerArgs
from fake_vmm import EvictParams, FakeTensor, NumpyInt
from pool_case import GRANULE, MAX_TOKENS, ROW, STRIPES, PoolHookTestCase

from kapsl_sglang_adapter.capacity import UnsupportedProfileError
from kapsl_sglang_adapter.sglang_pool import (
    POST_CAPTURE_ENV,
    UNBOUNDED_KV_BUDGET_BYTES,
    SglangPoolHost,
    SglangVmmBackend,
    SlotLists,
    lift_kv_budget,
    pool_facts,
    post_capture_enabled,
    shared_pool_details,
    shared_pool_launch_reasons,
)
from kapsl_sglang_adapter.shared_pool import RetirementDeferred, SharedPoolError


class SeamTestCase(PoolHookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheduler = self.start()
        runner = self.scheduler.tp_worker.model_runner
        self.owner = runner.token_to_kv_pool._post_capture_owner
        self.torch = self.sglang.modules["torch"]
        self.vmm = self.sglang.modules["sglang.srt.utils.cuda_vmm_utils"]

    def backend(self) -> SglangVmmBackend:
        return SglangVmmBackend(self.owner, vmm=self.vmm, torch=self.torch)

    def host(self) -> SglangPoolHost:
        return SglangPoolHost(
            self.scheduler, self.owner, torch=self.torch, evict_params=EvictParams
        )

    def descriptor(self) -> int:
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        self.addCleanup(os.close, write_end)
        return read_end


class VmmBackendTests(SeamTestCase):
    def test_map_imports_maps_and_releases_then_updates_the_arena(self) -> None:
        arena = self.owner._arena
        spec = self.owner._specs[2]
        released = self.sglang.cuda.released
        descriptor = self.descriptor()

        self.backend().map(2, 2 * GRANULE, GRANULE, descriptor)

        # Mapped at the buffer's own place in SGLang's reservation, directly
        # after what was already backed.
        self.assertEqual(
            self.sglang.events, [f"cuda:map:{spec.offset + 2 * GRANULE}:{GRANULE}"]
        )
        self.assertEqual(self.sglang.cuda.imported[-1], descriptor)
        # The imported handle is released once mapped; the caller's own
        # descriptor stays open.
        self.assertEqual(self.sglang.cuda.released, released + 1)
        os.fstat(descriptor)
        self.assertIn(
            (arena.base + spec.offset + 2 * GRANULE, GRANULE, None),
            arena._allocation._mappings,
        )
        self.assertEqual(spec.backed_to, 3 * GRANULE)
        self.assertEqual(arena._committed_by_offset[spec.offset], 3 * GRANULE)
        self.assertEqual(self.owner.backed_bytes, STRIPES * 2 * GRANULE + GRANULE)

    def test_map_must_continue_the_backed_prefix(self) -> None:
        with self.assertRaisesRegex(SharedPoolError, "buffer k1 is backed to 2048"):
            self.backend().map(1, 4 * GRANULE, GRANULE, self.descriptor())
        self.assertEqual(self.sglang.events, [])

    def test_a_failed_map_still_releases_the_imported_handle(self) -> None:
        released = self.sglang.cuda.released
        self.sglang.cuda.fail_map_at = self.sglang.cuda.map_calls + 1
        with self.assertRaisesRegex(RuntimeError, "cuMemMap"):
            self.backend().map(0, 2 * GRANULE, GRANULE, self.descriptor())
        self.assertEqual(self.sglang.cuda.released, released + 1)
        self.assertEqual(self.owner._specs[0].backed_to, 2 * GRANULE)

    def test_a_handle_that_cannot_be_released_is_an_error(self) -> None:
        self.sglang.cuda.fail_release = True
        with self.assertRaisesRegex(RuntimeError, r"cuMemRelease\(kapsl\)"):
            self.backend().map(0, 2 * GRANULE, GRANULE, self.descriptor())

    def test_unmap_removes_a_whole_mapping_from_the_end(self) -> None:
        arena = self.owner._arena
        spec = self.owner._specs[1]
        self.backend().unmap(1, GRANULE, GRANULE)
        self.assertEqual(
            self.sglang.events, [f"cuda:unmap:{spec.offset + GRANULE}:{GRANULE}"]
        )
        self.assertNotIn(
            (arena.base + spec.offset + GRANULE, GRANULE, None),
            arena._allocation._mappings,
        )
        self.assertEqual(spec.backed_to, GRANULE)
        self.assertEqual(arena._committed_by_offset[spec.offset], GRANULE)
        self.assertEqual(self.owner.backed_bytes, STRIPES * 2 * GRANULE - GRANULE)

    def test_unmap_refuses_anything_but_a_whole_trailing_mapping(self) -> None:
        cases = {
            "not the end of the prefix": (0, GRANULE),
            "half a mapping": (GRANULE + GRANULE // 2, GRANULE // 2),
            "two mappings at once": (0, 2 * GRANULE),
        }
        for name, (offset, length) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    SharedPoolError, "CUDA unmaps whole mappings only"
                ),
            ):
                self.backend().unmap(0, offset, length)
        self.assertEqual(self.sglang.events, [])

    def test_synchronize_waits_for_the_device(self) -> None:
        self.backend().synchronize()
        self.assertEqual(self.sglang.events, ["cuda:synchronize"])


class SlotListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = types.SimpleNamespace(
            free_pages=FakeTensor([3, 1, 2]), release_pages=FakeTensor([9, 4])
        )
        torch = types.SimpleNamespace(
            int64="int64",
            arange=lambda start, end, dtype, device: FakeTensor(
                range(start, end), device
            ),
            cat=lambda tensors: FakeTensor(
                tensors[0].values + tensors[1].values, tensors[0].device
            ),
        )
        self.slots = SlotLists(self.allocator, torch)

    def test_added_slots_follow_the_free_ones_on_the_same_device(self) -> None:
        self.slots.add(10, 12)
        self.assertEqual(self.allocator.free_pages.tolist(), [3, 1, 2, 10, 11, 12])
        self.assertEqual(self.allocator.free_pages.device, "cuda:0")
        self.assertEqual(self.allocator.release_pages.tolist(), [9, 4])

    def test_free_slots_above_a_bound_are_counted_on_both_lists(self) -> None:
        self.assertEqual(self.slots.free_above(2), 3)
        self.assertEqual(self.slots.free_above(9), 0)

    def test_slots_above_a_bound_are_dropped_from_both_lists(self) -> None:
        self.slots.drop_above(3)
        self.assertEqual(self.allocator.free_pages.tolist(), [3, 1, 2])
        self.assertEqual(self.allocator.release_pages.tolist(), [])


class PoolHostTests(SeamTestCase):
    def test_capacity_is_read_from_all_six_places(self) -> None:
        host = self.host()
        self.assertEqual(
            host.capacity_fields(),
            {
                "pool.size": 63,
                "allocator.size": 63,
                "model_runner.max_total_num_tokens": 63,
                "scheduler.max_total_num_tokens": 63,
                "pool_stats_observer.max_total_num_tokens": 63,
                "invariant_checker.max_total_num_tokens": 63,
            },
        )
        self.assertEqual(host.tokens(), 63)
        # SGLang reports sizes as NumPy integers; the adapter reads exact ints.
        self.scheduler.max_total_num_tokens = NumpyInt(63)
        self.assertIs(type(host.tokens()), int)

    def test_any_field_out_of_step_is_reported(self) -> None:
        host = self.host()
        self.scheduler.tp_worker.model_runner.max_total_num_tokens = 255
        with self.assertRaisesRegex(
            SharedPoolError, "model_runner.max_total_num_tokens': 255"
        ):
            host.tokens()

    def test_a_scheduler_without_the_expected_fields_is_refused(self) -> None:
        for name in ("pool_stats_observer", "invariant_checker", "tree_cache"):
            with self.subTest(name):
                saved = getattr(self.scheduler, name)
                delattr(self.scheduler, name)
                with self.assertRaisesRegex(UnsupportedProfileError, name):
                    self.host()
                setattr(self.scheduler, name, saved)
        self.scheduler.max_total_num_tokens = 63.0
        with self.assertRaises(UnsupportedProfileError):
            self.host()

    def test_publish_writes_the_frozen_observer_and_every_other_copy(self) -> None:
        host = self.host()
        observer = self.scheduler.pool_stats_observer
        with self.assertRaises(AttributeError):
            observer.max_total_num_tokens = 127  # frozen, as in SGLang

        host.publish(127)

        self.assert_capacity(self.scheduler, 127)
        self.assertEqual(self.owner._final_num_tokens, 127)
        self.assertEqual(self.free_slots(self.scheduler), list(range(1, 128)))
        for target in (127, 63):
            with (
                self.subTest(target),
                self.assertRaisesRegex(SharedPoolError, "cannot grow"),
            ):
                host.publish(target)

    def test_a_paged_allocators_page_count_follows_the_capacity(self) -> None:
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocator.num_pages = 63
        self.host().publish(127)
        self.assertEqual(allocator.num_pages, 127)

    def test_retire_needs_an_idle_server_and_a_free_tail(self) -> None:
        host = self.host()
        host.publish(127)
        allocator = self.scheduler.token_to_kv_pool_allocator

        self.scheduler.idle = False
        with self.assertRaisesRegex(RetirementDeferred, "not idle"):
            host.retire(63)
        self.scheduler.idle = True

        # A deferred group of frees is still open: not a step boundary.
        allocator.free_group = []
        with self.assertRaisesRegex(RetirementDeferred, "not idle"):
            host.retire(63)
        allocator.free_group = None

        held = allocator.alloc(70)
        with self.assertRaisesRegex(RetirementDeferred, "7 KV slot"):
            host.retire(63)
        self.assert_capacity(self.scheduler, 127)
        self.assertEqual(len(self.free_slots(self.scheduler)), 57)

        allocator.free(held)
        host.retire(63)
        self.assert_capacity(self.scheduler, 63)
        self.assertEqual(self.free_slots(self.scheduler), list(range(1, 64)))
        self.assertEqual(self.owner._final_num_tokens, 63)

    def test_retire_targets_must_shrink_the_pool(self) -> None:
        host = self.host()
        for target in (0, 63, 127):
            with (
                self.subTest(target),
                self.assertRaisesRegex(SharedPoolError, "cannot retire"),
            ):
                host.retire(target)

    def test_a_cache_that_evicts_nothing_does_not_spin(self) -> None:
        host = self.host()
        host.publish(127)
        cache = self.scheduler.tree_cache
        cache.cache(list(range(100, 110)))
        cache.evicts = False
        with self.assertRaisesRegex(RetirementDeferred, "10 KV slot"):
            host.retire(63)
        self.assertEqual(cache.evict_calls, [10])

    def test_a_free_list_with_more_tail_slots_than_the_pool_is_an_error(self) -> None:
        host = self.host()
        host.publish(127)
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocator.free(FakeTensor([127]))
        with self.assertRaisesRegex(SharedPoolError, "1 more slot"):
            host.retire(63)


class FactsTests(SeamTestCase):
    def test_facts_describe_the_owners_buffers(self) -> None:
        for spec in self.owner._specs:
            spec.backed_to = 0
        self.owner._arena._range_backed = 0
        self.owner._arena._allocation._mappings.clear()

        facts = pool_facts(self.owner)

        self.assertEqual(
            (facts.page_size, facts.reserved_tokens, facts.granularity_bytes),
            (1, MAX_TOKENS, GRANULE),
        )
        self.assertEqual(facts.element_type, "bfloat16")
        self.assertEqual(
            [buffer.name for buffer in facts.buffers], ["k0", "k1", "v0", "v1"]
        )
        first = facts.buffers[0]
        self.assertEqual(first.shape, (256, 2, 8))
        self.assertEqual(first.reserved_bytes, 256 * ROW)
        # NumPy-like integers are normalized before they reach a JSON document.
        self.assertIs(type(first.row_bytes), int)

    def test_an_arena_sglang_already_backed_is_refused(self) -> None:
        with self.assertRaisesRegex(UnsupportedProfileError, "already mapped memory"):
            pool_facts(self.owner)

    def test_an_owner_without_the_expected_state_is_refused(self) -> None:
        with self.assertRaisesRegex(UnsupportedProfileError, "does not expose"):
            pool_facts(types.SimpleNamespace(page_size=1))

    def test_details_identify_the_pool_the_scheduler_serves_from(self) -> None:
        details = shared_pool_details(self.scheduler, self.owner)
        self.assertEqual(details.allocator_class, "TokenToKVPoolAllocator")
        self.assertEqual(
            (details.allocator_page_size, details.kv_shard_widening), (1, 1)
        )
        self.assertFalse(details.quantized_kv)
        self.assertTrue(details.pool_is_attached)
        self.assertEqual(details.context_length, 16)
        self.assertEqual(details.host_cache_tiers, ())
        self.scheduler.enable_lmcache = True
        self.assertEqual(
            shared_pool_details(self.scheduler, self.owner).host_cache_tiers,
            ("enable_lmcache",),
        )
        self.assertFalse(shared_pool_details(self.scheduler, object()).pool_is_attached)
        del self.scheduler.model_config
        with self.assertRaisesRegex(UnsupportedProfileError, "model_config"):
            shared_pool_details(self.scheduler, self.owner)


class LaunchTests(unittest.TestCase):
    def test_post_capture_flag_is_read_like_sglang_reads_it(self) -> None:
        for value in ("1", "true", "True", "yes", "Y"):
            self.assertTrue(post_capture_enabled({POST_CAPTURE_ENV: value}), value)
        for value in ("0", "false", "no", "", "maybe"):
            self.assertFalse(post_capture_enabled({POST_CAPTURE_ENV: value}), value)
        self.assertFalse(post_capture_enabled({}))

    def test_a_supported_launch_has_no_reasons(self) -> None:
        arguments = ServerArgs(max_total_tokens=NumpyInt(255))
        self.assertEqual(shared_pool_launch_reasons(POOL_ENVIRON, arguments), [])

    def test_server_arguments_without_the_expected_fields_are_refused(self) -> None:
        with self.assertRaisesRegex(UnsupportedProfileError, "max_total_tokens"):
            shared_pool_launch_reasons(
                POOL_ENVIRON, types.SimpleNamespace(tp_size=1, pp_size=1, dp_size=1)
            )
        with self.assertRaises(UnsupportedProfileError):
            shared_pool_launch_reasons(POOL_ENVIRON, None)
        self.assertIn(
            "--max-total-tokens must be a positive integer",
            shared_pool_launch_reasons(
                POOL_ENVIRON, ServerArgs(max_total_tokens="255")
            ),
        )

    def test_the_budget_is_lifted_however_it_was_passed(self) -> None:
        configurator: Any = object()
        self.assertEqual(
            lift_kv_budget((configurator, 4096), {}, cap_tokens=None),
            ((configurator, UNBOUNDED_KV_BUDGET_BYTES), {}),
        )
        self.assertEqual(
            lift_kv_budget((configurator, 4096), {"cap_tokens": 255}, cap_tokens=63),
            ((configurator, UNBOUNDED_KV_BUDGET_BYTES), {"cap_tokens": 63}),
        )
        self.assertEqual(
            lift_kv_budget((configurator,), {"budget_bytes": 4096}, cap_tokens=None),
            ((configurator,), {"budget_bytes": UNBOUNDED_KV_BUDGET_BYTES}),
        )
        with self.assertRaisesRegex(UnsupportedProfileError, "without a byte budget"):
            lift_kv_budget((configurator,), {}, cap_tokens=None)


if __name__ == "__main__":
    unittest.main()
