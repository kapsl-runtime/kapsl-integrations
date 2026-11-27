from __future__ import annotations

import dataclasses
import logging
import unittest
from copy import deepcopy
from typing import Any

from fake_vmm import (
    FLOAT32,
    UINT8,
    FakePoolRuntime,
    PoolShape,
    RecordingBackend,
    pool_facts_for,
)
from test_pool_contract import registration

from kapsl_sglang_adapter.capacity import UnsupportedProfileError
from kapsl_sglang_adapter.pool_contract import validate_resize_operation
from kapsl_sglang_adapter.shared_pool import (
    SHARED_POOL_PROFILE_ID,
    PoolGeometry,
    SharedPoolError,
    StripedPool,
    adapter_profile,
    verify_binding,
)

GRANULE = 1024
ROW = 32


def setUpModule() -> None:
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


def geometry(shape: PoolShape | None = None, tokens: int = 255) -> PoolGeometry:
    return PoolGeometry.from_facts(pool_facts_for(shape or PoolShape(), tokens))


class GeometryTests(unittest.TestCase):
    def test_buffers_become_equal_stripes_of_one_row_blocks(self) -> None:
        described = geometry()
        # Stripe count is the number of K/V buffers; a block in one stripe is
        # one buffer row (row bytes x tokens per block).
        self.assertEqual(described.stripe_count, 2 * 2)
        self.assertEqual(described.stripe_names, ("k0", "k1", "v0", "v1"))
        self.assertEqual(described.stripe_block_bytes, ROW)
        self.assertEqual(described.bytes_per_block, 4 * ROW)
        # The reservation is the token maximum plus the padding slot.
        self.assertEqual(described.maximum_blocks, 256)
        self.assertEqual(described.resize_alignment_blocks, GRANULE // ROW)
        # Block 0 is SGLang's padding slot: B mapped blocks serve B - 1 tokens.
        self.assertEqual(described.tokens(64), 63)
        self.assertEqual(described.stripe_bytes(64), 2 * GRANULE)

    def test_the_documented_qwen_geometry(self) -> None:
        # Qwen2.5-1.5B from the design: 28 layers, 2 KV heads of 128, fp16.
        shape = PoolShape(layers=28, kv_heads=2, head_dim=128, granularity=2 << 20)
        described = geometry(shape, 4096 * 8 - 1)
        self.assertEqual(described.stripe_count, 56)
        self.assertEqual(described.stripe_block_bytes, 512)
        self.assertEqual(described.bytes_per_block, 28_672)
        # A 2 MiB granule holds 4,096 rows of one buffer.
        self.assertEqual(described.resize_alignment_blocks, 4096)

    def test_a_row_size_that_does_not_divide_the_granule(self) -> None:
        # 6 heads of 8 in bf16 is 96 bytes a row: 32 rows fill 3 granules.
        shape = PoolShape(kv_heads=6)
        described = geometry(shape, 32 * 4 - 1)
        self.assertEqual(described.resize_alignment_blocks, 32)
        with self.assertRaisesRegex(UnsupportedProfileError, r"k \* 32 - 1"):
            geometry(shape, 32 * 4)

    def test_layouts_outside_the_striped_profile_are_refused(self) -> None:
        def facts(**changes: Any):
            return dataclasses.replace(pool_facts_for(PoolShape(), 255), **changes)

        def buffers(mutate):
            mutated = list(pool_facts_for(PoolShape(), 255).buffers)
            mutate(mutated)
            return tuple(mutated)

        def replace(index: int, **changes: Any):
            def mutate(mutated: list) -> None:
                mutated[index] = dataclasses.replace(mutated[index], **changes)

            return mutate

        cases = {
            "paged allocator": (
                pool_facts_for(PoolShape(page_size=16), 255),
                "page size 16",
            ),
            "pages as rows (HND)": (
                pool_facts_for(PoolShape(tokens_per_row=16, page_size=16), 255),
                "not token slots",
            ),
            "K and V head dims differ": (
                pool_facts_for(PoolShape(v_head_dim=16), 255),
                "differ in shape",
            ),
            "quantized storage": (
                pool_facts_for(PoolShape(dtype=UINT8), 255),
                "uint8 is not covered",
            ),
            "maximum not on a granule": (
                pool_facts_for(PoolShape(), 256),
                "does not fill whole 1024-byte granules",
            ),
            "odd buffer count": (
                facts(buffers=buffers(lambda b: b.pop())),
                "one K and one V buffer per layer, found 3",
            ),
            "no buffers": (facts(buffers=()), "found 0"),
            "unexpected order": (
                facts(buffers=buffers(lambda b: b.reverse())),
                "not ordered",
            ),
            "already backed by SGLang": (
                facts(buffers=buffers(replace(1, backed_bytes=GRANULE))),
                "already backed",
            ),
            "misplaced buffer": (
                facts(buffers=buffers(replace(2, reservation_offset=7))),
                "granule-aligned range",
            ),
            "overlapping buffers": (
                facts(buffers=buffers(replace(1, reservation_offset=0))),
                "granule-aligned range",
            ),
            "past the reservation": (
                facts(reservation_bytes=4 * 8 * GRANULE),
                "past SGLang's address reservation",
            ),
            "span is not rows times row": (
                facts(buffers=buffers(replace(3, reserved_bytes=8 * GRANULE + 32))),
                "reserved span",
            ),
            "wrong row count": (facts(reserved_tokens=127), "is not \\(128 rows"),
            "zero granularity": (facts(granularity_bytes=0), "not positive"),
        }
        for name, (described, needle) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(UnsupportedProfileError, needle),
            ):
                PoolGeometry.from_facts(described)

    def test_too_many_buffers_for_one_binding(self) -> None:
        with self.assertRaisesRegex(UnsupportedProfileError, "exceed the 253 stripes"):
            geometry(PoolShape(layers=127))
        self.assertEqual(geometry(PoolShape(layers=126)).stripe_count, 252)

    def test_float32_pools_are_described(self) -> None:
        described = geometry(PoolShape(dtype=FLOAT32), 127)
        self.assertEqual(described.element_kind, "f32")
        self.assertEqual(described.stripe_block_bytes, 64)

    def test_the_profile_names_adapter_and_backend_builds_exactly(self) -> None:
        self.assertEqual(
            adapter_profile("0.3.0", "0.5.23"),
            {
                "adapter_id": "kapsl-sglang-adapter",
                "adapter_version": "0.3.0",
                "backend_version": "0.5.23",
                "profile_id": SHARED_POOL_PROFILE_ID,
            },
        )


class PoolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakePoolRuntime()
        self.addCleanup(self.close_descriptors)
        self.backend = RecordingBackend()
        self.geometry = geometry()

    def close_descriptors(self) -> None:
        self.runtime.close()

    def grant(self):
        return self.runtime.register_shared_pool(registration())

    def attached(self) -> StripedPool:
        receipt, handles = self.grant()
        pool = StripedPool(
            self.geometry, verify_binding(self.geometry, receipt), self.backend
        )
        pool.map_initial(handles)
        self.backend.events.clear()
        return pool

    def operation(self, actor: str = "worker") -> tuple[dict[str, Any], list[int]]:
        request = {
            "participant_epoch": self.runtime.epoch,
            "actor": {"role": "scheduler"}
            if actor == "scheduler"
            else {
                "role": "worker",
                "shard": {
                    "tensor_parallel_rank": 0,
                    "tensor_parallel_world_size": 1,
                    "pipeline_parallel_rank": 0,
                    "pipeline_parallel_world_size": 1,
                },
            },
            "applied_generation": 0,
        }
        operations, handles, _ = self.runtime.poll_resize(request)
        return operations[0], handles


class BindingTests(PoolTestCase):
    def refuse(self, mutate, needle: str) -> None:
        receipt, _ = self.grant()
        mutate(receipt["shared_pools"][0])
        with self.assertRaisesRegex(SharedPoolError, needle):
            verify_binding(self.geometry, receipt)

    def test_a_matching_binding_is_accepted(self) -> None:
        receipt, _ = self.grant()
        binding = verify_binding(self.geometry, receipt)
        self.assertEqual(
            (binding.minimum_blocks, binding.mapped_blocks, binding.maximum_blocks),
            (32, 64, 256),
        )
        # Two segments per stripe: the minimum and the headroom above it.
        self.assertEqual(len(binding.segments), 2 * 4)

    def test_a_binding_with_another_geometry_is_refused(self) -> None:
        self.refuse(
            lambda b: b["elastic"].update(stripe_count=2),
            "2 stripes for 4 K/V buffers",
        )
        # Twice the block size over the same four stripes: 64 bytes per block
        # in a stripe, but a buffer row is 32.
        self.refuse(
            lambda b: b.update(bytes_per_block=256),
            "64 bytes per block in a stripe, but a buffer row of 1 token\\(s\\) is 32",
        )
        self.refuse(
            lambda b: b.update(block_count=128), "128 blocks for a reservation of 256"
        )
        self.refuse(
            lambda b: b["elastic"].update(allocation_granularity_bytes=2048),
            "granularity 2048 bytes, but SGLang's arena uses 1024",
        )
        self.refuse(
            lambda b: b["elastic"].update(minimum_block_count=1),
            "no token besides the padding page",
        )

    def test_every_mismatch_is_reported_together(self) -> None:
        receipt, _ = self.grant()
        binding = receipt["shared_pools"][0]
        binding["block_count"] = 128
        binding["elastic"]["allocation_granularity_bytes"] = 2048
        with self.assertRaises(SharedPoolError) as raised:
            verify_binding(self.geometry, receipt)
        self.assertIn("128 blocks", str(raised.exception))
        self.assertIn("granularity 2048", str(raised.exception))


class MappingTests(PoolTestCase):
    def test_the_granted_prefix_is_mapped_into_every_stripe(self) -> None:
        receipt, handles = self.grant()
        pool = StripedPool(
            self.geometry, verify_binding(self.geometry, receipt), self.backend
        )
        self.assertEqual(pool.mapped_blocks, 0)
        pool.map_initial(handles)

        self.assertEqual(pool.mapped_blocks, 64)
        self.assertEqual(
            self.backend.events,
            [
                f"map:{stripe}:{offset}:{GRANULE}"
                for stripe in range(4)
                for offset in (0, GRANULE)
            ],
        )
        self.assertEqual(pool.segment_count(), 8)
        with self.assertRaisesRegex(SharedPoolError, "already mapped"):
            pool.map_initial(handles)

    def test_a_failed_mapping_is_rolled_back_in_every_stripe(self) -> None:
        receipt, handles = self.grant()
        pool = StripedPool(
            self.geometry, verify_binding(self.geometry, receipt), self.backend
        )
        self.backend.fail_map_at = 6
        with self.assertRaisesRegex(SharedPoolError, "after 5 of 8 segments"):
            pool.map_initial(handles)
        # No stripe keeps a prefix the others lack.
        self.assertEqual(self.backend.mapped, set())
        self.assertEqual((pool.mapped_blocks, pool.segment_count()), (0, 0))

    def test_growth_extends_every_stripe_from_the_mapped_prefix(self) -> None:
        pool = self.attached()
        self.runtime.request_resize(128)
        operation, handles = self.operation()
        pool.map_growth(operation, handles)
        self.assertEqual(pool.mapped_blocks, 128)
        self.assertEqual(
            self.backend.events,
            [f"map:{stripe}:{2 * GRANULE}:{2 * GRANULE}" for stripe in range(4)],
        )

    def test_growth_that_does_not_fit_the_pool_is_refused_unmapped(self) -> None:
        pool = self.attached()
        self.runtime.request_resize(128)
        operation, handles = self.operation()

        def changed(**changes: Any) -> dict[str, Any]:
            return {**deepcopy(operation), **changes}

        gap = deepcopy(operation)
        gap["from_block_count"] = 96
        for segment in gap["segments"]:
            segment["offset_bytes"] += GRANULE
            segment["length_bytes"] -= GRANULE
        gap = validate_resize_operation(gap)
        remapped = deepcopy(operation)
        remapped["segments"][0]["segment_id"] = f"{self.runtime.binding_id}:minimum:0"
        cases = {
            "other binding": (changed(binding_id="cuda:1"), "does not describe"),
            "other block size": (changed(bytes_per_block=256), "does not describe"),
            "other granularity": (
                changed(allocation_granularity_bytes=2048),
                "does not describe",
            ),
            "other stripe count": (changed(stripe_count=2), "does not describe"),
            "past the reservation": (
                changed(target_block_count=288),
                "outside \\[32, 256\\]",
            ),
            "not from the mapped prefix": (gap, "starts at 96 blocks, but 64"),
            "segment mapped twice": (remapped, "already mapped"),
        }
        for name, (broken, needle) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(SharedPoolError, needle):
                pool.map_growth(broken, handles)
        with self.assertRaisesRegex(SharedPoolError, "names OS handle 3 of 3"):
            pool.map_growth(operation, handles[:3])
        self.assertEqual(self.backend.events, [])
        self.assertEqual(pool.mapped_blocks, 64)

    def test_sglang_may_not_back_more_than_the_runtime_mapped(self) -> None:
        pool = self.attached()
        mapped = 64 * ROW
        pool.require_backed([ROW] * 4)
        pool.require_backed([mapped] * 4)
        with self.assertRaisesRegex(SharedPoolError, "buffer v0"):
            pool.require_backed([mapped, mapped, mapped + ROW, mapped])
        with self.assertRaisesRegex(SharedPoolError, "3 buffers of a pool with 4"):
            pool.require_backed([ROW] * 3)


class RetirementTests(PoolTestCase):
    def grown(self) -> StripedPool:
        pool = self.attached()
        self.runtime.request_resize(128)
        operation, handles = self.operation()
        pool.map_growth(operation, handles)
        self.runtime.resize = None
        self.runtime.mapped_blocks = 128
        self.runtime.mapped_segments.append(operation["segments"])
        self.backend.events.clear()
        return pool

    def test_whole_mappings_above_the_boundary_are_unmapped(self) -> None:
        pool = self.grown()
        self.runtime.request_resize(32)
        self.runtime.resize["stage"] = "unmap_workers"
        operation, _ = self.operation()

        pool.unmap_tail(operation)

        self.assertEqual(pool.mapped_blocks, 32)
        # The device is synchronized first, then each stripe loses its
        # mappings from the top down.
        self.assertEqual(self.backend.events[0], "synchronize")
        self.assertEqual(
            self.backend.events[1:],
            [
                event
                for stripe in range(4)
                for event in (
                    f"unmap:{stripe}:{2 * GRANULE}:{2 * GRANULE}",
                    f"unmap:{stripe}:{GRANULE}:{GRANULE}",
                )
            ],
        )
        self.assertEqual(self.backend.mapped, {(s, 0, GRANULE) for s in range(4)})

    def test_a_boundary_inside_a_mapping_is_refused(self) -> None:
        pool = self.grown()
        # 96 blocks is a granule boundary, but it falls inside the two-granule
        # mapping that growth added.
        with self.assertRaisesRegex(SharedPoolError, "CUDA unmaps whole mappings only"):
            pool.retirement_plan(96)
        self.assertEqual([s.offset_bytes for s in pool.retirement_plan(64)][:1], [2048])
        self.assertEqual(self.backend.events, [])
        self.assertEqual(pool.mapped_blocks, 128)

    def test_retirement_stays_within_the_minimum_and_the_mapped_prefix(self) -> None:
        pool = self.grown()
        for target in (0, 16, 128, 160):
            with (
                self.subTest(target),
                self.assertRaisesRegex(SharedPoolError, "cannot retire"),
            ):
                pool.retirement_plan(target)

    def test_unmap_must_name_exactly_the_mappings_held(self) -> None:
        pool = self.grown()
        self.runtime.request_resize(64)
        self.runtime.resize["stage"] = "unmap_workers"
        operation, _ = self.operation()

        renamed = deepcopy(operation)
        renamed["segments"][2]["segment_id"] = "someone-elses"
        stale = {**deepcopy(operation), "from_block_count": 256}
        for name, broken in {"renamed": renamed, "stale start": stale}.items():
            with self.subTest(name), self.assertRaises(SharedPoolError):
                pool.unmap_tail(broken)
        self.assertEqual(self.backend.events, [])
        self.assertEqual(pool.mapped_blocks, 128)

        pool.unmap_tail(operation)
        self.assertEqual(pool.mapped_blocks, 64)

    def test_a_failed_unmap_is_an_error(self) -> None:
        pool = self.grown()
        self.runtime.request_resize(64)
        self.runtime.resize["stage"] = "unmap_workers"
        operation, _ = self.operation()
        self.backend.fail_unmap = True
        with self.assertRaisesRegex(RuntimeError, "cuMemUnmap"):
            pool.unmap_tail(operation)


if __name__ == "__main__":
    unittest.main()
