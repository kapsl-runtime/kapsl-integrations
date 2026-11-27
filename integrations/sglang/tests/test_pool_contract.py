from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from fake_vmm import PoolShape, pool_facts_for

from kapsl_sglang_adapter import contract
from kapsl_sglang_adapter.contract import ContractValidationError
from kapsl_sglang_adapter.pool_contract import (
    MAX_POOL_STRIPES,
    SCHEDULER_ACTOR,
    STRIPED_POOL_ABI_VERSION,
    check_segment_coverage,
    make_attachment,
    make_resize_ack_request,
    make_resize_poll_request,
    single_rank_shard,
    striped_pool_registration,
    validate_resize_operation,
    validate_striped_pool_receipt,
    validate_striped_pool_registration,
    worker_actor,
)
from kapsl_sglang_adapter.shared_pool import PoolGeometry, adapter_profile

FIXTURES = Path(__file__).with_name("fixtures")
GRANULE = 1024
STRIPES = 4
# Bytes of one block in one stripe; 32 blocks fill one granule per stripe.
STRIPE_BLOCK = 32
GRANT = {
    "token": "kvg1_test",
    "geometry_digest": "sha256:" + "ab" * 32,
    "authority_generation": 7,
    "expires_at_unix_ms": 1_900_000_000_000,
}


def registration() -> dict[str, Any]:
    return json.loads(
        (FIXTURES / "sglang_striped_registration.json").read_text(encoding="utf-8")
    )


def stripe_segments(prefix: str, start: int, end: int, first_handle: int = 0):
    """One segment per stripe covering ``[start, end)`` of that stripe."""

    segments = []
    for stripe in range(STRIPES):
        segment = {
            "segment_id": f"{prefix}-{stripe}",
            "offset_bytes": start,
            "length_bytes": end - start,
            "handle_index": first_handle + stripe,
        }
        if stripe:
            segment["stripe"] = stripe
        segments.append(segment)
    return segments


def receipt() -> dict[str, Any]:
    """Four stripes; 32 blocks minimum, 64 mapped and 256 virtual per stripe."""

    return {
        "participant_id": "sglang:scheduler",
        "participant_epoch": 3,
        "shared_pools": [
            {
                "binding_id": "cuda:0",
                "capacity_pool_id": "sglang.pool.0",
                "generation": 1,
                "group_ids": ["sglang.kv.0"],
                "memory_domain": {"kind": "cuda", "device_id": 0},
                "block_count": 256,
                "bytes_per_block": STRIPE_BLOCK * STRIPES,
                "allocation_mode": "participant_managed",
                "transport": {"kind": "cuda_vmm"},
                "descriptor": "scm_rights:cuda-vmm-v1",
                "elastic": {
                    "minimum_block_count": 32,
                    "mapped_block_count": 64,
                    "maximum_block_count": 256,
                    "allocation_granularity_bytes": GRANULE,
                    "resize_alignment_blocks": 32,
                    "stripe_count": STRIPES,
                    "segments": stripe_segments("initial", 0, 64 * STRIPE_BLOCK),
                },
            }
        ],
    }


def growth() -> dict[str, Any]:
    """Growth from 64 to 128 blocks: one two-granule tail segment per stripe."""

    return {
        "participant_epoch": 3,
        "resize_generation": 7,
        "binding_id": "cuda:0",
        "stage": "map_workers",
        "from_block_count": 64,
        "target_block_count": 128,
        "bytes_per_block": STRIPE_BLOCK * STRIPES,
        "allocation_granularity_bytes": GRANULE,
        "stripe_count": STRIPES,
        "segments": stripe_segments("grow-7", 64 * STRIPE_BLOCK, 128 * STRIPE_BLOCK),
    }


class RegistrationTests(unittest.TestCase):
    def test_the_adapter_registers_the_pinned_document(self) -> None:
        geometry = PoolGeometry.from_facts(pool_facts_for(PoolShape(), 255))
        built = striped_pool_registration(
            "sglang:scheduler",
            "sha256:model",
            geometry.capacity_groups({"kind": "cuda", "device_id": 0}),
            geometry.topology("sha256:model"),
            adapter_profile("0.3.0", "0.5.23"),
        )
        self.assertEqual(built, registration())

    def test_it_is_a_striped_live_resize_shared_pool_over_cuda_vmm(self) -> None:
        document = registration()
        capabilities = document["capabilities"]
        self.assertEqual(capabilities["abi_version"], {"major": 1, "minor": 6})
        self.assertEqual(
            (
                capabilities["tier"],
                capabilities["metadata_mode"],
                capabilities["ownership"],
            ),
            ("shared_pool", "structured", "kapsl_runtime"),
        )
        self.assertEqual(
            set(capabilities["features"]),
            {
                "capacity_leasing",
                "direct_attention_access",
                "external_pool_attachment",
                "participant_block_selection",
                "live_pool_resize",
                "striped_elastic_pool",
            },
        )
        self.assertEqual(capabilities["transports"], [{"kind": "cuda_vmm"}])
        # The topology version must equal the capabilities version.
        self.assertEqual(document["topology"]["abi_version"], STRIPED_POOL_ABI_VERSION)
        self.assertEqual(document["backend"], "sglang")
        group = document["capacity_model"]["groups"][0]
        # One block is one buffer row across all 2 x layers buffers; the
        # reservation is the token maximum plus SGLang's padding slot.
        self.assertEqual(group["bytes_per_allocation"], 4 * 32)
        self.assertEqual(group["max_allocations"], 255 + 1)
        self.assertEqual(group["allocation_granularity_tokens"], 1)

    def test_the_opaque_tier_still_speaks_abi_1_5(self) -> None:
        self.assertEqual(contract.ABI_VERSION, {"major": 1, "minor": 5})
        self.assertEqual(contract.opaque_capabilities()["abi_version"]["minor"], 5)
        self.assertEqual(
            contract.make_envelope("req-1", "heartbeat")["abi_version"]["minor"], 5
        )
        # A 1.6 answer is only acceptable to a participant that speaks 1.6.
        answer = {
            "abi_version": {"major": 1, "minor": 6},
            "request_id": "r",
            "result": "ack",
        }
        with self.assertRaises(ContractValidationError):
            contract.validate_response(answer, "r")
        contract.validate_response(answer, "r", STRIPED_POOL_ABI_VERSION)
        with self.assertRaises(ContractValidationError):
            contract.validate_registration(registration())

    def test_a_provisioning_grant_adds_its_feature_and_proof_together(self) -> None:
        document = registration()
        granted = striped_pool_registration(
            document["participant_id"],
            document["model_fingerprint"],
            document["capacity_model"]["groups"],
            document["topology"],
            document["adapter_profile"],
            provisioning_grant=GRANT,
        )
        self.assertIn("provisioning_grant", granted["capabilities"]["features"])
        self.assertEqual(granted["provisioning_grant"], GRANT)

        proof_without_feature = {**document, "provisioning_grant": GRANT}
        feature_without_proof = deepcopy(granted)
        del feature_without_proof["provisioning_grant"]
        bad_digest = deepcopy(granted)
        bad_digest["provisioning_grant"]["geometry_digest"] = "sha256:short"
        for name, broken in {
            "proof without feature": proof_without_feature,
            "feature without proof": feature_without_proof,
            "bad digest": bad_digest,
        }.items():
            with self.subTest(name), self.assertRaises(ContractValidationError):
                validate_striped_pool_registration(broken)

    def test_registrations_outside_the_striped_shape_are_not_sent(self) -> None:
        def group(r):
            return r["capacity_model"]["groups"][0]

        def cache_group(r):
            return r["topology"]["cache_groups"][0]

        mutations = {
            "opaque abi": lambda r: r["capabilities"]["abi_version"].update(minor=5),
            "opaque tier": lambda r: r["capabilities"].update(tier="kv_connected"),
            "backend owned": lambda r: r["capabilities"].update(ownership="backend"),
            "no striping": lambda r: r["capabilities"]["features"].remove(
                "striped_elastic_pool"
            ),
            "no live resize": lambda r: r["capabilities"]["features"].remove(
                "live_pool_resize"
            ),
            "extra feature": lambda r: r["capabilities"]["features"].append(
                "prefix_lookup"
            ),
            "fixed transport": lambda r: r["capabilities"].update(
                transports=[{"kind": "cuda_ipc"}]
            ),
            "no profile": lambda r: r.pop("adapter_profile"),
            "empty profile field": lambda r: r["adapter_profile"].update(profile_id=""),
            "no topology": lambda r: r.pop("topology"),
            "topology at 1.5": lambda r: r["topology"]["abi_version"].update(minor=5),
            "other fingerprint": lambda r: r["topology"].update(
                model_fingerprint="sha256:other"
            ),
            "other group": lambda r: cache_group(r).update(group_id="sglang.kv.1"),
            "no layers": lambda r: cache_group(r).update(layers=[]),
            "duplicate layer": lambda r: cache_group(r).update(
                layers=[{"index": 0}, {"index": 0}]
            ),
            "opaque geometry": lambda r: cache_group(r)["geometry"].update(
                kind="opaque"
            ),
            "fp8 elements": lambda r: cache_group(r)["geometry"].update(
                element_type={"kind": "fp8_e4m3"}
            ),
            "sliding window": lambda r: cache_group(r).update(
                policy={"kind": "sliding_window", "window_tokens": 4096}
            ),
            "no maximum": lambda r: group(r).pop("max_allocations"),
            "host domain": lambda r: group(r).update(memory_domains=[{"kind": "host"}]),
            "two domains": lambda r: group(r)["memory_domains"].append(
                {"kind": "cuda", "device_id": 1}
            ),
            "two groups": lambda r: r["capacity_model"]["groups"].append(
                deepcopy(group(r))
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                document = registration()
                mutate(document)
                with self.assertRaises(ContractValidationError):
                    validate_striped_pool_registration(document)


class ReceiptTests(unittest.TestCase):
    def assert_refused(self, mutate, needle: str) -> None:
        document = receipt()
        mutate(document)
        with self.assertRaisesRegex(ContractValidationError, needle):
            validate_striped_pool_receipt(document, registration())

    def test_a_striped_receipt_with_per_stripe_segments_is_accepted(self) -> None:
        validated = validate_striped_pool_receipt(receipt(), registration())
        self.assertEqual(validated, receipt())
        # Several segments may back one stripe as long as they stay dense.
        split = receipt()
        segments = split["shared_pools"][0]["elastic"]["segments"]
        segments[2]["length_bytes"] = GRANULE
        segments.append(
            {
                "segment_id": "initial-2b",
                "stripe": 2,
                "offset_bytes": GRANULE,
                "length_bytes": GRANULE,
                "handle_index": 9,
            }
        )
        validate_striped_pool_receipt(split, registration())

    def test_the_receipt_must_answer_this_registration(self) -> None:
        def binding(r):
            return r["shared_pools"][0]

        cases = {
            "other participant": (
                lambda r: r.update(participant_id="vllm:0"),
                "participant_id",
            ),
            "no epoch": (lambda r: r.update(participant_epoch=0), "participant_epoch"),
            "no binding": (lambda r: r.update(shared_pools=[]), "exactly one binding"),
            "two bindings": (
                lambda r: r["shared_pools"].append(deepcopy(binding(r))),
                "exactly one binding",
            ),
            "other pool": (
                lambda r: binding(r).update(capacity_pool_id="vllm.pool.0"),
                "another capacity pool",
            ),
            "other group": (
                lambda r: binding(r).update(group_ids=["sglang.kv.1"]),
                "does not bind this capacity group",
            ),
            "other device": (
                lambda r: binding(r).update(
                    memory_domain={"kind": "cuda", "device_id": 1}
                ),
                "another memory domain",
            ),
            "fixed transport": (
                lambda r: binding(r).update(transport={"kind": "cuda_ipc"}),
                "unadvertised transport",
            ),
            "runtime leased": (
                lambda r: binding(r).pop("allocation_mode"),
                "allocation mode",
            ),
            "other block size": (
                lambda r: binding(r).update(bytes_per_block=256),
                "bytes_per_allocation",
            ),
            "more blocks than requested": (
                lambda r: (
                    binding(r).update(block_count=288),
                    binding(r)["elastic"].update(maximum_block_count=288),
                ),
                "more blocks than the registration requested",
            ),
            "not elastic": (lambda r: binding(r).pop("elastic"), "elastic"),
        }
        for name, (mutate, needle) in cases.items():
            with self.subTest(name):
                self.assert_refused(mutate, needle)

    def test_stripe_count_is_required_bounded_and_divides_the_block(self) -> None:
        def elastic(r):
            return r["shared_pools"][0]["elastic"]

        # A striped participant never receives a binding without stripes.
        self.assert_refused(
            lambda r: elastic(r).pop("stripe_count"), "must receive a stripe_count"
        )
        for invalid in (1, MAX_POOL_STRIPES + 1):
            with self.subTest(invalid):
                self.assert_refused(
                    lambda r, count=invalid: elastic(r).update(stripe_count=count),
                    "between 2 and 253",
                )
        self.assert_refused(
            lambda r: elastic(r).update(stripe_count=0), "positive integer"
        )
        self.assert_refused(
            lambda r: (elastic(r).update(stripe_count=3), elastic(r)["segments"].pop()),
            "divide evenly across stripes",
        )

    def test_sizes_must_align_to_the_granule_in_each_stripe(self) -> None:
        # In each case the whole binding is a whole number of granules, but
        # each stripe's share ends half way through a granule.
        def elastic(r):
            return r["shared_pools"][0]["elastic"]

        needle = r"\(per stripe\) must align to CUDA VMM granularity"
        self.assert_refused(
            lambda r: elastic(r).update(
                resize_alignment_blocks=16, minimum_block_count=16
            ),
            needle,
        )
        self.assert_refused(
            lambda r: elastic(r).update(
                resize_alignment_blocks=16, mapped_block_count=48
            ),
            needle,
        )
        self.assert_refused(
            lambda r: (
                r["shared_pools"][0].update(block_count=240),
                elastic(r).update(maximum_block_count=240),
            ),
            needle,
        )

    def test_elastic_counts_must_be_ordered_and_aligned(self) -> None:
        def elastic(r):
            return r["shared_pools"][0]["elastic"]

        needle = "aligned minimum and mapped capacity"
        self.assert_refused(lambda r: elastic(r).update(mapped_block_count=16), needle)
        self.assert_refused(lambda r: elastic(r).update(mapped_block_count=288), needle)
        self.assert_refused(
            lambda r: elastic(r).update(maximum_block_count=224), needle
        )
        self.assert_refused(
            lambda r: elastic(r).update(resize_alignment_blocks=48), needle
        )

    def test_every_stripe_must_densely_cover_the_same_mapped_prefix(self) -> None:
        def segments(r):
            return r["shared_pools"][0]["elastic"]["segments"]

        mapped = 64 * STRIPE_BLOCK
        cover = "exactly cover the byte range in every stripe"
        for missing in (1, 3):
            with self.subTest(missing=missing):
                self.assert_refused(
                    lambda r, index=missing: segments(r).pop(index), cover
                )
        self.assert_refused(
            lambda r: segments(r)[1].update(length_bytes=mapped - GRANULE), cover
        )
        self.assert_refused(
            lambda r: segments(r)[1].update(length_bytes=mapped + GRANULE), cover
        )

        # A segment cannot continue into the next stripe: offsets laid out as
        # one concatenated range are refused.
        def concatenate(r):
            for index, segment in enumerate(segments(r)):
                segment.pop("stripe", None)
                segment["offset_bytes"] = index * mapped

        self.assert_refused(concatenate, cover)
        self.assert_refused(
            lambda r: segments(r)[3].update(stripe=STRIPES),
            "must name a stripe of their binding",
        )
        self.assert_refused(
            lambda r: segments(r)[2].update(offset_bytes=GRANULE // 2),
            "granularity-aligned",
        )

    def test_segment_ids_and_handles_are_unique_across_stripes(self) -> None:
        def segments(r):
            return r["shared_pools"][0]["elastic"]["segments"]

        unique = "unique IDs and handles"
        self.assert_refused(
            lambda r: segments(r)[2].update(segment_id="initial-0"), unique
        )
        self.assert_refused(lambda r: segments(r)[2].update(handle_index=0), unique)

    def test_coverage_returns_segments_in_stripe_then_offset_order(self) -> None:
        shuffled = list(reversed(stripe_segments("s", 0, 2 * GRANULE)))
        ordered = check_segment_coverage(
            shuffled, stripes=STRIPES, granularity=GRANULE, start=0, end=2 * GRANULE
        )
        self.assertEqual([segment["stripe"] for segment in ordered], [0, 1, 2, 3])
        self.assertEqual(ordered[0]["segment_id"], "s-0")


class ResizeTests(unittest.TestCase):
    def assert_refused(self, operation: dict[str, Any], needle: str) -> None:
        with self.assertRaisesRegex(ContractValidationError, needle):
            validate_resize_operation(operation)

    def test_a_striped_resize_changes_every_stripe_by_the_same_tail(self) -> None:
        validated = validate_resize_operation(growth())
        self.assertEqual(
            [(s["stripe"], s["offset_bytes"]) for s in validated["segments"]],
            [(stripe, 64 * STRIPE_BLOCK) for stripe in range(STRIPES)],
        )

        activate = {**growth(), "stage": "activate_scheduler", "segments": []}
        self.assertEqual(validate_resize_operation(activate)["segments"], [])

        shrink = {
            **growth(),
            "stage": "unmap_workers",
            "from_block_count": 128,
            "target_block_count": 64,
        }
        validate_resize_operation(shrink)
        retire = {**shrink, "stage": "retire_scheduler", "segments": []}
        validate_resize_operation(retire)

    def test_resize_operations_are_checked_like_the_rust_contract(self) -> None:
        cover = "exactly cover the byte range in every stripe"
        one_stripe_short = growth()
        one_stripe_short["segments"].pop()
        self.assert_refused(one_stripe_short, cover)

        wrong_tail = growth()
        wrong_tail["segments"][1]["offset_bytes"] = 0
        self.assert_refused(wrong_tail, "dense in each stripe")

        cases = {
            "wrong direction": ({"stage": "unmap_workers"}, "direction"),
            "same size": ({"target_block_count": 64}, "must differ"),
            "unknown stage": ({"stage": "commit"}, "unsupported"),
            "no stripes": ({"stripe_count": None}, "must receive a stripe_count"),
            "unaligned target": ({"target_block_count": 80}, "align to CUDA VMM"),
            "scheduler with segments": (
                {"stage": "activate_scheduler"},
                "carry no segments",
            ),
            "worker without segments": ({"segments": []}, "non-empty"),
            "no epoch": ({"participant_epoch": 0}, "participant_epoch"),
            "no generation": ({"resize_generation": 0}, "resize_generation"),
        }
        for name, (changes, needle) in cases.items():
            with self.subTest(name):
                self.assert_refused({**growth(), **changes}, needle)

    def test_poll_and_acknowledgement_requests_match_the_rust_wire_shape(self) -> None:
        shard = single_rank_shard()
        self.assertEqual(
            make_resize_poll_request(
                participant_epoch=3, actor=SCHEDULER_ACTOR, applied_generation=0
            ),
            {
                "participant_epoch": 3,
                "actor": {"role": "scheduler"},
                "applied_generation": 0,
            },
        )
        self.assertEqual(
            make_resize_ack_request(
                participant_epoch=3,
                actor=worker_actor(shard),
                binding_id="cuda:0",
                resize_generation=7,
                stage="map_workers",
                applied_block_count=128,
            ),
            {
                "participant_epoch": 3,
                "actor": {
                    "role": "worker",
                    "shard": {
                        "tensor_parallel_rank": 0,
                        "tensor_parallel_world_size": 1,
                        "pipeline_parallel_rank": 0,
                        "pipeline_parallel_world_size": 1,
                    },
                },
                "binding_id": "cuda:0",
                "resize_generation": 7,
                "stage": "map_workers",
                "applied_block_count": 128,
            },
        )
        for broken in (
            {"role": "scheduler", "shard": shard},
            {"role": "worker"},
            {"role": "worker", "shard": {**shard, "tensor_parallel_rank": 1}},
            {"role": "planner"},
        ):
            with self.subTest(broken), self.assertRaises(ContractValidationError):
                make_resize_poll_request(
                    participant_epoch=3, actor=broken, applied_generation=0
                )
        with self.assertRaises(ContractValidationError):
            make_resize_ack_request(
                participant_epoch=3,
                actor=SCHEDULER_ACTOR,
                binding_id="cuda:0",
                resize_generation=7,
                stage="activate_scheduler",
                applied_block_count=0,
            )


class AttachmentTests(unittest.TestCase):
    def attachment(self, **changes: Any) -> dict[str, Any]:
        geometry = PoolGeometry.from_facts(pool_facts_for(PoolShape(), 255))
        arguments: dict[str, Any] = {
            "participant_epoch": 3,
            "binding_id": "cuda:0",
            "shard": single_rank_shard(),
            "profile": adapter_profile("0.3.0", "0.5.23"),
            "imported_bytes": 256 * 128,
            "mapped_bytes": 64 * 128,
            "views": geometry.attachment_views(),
        }
        arguments.update(changes)
        return make_attachment(**arguments)

    def test_one_view_per_buffer_names_its_layer_and_stripe(self) -> None:
        views = self.attachment()["views"]
        # Stripe order is SGLang's buffer order: k0, k1, v0, v1. Stripe 0 is
        # the wire default and stays off the wire.
        self.assertEqual(
            [(view["layer"], view.get("stripe")) for view in views],
            [
                ({"index": 0}, None),
                ({"index": 1}, 1),
                ({"index": 0}, 2),
                ({"index": 1}, 3),
            ],
        )
        # Offsets are relative to the stripe; each view is the whole stripe.
        self.assertEqual(
            {(view["offset_bytes"], view["length_bytes"]) for view in views},
            {(0, 256 * 32)},
        )
        # Totals stay sums over all stripes.
        self.assertEqual(self.attachment()["imported_bytes"], 4 * 256 * 32)

    def test_malformed_attachments_are_not_sent(self) -> None:
        views = self.attachment()["views"]
        duplicate = deepcopy(views)
        duplicate[1]["stripe"] = 2
        duplicate[1]["layer"] = {"index": 0}
        too_long = deepcopy(views)
        too_long[0]["length_bytes"] = 256 * 128 + 1
        cases = {
            "duplicate group/layer/stripe": {"views": duplicate},
            "view past the binding": {"views": too_long},
            "no views": {"views": []},
            "mapped past imported": {"mapped_bytes": 256 * 128 + 128},
            "nothing mapped": {"mapped_bytes": 0},
            "no epoch": {"participant_epoch": 0},
            "bad shard": {"shard": {**single_rank_shard(), "tensor_parallel_rank": 1}},
            "empty profile": {"profile": {}},
        }
        for name, changes in cases.items():
            with self.subTest(name), self.assertRaises(ContractValidationError):
                self.attachment(**changes)


if __name__ == "__main__":
    unittest.main()
