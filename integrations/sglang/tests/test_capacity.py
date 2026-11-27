import unittest
from dataclasses import replace

from kapsl_sglang_adapter.capacity import (
    GROUP_ID,
    POOL_ID,
    SchedulerFacts,
    UnsupportedProfileError,
    opaque_capacity_groups,
    pool_token_capacity,
    request_token_capacity,
    require_pinned_version,
    require_supported,
    unsupported_reasons,
)
from kapsl_sglang_adapter.contract import UINT32_MAX, opaque_registration

CUDA0 = [{"kind": "cuda", "device_id": 0}]

FACTS = SchedulerFacts(
    sglang_version="0.5.23",
    device="cuda",
    pool_class="MHATokenToKVPool",
    page_size=16,
    max_total_num_tokens=16_000,
    kv_pool_bytes=1_048_576_000,
    post_capture_sizing=False,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    disaggregation_mode="null",
    speculative_decoding=False,
)


class CapacityModelTests(unittest.TestCase):
    def test_one_page_is_one_allocation(self) -> None:
        (group,) = opaque_capacity_groups(FACTS, CUDA0)
        self.assertEqual(
            group,
            {
                "group_id": GROUP_ID,
                "pool_id": POOL_ID,
                "allocation_granularity_tokens": 16,
                "bytes_per_allocation": 1_048_576,
                "memory_domains": CUDA0,
                "max_allocations": 1000,
            },
        )
        opaque_registration("sglang:scheduler", "sha256:model", [group])

    def test_a_fully_leased_pool_is_charged_at_least_its_physical_bytes(self) -> None:
        # 1001 pages are allocated (one is SGLang's padding page) but only 1000
        # can be leased, so the per-page charge rounds up over the leasable ones.
        facts = replace(FACTS, kv_pool_bytes=1001 * 1_048_576 + 1)
        (group,) = opaque_capacity_groups(facts, CUDA0)
        self.assertGreaterEqual(
            group["bytes_per_allocation"] * group["max_allocations"],
            facts.kv_pool_bytes,
        )
        self.assertLess(
            (group["bytes_per_allocation"] - 1) * group["max_allocations"],
            facts.kv_pool_bytes,
        )

    def test_a_partial_trailing_page_is_not_leasable(self) -> None:
        (group,) = opaque_capacity_groups(
            replace(FACTS, max_total_num_tokens=16_015), CUDA0
        )
        self.assertEqual(group["max_allocations"], 1000)

    def test_the_configured_domain_must_match_the_scheduler_device(self) -> None:
        with self.assertRaisesRegex(UnsupportedProfileError, "'cuda' memory domain"):
            opaque_capacity_groups(FACTS, [{"kind": "host"}])
        with self.assertRaisesRegex(UnsupportedProfileError, "exactly one KV pool"):
            opaque_capacity_groups(
                FACTS,
                [{"kind": "cuda", "device_id": 0}, {"kind": "cuda", "device_id": 1}],
            )
        (group,) = opaque_capacity_groups(
            replace(FACTS, device="cpu"), [{"kind": "host"}]
        )
        self.assertEqual(group["memory_domains"], [{"kind": "host"}])

    def test_domains_are_copied_not_aliased(self) -> None:
        domains = [{"kind": "cuda", "device_id": 0}]
        (group,) = opaque_capacity_groups(FACTS, domains)
        group["memory_domains"][0]["device_id"] = 9
        self.assertEqual(domains, CUDA0)


class ProfileGateTests(unittest.TestCase):
    def test_the_designed_profile_is_supported(self) -> None:
        self.assertEqual(unsupported_reasons(FACTS), [])
        require_supported(FACTS)
        require_supported(replace(FACTS, pool_class="MLATokenToKVPool"))

    def test_every_departure_from_the_profile_is_refused(self) -> None:
        departures = {
            "not a pinned release": replace(FACTS, sglang_version="0.5.24"),
            "no Kapsl memory-domain mapping": replace(FACTS, device="xpu"),
            "HybridLinearKVPool is not modelled": replace(
                FACTS, pool_class="HybridLinearKVPool"
            ),
            "virtual upper bound": replace(FACTS, post_capture_sizing=True),
            "tensor-parallel size 2": replace(FACTS, tensor_parallel_size=2),
            "pipeline-parallel size 2": replace(FACTS, pipeline_parallel_size=2),
            "data-parallel size 4": replace(FACTS, data_parallel_size=4),
            "disaggregation mode 'decode'": replace(
                FACTS, disaggregation_mode="decode"
            ),
            "draft KV pool": replace(FACTS, speculative_decoding=True),
            "feature enable_hisparse": replace(
                FACTS, unsupported_features=("enable_hisparse",)
            ),
            "page_size must be a positive integer": replace(FACTS, page_size=0),
            "max_total_num_tokens must be a positive integer": replace(
                FACTS, max_total_num_tokens=True
            ),
            "kv_pool_bytes must be a positive integer": replace(
                FACTS, kv_pool_bytes=0
            ),
            "smaller than one page": replace(FACTS, max_total_num_tokens=8),
        }
        for reason, facts in departures.items():
            with self.subTest(reason):
                self.assertEqual(len(unsupported_reasons(facts)), 1)
                with self.assertRaisesRegex(UnsupportedProfileError, reason):
                    require_supported(facts)
                with self.assertRaises(UnsupportedProfileError):
                    opaque_capacity_groups(facts, CUDA0)

    def test_all_reasons_are_reported_together(self) -> None:
        facts = replace(
            FACTS,
            tensor_parallel_size=4,
            speculative_decoding=True,
            unsupported_features=("is_hybrid_swa", "dllm_config"),
        )
        self.assertEqual(len(unsupported_reasons(facts)), 4)

    def test_only_exact_pinned_releases_pass_the_version_gate(self) -> None:
        require_pinned_version("0.5.23")
        for version in ("0.5.22", "0.5.23.post1", "0.5.24", "0.0.0.dev0", ""):
            with self.subTest(version), self.assertRaises(UnsupportedProfileError):
                require_pinned_version(version)


class RequestCapacityTests(unittest.TestCase):
    def test_a_request_is_charged_its_prompt_plus_its_output_cap(self) -> None:
        self.assertEqual(
            request_token_capacity(
                input_tokens=100, max_new_tokens=28, pool_tokens=16_000
            ),
            128,
        )

    def test_capacity_is_clamped_to_what_the_pool_can_grant(self) -> None:
        self.assertEqual(
            request_token_capacity(
                input_tokens=100, max_new_tokens=1 << 30, pool_tokens=16_000
            ),
            16_000,
        )
        self.assertEqual(
            request_token_capacity(input_tokens=0, max_new_tokens=0, pool_tokens=16),
            1,
        )

    def test_nonsense_request_sizes_are_rejected(self) -> None:
        for arguments in (
            {"input_tokens": -1, "max_new_tokens": 1, "pool_tokens": 16},
            {"input_tokens": 1, "max_new_tokens": True, "pool_tokens": 16},
            {"input_tokens": 1, "max_new_tokens": 1, "pool_tokens": 0},
        ):
            with self.subTest(arguments), self.assertRaises(ValueError):
                request_token_capacity(**arguments)

    def test_pool_token_capacity_fits_the_reservation_wire_width(self) -> None:
        groups = opaque_capacity_groups(FACTS, CUDA0)
        self.assertEqual(pool_token_capacity(groups), 16_000)
        huge = [{**groups[0], "max_allocations": 1 << 40}]
        self.assertEqual(pool_token_capacity(huge), UINT32_MAX)


if __name__ == "__main__":
    unittest.main()
