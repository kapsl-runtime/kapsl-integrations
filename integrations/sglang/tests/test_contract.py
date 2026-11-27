import json
import unittest
from copy import deepcopy
from pathlib import Path

from kapsl_sglang_adapter.contract import (
    ABI_VERSION,
    UINT32_MAX,
    ContractValidationError,
    accepts_version,
    make_envelope,
    make_reserve_request,
    opaque_registration,
    validate_lease,
    validate_registration,
    validate_registration_receipt,
    validate_reserve_request,
    validate_response,
)

FIXTURES = Path(__file__).with_name("fixtures")


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def key_shape(value):
    """Recursive key structure of a JSON document, ignoring leaf values."""

    if isinstance(value, dict):
        return {key: key_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [key_shape(item) for item in value]
    return type(value).__name__


def reserve_request(**overrides):
    arguments = {
        "request_id": "rid-1",
        "sequence_id": "rid-1",
        "token_capacity": 128,
        "group_ids": ["sglang.kv.0"],
        "ttl_ms": 30_000,
    }
    arguments.update(overrides)
    return make_reserve_request(**arguments)


def lease_for(request, **overrides):
    lease = {
        "lease_id": "kapsl-kv-0000000000000001",
        "sequence": dict(request["sequence"]),
        "groups": [{**group, "blocks": []} for group in request["groups"]],
        "expires_at_unix_ms": 1_800_000_000_000,
    }
    lease.update(overrides)
    return lease


class RegistrationTests(unittest.TestCase):
    def test_builder_reproduces_the_kapsl_kv_abi_fixture(self) -> None:
        # The SDK asserts this file equals the Rust serialization of an opaque
        # KvParticipantRegistration, so equality here pins the wire shape.
        fixture = load_fixture("opaque_registration.json")
        group = fixture["capacity_model"]["groups"][0]
        registration = opaque_registration(
            fixture["participant_id"],
            fixture["model_fingerprint"],
            [group],
            backend=fixture["backend"],
        )
        self.assertEqual(registration, fixture)

    def test_sglang_registration_has_the_wire_shape_of_the_sdk_fixture(self) -> None:
        sdk = load_fixture("opaque_registration.json")
        sglang = load_fixture("sglang_opaque_registration.json")
        validate_registration(sglang)
        self.assertEqual(key_shape(sglang), key_shape(sdk))
        self.assertEqual(sglang["capabilities"], sdk["capabilities"])
        self.assertEqual(sglang["backend"], "sglang")

    def test_default_backend_is_sglang(self) -> None:
        fixture = load_fixture("sglang_opaque_registration.json")
        registration = opaque_registration(
            fixture["participant_id"],
            fixture["model_fingerprint"],
            fixture["capacity_model"]["groups"],
        )
        self.assertEqual(registration, fixture)

    def test_only_the_opaque_tier_can_be_registered(self) -> None:
        base = load_fixture("sglang_opaque_registration.json")
        mutations = {
            "shared tier": lambda r: r["capabilities"].update(tier="shared_pool"),
            "structured metadata": lambda r: r["capabilities"].update(
                metadata_mode="structured"
            ),
            "runtime ownership": lambda r: r["capabilities"].update(
                ownership="kapsl_runtime"
            ),
            "extra feature": lambda r: r["capabilities"]["features"].append(
                "prefix_lookup"
            ),
            "extra transport": lambda r: r["capabilities"]["transports"].append(
                {"kind": "cuda_ipc"}
            ),
            "topology": lambda r: r.update(topology={}),
            "adapter profile": lambda r: r.update(adapter_profile={}),
            "provisioning grant": lambda r: r.update(provisioning_grant={}),
            "newer abi": lambda r: r["capabilities"]["abi_version"].update(minor=6),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                registration = deepcopy(base)
                mutate(registration)
                with self.assertRaises(ContractValidationError):
                    validate_registration(registration)

    def test_capacity_model_is_checked_before_it_is_sent(self) -> None:
        base = load_fixture("sglang_opaque_registration.json")
        group = base["capacity_model"]["groups"][0]
        mutations = {
            "no groups": lambda r: r["capacity_model"].update(groups=[]),
            "empty identity": lambda r: r.update(model_fingerprint=" "),
            "zero granularity": lambda g: g.update(allocation_granularity_tokens=0),
            "granularity wider than u32": lambda g: g.update(
                allocation_granularity_tokens=UINT32_MAX + 1
            ),
            "boolean bytes": lambda g: g.update(bytes_per_allocation=True),
            "zero max allocations": lambda g: g.update(max_allocations=0),
            "no domains": lambda g: g.update(memory_domains=[]),
            "duplicate domains": lambda g: g.update(
                memory_domains=[{"kind": "host"}, {"kind": "host"}]
            ),
            "cuda without device": lambda g: g.update(
                memory_domains=[{"kind": "cuda"}]
            ),
            "unknown domain": lambda g: g.update(memory_domains=[{"kind": "tape"}]),
            "provider without name": lambda g: g.update(
                memory_domains=[{"kind": "provider"}]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                registration = deepcopy(base)
                target = (
                    registration
                    if name in {"no groups", "empty identity"}
                    else registration["capacity_model"]["groups"][0]
                )
                mutate(target)
                with self.assertRaises(ContractValidationError):
                    validate_registration(registration)

        duplicate = deepcopy(base)
        duplicate["capacity_model"]["groups"].append(dict(group))
        with self.assertRaises(ContractValidationError):
            validate_registration(duplicate)

        split_pool = deepcopy(base)
        split_pool["capacity_model"]["groups"].append(
            {**group, "group_id": "sglang.kv.1", "memory_domains": [{"kind": "host"}]}
        )
        with self.assertRaises(ContractValidationError):
            validate_registration(split_pool)

    def test_max_allocations_is_optional_for_an_opaque_group(self) -> None:
        registration = load_fixture("sglang_opaque_registration.json")
        del registration["capacity_model"]["groups"][0]["max_allocations"]
        validate_registration(registration)


class ReservationTests(unittest.TestCase):
    def test_reserve_request_matches_the_rust_wire_shape(self) -> None:
        self.assertEqual(
            reserve_request(priority=-2),
            {
                "sequence": {"request_id": "rid-1", "sequence_id": "rid-1"},
                "groups": [{"group_id": "sglang.kv.0", "token_capacity": 128}],
                "priority": -2,
                "ttl_ms": 30_000,
            },
        )
        self.assertNotIn("ttl_ms", reserve_request(ttl_ms=None))

    def test_invalid_reservations_are_rejected_locally(self) -> None:
        for name, overrides in {
            "zero capacity": {"token_capacity": 0},
            "capacity wider than u32": {"token_capacity": UINT32_MAX + 1},
            "boolean capacity": {"token_capacity": True},
            "empty request id": {"request_id": ""},
            "no groups": {"group_ids": []},
            "duplicate groups": {"group_ids": ["sglang.kv.0", "sglang.kv.0"]},
            "priority wider than i32": {"priority": 2**31},
            "boolean priority": {"priority": True},
            "zero ttl": {"ttl_ms": 0},
        }.items():
            with self.subTest(name), self.assertRaises(ContractValidationError):
                reserve_request(**overrides)

        with_prefix = reserve_request()
        with_prefix["prefix"] = {"namespace": "n"}
        with self.assertRaises(ContractValidationError):
            validate_reserve_request(with_prefix)

    def test_lease_must_answer_the_reservation(self) -> None:
        request = reserve_request()
        lease = lease_for(request)
        self.assertEqual(validate_lease(lease, request), lease)
        self.assertIsNot(validate_lease(lease, request), lease)

        larger_grant = lease_for(request)
        larger_grant["groups"][0]["token_capacity"] = 256
        validate_lease(larger_grant, request)

        for name, mutate in {
            "other sequence": lambda l: l["sequence"].update(request_id="rid-2"),
            "other group": lambda l: l["groups"][0].update(group_id="vllm.group.0"),
            "smaller grant": lambda l: l["groups"][0].update(token_capacity=64),
            "block handles": lambda l: l["groups"][0].update(
                blocks=[
                    {
                        "kind": "runtime_pool",
                        "pool_id": "p",
                        "block_index": 0,
                        "generation": 1,
                    }
                ]
            ),
            "blocks not a list": lambda l: l["groups"][0].update(blocks="none"),
            "empty lease id": lambda l: l.update(lease_id=""),
            "no groups": lambda l: l.update(groups=[]),
            "duplicate groups": lambda l: l["groups"].append(dict(l["groups"][0])),
            "zero expiry": lambda l: l.update(expires_at_unix_ms=0),
        }.items():
            with self.subTest(name):
                broken = lease_for(request)
                mutate(broken)
                with self.assertRaises(ContractValidationError):
                    validate_lease(broken, request)

    def test_lease_blocks_field_may_be_omitted(self) -> None:
        request = reserve_request()
        lease = lease_for(request)
        del lease["groups"][0]["blocks"]
        del lease["expires_at_unix_ms"]
        validate_lease(lease, request)


class EnvelopeTests(unittest.TestCase):
    def test_receipt_must_be_opaque_and_addressed_to_this_participant(self) -> None:
        receipt = {"participant_id": "sglang:scheduler", "participant_epoch": 3}
        self.assertEqual(
            validate_registration_receipt(receipt, "sglang:scheduler"), receipt
        )
        validate_registration_receipt(
            {**receipt, "shared_pools": []}, "sglang:scheduler"
        )
        for name, broken in {
            "other participant": {**receipt, "participant_id": "vllm-0"},
            "zero epoch": {**receipt, "participant_epoch": 0},
            "shared pools": {**receipt, "shared_pools": [{"binding_id": "cuda:0"}]},
            "pools not a list": {**receipt, "shared_pools": {}},
        }.items():
            with self.subTest(name), self.assertRaises(ContractValidationError):
                validate_registration_receipt(broken, "sglang:scheduler")

    def test_request_envelope_flattens_the_operation_payload(self) -> None:
        self.assertEqual(
            make_envelope("req-1", "heartbeat", participant_id="sglang:scheduler"),
            {
                "abi_version": {"major": 1, "minor": 5},
                "request_id": "req-1",
                "operation": "heartbeat",
                "participant_id": "sglang:scheduler",
            },
        )
        with self.assertRaises(ContractValidationError):
            make_envelope("", "heartbeat")

    def test_response_envelope_is_version_and_correlation_checked(self) -> None:
        response = {
            "abi_version": dict(ABI_VERSION),
            "request_id": "req-1",
            "result": "ack",
        }
        validate_response(response, "req-1")
        older_minor = {**response, "abi_version": {"major": 1, "minor": 0}}
        validate_response(older_minor, "req-1")
        for name, broken in {
            "other request": {**response, "request_id": "req-2"},
            "newer minor": {**response, "abi_version": {"major": 1, "minor": 6}},
            "other major": {**response, "abi_version": {"major": 2, "minor": 0}},
            "missing version": {"request_id": "req-1", "result": "ack"},
            "unknown result": {**response, "result": "granted"},
        }.items():
            with self.subTest(name), self.assertRaises(ContractValidationError):
                validate_response(broken, "req-1")

    def test_version_negotiation_rejects_non_integer_minors(self) -> None:
        self.assertTrue(accepts_version({"major": 1, "minor": 5}))
        self.assertFalse(accepts_version({"major": 1, "minor": True}))
        self.assertFalse(accepts_version({"major": 1, "minor": "5"}))
        self.assertFalse(accepts_version({"major": 1}))


if __name__ == "__main__":
    unittest.main()
