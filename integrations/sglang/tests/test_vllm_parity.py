"""Guard against the opaque contract drifting from ``integrations/vllm``.

This package carries a reduced copy of the vLLM connector's contract module
(see the README for why). Until both consume one shared, released client, this
test compares the two wherever they overlap. It loads the sibling module by
file path, so neither package depends on the other at runtime, and skips when
the sibling is not present (for example in a source distribution).
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from kapsl_sglang_adapter import contract

VLLM_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "vllm"
    / "src"
    / "kapsl_vllm_connector"
    / "contract.py"
)
FIXTURES = Path(__file__).with_name("fixtures")


def load_vllm_contract():
    spec = importlib.util.spec_from_file_location(
        "_kapsl_vllm_contract_for_parity", VLLM_CONTRACT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(VLLM_CONTRACT.exists(), "integrations/vllm is not present")
class VllmContractParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vllm = load_vllm_contract()

    def test_both_adapters_speak_the_same_abi_version(self) -> None:
        self.assertEqual(contract.ABI_VERSION, self.vllm.ABI_VERSION)
        for peer in ({"major": 1, "minor": 5}, {"major": 1, "minor": 6}, {"major": 2}):
            self.assertEqual(
                contract.accepts_version(peer), self.vllm.accepts_version(peer)
            )

    def test_opaque_capabilities_and_registration_are_identical(self) -> None:
        self.assertEqual(
            contract.opaque_capabilities(), self.vllm.opaque_capabilities()
        )
        fixture = json.loads(
            (FIXTURES / "opaque_registration.json").read_text(encoding="utf-8")
        )
        arguments = (
            fixture["participant_id"],
            fixture["model_fingerprint"],
            fixture["capacity_model"]["groups"],
        )
        self.assertEqual(
            contract.opaque_registration(*arguments, backend="vllm"),
            self.vllm.opaque_registration(*arguments, backend="vllm"),
        )

    def test_the_sglang_registration_passes_the_vllm_validator(self) -> None:
        registration = json.loads(
            (FIXTURES / "sglang_opaque_registration.json").read_text(encoding="utf-8")
        )
        self.vllm.validate_registration(registration)

    def test_reserve_requests_and_envelopes_are_identical(self) -> None:
        arguments = {
            "request_id": "rid-1",
            "sequence_id": "rid-1",
            "token_capacity": 128,
            "group_ids": ["g.0"],
            "priority": 3,
            "ttl_ms": 30_000,
        }
        request = contract.make_reserve_request(**arguments)
        self.assertEqual(request, self.vllm.make_reserve_request(**arguments))
        payload = {"participant_id": "p", "request": request}
        self.assertEqual(
            contract.make_envelope("req-1", "reserve", **payload),
            self.vllm.make_envelope("req-1", "reserve", **payload),
        )

    def test_runtime_answers_are_accepted_by_both(self) -> None:
        request = contract.make_reserve_request(
            request_id="rid-1",
            sequence_id="rid-1",
            token_capacity=128,
            group_ids=["g.0"],
        )
        lease = {
            "lease_id": "kapsl-kv-0000000000000001",
            "sequence": {"request_id": "rid-1", "sequence_id": "rid-1"},
            "groups": [{"group_id": "g.0", "token_capacity": 128, "blocks": []}],
            "expires_at_unix_ms": 1_800_000_000_000,
        }
        self.assertEqual(
            contract.validate_lease(lease, request), self.vllm.validate_lease(lease)
        )
        receipt = {"participant_id": "p", "participant_epoch": 1, "shared_pools": []}
        self.assertEqual(
            contract.validate_registration_receipt(receipt, "p"),
            self.vllm.validate_registration_receipt(receipt, "p"),
        )
        response = {
            "abi_version": {"major": 1, "minor": 5},
            "request_id": "req-1",
            "result": "lease",
            "lease": lease,
        }
        contract.validate_response(response, "req-1")
        self.vllm.validate_response(response, "req-1")


if __name__ == "__main__":
    unittest.main()
