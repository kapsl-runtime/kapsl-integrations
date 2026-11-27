import json
import unittest
from pathlib import Path

from kapsl_sglang_adapter.participant_config import (
    MODE_OPAQUE,
    MODE_SHARED_POOL,
    PARTICIPANT_CONFIG_ENV,
    ParticipantConfig,
    ParticipantConfigError,
    load_participant_config,
    parse_participant_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ParticipantConfigTests(unittest.TestCase):
    """The reader against verbatim copies of the ``kapsl-kv-abi`` fixtures."""

    def test_opaque_fixture(self) -> None:
        config = parse_participant_config(fixture("participant_config_opaque.json"))
        self.assertEqual(
            config,
            ParticipantConfig(
                control_endpoint="unix:///run/kapsl/kv-control.sock",
                participant_id="kapsl-model-1-0",
                model_fingerprint="sha256:model",
                mode=MODE_OPAQUE,
                memory_domains=({"kind": "cuda", "device_id": 0},),
                lease_ttl_ms=30000,
            ),
        )
        self.assertEqual(config.rank_device_map, {})
        self.assertIsNone(config.timeout_ms)
        self.assertIsNone(config.provisioning_grant)

    def test_shared_pool_fixture(self) -> None:
        document = fixture("participant_config_shared_pool.json")
        config = parse_participant_config(document)
        self.assertEqual(config.mode, MODE_SHARED_POOL)
        self.assertEqual(
            config.memory_domains,
            ({"kind": "cuda", "device_id": 2}, {"kind": "cuda", "device_id": 3}),
        )
        self.assertEqual(config.rank_device_map, {"0": 2, "1": 3})
        self.assertEqual(config.timeout_ms, 2000)
        self.assertTrue(config.live_resize)
        self.assertFalse(config.vmm_conformance)
        self.assertEqual(
            config.provisioning_grant, document["kapsl_provisioning_grant"]
        )

    def test_the_document_is_read_from_one_environment_variable(self) -> None:
        self.assertEqual(PARTICIPANT_CONFIG_ENV, "KAPSL_KV_PARTICIPANT_CONFIG")
        text = (FIXTURES / "participant_config_opaque.json").read_text(encoding="utf-8")
        config = load_participant_config({PARTICIPANT_CONFIG_ENV: text})
        self.assertEqual(config.participant_id, "kapsl-model-1-0")

        for environ in ({}, {PARTICIPANT_CONFIG_ENV: "  "}):
            with self.subTest(environ=environ):
                with self.assertRaisesRegex(
                    ParticipantConfigError, f"{PARTICIPANT_CONFIG_ENV} is required"
                ):
                    load_participant_config(environ)
        for raw in ("{not json", "[]", '"text"'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    ParticipantConfigError, PARTICIPANT_CONFIG_ENV
                ):
                    load_participant_config({PARTICIPANT_CONFIG_ENV: raw})

    def test_unknown_fields_are_ignored_and_optional_fields_default(self) -> None:
        document = {
            "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
            "kapsl_participant_id": " kapsl-model-1-0 ",
            "kapsl_model_fingerprint": "sha256:model",
            "kapsl_kv_mode": "opaque",
            "kapsl_memory_domains": [{"kind": "host"}],
            "kapsl_lease_ttl_ms": 1000,
            "kapsl_added_by_a_newer_runtime": True,
        }
        config = parse_participant_config(document)
        self.assertEqual(config.participant_id, "kapsl-model-1-0")
        self.assertFalse(config.live_resize or config.vmm_conformance)

    def test_every_required_field_is_enforced(self) -> None:
        document = fixture("participant_config_opaque.json")
        for name in (
            "kapsl_control_endpoint",
            "kapsl_participant_id",
            "kapsl_model_fingerprint",
            "kapsl_kv_mode",
            "kapsl_memory_domains",
            "kapsl_lease_ttl_ms",
        ):
            for value in (None, "", "   "):
                changed = dict(document)
                if value is None:
                    del changed[name]
                else:
                    changed[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ParticipantConfigError, name):
                        parse_participant_config(changed)

    def test_inconsistent_documents_are_refused(self) -> None:
        shared = fixture("participant_config_shared_pool.json")
        opaque = fixture("participant_config_opaque.json")
        grant = shared["kapsl_provisioning_grant"]
        cases = [
            (shared, "kapsl_kv_mode", "something_else", "not a known mode"),
            (shared, "kapsl_memory_domains", [], "kapsl_memory_domains"),
            (
                shared,
                "kapsl_memory_domains",
                [{"kind": "cuda", "device_id": 2}] * 2,
                "unique",
            ),
            (shared, "kapsl_memory_domains", [{"kind": "tape"}], "unknown"),
            (shared, "kapsl_memory_domains", "cuda:0", "kapsl_memory_domains"),
            (shared, "kapsl_rank_device_map", {"rank0": 0}, "decimal ranks"),
            (shared, "kapsl_rank_device_map", {"0": -1}, "device IDs"),
            (shared, "kapsl_rank_device_map", [2, 3], "must be an object"),
            (shared, "kapsl_lease_ttl_ms", 0, "kapsl_lease_ttl_ms"),
            (shared, "kapsl_lease_ttl_ms", True, "kapsl_lease_ttl_ms"),
            (shared, "kapsl_lease_ttl_ms", "30000", "kapsl_lease_ttl_ms"),
            (shared, "kapsl_timeout_ms", 0, "kapsl_timeout_ms"),
            (shared, "kapsl_live_resize", "yes", "must be a boolean"),
            (shared, "kapsl_provisioning_grant", {**grant, "token": ""}, "grant"),
            (
                shared,
                "kapsl_provisioning_grant",
                {**grant, "geometry_digest": "sha256:abc"},
                "grant",
            ),
            (
                shared,
                "kapsl_provisioning_grant",
                {**grant, "authority_generation": 0},
                "grant",
            ),
            (shared, "kapsl_provisioning_grant", "kvg1_test", "grant"),
            (opaque, "kapsl_live_resize", True, "shared_pool mode"),
            (opaque, "kapsl_provisioning_grant", grant, "shared_pool mode"),
        ]
        for document, name, value, expected in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ParticipantConfigError, expected):
                    parse_participant_config({**document, name: value})

        conformance = {
            **shared,
            "kapsl_live_resize": False,
            "kapsl_vmm_conformance": True,
        }
        with self.assertRaisesRegex(ParticipantConfigError, "requires"):
            parse_participant_config(conformance)


if __name__ == "__main__":
    unittest.main()
