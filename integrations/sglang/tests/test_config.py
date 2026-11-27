import unittest

from fake_sglang import ENVIRON, POOL_ENVIRON, environ, pool_environ
from kapsl_sglang_adapter.config import (
    DEFAULT_DENIAL_BACKOFF_MS,
    DEFAULT_RESIZE_POLL_MS,
    DEFAULT_TIMEOUT_MS,
    PARTICIPANT_CONFIG_ENV,
    ConfigError,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_runtime_identity_with_adapter_defaults(self) -> None:
        config = load_config(ENVIRON)
        self.assertEqual(config.control_endpoint, "unix:///run/kapsl/kv-control.sock")
        self.assertEqual(config.model_fingerprint, "sha256:model")
        self.assertEqual(config.memory_domains, ({"kind": "cuda", "device_id": 0},))
        self.assertEqual(config.participant_base, "sglang")
        self.assertEqual(config.participant_id, "sglang:scheduler")
        self.assertEqual(config.lease_ttl_ms, 30000)
        self.assertEqual(config.timeout_ms, DEFAULT_TIMEOUT_MS)
        self.assertEqual(config.denial_backoff_ms, DEFAULT_DENIAL_BACKOFF_MS)

    def test_runtime_supplied_values_and_local_tuning(self) -> None:
        config = load_config(
            {
                **environ(
                    kapsl_participant_id="kapsl-demo-qwen-0",
                    kapsl_lease_ttl_ms=5000,
                    kapsl_timeout_ms=250,
                    kapsl_rank_device_map={"0": 0},
                ),
                "KAPSL_SGLANG_DENIAL_BACKOFF_MS": "0",
            }
        )
        self.assertEqual(config.participant_id, "kapsl-demo-qwen-0:scheduler")
        self.assertEqual(config.lease_ttl_ms, 5000)
        self.assertEqual(config.timeout_ms, 250)
        self.assertEqual(config.denial_backoff_ms, 0)

    def test_the_document_is_the_only_identity_channel(self) -> None:
        self.assertEqual(list(ENVIRON), [PARTICIPANT_CONFIG_ENV])
        # The names an earlier draft invented configure nothing.
        legacy = {
            "KAPSL_KV_CONTROL_ENDPOINT": "unix:///run/kapsl/kv-control.sock",
            "KAPSL_KV_MODEL_FINGERPRINT": "sha256:model",
            "KAPSL_KV_MEMORY_DOMAINS": '[{"kind": "cuda", "device_id": 0}]',
        }
        with self.assertRaisesRegex(ConfigError, PARTICIPANT_CONFIG_ENV):
            load_config(legacy)
        self.assertEqual(load_config({**legacy, **ENVIRON}), load_config(ENVIRON))

    def test_a_missing_or_incomplete_document_is_refused(self) -> None:
        for name in (
            "kapsl_control_endpoint",
            "kapsl_participant_id",
            "kapsl_model_fingerprint",
            "kapsl_kv_mode",
            "kapsl_memory_domains",
            "kapsl_lease_ttl_ms",
        ):
            with self.subTest(name), self.assertRaisesRegex(ConfigError, name):
                load_config(environ(**{name: None}))

    def test_values_this_adapter_cannot_honour_are_refused(self) -> None:
        invalid = {
            "kapsl_control_endpoint": "tcp://127.0.0.1:9000",
            "kapsl_memory_domains": "cuda:0",
            "kapsl_participant_id": "  ",
            "kapsl_lease_ttl_ms": 999,
            "kapsl_timeout_ms": 0,
        }
        for name, value in invalid.items():
            with self.subTest(name), self.assertRaisesRegex(ConfigError, name):
                load_config(environ(**{name: value}))

        for raw in ("-1", "soon"):
            with self.subTest(raw), self.assertRaisesRegex(
                ConfigError, "KAPSL_SGLANG_DENIAL_BACKOFF_MS"
            ):
                load_config({**ENVIRON, "KAPSL_SGLANG_DENIAL_BACKOFF_MS": raw})

    def test_tiers_that_are_not_implemented_are_refused(self) -> None:
        # A fixed (CUDA IPC) shared pool is not implemented; only the elastic one.
        with self.assertRaisesRegex(ConfigError, "requires kapsl_live_resize"):
            load_config(environ(kapsl_kv_mode="shared_pool"))
        for mode in ("elastic", "off"):
            with self.subTest(mode), self.assertRaisesRegex(
                ConfigError, "kapsl_kv_mode"
            ):
                load_config(environ(kapsl_kv_mode=mode))
        with self.assertRaisesRegex(ConfigError, "shared_pool mode"):
            load_config(environ(kapsl_live_resize=True))

    def test_the_opaque_tier_is_the_default_and_carries_no_pool_settings(self) -> None:
        config = load_config(ENVIRON)
        self.assertEqual(config.mode, "opaque")
        self.assertFalse(config.shared_pool)
        self.assertIsNone(config.provisioning_grant)

    def test_the_elastic_shared_pool_is_selected_by_the_document(self) -> None:
        config = load_config(POOL_ENVIRON)
        self.assertTrue(config.shared_pool)
        self.assertEqual(config.mode, "shared_pool")
        self.assertIsNone(config.provisioning_grant)
        self.assertEqual(config.resize_poll_ms, DEFAULT_RESIZE_POLL_MS)
        self.assertEqual(config.participant_id, "sglang:scheduler")

        grant = {
            "token": "kvg1_test",
            "geometry_digest": "sha256:" + "0" * 64,
            "authority_generation": 7,
            "expires_at_unix_ms": 1_900_000_000_000,
        }
        granted = load_config(
            {
                **pool_environ(
                    kapsl_provisioning_grant=grant, kapsl_rank_device_map={"0": 0}
                ),
                "KAPSL_SGLANG_RESIZE_POLL_MS": "25",
            }
        )
        self.assertEqual(granted.provisioning_grant, grant)
        self.assertEqual(granted.resize_poll_ms, 25)

    def test_shared_pool_shapes_that_are_not_implemented_are_refused(self) -> None:
        two_devices = [
            {"kind": "cuda", "device_id": 0},
            {"kind": "cuda", "device_id": 1},
        ]
        cases = {
            "conformance mode": (
                {"kapsl_vmm_conformance": True},
                "kapsl_vmm_conformance is not implemented",
            ),
            "two devices": (
                {"kapsl_memory_domains": two_devices},
                "exactly one cuda entry",
            ),
            "host memory": (
                {"kapsl_memory_domains": [{"kind": "host"}]},
                "exactly one cuda entry",
            ),
            "second rank": (
                {"kapsl_rank_device_map": {"0": 0, "1": 1}},
                "may only map rank 0",
            ),
            "rank on another device": (
                {"kapsl_rank_device_map": {"0": 3}},
                "may only map rank 0",
            ),
        }
        for name, (fields, needle) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ConfigError, needle):
                load_config(pool_environ(**fields))
        for raw in ("0", "-5", "often"):
            with self.subTest(raw), self.assertRaisesRegex(
                ConfigError, "KAPSL_SGLANG_RESIZE_POLL_MS"
            ):
                load_config({**POOL_ENVIRON, "KAPSL_SGLANG_RESIZE_POLL_MS": raw})


if __name__ == "__main__":
    unittest.main()
