import unittest

from fake_sglang import ENVIRON, environ
from kapsl_sglang_adapter.config import (
    DEFAULT_DENIAL_BACKOFF_MS,
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
        with self.assertRaisesRegex(ConfigError, "supports only 'opaque'"):
            load_config(environ(kapsl_kv_mode="shared_pool"))
        for mode in ("elastic", "off"):
            with self.subTest(mode), self.assertRaisesRegex(
                ConfigError, "kapsl_kv_mode"
            ):
                load_config(environ(kapsl_kv_mode=mode))
        with self.assertRaisesRegex(ConfigError, "shared_pool mode"):
            load_config(environ(kapsl_live_resize=True))


if __name__ == "__main__":
    unittest.main()
