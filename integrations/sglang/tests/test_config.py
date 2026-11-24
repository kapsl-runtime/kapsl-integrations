import unittest

from kapsl_sglang_adapter.config import (
    DEFAULT_DENIAL_BACKOFF_MS,
    DEFAULT_LEASE_TTL_MS,
    DEFAULT_TIMEOUT_MS,
    ConfigError,
    load_config,
)

REQUIRED = {
    "KAPSL_KV_CONTROL_ENDPOINT": "unix:///run/kapsl/kv-control.sock",
    "KAPSL_KV_MODEL_FINGERPRINT": "sha256:model",
    "KAPSL_KV_MEMORY_DOMAINS": '[{"kind": "cuda", "device_id": 0}]',
}


class ConfigTests(unittest.TestCase):
    def test_required_identity_with_defaults(self) -> None:
        config = load_config(REQUIRED)
        self.assertEqual(config.control_endpoint, "unix:///run/kapsl/kv-control.sock")
        self.assertEqual(config.model_fingerprint, "sha256:model")
        self.assertEqual(config.memory_domains, ({"kind": "cuda", "device_id": 0},))
        self.assertEqual(config.participant_base, "sglang")
        self.assertEqual(config.participant_id, "sglang:scheduler")
        self.assertEqual(config.lease_ttl_ms, DEFAULT_LEASE_TTL_MS)
        self.assertEqual(config.timeout_ms, DEFAULT_TIMEOUT_MS)
        self.assertEqual(config.denial_backoff_ms, DEFAULT_DENIAL_BACKOFF_MS)

    def test_launcher_supplied_overrides(self) -> None:
        config = load_config(
            {
                **REQUIRED,
                "KAPSL_KV_PARTICIPANT_ID": " kapsl-demo-qwen-0 ",
                "KAPSL_KV_MODE": "Opaque",
                "KAPSL_KV_LEASE_TTL_MS": "5000",
                "KAPSL_KV_TIMEOUT_MS": "250",
                "KAPSL_SGLANG_DENIAL_BACKOFF_MS": "0",
            }
        )
        self.assertEqual(config.participant_id, "kapsl-demo-qwen-0:scheduler")
        self.assertEqual(config.lease_ttl_ms, 5000)
        self.assertEqual(config.timeout_ms, 250)
        self.assertEqual(config.denial_backoff_ms, 0)

    def test_every_required_variable_is_enforced(self) -> None:
        for name in REQUIRED:
            for value in (None, "", "   "):
                environ = dict(REQUIRED)
                if value is None:
                    del environ[name]
                else:
                    environ[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ConfigError, name):
                        load_config(environ)

    def test_invalid_values_are_refused(self) -> None:
        invalid = {
            "KAPSL_KV_CONTROL_ENDPOINT": "tcp://127.0.0.1:9000",
            "KAPSL_KV_MEMORY_DOMAINS": "cuda:0",
            "KAPSL_KV_PARTICIPANT_ID": "  ",
            "KAPSL_KV_LEASE_TTL_MS": "999",
            "KAPSL_KV_TIMEOUT_MS": "0",
            "KAPSL_SGLANG_DENIAL_BACKOFF_MS": "-1",
        }
        for name, value in invalid.items():
            with self.subTest(name), self.assertRaisesRegex(ConfigError, name):
                load_config({**REQUIRED, name: value})

        for domains in ("[]", "{}", '[{"kind": "cuda"}]', '[{"kind": "tape"}]', "[1]"):
            with self.subTest(domains), self.assertRaisesRegex(
                ConfigError, "KAPSL_KV_MEMORY_DOMAINS"
            ):
                load_config({**REQUIRED, "KAPSL_KV_MEMORY_DOMAINS": domains})

        with self.assertRaisesRegex(ConfigError, "KAPSL_KV_LEASE_TTL_MS"):
            load_config({**REQUIRED, "KAPSL_KV_LEASE_TTL_MS": "thirty seconds"})

    def test_tiers_that_are_not_implemented_are_refused(self) -> None:
        for mode in ("shared_pool", "elastic", "off"):
            with self.subTest(mode), self.assertRaisesRegex(ConfigError, "opaque"):
                load_config({**REQUIRED, "KAPSL_KV_MODE": mode})


if __name__ == "__main__":
    unittest.main()
