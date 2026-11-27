from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from kapsl_vllm_connector.client import KapslKvControlClient, KapslKvControlError
from kapsl_vllm_connector.contract import make_reserve_request


def _transport() -> KapslKvControlError:
    return KapslKvControlError(
        "KV control request failed: timed out", kind="transport"
    )


def _lease(lease_id: str = "kapsl-kv-1") -> dict[str, Any]:
    return {
        "lease_id": lease_id,
        "sequence": {"request_id": "request-1", "sequence_id": "request-1"},
        "groups": [
            {"group_id": "vllm.group.0", "token_capacity": 4096, "blocks": []}
        ],
        "expires_at_unix_ms": 1,
    }


class ClientTransportRetryTests(unittest.TestCase):
    def _client(self, pauses: list[float]) -> KapslKvControlClient:
        return KapslKvControlClient(
            "unix:///tmp/kapsl-test-kv.sock",
            "vllm-0",
            transport_retry_delays_seconds=(0.1, 0.2),
            sleep=pauses.append,
        )

    def _reserve(self, client: KapslKvControlClient) -> dict[str, Any]:
        return client.reserve(
            make_reserve_request(
                request_id="request-1",
                sequence_id="request-1",
                token_capacity=4096,
                group_ids=["vllm.group.0"],
            )
        )

    def test_reserve_repeats_after_a_transport_failure(self) -> None:
        pauses: list[float] = []
        client = self._client(pauses)
        replies = [_transport(), {"result": "lease", "lease": _lease()}]
        with patch.object(client, "_rpc", side_effect=replies) as rpc:
            lease = self._reserve(client)

        self.assertEqual(lease["lease_id"], "kapsl-kv-1")
        self.assertEqual(rpc.call_count, 2)
        self.assertEqual(pauses, [0.1])
        # The repeat carries the same sequence, which is what makes it safe.
        self.assertEqual(rpc.call_args_list[0], rpc.call_args_list[1])

    def test_commit_gives_up_after_the_configured_attempts(self) -> None:
        pauses: list[float] = []
        client = self._client(pauses)
        with patch.object(
            client, "_rpc", side_effect=[_transport(), _transport(), _transport()]
        ) as rpc:
            with self.assertRaises(KapslKvControlError) as caught:
                client.commit("kapsl-kv-1", 128)

        self.assertEqual(caught.exception.kind, "transport")
        self.assertEqual(rpc.call_count, 3)
        self.assertEqual(pauses, [0.1, 0.2])

    def test_coordinator_answers_are_never_repeated(self) -> None:
        pauses: list[float] = []
        client = self._client(pauses)
        rejection = KapslKvControlError("budget exhausted", kind="capacity_exhausted")
        with patch.object(client, "_rpc", side_effect=[rejection]) as rpc:
            with self.assertRaises(KapslKvControlError) as caught:
                self._reserve(client)

        self.assertIs(caught.exception, rejection)
        self.assertEqual(rpc.call_count, 1)
        self.assertEqual(pauses, [])

    def test_release_accepts_not_found_after_a_lost_reply(self) -> None:
        pauses: list[float] = []
        client = self._client(pauses)
        gone = KapslKvControlError("KV lease does not exist", kind="not_found")
        with patch.object(client, "_rpc", side_effect=[_transport(), gone]) as rpc:
            client.release("kapsl-kv-1")

        self.assertEqual(rpc.call_count, 2)
        self.assertEqual(pauses, [0.1])

    def test_release_reports_not_found_when_no_reply_was_lost(self) -> None:
        client = self._client([])
        gone = KapslKvControlError("KV lease does not exist", kind="not_found")
        with patch.object(client, "_rpc", side_effect=[gone]):
            with self.assertRaises(KapslKvControlError) as caught:
                client.release("kapsl-kv-1")

        self.assertEqual(caught.exception.kind, "not_found")

    def test_release_repeats_until_acknowledged(self) -> None:
        pauses: list[float] = []
        client = self._client(pauses)
        replies = [_transport(), _transport(), {"result": "ack"}]
        with patch.object(client, "_rpc", side_effect=replies) as rpc:
            client.release("kapsl-kv-1")

        self.assertEqual(rpc.call_count, 3)
        self.assertEqual(pauses, [0.1, 0.2])

    def test_negative_retry_delays_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            KapslKvControlClient(
                "unix:///tmp/kapsl-test-kv.sock",
                "vllm-0",
                transport_retry_delays_seconds=(-1.0,),
            )


if __name__ == "__main__":
    unittest.main()
