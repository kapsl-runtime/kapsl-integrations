from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kapsl_sglang_adapter.client import (
    KapslKvControlClient,
    KapslKvControlError,
    parse_unix_endpoint,
)
from kapsl_sglang_adapter.contract import ABI_VERSION, make_reserve_request

ENDPOINT = "unix:///run/kapsl/kv-control.sock"
PARTICIPANT = "sglang:scheduler"
FIXTURES = Path(__file__).with_name("fixtures")
HANG = object()


def registration() -> dict[str, Any]:
    return json.loads(
        (FIXTURES / "sglang_opaque_registration.json").read_text(encoding="utf-8")
    )


def reserve_request() -> dict[str, Any]:
    return make_reserve_request(
        request_id="rid-1",
        sequence_id="rid-1",
        token_capacity=128,
        group_ids=["sglang.kv.0"],
        priority=1,
        ttl_ms=30_000,
    )


def coordinator_reply(request: dict[str, Any]) -> dict[str, Any]:
    """Answer the way the runtime's opaque coordinator serializes responses."""

    base = {"abi_version": dict(ABI_VERSION), "request_id": request["request_id"]}
    operation = request["operation"]
    if operation == "register":
        return {
            **base,
            "result": "registered",
            "receipt": {
                "participant_id": request["registration"]["participant_id"],
                "participant_epoch": 7,
                "shared_pools": [],
            },
        }
    if operation == "reserve":
        return {
            **base,
            "result": "lease",
            "lease": {
                "lease_id": "kapsl-kv-0000000000000001",
                "sequence": request["request"]["sequence"],
                "groups": [
                    {**group, "blocks": []} for group in request["request"]["groups"]
                ],
                "expires_at_unix_ms": 1_800_000_000_000,
            },
        }
    return {**base, "result": "ack"}


def read_line(connection: socket.socket) -> bytes:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
    return bytes(data)


class FakeCoordinator:
    """Serves one control request per connection, like the runtime listener."""

    def __init__(self, respond=coordinator_reply) -> None:
        self.requests: list[dict[str, Any]] = []
        self.frames: list[bytes] = []
        self.errors: list[BaseException] = []
        self._respond = respond
        self._threads: list[threading.Thread] = []
        self._release_hung = threading.Event()

    def connect(self) -> socket.socket:
        client, server = socket.socketpair()
        self._serve_in_thread(server)
        return client

    def _serve_in_thread(self, connection: socket.socket) -> None:
        thread = threading.Thread(target=self._serve, args=(connection,), daemon=True)
        thread.start()
        self._threads.append(thread)

    def _serve(self, connection: socket.socket) -> None:
        try:
            with connection:
                frame = read_line(connection)
                self.frames.append(frame)
                request = json.loads(frame)
                self.requests.append(request)
                reply = self._respond(request)
                if reply is HANG:
                    self._release_hung.wait(timeout=5)
                elif isinstance(reply, bytes):
                    connection.sendall(reply)
                else:
                    connection.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        except BaseException as error:  # surfaced by close()
            self.errors.append(error)

    def close(self) -> None:
        self._release_hung.set()
        for thread in self._threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self._threads):
            raise AssertionError("fake coordinator did not finish")
        if self.errors:
            raise self.errors[0]


class ClientTestCase(unittest.TestCase):
    def coordinator(self, respond=coordinator_reply) -> FakeCoordinator:
        coordinator = FakeCoordinator(respond)
        self.addCleanup(coordinator.close)
        return coordinator

    def client(
        self, coordinator: FakeCoordinator, **options: Any
    ) -> KapslKvControlClient:
        ids = iter(f"req-{index}" for index in range(1, 100))
        return KapslKvControlClient(
            ENDPOINT,
            PARTICIPANT,
            request_id_factory=lambda: next(ids),
            connect=coordinator.connect,
            **options,
        )


class WireShapeTests(ClientTestCase):
    def test_lifecycle_requests_match_the_rust_control_envelopes(self) -> None:
        coordinator = self.coordinator()
        client = self.client(coordinator)

        receipt = client.register(registration())
        lease = client.reserve(reserve_request())
        client.heartbeat()
        client.commit(lease["lease_id"], 117)
        client.release(lease["lease_id"])
        coordinator.close()

        version = {"major": 1, "minor": 5}
        self.assertEqual(receipt["participant_epoch"], 7)
        self.assertEqual(
            coordinator.requests,
            [
                {
                    "abi_version": version,
                    "request_id": "req-1",
                    "operation": "register",
                    "registration": registration(),
                },
                {
                    "abi_version": version,
                    "request_id": "req-2",
                    "operation": "reserve",
                    "participant_id": PARTICIPANT,
                    "request": {
                        "sequence": {"request_id": "rid-1", "sequence_id": "rid-1"},
                        "groups": [
                            {"group_id": "sglang.kv.0", "token_capacity": 128}
                        ],
                        "priority": 1,
                        "ttl_ms": 30_000,
                    },
                },
                {
                    "abi_version": version,
                    "request_id": "req-3",
                    "operation": "heartbeat",
                    "participant_id": PARTICIPANT,
                },
                {
                    "abi_version": version,
                    "request_id": "req-4",
                    "operation": "commit",
                    "participant_id": PARTICIPANT,
                    "request": {
                        "lease_id": "kapsl-kv-0000000000000001",
                        "computed_tokens": 117,
                    },
                },
                {
                    "abi_version": version,
                    "request_id": "req-5",
                    "operation": "release",
                    "participant_id": PARTICIPANT,
                    "lease_id": "kapsl-kv-0000000000000001",
                },
            ],
        )

    def test_each_request_is_one_newline_terminated_json_line(self) -> None:
        coordinator = self.coordinator()
        self.client(coordinator).heartbeat()
        coordinator.close()

        (frame,) = coordinator.frames
        self.assertTrue(frame.endswith(b"\n"))
        self.assertEqual(frame.count(b"\n"), 1)
        self.assertEqual(
            frame[:-1],
            json.dumps(
                coordinator.requests[0], separators=(",", ":"), sort_keys=True
            ).encode("utf-8"),
        )

    def test_registration_must_be_for_this_client(self) -> None:
        coordinator = self.coordinator()
        other = registration()
        other["participant_id"] = "sglang:other"
        with self.assertRaises(ValueError):
            self.client(coordinator).register(other)
        self.assertEqual(coordinator.requests, [])


class FailureTests(ClientTestCase):
    def error_reply(self, kind: str, **fields: Any):
        def respond(request: dict[str, Any]) -> dict[str, Any]:
            return {
                "abi_version": dict(ABI_VERSION),
                "request_id": request["request_id"],
                "result": "error",
                "error": {"kind": kind, **fields},
            }

        return respond

    def test_capacity_denial_keeps_its_machine_readable_kind(self) -> None:
        coordinator = self.coordinator(
            self.error_reply("capacity_exhausted", message="device KV budget exhausted")
        )
        with self.assertRaises(KapslKvControlError) as raised:
            self.client(coordinator).reserve(reserve_request())
        self.assertEqual(raised.exception.kind, "capacity_exhausted")
        self.assertEqual(str(raised.exception), "device KV budget exhausted")

    def test_remote_errors_without_a_message_are_still_reported(self) -> None:
        coordinator = self.coordinator(
            self.error_reply("unsupported", operation="reserve_prefix")
        )
        with self.assertRaises(KapslKvControlError) as raised:
            self.client(coordinator).heartbeat()
        self.assertEqual(raised.exception.kind, "unsupported")
        self.assertEqual(str(raised.exception), "reserve_prefix")

    def test_unreachable_runtime_is_a_transport_error(self) -> None:
        def refuse() -> socket.socket:
            raise ConnectionRefusedError("no listener")

        client = KapslKvControlClient(ENDPOINT, PARTICIPANT, connect=refuse)
        with self.assertRaises(KapslKvControlError) as raised:
            client.heartbeat()
        self.assertEqual(raised.exception.kind, "transport")

    def test_default_connector_reports_a_missing_socket_as_transport(self) -> None:
        # Holds with or without AF_UNIX support on the host platform.
        client = KapslKvControlClient(
            "unix:///nonexistent-kapsl-test-dir/kv-control.sock",
            PARTICIPANT,
            timeout_seconds=0.5,
        )
        with self.assertRaises(KapslKvControlError) as raised:
            client.heartbeat()
        self.assertEqual(raised.exception.kind, "transport")

    def test_unresponsive_runtime_times_out_as_a_transport_error(self) -> None:
        coordinator = self.coordinator(lambda request: HANG)
        client = self.client(coordinator, timeout_seconds=0.05)
        with self.assertRaises(KapslKvControlError) as raised:
            client.heartbeat()
        self.assertEqual(raised.exception.kind, "transport")

    def test_malformed_responses_are_protocol_errors(self) -> None:
        def reply(**changes: Any):
            def respond(request: dict[str, Any]) -> dict[str, Any]:
                return {**coordinator_reply(request), **changes}

            return respond

        cases = {
            "not json": lambda request: b"not json\n",
            "not an object": lambda request: b"[1, 2]\n",
            "closed early": lambda request: b"",
            "empty line": lambda request: b"\n",
            "wrong correlation": reply(request_id="someone-else"),
            "newer abi": reply(abi_version={"major": 1, "minor": 6}),
            "unknown result": reply(result="granted"),
            "error without detail": reply(result="error"),
        }
        for name, respond in cases.items():
            with self.subTest(name):
                coordinator = FakeCoordinator(respond)
                with self.assertRaises(KapslKvControlError) as raised:
                    self.client(coordinator).heartbeat()
                coordinator.close()
                self.assertEqual(raised.exception.kind, "protocol")

    def test_an_operation_rejects_the_wrong_result_type(self) -> None:
        def always_ack(request: dict[str, Any]) -> dict[str, Any]:
            return {
                "abi_version": dict(ABI_VERSION),
                "request_id": request["request_id"],
                "result": "ack",
            }

        coordinator = self.coordinator(always_ack)
        client = self.client(coordinator)
        with self.assertRaises(KapslKvControlError) as raised:
            client.reserve(reserve_request())
        self.assertEqual(raised.exception.kind, "protocol")
        with self.assertRaises(KapslKvControlError):
            client.register(registration())

    def test_a_lease_that_is_not_opaque_is_refused(self) -> None:
        def with_blocks(request: dict[str, Any]) -> dict[str, Any]:
            reply = coordinator_reply(request)
            reply["lease"]["groups"][0]["blocks"] = [
                {
                    "kind": "runtime_pool",
                    "pool_id": "sglang.pool.0",
                    "block_index": 0,
                    "generation": 1,
                }
            ]
            return reply

        def other_sequence(request: dict[str, Any]) -> dict[str, Any]:
            reply = coordinator_reply(request)
            reply["lease"]["sequence"] = {"request_id": "x", "sequence_id": "x"}
            return reply

        for respond in (with_blocks, other_sequence):
            with self.subTest(respond.__name__):
                coordinator = FakeCoordinator(respond)
                with self.assertRaises(KapslKvControlError) as raised:
                    self.client(coordinator).reserve(reserve_request())
                coordinator.close()
                self.assertEqual(raised.exception.kind, "protocol")

    def test_a_receipt_with_shared_pool_bindings_is_refused(self) -> None:
        def shared(request: dict[str, Any]) -> dict[str, Any]:
            reply = coordinator_reply(request)
            reply["receipt"]["shared_pools"] = [{"binding_id": "cuda:0"}]
            return reply

        coordinator = self.coordinator(shared)
        with self.assertRaises(KapslKvControlError) as raised:
            self.client(coordinator).register(registration())
        self.assertEqual(raised.exception.kind, "protocol")

    def test_frames_are_bounded_in_both_directions(self) -> None:
        coordinator = self.coordinator()
        with self.assertRaises(KapslKvControlError) as raised:
            self.client(coordinator, max_frame_bytes=64).register(registration())
        self.assertEqual(raised.exception.kind, "protocol")
        self.assertEqual(coordinator.requests, [])

        # An otherwise valid acknowledgement that is larger than the bound.
        oversized = self.coordinator(
            lambda request: {**coordinator_reply(request), "padding": "x" * 8192}
        )
        with self.assertRaisesRegex(
            KapslKvControlError, "exceeds maximum frame size"
        ) as raised:
            self.client(oversized, max_frame_bytes=4096).heartbeat()
        self.assertEqual(raised.exception.kind, "protocol")
        self.client(oversized, max_frame_bytes=16384).heartbeat()

    def test_local_argument_errors_do_not_reach_the_runtime(self) -> None:
        coordinator = self.coordinator()
        client = self.client(coordinator)
        with self.assertRaises(ValueError):
            client.commit("lease", -1)
        with self.assertRaises(ValueError):
            client.commit(" ", 1)
        with self.assertRaises(ValueError):
            client.release("")
        self.assertEqual(coordinator.requests, [])


class EndpointTests(unittest.TestCase):
    def test_only_absolute_unix_urls_are_accepted(self) -> None:
        self.assertEqual(parse_unix_endpoint(ENDPOINT), "/run/kapsl/kv-control.sock")
        self.assertEqual(
            parse_unix_endpoint("unix:///run/kapsl/kv%20control.sock"),
            "/run/kapsl/kv control.sock",
        )
        for endpoint in (
            "tcp://127.0.0.1:9000",
            "unix://host/run/kapsl.sock",
            "unix:///",
            "unix:///run/kapsl.sock?x=1",
            "/run/kapsl.sock",
            "",
        ):
            with self.subTest(endpoint), self.assertRaises(ValueError):
                parse_unix_endpoint(endpoint)

    def test_constructor_validates_its_arguments(self) -> None:
        for options in (
            {"participant_id": " "},
            {"timeout_seconds": 0},
            {"max_frame_bytes": 0},
        ):
            arguments = {"endpoint": ENDPOINT, "participant_id": PARTICIPANT, **options}
            with self.subTest(options), self.assertRaises(ValueError):
                KapslKvControlClient(**arguments)


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix domain sockets")
class UnixSocketTests(unittest.TestCase):
    def test_round_trip_over_a_real_unix_socket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv-control.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            listener.listen(2)
            coordinator = FakeCoordinator()

            def accept() -> None:
                for _ in range(2):
                    connection, _ = listener.accept()
                    coordinator._serve(connection)

            thread = threading.Thread(target=accept, daemon=True)
            thread.start()
            client = KapslKvControlClient(f"unix://{path}", PARTICIPANT)
            try:
                client.register(registration())
                lease = client.reserve(reserve_request())
            finally:
                thread.join(timeout=5)
                listener.close()
            coordinator.close()

        self.assertEqual(lease["lease_id"], "kapsl-kv-0000000000000001")
        self.assertEqual(
            [request["operation"] for request in coordinator.requests],
            ["register", "reserve"],
        )


if __name__ == "__main__":
    unittest.main()
