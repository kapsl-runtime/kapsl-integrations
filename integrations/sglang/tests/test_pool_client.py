"""Shared-pool control operations and OS handle passing over a real socket."""

from __future__ import annotations

import array
import json
import os
import socket
import threading
import unittest
from typing import Any
from unittest import mock

from test_pool_contract import growth, receipt, registration

from kapsl_sglang_adapter import client as client_module
from kapsl_sglang_adapter.client import KapslKvControlClient, KapslKvControlError
from kapsl_sglang_adapter.contract import ABI_VERSION
from kapsl_sglang_adapter.pool_contract import (
    SCHEDULER_ACTOR,
    STRIPED_POOL_ABI_VERSION,
    make_resize_ack_request,
    make_resize_poll_request,
    single_rank_shard,
    worker_actor,
)

ENDPOINT = "unix:///run/kapsl/kv-control.sock"
PARTICIPANT = "sglang:scheduler"
CAN_PASS_HANDLES = hasattr(socket, "SCM_RIGHTS") and hasattr(socket.socket, "sendmsg")


class Pipes:
    """Descriptors to send, each the read end of a pipe.

    A descriptor passed over a socket is duplicated into the receiver. The
    write end reports a broken pipe once every copy of the read end is closed,
    which is how these tests see that the client closed what it received.
    """

    def __init__(self, count: int) -> None:
        self._pipes = [os.pipe() for _ in range(count)]

    @property
    def read_ends(self) -> list[int]:
        return [read_end for read_end, _ in self._pipes]

    def close_sender_copies(self) -> None:
        for read_end, _ in self._pipes:
            os.close(read_end)

    def receiver_copies_open(self) -> int:
        """How many pipes still have a read end open somewhere."""

        still_open = 0
        for _, write_end in self._pipes:
            try:
                os.write(write_end, b"x")
            except BrokenPipeError:
                continue
            still_open += 1
        return still_open

    def close(self) -> None:
        for _, write_end in self._pipes:
            os.close(write_end)


class HandleCoordinator:
    """Serves one request per connection and may attach descriptors."""

    def __init__(self, respond) -> None:
        self.requests: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []
        self._respond = respond
        self._threads: list[threading.Thread] = []

    def connect(self) -> socket.socket:
        client, server = socket.socketpair()
        thread = threading.Thread(target=self._serve, args=(server,), daemon=True)
        thread.start()
        self._threads.append(thread)
        return client

    def _serve(self, connection: socket.socket) -> None:
        try:
            with connection:
                frame = bytearray()
                while not frame.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    frame += chunk
                request = json.loads(frame)
                self.requests.append(request)
                reply, descriptors = self._respond(request)
                data = json.dumps(reply).encode("utf-8") + b"\n"
                if descriptors:
                    # All handles travel with the frame in one SCM_RIGHTS message.
                    sent = connection.sendmsg(
                        [data],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                array.array("i", descriptors),
                            )
                        ],
                    )
                    data = data[sent:]
                if data:
                    # The client may hang up as soon as it has the whole frame.
                    connection.sendall(data)
        except BaseException as error:  # surfaced by close()
            self.errors.append(error)

    def close(self) -> None:
        for thread in self._threads:
            thread.join(timeout=5)
        if self.errors:
            raise self.errors[0]


@unittest.skipUnless(CAN_PASS_HANDLES, "this platform cannot pass OS handles")
class HandleTestCase(unittest.TestCase):
    def pipes(self, count: int) -> Pipes:
        pipes = Pipes(count)
        self.addCleanup(pipes.close)
        return pipes

    def client(self, respond, **options: Any) -> KapslKvControlClient:
        self.coordinator = HandleCoordinator(respond)
        self.addCleanup(self.coordinator.close)
        ids = iter(f"req-{index}" for index in range(1, 100))
        options.setdefault("abi_version", STRIPED_POOL_ABI_VERSION)
        return KapslKvControlClient(
            ENDPOINT,
            PARTICIPANT,
            request_id_factory=lambda: next(ids),
            connect=self.coordinator.connect,
            **options,
        )

    @staticmethod
    def reply(request: dict[str, Any], **body: Any) -> dict[str, Any]:
        return {
            "abi_version": request["abi_version"],
            "request_id": request["request_id"],
            **body,
        }

    def finish(self, pipes: Pipes) -> None:
        """Wait for the coordinator and drop its own copies of the pipes."""

        self.coordinator.close()
        pipes.close_sender_copies()


class RegistrationTests(HandleTestCase):
    def registered(self, pipes: Pipes, mutate=None):
        def respond(request: dict[str, Any]):
            document = receipt()
            if mutate is not None:
                mutate(document)
            return (
                self.reply(request, result="registered", receipt=document),
                pipes.read_ends,
            )

        return self.client(respond)

    def test_registration_returns_the_receipt_and_its_handles(self) -> None:
        pipes = self.pipes(4)
        client = self.registered(pipes)

        granted, handles = client.register_shared_pool(registration())
        self.finish(pipes)

        request = self.coordinator.requests[0]
        self.assertEqual(request["abi_version"], {"major": 1, "minor": 6})
        self.assertEqual(request["operation"], "register")
        self.assertEqual(request["registration"], registration())
        self.assertEqual(granted, receipt())
        # One descriptor per segment, in handle-index order, owned by the caller.
        self.assertEqual(len(handles), 4)
        self.assertEqual(len(set(handles)), 4)
        for handle in handles:
            os.fstat(handle)
        self.assertEqual(pipes.receiver_copies_open(), 4)
        client_module.close_handles(handles)
        self.assertEqual(handles, [])
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_handles_that_do_not_match_the_segments_are_closed(self) -> None:
        for count in (3, 5):
            with self.subTest(count=count):
                pipes = self.pipes(count)
                client = self.registered(pipes)
                with self.assertRaises(KapslKvControlError) as raised:
                    client.register_shared_pool(registration())
                self.finish(pipes)
                self.assertEqual(raised.exception.kind, "protocol")
                self.assertIn("OS handles arrived", str(raised.exception))
                self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_an_invalid_receipt_is_a_protocol_error_and_leaks_nothing(self) -> None:
        def not_striped(document: dict[str, Any]) -> None:
            del document["shared_pools"][0]["elastic"]["stripe_count"]

        pipes = self.pipes(4)
        client = self.registered(pipes, not_striped)
        with self.assertRaises(KapslKvControlError) as raised:
            client.register_shared_pool(registration())
        self.finish(pipes)
        self.assertEqual(raised.exception.kind, "protocol")
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_a_runtime_rejection_keeps_its_kind_and_leaks_nothing(self) -> None:
        pipes = self.pipes(2)

        def respond(request: dict[str, Any]):
            error = {
                "kind": "invalid_capabilities",
                "message": "shared-pool adapter profile is not allowlisted",
            }
            return self.reply(request, result="error", error=error), pipes.read_ends

        client = self.client(respond)
        with self.assertRaises(KapslKvControlError) as raised:
            client.register_shared_pool(registration())
        self.finish(pipes)
        self.assertEqual(raised.exception.kind, "invalid_capabilities")
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_a_response_from_a_newer_abi_is_refused(self) -> None:
        pipes = self.pipes(4)

        def respond(request: dict[str, Any]):
            reply = self.reply(request, result="registered", receipt=receipt())
            reply["abi_version"] = {"major": 1, "minor": 7}
            return reply, pipes.read_ends

        client = self.client(respond)
        with self.assertRaises(KapslKvControlError) as raised:
            client.register_shared_pool(registration())
        self.finish(pipes)
        self.assertEqual(raised.exception.kind, "protocol")
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_more_handles_than_one_response_may_carry_are_refused(self) -> None:
        pipes = self.pipes(6)
        client = self.registered(pipes)
        with mock.patch.object(client_module, "MAX_RESPONSE_HANDLES", 4):
            with self.assertRaises(KapslKvControlError) as raised:
                client.register_shared_pool(registration())
        self.finish(pipes)
        self.assertIn("too many OS handles", str(raised.exception))
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_a_shared_pool_registration_needs_the_striped_shape(self) -> None:
        client = self.client(lambda request: (self.reply(request, result="ack"), []))
        opaque = json.loads(
            (
                __import__("pathlib").Path(__file__).with_name("fixtures")
                / "sglang_opaque_registration.json"
            ).read_text(encoding="utf-8")
        )
        with self.assertRaises(ValueError):
            client.register_shared_pool(opaque)
        other = {**registration(), "participant_id": "sglang:other"}
        with self.assertRaisesRegex(ValueError, "does not match this client"):
            client.register_shared_pool(other)
        self.assertEqual(self.coordinator.requests, [])


class ResizeTests(HandleTestCase):
    def poll_request(self, actor: dict[str, Any]) -> dict[str, Any]:
        return make_resize_poll_request(
            participant_epoch=3, actor=actor, applied_generation=6
        )

    def test_an_idle_poll_reports_nothing_pending(self) -> None:
        client = self.client(
            lambda request: (
                self.reply(request, result="resize", pending=False, operations=[]),
                [],
            )
        )
        self.assertEqual(
            client.poll_resize(self.poll_request(SCHEDULER_ACTOR)), ([], [], False)
        )
        request = self.coordinator.requests[0]
        self.assertEqual(request["operation"], "resize_poll")
        self.assertEqual(request["participant_id"], PARTICIPANT)
        self.assertEqual(
            request["request"],
            {
                "participant_epoch": 3,
                "actor": {"role": "scheduler"},
                "applied_generation": 6,
            },
        )

    def test_a_map_operation_arrives_with_one_handle_per_stripe(self) -> None:
        pipes = self.pipes(4)
        client = self.client(
            lambda request: (
                self.reply(
                    request, result="resize", pending=True, operations=[growth()]
                ),
                pipes.read_ends,
            )
        )
        actor = worker_actor(single_rank_shard())
        operations, handles, pending = client.poll_resize(self.poll_request(actor))
        self.finish(pipes)
        self.assertTrue(pending)
        self.assertEqual(operations[0]["stage"], "map_workers")
        # Segments come back in stripe order with the stripe filled in.
        self.assertEqual(
            [segment["stripe"] for segment in operations[0]["segments"]], [0, 1, 2, 3]
        )
        self.assertEqual(len(handles), 4)
        client_module.close_handles(handles)
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_malformed_resize_answers_are_protocol_errors(self) -> None:
        activate = {**growth(), "stage": "activate_scheduler", "segments": []}
        broken = growth()
        broken["segments"].pop()
        cases = {
            "operations without a pending resize": (
                {"pending": False, "operations": [activate]},
                0,
            ),
            "pending is not a boolean": ({"pending": 1, "operations": []}, 0),
            "operations is not a list": ({"pending": True, "operations": {}}, 0),
            "a stripe is missing its segment": (
                {"pending": True, "operations": [broken]},
                3,
            ),
            "handles on a scheduler stage": (
                {"pending": True, "operations": [activate]},
                2,
            ),
            "map without its handles": ({"pending": True, "operations": [growth()]}, 0),
        }
        for name, (body, descriptors) in cases.items():
            with self.subTest(name):
                pipes = self.pipes(descriptors)
                client = self.client(
                    lambda request, body=body, pipes=pipes: (
                        self.reply(request, result="resize", **body),
                        pipes.read_ends,
                    )
                )
                with self.assertRaises(KapslKvControlError) as raised:
                    client.poll_resize(self.poll_request(SCHEDULER_ACTOR))
                self.finish(pipes)
                self.assertEqual(raised.exception.kind, "protocol")
                self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_lifecycle_requests_match_the_rust_control_envelopes(self) -> None:
        client = self.client(lambda request: (self.reply(request, result="ack"), []))
        acknowledgement = make_resize_ack_request(
            participant_epoch=3,
            actor=SCHEDULER_ACTOR,
            binding_id="cuda:0",
            resize_generation=7,
            stage="activate_scheduler",
            applied_block_count=128,
        )
        client.activate(3)
        client.ack_resize(acknowledgement)
        self.coordinator.close()
        version = {"major": 1, "minor": 6}
        self.assertEqual(
            self.coordinator.requests,
            [
                {
                    "abi_version": version,
                    "request_id": "req-1",
                    "operation": "activate",
                    "participant_id": PARTICIPANT,
                    "participant_epoch": 3,
                },
                {
                    "abi_version": version,
                    "request_id": "req-2",
                    "operation": "resize_ack",
                    "participant_id": PARTICIPANT,
                    "request": acknowledgement,
                },
            ],
        )
        with self.assertRaises(ValueError):
            client.activate(0)


@unittest.skipUnless(CAN_PASS_HANDLES, "this platform cannot pass OS handles")
class OpaqueClientTests(HandleTestCase):
    def test_an_opaque_client_never_takes_handles_a_peer_attaches(self) -> None:
        pipes = self.pipes(3)
        client = self.client(
            lambda request: (self.reply(request, result="ack"), pipes.read_ends),
            abi_version=ABI_VERSION,
        )
        client.heartbeat()
        self.finish(pipes)
        self.assertEqual(self.coordinator.requests[0]["abi_version"]["minor"], 5)
        # Read with plain recv: the kernel discarded the descriptors.
        self.assertEqual(pipes.receiver_copies_open(), 0)

    def test_shared_pool_operations_that_carry_no_handles_take_none(self) -> None:
        pipes = self.pipes(2)
        client = self.client(
            lambda request: (self.reply(request, result="ack"), pipes.read_ends)
        )
        client.activate(3)
        self.finish(pipes)
        self.assertEqual(pipes.receiver_copies_open(), 0)


if __name__ == "__main__":
    unittest.main()
