"""Client for Kapsl's newline-delimited JSON KV control protocol."""

from __future__ import annotations

import array
import json
import os
import socket
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from .contract import (
    ABI_VERSION,
    ContractValidationError,
    make_envelope_for,
    validate_lease,
    validate_registration,
    validate_registration_receipt,
    validate_reserve_request,
    validate_response,
)
from .pool_contract import (
    MAX_POOL_STRIPES,
    STAGE_MAP_WORKERS,
    validate_attachment,
    validate_resize_ack_request,
    validate_resize_operation,
    validate_resize_poll_request,
    validate_striped_pool_receipt,
    validate_striped_pool_registration,
)

Connector = Callable[[], socket.socket]

# One response carries at most this many OS handles (Linux ``SCM_MAX_FD``).
MAX_RESPONSE_HANDLES = MAX_POOL_STRIPES


class KapslKvControlError(RuntimeError):
    """Control-plane connection, protocol, or coordinator error.

    ``kind`` is the runtime's machine-readable error kind for a remote
    rejection (for example ``capacity_exhausted``), ``transport`` when the
    runtime could not be reached, and ``protocol`` when its answer was not a
    valid contract message.
    """

    def __init__(self, message: str, *, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


class KapslKvControlClient:
    """Synchronous client used at SGLang scheduler lifecycle boundaries.

    A fresh connection is used for every operation, so the client holds no
    socket across SGLang's spawned processes and a retry is unambiguous through
    the envelope request ID.

    The opaque tier never receives OS handles, so its frames are read with
    plain ``recv``; descriptors a peer attached anyway are discarded by the
    kernel instead of leaking into this process. Only the two shared-pool
    operations that the contract lets carry handles (registration and resize
    poll) read ancillary data, and every descriptor they receive is either
    returned to the caller, who then owns it, or closed here.

    ``abi_version`` is the contract version this participant speaks: 1.5 for
    the opaque tier, 1.6 for a striped shared pool.
    """

    def __init__(
        self,
        endpoint: str,
        participant_id: str,
        *,
        timeout_seconds: float = 2.0,
        max_frame_bytes: int = 1024 * 1024,
        request_id_factory: Callable[[], str] | None = None,
        connect: Connector | None = None,
        abi_version: Mapping[str, int] = ABI_VERSION,
    ) -> None:
        self.socket_path = parse_unix_endpoint(endpoint)
        if not participant_id.strip():
            raise ValueError("participant_id must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.participant_id = participant_id
        self.timeout_seconds = timeout_seconds
        self.max_frame_bytes = max_frame_bytes
        self.abi_version = dict(abi_version)
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._connect = connect or self._connect_unix

    def register(self, registration: Mapping[str, Any]) -> dict[str, Any]:
        validate_registration(registration)
        if registration.get("participant_id") != self.participant_id:
            raise ValueError("registration participant_id does not match this client")
        response = self._rpc("register", registration=dict(registration))
        self._expect(response, "registered")
        try:
            return validate_registration_receipt(
                _as_mapping(response.get("receipt"), "receipt"), self.participant_id
            )
        except ContractValidationError as error:
            raise KapslKvControlError(
                f"invalid registration receipt: {error}", kind="protocol"
            ) from error

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validate_reserve_request(request)
        response = self._rpc(
            "reserve", participant_id=self.participant_id, request=dict(request)
        )
        self._expect(response, "lease")
        try:
            return validate_lease(_as_mapping(response.get("lease"), "lease"), request)
        except ContractValidationError as error:
            raise KapslKvControlError(
                f"invalid lease response: {error}", kind="protocol"
            ) from error

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        if computed_tokens < 0:
            raise ValueError("computed_tokens cannot be negative")
        response = self._rpc(
            "commit",
            participant_id=self.participant_id,
            request={
                "lease_id": _required(lease_id, "lease_id"),
                "computed_tokens": int(computed_tokens),
            },
        )
        self._expect(response, "ack")

    def heartbeat(self) -> None:
        response = self._rpc("heartbeat", participant_id=self.participant_id)
        self._expect(response, "ack")

    def release(self, lease_id: str) -> None:
        response = self._rpc(
            "release",
            participant_id=self.participant_id,
            lease_id=_required(lease_id, "lease_id"),
        )
        self._expect(response, "ack")

    # Shared pool ----------------------------------------------------------------

    def register_shared_pool(
        self, registration: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[int]]:
        """Register a striped shared pool.

        Returns the receipt and the OS handles of its segments, indexed by each
        segment's ``handle_index``. The caller owns the handles and closes
        them once the segments are mapped.
        """

        validate_striped_pool_registration(registration)
        if registration.get("participant_id") != self.participant_id:
            raise ValueError("registration participant_id does not match this client")
        response, handles = self._rpc_with_handles(
            "register", registration=dict(registration)
        )
        try:
            self._expect(response, "registered")
            try:
                receipt = validate_striped_pool_receipt(
                    _as_mapping(response.get("receipt"), "receipt"), registration
                )
                _require_exact_handles(
                    receipt["shared_pools"][0]["elastic"]["segments"], handles
                )
            except ContractValidationError as error:
                raise KapslKvControlError(
                    f"invalid registration receipt: {error}", kind="protocol"
                ) from error
        except BaseException:
            close_handles(handles)
            raise
        return receipt, handles

    def attach(self, attachment: Mapping[str, Any]) -> None:
        validate_attachment(attachment)
        response = self._rpc(
            "attach", participant_id=self.participant_id, attachment=dict(attachment)
        )
        self._expect(response, "ack")

    def activate(self, participant_epoch: int) -> None:
        if isinstance(participant_epoch, bool) or participant_epoch <= 0:
            raise ValueError("participant_epoch must be positive")
        response = self._rpc(
            "activate",
            participant_id=self.participant_id,
            participant_epoch=int(participant_epoch),
        )
        self._expect(response, "ack")

    def poll_resize(
        self, request: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[int], bool]:
        """Ask for resize operations addressed to the polling actor.

        Returns the validated operations, the OS handles of the segments a
        ``map_workers`` operation maps (owned by the caller), and whether the
        runtime has a resize transaction in progress.
        """

        validate_resize_poll_request(request)
        response, handles = self._rpc_with_handles(
            "resize_poll", participant_id=self.participant_id, request=dict(request)
        )
        try:
            self._expect(response, "resize")
            try:
                pending = response.get("pending")
                if not isinstance(pending, bool):
                    raise ContractValidationError("resize pending must be a boolean")
                raw_operations = response.get("operations", [])
                if not isinstance(raw_operations, list):
                    raise ContractValidationError("resize operations must be a list")
                operations = [
                    validate_resize_operation(_as_mapping(raw, "resize operation"))
                    for raw in raw_operations
                ]
                if operations and not pending:
                    raise ContractValidationError(
                        "resize operations require a pending resize transaction"
                    )
                _require_exact_handles(
                    [
                        segment
                        for operation in operations
                        if operation["stage"] == STAGE_MAP_WORKERS
                        for segment in operation["segments"]
                    ],
                    handles,
                )
            except ContractValidationError as error:
                raise KapslKvControlError(
                    f"invalid resize operation: {error}", kind="protocol"
                ) from error
        except BaseException:
            close_handles(handles)
            raise
        return operations, handles, pending

    def ack_resize(self, request: Mapping[str, Any]) -> None:
        validate_resize_ack_request(request)
        response = self._rpc(
            "resize_ack", participant_id=self.participant_id, request=dict(request)
        )
        self._expect(response, "ack")

    # Transport ------------------------------------------------------------------

    def _rpc(self, operation: str, **payload: Any) -> dict[str, Any]:
        request_id, frame = self._encode(operation, payload)
        try:
            with self._connect() as connection:
                connection.settimeout(self.timeout_seconds)
                connection.sendall(frame + b"\n")
                raw_response = _read_frame(connection, self.max_frame_bytes)
        except OSError as error:  # includes socket timeouts
            raise KapslKvControlError(
                f"KV control request '{operation}' failed: {error}", kind="transport"
            ) from error
        return self._decode(raw_response, request_id)

    def _rpc_with_handles(
        self, operation: str, **payload: Any
    ) -> tuple[dict[str, Any], list[int]]:
        request_id, frame = self._encode(operation, payload)
        handles: list[int] = []
        try:
            try:
                with self._connect() as connection:
                    connection.settimeout(self.timeout_seconds)
                    connection.sendall(frame + b"\n")
                    raw_response = _read_frame_with_handles(
                        connection, self.max_frame_bytes, handles
                    )
            except OSError as error:  # includes socket timeouts
                raise KapslKvControlError(
                    f"KV control request '{operation}' failed: {error}",
                    kind="transport",
                ) from error
            return self._decode(raw_response, request_id), handles
        except BaseException:
            close_handles(handles)
            raise

    def _encode(self, operation: str, payload: Mapping[str, Any]) -> tuple[str, bytes]:
        request_id = self._request_id_factory()
        envelope = make_envelope_for(
            self.abi_version, request_id, operation, **payload
        )
        frame = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        if len(frame) + 1 > self.max_frame_bytes:
            raise KapslKvControlError(
                "KV control request exceeds maximum frame size", kind="protocol"
            )
        return request_id, frame

    def _decode(self, raw_response: bytes, request_id: str) -> dict[str, Any]:
        try:
            response = dict(_as_mapping(json.loads(raw_response), "response"))
            validate_response(response, request_id, self.abi_version)
        except ValueError as error:
            # Covers undecodable bytes, malformed JSON, and contract violations.
            raise KapslKvControlError(
                f"invalid KV control response: {error}", kind="protocol"
            ) from error

        if response["result"] == "error":
            try:
                remote = _as_mapping(response.get("error"), "error")
            except ContractValidationError as error:
                raise KapslKvControlError(
                    f"invalid KV control response: {error}", kind="protocol"
                ) from error
            kind = str(remote.get("kind", "internal"))
            message = str(remote.get("message") or remote.get("operation") or kind)
            raise KapslKvControlError(message, kind=kind)
        return response

    def _connect_unix(self) -> socket.socket:
        family = getattr(socket, "AF_UNIX", None)
        if family is None:
            raise OSError("this platform has no Unix domain sockets")
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            connection.settimeout(self.timeout_seconds)
            connection.connect(self.socket_path)
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _expect(response: Mapping[str, Any], expected: str) -> None:
        result = response.get("result")
        if result != expected:
            raise KapslKvControlError(
                f"unexpected KV control result '{result}', expected {expected}",
                kind="protocol",
            )


def parse_unix_endpoint(endpoint: str) -> str:
    """Return the socket path of an absolute ``unix:///`` control endpoint."""

    parsed = urlsplit(endpoint)
    if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("control endpoint must be an absolute unix:/// socket URL")
    path = unquote(parsed.path)
    if not path.startswith("/") or path == "/":
        raise ValueError("control endpoint must contain an absolute socket path")
    return path


def close_handles(handles: list[int]) -> None:
    """Close received OS handles and empty the list. Never raises."""

    while handles:
        try:
            os.close(handles.pop())
        except OSError:
            pass


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> bytes:
    frame = bytearray()
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            raise KapslKvControlError(
                "KV control peer closed before a complete frame", kind="protocol"
            )
        newline = chunk.find(b"\n")
        frame += chunk if newline < 0 else chunk[:newline]
        if len(frame) > max_frame_bytes:
            raise KapslKvControlError(
                "KV control response exceeds maximum frame size", kind="protocol"
            )
        if newline >= 0:
            break
    if not frame:
        raise KapslKvControlError("KV control response is empty", kind="protocol")
    return bytes(frame)


def _read_frame_with_handles(
    connection: socket.socket, max_frame_bytes: int, handles: list[int]
) -> bytes:
    """Read one frame and append the descriptors that arrived with it.

    ``handles`` is filled as descriptors arrive, so a caller that owns the list
    can close them whatever happens after.
    """

    if not hasattr(socket, "SCM_RIGHTS") or not hasattr(connection, "recvmsg"):
        raise OSError("this platform cannot receive OS handles over a socket")
    descriptor_size = array.array("i").itemsize
    ancillary_size = socket.CMSG_SPACE(MAX_RESPONSE_HANDLES * descriptor_size)
    frame = bytearray()
    while True:
        chunk, ancillary, flags, _ = connection.recvmsg(65536, ancillary_size)
        for level, kind, payload in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                raise KapslKvControlError(
                    "KV control response carried unsupported ancillary data",
                    kind="protocol",
                )
            descriptors = array.array("i")
            descriptors.frombytes(
                payload[: len(payload) - len(payload) % descriptor_size]
            )
            handles.extend(int(descriptor) for descriptor in descriptors)
        if flags & socket.MSG_CTRUNC or len(handles) > MAX_RESPONSE_HANDLES:
            raise KapslKvControlError(
                "KV control response carries too many OS handles", kind="protocol"
            )
        if not chunk:
            raise KapslKvControlError(
                "KV control peer closed before a complete frame", kind="protocol"
            )
        newline = chunk.find(b"\n")
        frame += chunk if newline < 0 else chunk[:newline]
        if len(frame) > max_frame_bytes:
            raise KapslKvControlError(
                "KV control response exceeds maximum frame size", kind="protocol"
            )
        if newline >= 0:
            break
    if not frame:
        raise KapslKvControlError("KV control response is empty", kind="protocol")
    return bytes(frame)


def _require_exact_handles(
    segments: list[Mapping[str, Any]], handles: list[int]
) -> None:
    referenced = {int(segment["handle_index"]) for segment in segments}
    if referenced != set(range(len(handles))):
        raise ContractValidationError(
            f"{len(handles)} OS handles arrived for segments that reference "
            f"{len(referenced)} handle indices"
        )


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value
