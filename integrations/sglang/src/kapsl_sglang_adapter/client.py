"""Client for Kapsl's newline-delimited JSON KV control protocol (opaque tier)."""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from .contract import (
    ContractValidationError,
    make_envelope,
    validate_lease,
    validate_registration,
    validate_registration_receipt,
    validate_reserve_request,
    validate_response,
)

Connector = Callable[[], socket.socket]


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
    the envelope request ID. The opaque tier never receives OS handles, so
    frames are read with plain ``recv``; descriptors a peer attached anyway are
    discarded by the kernel instead of leaking into this process.
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

    def _rpc(self, operation: str, **payload: Any) -> dict[str, Any]:
        request_id = self._request_id_factory()
        envelope = make_envelope(request_id, operation, **payload)
        frame = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        if len(frame) + 1 > self.max_frame_bytes:
            raise KapslKvControlError(
                "KV control request exceeds maximum frame size", kind="protocol"
            )

        try:
            with self._connect() as connection:
                connection.settimeout(self.timeout_seconds)
                connection.sendall(frame + b"\n")
                raw_response = _read_frame(connection, self.max_frame_bytes)
        except OSError as error:  # includes socket timeouts
            raise KapslKvControlError(
                f"KV control request '{operation}' failed: {error}", kind="transport"
            ) from error

        try:
            response = dict(_as_mapping(json.loads(raw_response), "response"))
            validate_response(response, request_id)
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


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value
