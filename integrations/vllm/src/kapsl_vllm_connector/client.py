"""Transport adapter for Kapsl's newline-delimited JSON KV control protocol."""

from __future__ import annotations

import json
import array
import os
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlsplit

from .contract import (
    ContractValidationError,
    make_envelope,
    validate_resize_ack_request,
    validate_resize_operation,
    validate_resize_poll_request,
    validate_lease,
    validate_registration_receipt,
    validate_registration,
    validate_reserve_request,
    validate_response,
    validate_shared_pool_attachment,
    validate_shared_pool_detach_request,
)


class KapslKvControlError(RuntimeError):
    """Control-plane connection, protocol, or coordinator error."""

    def __init__(self, message: str, *, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


# Pauses between attempts of an idempotent request whose transport failed.
# The coordinator serializes every participant behind one lock, so a healthy
# participant can wait out another one's slow lifecycle step. Four attempts at
# the default two-second timeout bound the wait near nine seconds, well inside
# the thirty-second lease TTL.
DEFAULT_TRANSPORT_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.05, 0.25, 1.0)


class KapslKvControlClient:
    """Synchronous client used at vLLM scheduler lifecycle boundaries.

    A fresh Unix connection is used for every operation. This avoids sharing a
    socket across vLLM's forked scheduler/worker processes.

    Lease operations that the coordinator treats as idempotent (reserve,
    commit, release) are repeated after a transport failure instead of being
    surfaced to vLLM, where one timed-out request would end the engine.
    """

    def __init__(
        self,
        endpoint: str,
        participant_id: str,
        *,
        timeout_seconds: float = 2.0,
        max_frame_bytes: int = 1024 * 1024,
        request_id_factory: Callable[[], str] | None = None,
        transport_retry_delays_seconds: Sequence[float] = (
            DEFAULT_TRANSPORT_RETRY_DELAYS_SECONDS
        ),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.socket_path = _unix_socket_path(endpoint)
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
        retry_delays = tuple(float(delay) for delay in transport_retry_delays_seconds)
        if any(delay < 0 for delay in retry_delays):
            raise ValueError("transport retry delays cannot be negative")
        self._transport_retry_delays_seconds = retry_delays
        self._sleep = sleep

    def register(self, registration: Mapping[str, Any]) -> dict[str, Any]:
        receipt, handles = self.register_with_handles(registration)
        _close_handles(handles)
        return receipt

    def register_with_handles(
        self, registration: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[int]]:
        validate_registration(registration)
        response, handles = self._rpc_with_handles(
            "register", registration=dict(registration)
        )
        try:
            self._expect(response, {"registered"})
            receipt = validate_registration_receipt(
                _as_mapping(response.get("receipt"), "receipt"),
                self.participant_id,
            )
            tier = _as_mapping(
                registration.get("capabilities"), "capabilities"
            ).get("tier")
            if tier == "shared_pool" and not receipt.get("shared_pools"):
                raise ContractValidationError(
                    "shared_pool vLLM registration received no physical bindings"
                )
            if tier != "shared_pool" and receipt.get("shared_pools"):
                raise ContractValidationError(
                    "opaque vLLM registration cannot receive shared-pool bindings"
                )
            live_resize = "live_pool_resize" in _as_mapping(
                registration.get("capabilities"), "capabilities"
            ).get("features", [])
            receipt_live = any(
                _as_mapping(pool, "shared pool").get("elastic") is not None
                for pool in receipt.get("shared_pools", [])
            )
            if live_resize != receipt_live:
                raise ContractValidationError(
                    "registration receipt live-resize mode does not match the request"
                )
            if live_resize:
                referenced = {
                    int(segment["handle_index"])
                    for pool in receipt.get("shared_pools", [])
                    for segment in _as_mapping(
                        _as_mapping(pool, "shared pool").get("elastic"),
                        "elastic shared pool",
                    ).get("segments", [])
                }
                if referenced != set(range(len(handles))):
                    raise ContractValidationError(
                        "registration VMM handles do not exactly match segment indices"
                    )
            elif handles:
                raise ContractValidationError(
                    "fixed shared-pool registration returned unexpected OS handles"
                )
            return receipt, handles
        except ContractValidationError as error:
            _close_handles(handles)
            raise KapslKvControlError(
                f"invalid registration receipt: {error}"
            ) from error
        except Exception:
            _close_handles(handles)
            raise

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validate_reserve_request(request)
        # The coordinator returns the existing lease when the same sequence
        # repeats an identical reservation, so a lost reply is safe to repeat.
        response = self._rpc_idempotent(
            "reserve",
            participant_id=self.participant_id,
            request=dict(request),
        )
        self._expect(response, {"lease"})
        try:
            return validate_lease(_as_mapping(response.get("lease"), "lease"))
        except ContractValidationError as error:
            raise KapslKvControlError(f"invalid lease response: {error}") from error

    def attach(self, attachment: Mapping[str, Any]) -> None:
        validate_shared_pool_attachment(attachment)
        response = self._rpc(
            "attach",
            participant_id=self.participant_id,
            attachment=dict(attachment),
        )
        self._expect(response, {"ack"})

    def activate(self, participant_epoch: int) -> None:
        if participant_epoch <= 0:
            raise ValueError("participant_epoch must be positive")
        response = self._rpc(
            "activate",
            participant_id=self.participant_id,
            participant_epoch=int(participant_epoch),
        )
        self._expect(response, {"ack"})

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        if not lease_id.strip():
            raise ValueError("lease_id must not be empty")
        if computed_tokens < 0:
            raise ValueError("computed_tokens cannot be negative")
        response = self._rpc_idempotent(
            "commit",
            participant_id=self.participant_id,
            request={
                "lease_id": lease_id,
                "computed_tokens": int(computed_tokens),
            },
        )
        self._expect(response, {"ack"})

    def touch(self, lease_id: str) -> None:
        response = self._rpc(
            "touch",
            participant_id=self.participant_id,
            lease_id=_required(lease_id, "lease_id"),
        )
        self._expect(response, {"ack"})

    def heartbeat(self) -> None:
        response = self._rpc(
            "heartbeat",
            participant_id=self.participant_id,
        )
        self._expect(response, {"ack"})

    def release(
        self, lease_id: str, *, completion: Mapping[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "participant_id": self.participant_id,
            "lease_id": _required(lease_id, "lease_id"),
        }
        if completion is not None:
            payload["completion"] = dict(completion)
        reply_lost = False
        for delay in (*self._transport_retry_delays_seconds, None):
            try:
                response = self._rpc("release", **payload)
            except KapslKvControlError as error:
                if error.kind == "not_found" and reply_lost:
                    # An earlier attempt released the lease and only its reply
                    # was lost; a lease that expired meanwhile is gone as well.
                    return
                if error.kind != "transport" or delay is None:
                    raise
                reply_lost = True
                self._sleep(delay)
                continue
            self._expect(response, {"ack"})
            return

    def detach(self, request: Mapping[str, Any]) -> None:
        validate_shared_pool_detach_request(request)
        response = self._rpc(
            "detach",
            participant_id=self.participant_id,
            request=dict(request),
        )
        self._expect(response, {"ack"})

    def poll_resize(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        operations, handles = self.poll_resize_with_handles(request)
        _close_handles(handles)
        return operations

    def poll_resize_with_handles(
        self, request: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        operations, handles, _ = self.poll_resize_state_with_handles(request)
        return operations, handles

    def poll_resize_state_with_handles(
        self, request: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[int], bool]:
        validate_resize_poll_request(request)
        response, handles = self._rpc_with_handles(
            "resize_poll",
            participant_id=self.participant_id,
            request=dict(request),
        )
        try:
            self._expect(response, {"resize"})
            pending = response.get("pending")
            if not isinstance(pending, bool):
                raise ContractValidationError("resize pending must be a boolean")
            operations = response.get("operations", [])
            if not isinstance(operations, list):
                raise ContractValidationError("resize operations must be a list")
            operations = [
                validate_resize_operation(
                    _as_mapping(operation, "resize operation")
                )
                for operation in operations
            ]
            referenced = {
                int(segment["handle_index"])
                for operation in operations
                if operation["stage"] == "map_workers"
                for segment in operation.get("segments", [])
            }
            if referenced != set(range(len(handles))):
                raise ContractValidationError(
                    "resize VMM handles do not exactly match map segment indices"
                )
            if operations and not pending:
                raise ContractValidationError(
                    "resize operations require a pending transaction"
                )
            return operations, handles, pending
        except ContractValidationError as error:
            _close_handles(handles)
            raise KapslKvControlError(
                f"invalid resize operation: {error}"
            ) from error
        except Exception:
            _close_handles(handles)
            raise

    def ack_resize(self, request: Mapping[str, Any]) -> None:
        validate_resize_ack_request(request)
        response = self._rpc(
            "resize_ack",
            participant_id=self.participant_id,
            request=dict(request),
        )
        self._expect(response, {"ack"})

    def _rpc_idempotent(self, operation: str, **payload: Any) -> dict[str, Any]:
        """Repeat a request the coordinator answers identically when replayed."""

        for delay in self._transport_retry_delays_seconds:
            try:
                return self._rpc(operation, **payload)
            except KapslKvControlError as error:
                if error.kind != "transport":
                    raise
            self._sleep(delay)
        return self._rpc(operation, **payload)

    def _rpc(self, operation: str, **payload: Any) -> dict[str, Any]:
        response, handles = self._rpc_with_handles(operation, **payload)
        if handles:
            _close_handles(handles)
            raise KapslKvControlError(
                f"KV control response '{operation}' carried unexpected OS handles"
            )
        return response

    def _rpc_with_handles(
        self, operation: str, **payload: Any
    ) -> tuple[dict[str, Any], list[int]]:
        request_id = self._request_id_factory()
        envelope = make_envelope(request_id, operation, **payload)
        frame = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(frame) + 1 > self.max_frame_bytes:
            raise KapslKvControlError("KV control request exceeds maximum frame size")

        handles: list[int] = []
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(self.socket_path)
                connection.sendall(frame + b"\n")
                raw_response, handles = _read_frame_with_handles(
                    connection, self.max_frame_bytes
                )
        except (OSError, TimeoutError) as error:
            _close_handles(handles)
            raise KapslKvControlError(
                f"KV control request '{operation}' failed: {error}", kind="transport"
            ) from error

        try:
            decoded = json.loads(raw_response)
            response = dict(_as_mapping(decoded, "response"))
            validate_response(response, request_id)
        except (UnicodeDecodeError, json.JSONDecodeError, ContractValidationError) as error:
            _close_handles(handles)
            raise KapslKvControlError(f"invalid KV control response: {error}") from error

        if response["result"] == "error":
            _close_handles(handles)
            remote = _as_mapping(response.get("error"), "error")
            kind = str(remote.get("kind", "internal"))
            message = str(remote.get("message") or remote.get("operation") or kind)
            raise KapslKvControlError(message, kind=kind)
        return response, handles

    @staticmethod
    def _expect(response: Mapping[str, Any], allowed: set[str]) -> None:
        result = response.get("result")
        if result not in allowed:
            expected = ", ".join(sorted(allowed))
            raise KapslKvControlError(
                f"unexpected KV control result '{result}', expected {expected}"
            )


def _unix_socket_path(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("kapsl_control_endpoint must be an absolute unix:/// socket URL")
    path = unquote(parsed.path)
    if not path.startswith("/") or path == "/":
        raise ValueError("kapsl_control_endpoint must contain an absolute socket path")
    return path


def _read_frame_with_handles(
    connection: socket.socket, max_frame_bytes: int
) -> tuple[bytes, list[int]]:
    chunks: list[bytes] = []
    handles: list[int] = []
    size = 0
    while True:
        chunk, ancillary, _, _ = connection.recvmsg(
            min(65536, max_frame_bytes - size + 1),
            socket.CMSG_SPACE(64 * array.array("i").itemsize),
        )
        for level, kind, payload in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                _close_handles(handles)
                raise KapslKvControlError(
                    "KV control response carried unsupported ancillary data"
                )
            descriptors = array.array("i")
            usable = len(payload) - (len(payload) % descriptors.itemsize)
            descriptors.frombytes(payload[:usable])
            handles.extend(int(descriptor) for descriptor in descriptors)
            if len(handles) > 64:
                _close_handles(handles)
                raise KapslKvControlError(
                    "KV control response carries too many OS handles"
                )
        if not chunk:
            _close_handles(handles)
            raise KapslKvControlError("KV control peer closed before a complete frame")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            break
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_frame_bytes:
            _close_handles(handles)
            raise KapslKvControlError("KV control response exceeds maximum frame size")
    frame = b"".join(chunks)
    if not frame:
        _close_handles(handles)
        raise KapslKvControlError("KV control response is empty")
    if len(frame) > max_frame_bytes:
        _close_handles(handles)
        raise KapslKvControlError("KV control response exceeds maximum frame size")
    return frame, handles


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> bytes:
    frame, handles = _read_frame_with_handles(connection, max_frame_bytes)
    _close_handles(handles)
    return frame


def _close_handles(handles: list[int]) -> None:
    for descriptor in handles:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _required(value: str, field: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value
