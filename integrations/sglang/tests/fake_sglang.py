"""Minimal stand-ins for the SGLang objects the adapter hooks touch.

These fakes exist so the host suite can run without SGLang. They reproduce
only the attribute names and call shapes the adapter was designed against
(SGLang 0.5.23); they are not evidence that a real server behaves this way.
The registry implements the hook semantics SGLang documents for its plugin
framework (BEFORE may rewrite arguments, AFTER may replace the result, AROUND
receives the original callable) and was written from that description.
"""

from __future__ import annotations

import enum
import functools
import json
import pkgutil
import sys
import types
from typing import Any, Callable

PINNED_VERSION = "0.5.23"

PARTICIPANT_CONFIG_ENV = "KAPSL_KV_PARTICIPANT_CONFIG"

# What a managing runtime hands the adapter for one opaque participant.
PARTICIPANT_DOCUMENT = {
    "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
    "kapsl_participant_id": "sglang",
    "kapsl_model_fingerprint": "sha256:model",
    "kapsl_kv_mode": "opaque",
    "kapsl_memory_domains": [{"kind": "cuda", "device_id": 0}],
    "kapsl_lease_ttl_ms": 30000,
}


def environ(**fields: Any) -> dict[str, str]:
    """Launch environment for the document with ``fields`` changed.

    A field set to ``None`` is left out of the document.
    """

    document = {**PARTICIPANT_DOCUMENT, **fields}
    return {
        PARTICIPANT_CONFIG_ENV: json.dumps(
            {name: value for name, value in document.items() if value is not None}
        )
    }


ENVIRON = environ()

_MODULES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.managers",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.schedule_policy",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.memory_pool",
    "sglang.srt.mem_cache.common",
    "sglang.srt.plugins",
    "sglang.srt.plugins.hook_registry",
)


class HookType(enum.Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


class HookRegistry:
    registered: list[tuple[str, Callable[..., Any], HookType]] = []

    @classmethod
    def register(
        cls, target: str, hook: Callable[..., Any], hook_type: HookType = HookType.AFTER
    ) -> None:
        cls.registered.append((target, hook, hook_type))

    @classmethod
    def apply_hooks(cls, *, skip: tuple[str, ...] = ()) -> None:
        for target, hook, hook_type in cls.registered:
            if target in skip:
                continue
            owner_path, attribute = target.rsplit(".", 1)
            owner = pkgutil.resolve_name(owner_path)
            setattr(owner, attribute, _wrap(getattr(owner, attribute), hook, hook_type))

    @classmethod
    def reset(cls) -> None:
        cls.registered = []


def _wrap(original: Callable[..., Any], hook: Callable[..., Any], hook_type: HookType):
    if hook_type is HookType.BEFORE:

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rewritten = hook(*args, **kwargs)
            if rewritten is not None:
                args, kwargs = rewritten
            return original(*args, **kwargs)

    elif hook_type is HookType.AFTER:

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            replaced = hook(result, *args, **kwargs)
            return result if replaced is None else replaced

    elif hook_type is HookType.AROUND:

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return hook(original, *args, **kwargs)

    else:
        raise ValueError(f"unsupported fake hook type {hook_type}")
    return wrapper


class AddReqResult(enum.Enum):
    CONTINUE = enum.auto()
    NO_TOKEN = enum.auto()
    OTHER = enum.auto()


class DisaggregationMode(enum.Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class SpecAlgorithm:
    def __init__(self, none: bool = True) -> None:
        self._none = none

    def is_none(self) -> bool:
        return self._none


class ReqKvInfo:
    def __init__(self) -> None:
        self.req_pool_idx: int | None = None

    @property
    def holds_kv(self) -> bool:
        return self.req_pool_idx is not None


class SamplingParams:
    def __init__(self, max_new_tokens: int | None) -> None:
        self.max_new_tokens = max_new_tokens


class Req:
    def __init__(
        self,
        rid: Any,
        *,
        input_tokens: int = 100,
        max_new_tokens: int | None = 28,
        priority: int | None = None,
    ) -> None:
        self.rid = rid
        self.origin_input_ids = list(range(input_tokens))
        self.output_ids: list[int] = []
        self.sampling_params = SamplingParams(max_new_tokens)
        self.priority = priority
        self.kv = ReqKvInfo()


class MHATokenToKVPool:
    post_capture_active = False

    def __init__(self, size_report: Any) -> None:
        self._size_report = size_report

    def get_kv_size_bytes(self) -> Any:
        return self._size_report


class HybridLinearKVPool(MHATokenToKVPool):
    pass


class Allocator:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def get_kvcache(self) -> Any:
        return self._pool


class ServerArgs:
    def __init__(self, pp_size: int = 1, dp_size: int = 1) -> None:
        self.pp_size = pp_size
        self.dp_size = dp_size


def configure_scheduler(scheduler: Any, server_args: ServerArgs | None = None) -> Any:
    """Give ``scheduler`` the attributes a constructed 0.5.23 scheduler has."""

    scheduler.server_args = server_args or ServerArgs()
    scheduler.device = "cuda"
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 8192
    scheduler.tp_size = 1
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.spec_algorithm = SpecAlgorithm()
    # Pools report K and V bytes separately; 8192 pages of 128 KiB in total.
    scheduler.token_to_kv_pool_allocator = Allocator(
        MHATokenToKVPool((8192 * 65536, 8192 * 65536))
    )
    scheduler.is_hybrid_swa = False
    scheduler.is_hybrid_ssm = False
    scheduler.enable_hisparse = False
    scheduler.enable_unified_memory = False
    scheduler.enable_pdmux = False
    scheduler.attn_dp_enabled = False
    scheduler.enable_session_radix_cache = False
    scheduler.dllm_config = None
    return scheduler


class FakeSglang:
    """A fresh fake ``sglang`` module tree installed into ``sys.modules``."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.modules = {name: types.ModuleType(name) for name in _MODULES}
        for name, module in self.modules.items():
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(self.modules[parent], child, module)
        fake = self

        class ReqToTokenPool:
            def __init__(self) -> None:
                self._next_row = 0

            def alloc(self, reqs: list[Req]) -> list[int]:
                for req in reqs:
                    if not req.kv.holds_kv:
                        req.kv.req_pool_idx = self._next_row
                        self._next_row += 1
                        fake.events.append(f"row:{req.rid}")
                return [req.kv.req_pool_idx for req in reqs]

        class PrefillAdder:
            def __init__(self) -> None:
                self.can_run_list: list[Req] = []
                self.accept = True
                self.error: BaseException | None = None

            def add_one_req(
                self, req: Req, has_chunked_req: bool, truncation_align_size: Any
            ) -> AddReqResult:
                del has_chunked_req, truncation_align_size
                fake.events.append(f"sglang_admission:{req.rid}")
                if self.error is not None:
                    raise self.error
                if not self.accept:
                    return AddReqResult.NO_TOKEN
                self.can_run_list.append(req)
                return AddReqResult.CONTINUE

        def release_kv_cache(req: Req, tree_cache: Any, *, checkpoint: bool) -> None:
            del tree_cache, checkpoint
            req.kv.req_pool_idx = None
            fake.events.append(f"row_released:{req.rid}")

        class Scheduler:
            warm_up_request: Req | None = None

            def __init__(self, server_args: ServerArgs, port_args: Any) -> None:
                del port_args
                fake.events.append("model_loaded")
                configure_scheduler(self, server_args)
                if Scheduler.warm_up_request is not None:
                    # Resolved through the module so an installed hook is hit.
                    fake.memory_pool.ReqToTokenPool().alloc(
                        [Scheduler.warm_up_request]
                    )

        self.scheduler = self.modules["sglang.srt.managers.scheduler"]
        self.schedule_policy = self.modules["sglang.srt.managers.schedule_policy"]
        self.memory_pool = self.modules["sglang.srt.mem_cache.memory_pool"]
        self.common = self.modules["sglang.srt.mem_cache.common"]
        self.hook_registry = self.modules["sglang.srt.plugins.hook_registry"]
        self.scheduler.Scheduler = Scheduler
        self.schedule_policy.PrefillAdder = PrefillAdder
        self.schedule_policy.AddReqResult = AddReqResult
        self.memory_pool.ReqToTokenPool = ReqToTokenPool
        self.common.release_kv_cache = release_kv_cache
        self.hook_registry.HookRegistry = HookRegistry
        self.hook_registry.HookType = HookType

    def install(self) -> "FakeSglang":
        HookRegistry.reset()
        self._previous = {name: sys.modules.get(name) for name in _MODULES}
        sys.modules.update(self.modules)
        return self

    def uninstall(self) -> None:
        HookRegistry.reset()
        for name, previous in self._previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class FakeClient:
    """Records control operations and answers them like an opaque coordinator."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.errors: dict[str, BaseException] = {}
        self._next_lease = 1

    def _record(self, operation: str, *details: Any) -> None:
        self.calls.append((operation, *details))
        error = self.errors.get(operation)
        if error is not None:
            raise error

    def operations(self) -> list[str]:
        return [call[0] for call in self.calls]

    def register(self, registration: dict[str, Any]) -> dict[str, Any]:
        self._record("register", registration)
        return {
            "participant_id": registration["participant_id"],
            "participant_epoch": 1,
        }

    def reserve(self, request: dict[str, Any]) -> dict[str, Any]:
        self._record("reserve", request)
        lease_id = f"kapsl-kv-{self._next_lease:016x}"
        self._next_lease += 1
        return {
            "lease_id": lease_id,
            "sequence": dict(request["sequence"]),
            "groups": [{**group, "blocks": []} for group in request["groups"]],
        }

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        self._record("commit", lease_id, computed_tokens)

    def heartbeat(self) -> None:
        self._record("heartbeat")

    def release(self, lease_id: str) -> None:
        self._record("release", lease_id)
