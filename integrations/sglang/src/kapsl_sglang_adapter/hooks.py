"""SGLang-facing layer: pinned hook targets and the functions installed on them.

SGLang has no KV-connector interface. It does have a documented general plugin
framework (``sglang.srt.plugins``): an entry point is called in every SGLang
process and registers before/after/around hooks on dotted targets. This module
is that plugin. Everything SGLang-specific in the adapter lives here, and the
hook functions only duck-type the objects SGLang passes them, so they are
exercised by host tests without SGLang installed.

Lifecycle (one scheduler per process)::

    inactive --Scheduler.__init__ entered--> starting --registered--> serving
        any startup error ---------------------------------------> failed

* ``Scheduler.__init__`` (around): validate configuration and hook
  installation before the model loads; register the participant after the KV
  pool exists. Any failure propagates and SGLang terminates the server.
* ``PrefillAdder.add_one_req`` (around): take a capacity lease before SGLang's
  own admission; a Kapsl denial keeps the request queued.
* ``ReqToTokenPool.alloc`` (before): refuse to give a request a KV row unless
  it holds a lease. This turns an admission path the adapter does not hook
  into a hard failure instead of an ungoverned allocation.
* ``release_kv_cache`` (after): report completion and return the lease when the
  request's KV row is given back, for any reason.

Those four are all the opaque tier installs. When the participant
configuration asks for the elastic shared pool, four more are registered, on
SGLang's own VMM-backed KV pool (``mem_cache/kv_vmm_backing.py``):

* ``KVCacheConfigurator.config_from_budget`` (before): size the address
  reservation from ``--max-total-tokens`` alone and the backed pool from what
  the runtime mapped, instead of from free GPU memory.
* ``KvVmmBufferOwner._back_spans`` (around): the one place SGLang backs its KV
  buffers. On its first call the pool is registered and the runtime's segments
  are mapped; SGLang's own mapping is never run.
* ``KvVmmArena.commit_range`` (around): refuse. Nothing may create KV memory
  in this mode except the runtime.
* ``Scheduler.get_next_batch_to_run`` (before): between scheduler steps, apply
  a resize the runtime has pending.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import operator
import os
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from ._version import ADAPTER_VERSION
from .capacity import (
    SchedulerFacts,
    UnsupportedProfileError,
    require_pinned_version,
    require_shared_pool_supported,
)
from .client import KapslKvControlClient, KapslKvControlError
from .config import AdapterConfig, ConfigError, load_config
from .participant import KvParticipant
from .pool_contract import STRIPED_POOL_ABI_VERSION
from .pool_participant import SharedPoolParticipant
from .sglang_pool import (
    SglangPoolHost,
    SglangVmmBackend,
    lift_kv_budget,
    pool_facts,
    shared_pool_details,
    shared_pool_launch_reasons,
)
from .shared_pool import SharedPoolError

logger = logging.getLogger(__name__)

PLUGIN_NAME = "kapsl_kv"
PLUGIN_ENTRY_POINT = "kapsl_sglang_adapter.hooks:activate"
ENTRY_POINT_GROUP = "sglang.srt.plugins"

SCHEDULER_INIT_TARGET = "sglang.srt.managers.scheduler.Scheduler.__init__"
ADMISSION_TARGET = "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req"
ROW_ALLOC_TARGET = "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.alloc"
ROW_RELEASE_TARGET = "sglang.srt.mem_cache.common.release_kv_cache"
ADMISSION_MODULE = "sglang.srt.managers.schedule_policy"

# Shared-pool tier only.
KV_BUDGET_TARGET = (
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.config_from_budget"
)
POOL_BACKING_TARGET = "sglang.srt.mem_cache.kv_vmm_backing.KvVmmBufferOwner._back_spans"
ARENA_COMMIT_TARGET = "sglang.srt.mem_cache.kv_vmm_backing.KvVmmArena.commit_range"
SCHEDULER_STEP_TARGET = "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run"
VMM_UTILS_MODULE = "sglang.srt.utils.cuda_vmm_utils"
PREFIX_CACHE_MODULE = "sglang.srt.mem_cache.base_prefix_cache"

# Scheduler attributes that, when truthy, mean KV is allocated outside the one
# leased request row this adapter accounts for.
UNSUPPORTED_FEATURE_FLAGS: tuple[str, ...] = (
    "is_hybrid_swa",
    "is_hybrid_ssm",
    "enable_hisparse",
    "enable_unified_memory",
    "enable_pdmux",
    "attn_dp_enabled",
    "enable_session_radix_cache",
    "dllm_config",
)

_INACTIVE = "inactive"
_STARTING = "starting"
_SERVING = "serving"
_FAILED = "failed"


class HookInstallError(RuntimeError):
    """A pinned hook target is missing, moved, or not wrapped by this adapter."""


class AdapterStateError(RuntimeError):
    """A hook ran while the adapter was not governing this process."""


@dataclass
class _PoolSetup:
    """The shared pool while SGLang builds it: who registers it, and for what."""

    participant: SharedPoolParticipant
    sglang_version: str
    # SGLang's ``KvVmmBufferOwner`` once its buffers are backed by the runtime.
    owner: Any = None

    def back_spans(self, owner: Any, span_bytes: Any) -> None:
        if self.owner is None:
            backend = SglangVmmBackend(
                owner,
                vmm=importlib.import_module(VMM_UTILS_MODULE),
                torch=importlib.import_module("torch"),
            )
            self.participant.attach_pool(
                pool_facts(owner), backend, sglang_version=self.sglang_version
            )
            self.owner = owner
        elif owner is not self.owner:
            raise SharedPoolError(
                "SGLang created a second VMM-backed KV pool; the striped profile "
                "covers exactly one"
            )
        pool = self.participant.pool
        if pool is None:
            raise SharedPoolError("the Kapsl KV pool was not attached")
        pool.require_backed(list(span_bytes))


@dataclass
class _AdapterState:
    phase: str = _INACTIVE
    participant: KvParticipant | None = None
    denied_result: Any = None
    pool_setup: _PoolSetup | None = None


_STATE = _AdapterState()


# Hook functions ---------------------------------------------------------------


def around_scheduler_init(
    original: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> Any:
    if _STATE.phase != _INACTIVE:
        _STATE.phase = _FAILED
        raise AdapterStateError(
            "the Kapsl adapter governs one SGLang scheduler per process"
        )
    _STATE.phase = _STARTING
    try:
        # Everything that can be checked before the model loads is checked
        # first, so a misconfigured launch fails in seconds.
        config = load_config(os.environ)
        version = installed_sglang_version()
        require_pinned_version(version)
        verify_hooks_applied(active_hooks(config))
        denied_result = resolve_denied_result()
        if config.shared_pool:
            result, participant = _start_with_shared_pool(
                config, version, original, scheduler, args, kwargs
            )
        else:
            result = original(scheduler, *args, **kwargs)
            participant = build_participant(config)
            participant.register(scheduler_facts(scheduler, sglang_version=version))
    except BaseException:
        _STATE.phase = _FAILED
        raise
    _STATE.participant = participant
    _STATE.denied_result = denied_result
    _STATE.phase = _SERVING
    participant.start_heartbeat()
    atexit.register(participant.shutdown)
    return result


def around_add_one_req(
    original: Callable[..., Any], adder: Any, req: Any, *args: Any, **kwargs: Any
) -> Any:
    participant = _serving_participant("request admission")
    request_id = _request_id(req)
    if participant.holds(request_id):
        # The request already owns a leased KV row; SGLang is revisiting it.
        return original(adder, req, *args, **kwargs)
    admitted = participant.admit(
        request_id,
        input_tokens=_input_tokens(req),
        max_new_tokens=_max_new_tokens(req),
        priority=_priority(req),
    )
    if not admitted:
        # Indistinguishable, to SGLang, from its own admission budget being
        # spent: the request stays in the waiting queue and is offered again.
        return _STATE.denied_result
    try:
        result = original(adder, req, *args, **kwargs)
        accepted = _accepted_by_sglang(adder, req)
    except BaseException:
        participant.cancel(request_id)
        raise
    if not accepted:
        participant.cancel(request_id)
    return result


def before_alloc_request_rows(
    pool: Any, reqs: Iterable[Any], *args: Any, **kwargs: Any
) -> None:
    del pool, args, kwargs
    if _STATE.phase == _STARTING:
        # Scheduler construction (warm-up, graph capture) precedes registration
        # and serves no client request.
        return None
    participant = _serving_participant("KV row allocation")
    for req in reqs:
        if not _holds_row(req):
            participant.require_lease(_request_id(req))
    return None


def after_release_kv_cache(result: Any, req: Any, *args: Any, **kwargs: Any) -> None:
    del result, args, kwargs
    participant = _STATE.participant
    if participant is None:
        return None
    try:
        participant.release(_request_id(req), computed_tokens=_computed_tokens(req))
    except Exception as error:  # a release path must never raise into SGLang
        participant.fail(error)
    return None


def _start_with_shared_pool(
    config: AdapterConfig,
    version: str,
    original: Callable[..., Any],
    scheduler: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, KvParticipant]:
    """Build the scheduler on a pool the runtime backs, then activate it.

    The pool is registered and mapped from inside ``original``, when SGLang
    backs its KV buffers for the first time. The runtime grants no lease until
    the activation at the end, so a scheduler that turns out to be outside the
    profile never serves.
    """

    server_args = args[0] if args else kwargs.get("server_args")
    reasons = shared_pool_launch_reasons(os.environ, server_args)
    if reasons:
        raise UnsupportedProfileError(
            "SGLang launch is outside the Kapsl shared-pool profile: "
            + "; ".join(reasons)
        )
    participant = build_participant(config)
    if not isinstance(participant, SharedPoolParticipant):
        raise AdapterStateError("shared_pool mode needs a shared-pool participant")
    setup = _PoolSetup(participant, version)
    _STATE.pool_setup = setup
    result = original(scheduler, *args, **kwargs)
    if setup.owner is None:
        raise UnsupportedProfileError(
            "SGLang built its scheduler without a VMM-backed KV pool, so there "
            "is nothing for the Kapsl runtime to back"
        )
    require_shared_pool_supported(
        scheduler_facts(scheduler, sglang_version=version),
        shared_pool_details(scheduler, setup.owner),
        minimum_tokens=participant.minimum_tokens(),
    )
    host = SglangPoolHost(
        scheduler,
        setup.owner,
        torch=importlib.import_module("torch"),
        evict_params=importlib.import_module(PREFIX_CACHE_MODULE).EvictParams,
    )
    participant.activate(host)
    return result, participant


def before_config_from_budget(*args: Any, **kwargs: Any) -> Any:
    setup = _STATE.pool_setup
    if setup is None:
        return None
    if kwargs.get("cap_tokens") is None:
        # First call: the address reservation, sized by --max-total-tokens.
        return lift_kv_budget(args, kwargs, cap_tokens=None)
    # After graph capture: back exactly what the runtime mapped.
    return lift_kv_budget(
        args, kwargs, cap_tokens=setup.participant.startup_tokens()
    )


def around_back_spans(
    original: Callable[..., Any], owner: Any, span_bytes: Any, *args: Any, **kwargs: Any
) -> None:
    # SGLang's own implementation creates and maps memory; it is never run.
    del original, args, kwargs
    setup = _STATE.pool_setup
    if setup is None or _STATE.phase == _FAILED:
        raise AdapterStateError(
            "SGLang backed a VMM KV pool outside a Kapsl-governed scheduler start"
        )
    setup.back_spans(owner, span_bytes)
    return None


def around_arena_commit_range(
    original: Callable[..., Any], arena: Any, *args: Any, **kwargs: Any
) -> None:
    del original, arena, args, kwargs
    raise AdapterStateError(
        "SGLang attempted to create KV memory itself; in shared_pool mode only "
        "the Kapsl runtime backs the pool"
    )


def before_scheduler_step(scheduler: Any, *args: Any, **kwargs: Any) -> None:
    del scheduler, args, kwargs
    participant = _serving_participant("a scheduler step")
    if isinstance(participant, SharedPoolParticipant):
        participant.service_resize()
    return None


@dataclass(frozen=True)
class HookSpec:
    target: str
    kind: str  # name of an ``sglang.srt.plugins.hook_registry.HookType`` member
    function: Callable[..., Any]


HOOKS: tuple[HookSpec, ...] = (
    HookSpec(SCHEDULER_INIT_TARGET, "AROUND", around_scheduler_init),
    HookSpec(ADMISSION_TARGET, "AROUND", around_add_one_req),
    HookSpec(ROW_ALLOC_TARGET, "BEFORE", before_alloc_request_rows),
    HookSpec(ROW_RELEASE_TARGET, "AFTER", after_release_kv_cache),
)

# Installed in addition, and only when the shared-pool tier is configured.
SHARED_POOL_HOOKS: tuple[HookSpec, ...] = (
    HookSpec(KV_BUDGET_TARGET, "BEFORE", before_config_from_budget),
    HookSpec(POOL_BACKING_TARGET, "AROUND", around_back_spans),
    HookSpec(ARENA_COMMIT_TARGET, "AROUND", around_arena_commit_range),
    HookSpec(SCHEDULER_STEP_TARGET, "BEFORE", before_scheduler_step),
)


def active_hooks(config: AdapterConfig) -> tuple[HookSpec, ...]:
    """The hooks the configured tier needs: the opaque four, or all eight."""

    return HOOKS + SHARED_POOL_HOOKS if config.shared_pool else HOOKS


def hooks_for_environment(environ: Any) -> tuple[HookSpec, ...]:
    """The hooks to install for the tier ``environ`` configures.

    An unreadable configuration selects the opaque set; the scheduler hook
    then refuses to start on that same configuration.
    """

    try:
        return active_hooks(load_config(environ))
    except ConfigError:
        return HOOKS


# Plugin entry point -------------------------------------------------------------


def activate() -> None:
    """``sglang.srt.plugins`` entry point; SGLang calls it in every process.

    It only registers hooks. SGLang logs and ignores an exception raised by a
    plugin, so nothing that must fail closed is decided here: configuration and
    hook installation are verified inside the scheduler-construction hook,
    where an exception does stop the server.
    """

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for spec in hooks_for_environment(os.environ):
        HookRegistry.register(spec.target, spec.function, getattr(HookType, spec.kind))


def verify_hooks_applied(
    hooks: tuple[HookSpec, ...] | None = None,
    resolve: Callable[[str], Any] = pkgutil.resolve_name,
) -> None:
    """Fail unless every pinned target is wrapped by this adapter's hook.

    ``hooks`` defaults to the set the environment's configuration selects.
    """

    problems: list[str] = []
    for spec in hooks_for_environment(os.environ) if hooks is None else hooks:
        try:
            installed = resolve(spec.target)
        except (ImportError, AttributeError, ValueError) as error:
            problems.append(f"{spec.target} cannot be resolved ({error})")
            continue
        if not _wraps(installed, spec.function):
            problems.append(
                f"{spec.target} is not wrapped by {spec.function.__name__}"
            )
    if problems:
        raise HookInstallError(
            "Kapsl SGLang hooks are not fully installed: " + "; ".join(problems)
        )


def resolve_denied_result() -> Any:
    """SGLang's own "stop adding requests this pass" admission result."""

    try:
        return importlib.import_module(ADMISSION_MODULE).AddReqResult.OTHER
    except (ImportError, AttributeError) as error:
        raise HookInstallError(
            f"{ADMISSION_MODULE}.AddReqResult.OTHER is unavailable ({error})"
        ) from error


def installed_sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError as error:
        raise UnsupportedProfileError("SGLang is not installed") from error


def build_participant(config: AdapterConfig) -> KvParticipant:
    if config.shared_pool:
        return SharedPoolParticipant(
            config,
            KapslKvControlClient(
                config.control_endpoint,
                config.participant_id,
                timeout_seconds=config.timeout_ms / 1000.0,
                abi_version=STRIPED_POOL_ABI_VERSION,
            ),
            adapter_version=ADAPTER_VERSION,
        )
    client = KapslKvControlClient(
        config.control_endpoint,
        config.participant_id,
        timeout_seconds=config.timeout_ms / 1000.0,
    )
    return KvParticipant(config, client)


def scheduler_facts(scheduler: Any, *, sglang_version: str) -> SchedulerFacts:
    """Reduce a constructed SGLang scheduler to the facts the adapter governs by.

    A missing attribute means SGLang's internals moved; that is reported as an
    unsupported profile rather than guessed around.
    """

    try:
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        server_args = scheduler.server_args
        return SchedulerFacts(
            sglang_version=sglang_version,
            device=str(scheduler.device),
            pool_class=type(pool).__name__,
            page_size=scheduler.page_size,
            max_total_num_tokens=scheduler.max_total_num_tokens,
            kv_pool_bytes=_total_bytes(pool.get_kv_size_bytes()),
            post_capture_sizing=bool(pool.post_capture_active),
            tensor_parallel_size=scheduler.tp_size,
            pipeline_parallel_size=server_args.pp_size,
            data_parallel_size=server_args.dp_size,
            disaggregation_mode=str(scheduler.disaggregation_mode.value),
            speculative_decoding=not scheduler.spec_algorithm.is_none(),
            unsupported_features=tuple(
                name
                for name in UNSUPPORTED_FEATURE_FLAGS
                if getattr(scheduler, name)
            ),
        )
    except (AttributeError, TypeError) as error:
        raise UnsupportedProfileError(
            f"SGLang scheduler does not expose the expected KV facts ({error})"
        ) from error


def reset_for_tests() -> None:
    _STATE.phase = _INACTIVE
    _STATE.participant = None
    _STATE.denied_result = None
    _STATE.pool_setup = None


# Internals ------------------------------------------------------------------------


def _serving_participant(action: str) -> KvParticipant:
    participant = _STATE.participant
    if _STATE.phase != _SERVING or participant is None:
        raise AdapterStateError(
            f"SGLang attempted {action} without a registered Kapsl KV participant "
            f"(adapter phase: {_STATE.phase})"
        )
    return participant


def _wraps(installed: Any, hook: Callable[..., Any]) -> bool:
    """Whether ``hook`` is captured anywhere in ``installed``'s wrapper chain."""

    seen: set[int] = set()
    current = installed
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for cell in getattr(current, "__closure__", None) or ():
            try:
                if cell.cell_contents is hook:
                    return True
            except ValueError:  # empty cell
                continue
        current = getattr(current, "__wrapped__", None)
    return False


def _total_bytes(size: Any) -> int:
    # Pools report either one figure or a (K bytes, V bytes) pair. SGLang
    # 0.5.23 reports NumPy integers, so any exact integer counts (anything
    # operator.index accepts); a bool, a float or a negative size does not.
    parts = size if isinstance(size, tuple) else (size,)
    total = 0
    for part in parts:
        if isinstance(part, bool):
            raise TypeError(f"unexpected KV pool size report {size!r}")
        try:
            value = operator.index(part)
        except TypeError:
            raise TypeError(f"unexpected KV pool size report {size!r}") from None
        if value < 0:
            raise TypeError(f"unexpected KV pool size report {size!r}")
        total += value
    if not parts:
        raise TypeError(f"unexpected KV pool size report {size!r}")
    return total


def _request_id(req: Any) -> str:
    request_id = req.rid
    if not isinstance(request_id, str) or not request_id.strip():
        raise KapslKvControlError("SGLang request has no usable rid", kind="adapter")
    return request_id


def _input_tokens(req: Any) -> int:
    return len(req.origin_input_ids)


def _max_new_tokens(req: Any) -> int:
    value = req.sampling_params.max_new_tokens
    return 0 if value is None else max(0, int(value))


def _priority(req: Any) -> int:
    value = getattr(req, "priority", None)
    return 0 if value is None else int(value)


def _computed_tokens(req: Any) -> int:
    return len(req.origin_input_ids) + len(req.output_ids)


def _holds_row(req: Any) -> bool:
    return bool(req.kv.holds_kv)


def _accepted_by_sglang(adder: Any, req: Any) -> bool:
    # The scheduler's own test for whether add_one_req admitted the request.
    can_run_list = adder.can_run_list
    return len(can_run_list) > 0 and can_run_list[-1] is req
