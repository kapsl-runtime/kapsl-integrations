# Kapsl SGLang adapter

An out-of-tree adapter that makes an SGLang server a Kapsl **KV-connected
opaque** participant: it registers the scheduler's KV pool with Kapsl Runtime,
takes a capacity lease before SGLang may give a request KV memory, reports
completion when that memory is returned, and stops the server rather than let
it serve ungoverned.

**Status: design plus a host-tested first slice.** Nothing in this directory
has run against a real SGLang server or a GPU. The host tests exercise the
wire contract, the control client, the lifecycle logic, and the hook functions
against stand-ins for SGLang's objects. They show the adapter is internally
consistent and matches the contract; they do not show that SGLang 0.5.23
behaves the way its source was read. The package is not published and no
profile is certified.

No SGLang-specific contract is added to the engine. The adapter speaks the
existing transport-neutral KV control protocol (`kapsl-kv-abi` 1.5).

## Tiers

| Tier | State |
|---|---|
| `kv_connected`, opaque metadata, backend-owned KV | implemented here, host-tested only |
| `shared_pool`, fixed (CUDA IPC) | not started |
| `shared_pool`, elastic (CUDA VMM) | not started |
| prefix lookup, restore, transfer | not advertised |

In the opaque tier SGLang keeps its own allocator, radix cache, and tensor
layout. Kapsl gains admission and device-wide byte accounting for in-flight
requests. SGLang still allocates its whole KV pool at startup, so a lease is a
claim against Kapsl's budget for that device, not a physical allocation;
whoever launches the server must bound and account for the pool itself with
`--mem-fraction-static` or `--max-total-tokens`.

For the later shared-pool tiers the likely seam is SGLang's own CUDA-VMM KV
backing (`mem_cache/kv_vmm_backing.py`, `KvVmmArena` and `KvVmmBufferOwner`,
behind `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`). That is an observation from
reading the source, not a design.

## What SGLang offers

Designed against SGLang **0.5.23** (tag `v0.5.23`, commit
`297e54fd105e974cdf401e5b4d20253a22c16f90`). Paths below are under
`python/sglang/srt/` at that commit.

SGLang has no KV-connector interface comparable to vLLM's
`KVConnectorBase_V1`. Its `connector/` package, `BaseKVConnector` included,
loads model weights from remote stores and is unrelated to the KV cache. What
SGLang does have is a documented general plugin framework:

- `plugins/__init__.py:103` `load_plugins()` discovers `sglang.srt.plugins`
  entry points and calls each in every process (the main process, and each
  scheduler subprocess at `managers/scheduler.py:6323`), then applies
  registered hooks.
- `plugins/hook_registry.py` lets a plugin register before, after, around, or
  replace hooks on any dotted target.
- The loader is fail-open: a plugin that fails to load or raises
  (`plugins/__init__.py:84`, `:136`) and a hook target that cannot be patched
  (`plugins/hook_registry.py:166`) are logged and skipped. `SGLANG_PLUGINS`
  can filter a plugin out.

The relevant lifecycle points:

| Concern | Where |
|---|---|
| KV pool sizing | `mem_cache/kv_cache_configurator.py:2990` (free memory less the `--mem-fraction-static` slack), `:3090` (`--max-total-tokens` cap); allocated once from `Scheduler.__init__` via `managers/scheduler.py:1098` |
| Pool geometry | `mem_cache/memory_pool.py:2184` `MHATokenToKVPool`, `:2753` `get_kv_size_bytes`; `size + page_size` slots, the extra page is padding |
| Admission to run | `managers/scheduler.py:4270` calls `PrefillAdder.add_one_req` (`managers/schedule_policy.py:1395`); a result other than `CONTINUE` leaves the request queued |
| KV row allocation | `mem_cache/allocation.py:342` calls `ReqToTokenPool.alloc` (`mem_cache/memory_pool.py:419`) |
| KV row release | `mem_cache/common.py:400` `release_kv_cache`, the function through which every finish, abort, retraction, and preemption path returns a request's row |
| Scheduler failure | `managers/scheduler.py:6370` logs and sends `SIGQUIT` to the main process, whose default handler kills the process tree (`entrypoints/engine.py:1976`) |

## Hook strategy

The adapter is an `sglang.srt.plugins` entry point (`kapsl_kv`) that registers
four hooks through SGLang's registry. All four targets are pinned by name and
re-verified at startup.

| Target | Hook | Purpose |
|---|---|---|
| `managers.scheduler.Scheduler.__init__` | around | Before the model loads: validate configuration, the SGLang version pin, and that all four hooks are installed. After the pool exists: register the participant. Any failure propagates. |
| `managers.schedule_policy.PrefillAdder.add_one_req` | around | Reserve a lease before SGLang's own admission. A Kapsl capacity denial returns `AddReqResult.OTHER`, so the request stays in the waiting queue exactly as if SGLang's own budget were spent. If SGLang then declines or raises, the lease is returned. |
| `mem_cache.memory_pool.ReqToTokenPool.alloc` | before | Refuse to give a request a KV row unless it holds a lease. An admission path the adapter does not hook becomes a scheduler failure, not an ungoverned allocation. |
| `mem_cache.common.release_kv_cache` | after | Commit computed tokens and release the lease when the row is returned, for any reason. |

A lease therefore lives exactly as long as the request owns a KV row. A request
aborted while queued never had one. A retracted request returns its lease and
reserves again on re-admission. Leak-freedom follows from SGLang's own row
accounting rather than from enumerating its abort paths.

Failure policy:

- Allocation paths raise; SGLang's handler then terminates the server. This
  covers registration failure, any reserve error other than
  `capacity_exhausted`, a failed heartbeat while leases are live, and a row
  requested without a lease.
- Release paths never raise into SGLang's cleanup. A failure there is latched
  and raised at the next allocation.
- A capacity denial is not a failure. Further reserves are suppressed for
  `KAPSL_SGLANG_DENIAL_BACKOFF_MS` so a blocked queue does not flood the
  control socket.

The plugin loader's fail-open behaviour is not something a plugin can fix from
inside, so `kapsl-sglang-serve` runs the loader in the launcher process first
and refuses to start unless the plugin is installed exactly once, is not
filtered by `SGLANG_PLUGINS`, the SGLang version is pinned, and every target is
wrapped. It then runs `sglang.launch_server` unchanged. A managing runtime must
additionally treat participant registration as a readiness condition (see
below); that is the only check that covers a scheduler subprocess whose hooks
silently failed to apply.

### Alternatives considered

- **Raw monkeypatching** (`sitecustomize`, import hooks): same targets, none of
  the framework's per-process loading. Rejected.
- **Replacing `Scheduler` with a subclass** through a `REPLACE` hook, overriding
  `get_prefill_adder_class()` (`managers/scheduler.py:2462`): closer to an
  intended extension point, but only one plugin can replace a class, the
  subclass cannot be defined or tested without SGLang installed, and it still
  needs the two `mem_cache` hooks.
- **Leasing once at queue entry** (`managers/scheduler.py:3480`), the closest
  analogue of the one-shot request hook the vLLM connector uses: one reserve
  per request and a fast failure on denial, but queued requests would hold
  device budget, and a queued request can leave through several unrelated
  abort paths (`managers/scheduler.py:5675`), each a potential lease leak that
  the participant heartbeat would keep alive.
- **Upstream hooks**: the durable answer. A small admission-gate and
  row-lifecycle callback interface in SGLang would replace all four patches.
  Not attempted here.
- **A fork**: rejected by policy.

### Risks

- The targets are internal functions. Between `v0.5.23` and `main` four weeks
  later (`8c0a524`), `managers/scheduler.py` alone changed by about 2,600
  lines; the four targets kept their names and signatures, but `main` already
  adds a `PrefillAdder` subclass. The adapter accepts only exact pinned
  versions, and the row guard is what turns an unhooked admission path into a
  hard failure.
- Each admitted request costs three synchronous control round trips on the
  scheduler thread (reserve, commit, release). The latency cost is unmeasured.
- Termination on failure depends on SGLang's default `SIGQUIT` handling. A
  custom handler must preserve it.
- A denied request waits in SGLang's queue until capacity returns or a
  configured SGLang queue limit or timeout removes it. It is not failed fast.
- After a runtime restart the participant is unknown to the new runtime. The
  adapter does not re-register; the next reserve fails and the server stops.

## Supported profile

Startup fails unless the scheduler matches all of the following. Each limit
exists because the adapter's accounting would otherwise be wrong, not because
the configuration was tested and found broken.

- SGLang exactly `0.5.23`.
- One scheduler rank: tensor, pipeline, and data parallel sizes all 1. With
  more ranks every rank schedules a broadcast copy of each request
  (`scheduler_components/request_receiver.py:163`) and a Kapsl decision would
  have to be made once and replicated.
- No PD disaggregation, speculative decoding, diffusion LLM, PD multiplexing,
  DP attention, HiSparse, unified memory, session radix cache, or hybrid
  sliding-window or recurrent-state pools.
- KV pool class `MHATokenToKVPool` or `MLATokenToKVPool`, without post-capture
  sizing.
- Device `cuda` with one `cuda` memory domain, or `cpu` with a `host` domain.

Known gap inside that profile: beam search is a per-request option, so it
cannot be refused at startup. Its member rows are allocated outside the leased
request row and are not accounted.

## Capacity model

One capacity group, `sglang.kv.0` in pool `sglang.pool.0`:

- `allocation_granularity_tokens` is SGLang's `page_size`;
- `max_allocations` is `max_total_num_tokens // page_size`;
- `bytes_per_allocation` is the pool's `get_kv_size_bytes()` total divided by
  those pages, rounded up, so a fully leased pool is charged at least its
  physical bytes including the padding page.

A request reserves its prompt length plus `max_new_tokens`, clamped to the
pool. Prefix-cache sharing is not discounted. SGLang's radix cache keeps
finished requests' tokens until evicted; those are inside the static pool and
are not leased.

## Install and configure

Install into the same environment as SGLang 0.5.23. Installation alone opts
every server launched from that environment into governance, because SGLang
discovers the entry point automatically.

The participant's identity arrives as one JSON document in the
`KAPSL_KV_PARTICIPANT_CONFIG` environment variable. It is the participant
configuration of the KV contract (`KvParticipantConfig` in `kapsl-kv-abi`), with
the field names the vLLM connector reads from `kv_connector_extra_config`.
Kapsl Runtime writes it when it launches the server; an operator running the
server by hand writes the same document.

| Field | Required | Meaning |
|---|---|---|
| `kapsl_control_endpoint` | yes | `unix:///run/kapsl/kv-control.sock` |
| `kapsl_participant_id` | yes | base identity; registered as `<base>:scheduler` |
| `kapsl_model_fingerprint` | yes | model identity, as given to other participants |
| `kapsl_kv_mode` | yes | only `opaque` is accepted |
| `kapsl_memory_domains` | yes | one domain, e.g. `[{"kind":"cuda","device_id":0}]`, in the runtime's device namespace |
| `kapsl_lease_ttl_ms` | yes | minimum 1000; must not exceed the runtime maximum |
| `kapsl_timeout_ms` | no | control round-trip timeout, default 2000 |

`kapsl_live_resize` and `kapsl_provisioning_grant` belong to the shared-pool
tiers and are refused. Other fields are ignored.
`tests/fixtures/participant_config_*.json` are verbatim copies of the
`kapsl-kv-abi` fixtures.

One adapter-local variable is not part of that document:
`KAPSL_SGLANG_DENIAL_BACKOFF_MS` (default 100).

Configuration is environmental because SGLang has no per-plugin configuration
channel and its scheduler runs in a spawned subprocess.

```bash
export KAPSL_KV_PARTICIPANT_CONFIG='{
  "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
  "kapsl_participant_id": "sglang-0",
  "kapsl_model_fingerprint": "sha256:<model>",
  "kapsl_kv_mode": "opaque",
  "kapsl_memory_domains": [{"kind": "cuda", "device_id": 0}],
  "kapsl_lease_ttl_ms": 30000
}'
kapsl-sglang-serve --model-path <model> --host 127.0.0.1 --port 30000 \
  --max-total-tokens <n>
```

## What the engine needs

Reading the runtime's KV control listener, the opaque path should accept this
adapter unchanged: opaque registration has no backend allowlist, and each
reservation is charged against the domain's external-lease budget. That has
not been exercised. Running SGLang as a *managed* backend needs generic pieces
that today exist only in vLLM-specific form (`ManagedVllm*` under
`kapsl-cli/src/runtime/managed/`):

1. A managed out-of-process server launcher that is not named for vLLM:
   command and environment, loopback port, health probe, supervision, restart.
2. A readiness rule for opaque participants. Managed readiness currently
   requires a shared-pool activation, so an opaque registration can never
   become routable. The neutral rule is health plus a registered participant
   with a fresh epoch.
3. The identity handoff above (endpoint, participant base, fingerprint,
   domains, TTL) through the environment.
4. Admission of the server's static GPU footprint at model load, and the
   matching `--max-total-tokens` or `--mem-fraction-static` argument.
5. Backend selection and the OpenAI bridge keyed on a managed-server kind
   rather than on `vllm`. SGLang serves `/health`, `/v1/completions`, and
   `/v1/chat/completions`.

Provisioning grants are checked against the literal backend `vllm`; that only
matters for a later shared-pool tier. The engine's vLLM bridge does not pass
its request ID upstream, so an engine request cannot be matched to a lease's
`sequence.request_id`; the same would hold for SGLang, whose leases carry
SGLang's own `rid`.

## What "certified" would require

On a Linux GPU host, against the exact SGLang wheel, in the stable-release
conformance workflow only:

- The launcher starts a real server; hooks are confirmed applied in every
  process; the registered capacity model is compared with actual KV tensor
  bytes and the NVML delta.
- Every admitted request holds exactly one lease from before row allocation
  until row release, across streaming and unary requests, batch requests,
  embeddings, grammar-constrained requests, chunked prefill, client aborts in
  each phase, queue timeouts, priority preemption, retraction under pool
  pressure, and shutdown. Zero leases remain when idle, and leases are
  reclaimed by TTL after `SIGKILL`.
- With the Kapsl budget exhausted, no row is allocated without a lease, queued
  requests proceed when capacity returns, and control traffic stays bounded.
- Fail-closed cases: no runtime at startup, runtime lost while serving, plugin
  filtered out, version drift, a scheduler outside the profile.
- Throughput and time-to-first-token against an unmanaged server at several
  concurrencies, with the same budgets used for vLLM.
- Mixed admission with another governed backend under one authority.

## Code shared with `integrations/vllm`

`contract.py` and `client.py` overlap with the vLLM connector's modules of the
same names. They are a deliberate, reduced copy (about 600 lines against about
1,330): only the opaque subset, with shared-pool, resize, and OS-handle passing
removed, and with wire-width and lease-against-request checks added. Depending
on `kapsl-vllm-connector` would pull a vLLM-named package into SGLang
environments, and nothing here should import vLLM code.

`tests/test_vllm_parity.py` loads the vLLM contract module by path and fails
if the two disagree on the overlapping messages. The proposed end state is a
dependency-free `kapsl-kv-client` Python package released from `kapsl-sdk`,
which already owns the contract and its fixtures, consumed by both adapters.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

`tests/fixtures/opaque_registration.json` is a verbatim copy of the
`kapsl-kv-abi` fixture the SDK checks against its Rust serialization.
`sglang_opaque_registration.json` is this adapter's own registration; it has
the same shape but has not been checked by the SDK's Rust tests. The control
envelope shapes are transcribed from `kapsl-kv-abi`'s `contract/control.rs`,
for which the SDK publishes no fixtures.
