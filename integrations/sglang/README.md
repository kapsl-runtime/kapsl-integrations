# Kapsl SGLang adapter

An out-of-tree adapter that makes an SGLang server a Kapsl KV participant. It
registers the scheduler's KV pool with Kapsl Runtime, takes a capacity lease
before SGLang may give a request KV memory, reports completion when that
memory is returned, and stops the server rather than let it serve ungoverned.

It has two tiers. In the **opaque** tier, the default, SGLang allocates and
owns its KV pool and Kapsl only leases capacity. In the **elastic shared
pool** tier the Kapsl runtime owns the physical KV memory: the adapter maps
runtime-exported segments into SGLang's KV buffers, grows the pool on demand
and gives memory back when the server is idle.

**Status.** The opaque tier ran against a real SGLang 0.5.23 server on one RTX
3090 on 2026-11-26 (phase 0 in kapsl-engine
`docs/SGLANG_SHARED_POOL_DESIGN.md`). **The elastic shared-pool tier has not
run against a real SGLang server, torch, or a GPU.** It is tested on a host
against stand-ins for SGLang's objects and for the CUDA driver. Those tests
show the adapter is internally consistent and matches the contract and the
SGLang source it was written from; they do not show that SGLang 0.5.23 or CUDA
behaves the way that source was read. "GPU checks still to run" lists what a
GPU run has to confirm. The shared-pool tier also needs `kapsl-kv-abi` 1.6
(striped pools, kapsl-sdk #228, not merged) and engine support that does not
exist yet. The package is not published and no profile is certified.

No SGLang-specific contract is added to the engine. The opaque tier speaks the
transport-neutral KV control protocol at `kapsl-kv-abi` 1.5, the shared-pool
tier at 1.6.

## Tiers

| Tier | Selected by | State |
|---|---|---|
| `kv_connected`, opaque metadata, backend-owned KV | `kapsl_kv_mode: opaque` (default) | ran on one GPU (phase 0) |
| `shared_pool`, elastic (CUDA VMM), striped | `kapsl_kv_mode: shared_pool` with `kapsl_live_resize: true` | host-tested only |
| `shared_pool`, fixed (CUDA IPC) | | not implemented; refused |
| prefix lookup, restore, transfer | | not advertised |

In the opaque tier SGLang keeps its own allocator, radix cache, and tensor
layout. Kapsl gains admission and device-wide byte accounting for in-flight
requests. SGLang still allocates its whole KV pool at startup, so a lease is a
claim against Kapsl's budget for that device, not a physical allocation;
whoever launches the server must bound and account for the pool itself with
`--mem-fraction-static` or `--max-total-tokens`.

In the shared-pool tier SGLang still keeps its allocator, radix cache, and
tensor layout, and still chooses which slot a token goes in. What changes is
where the bytes come from: SGLang reserves address space for the largest pool
it may ever have, and the runtime backs a prefix of it with memory it created,
zeroed and exported. Nothing else changes for a request: it takes a lease as
in the opaque tier.

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

For the shared pool, SGLang also has its own CUDA virtual-memory backing for
the KV pool, off by default behind `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`:

| Concern | Where |
|---|---|
| Address reservation and per-buffer backing | `mem_cache/kv_vmm_backing.py`: `KvVmmArena` (one reservation per device, `commit_range`), `KvVmmBufferOwner` (`_back_spans`, called by `ensure_prefix` and `finalize`) |
| Mapping memory it did not create | `utils/cuda_vmm_utils.py`: `import_peer_handle` (a POSIX descriptor to a CUDA handle), `VmmReservation.map_existing` |
| Reservation and final size | `KVCacheConfigurator.config_from_budget`, called once for the reservation and once after graph capture (`model_executor/model_runner_components/kv_pool_runtime.py`) |
| Free slots | `mem_cache/allocator/token.py` `TokenToKVPoolAllocator`: `free_pages` and `release_pages`; `resize()` clears every allocation, so it cannot be used on a live pool |
| Where capacity is cached | the pool's `size`, the allocator's `size`, `ModelRunner.max_total_num_tokens`, `Scheduler.max_total_num_tokens`, `Scheduler.pool_stats_observer` (a frozen, slotted dataclass) and `Scheduler.invariant_checker` |

## Hook strategy

The adapter is an `sglang.srt.plugins` entry point (`kapsl_kv`) that registers
its hooks through SGLang's registry. All targets are pinned by name and
re-verified at startup. The opaque tier installs four:

| Target | Hook | Purpose |
|---|---|---|
| `managers.scheduler.Scheduler.__init__` | around | Before the model loads: validate configuration, the SGLang version pin, and that every hook is installed. After the pool exists: register the participant. Any failure propagates. |
| `managers.schedule_policy.PrefillAdder.add_one_req` | around | Reserve a lease before SGLang's own admission. A Kapsl capacity denial returns `AddReqResult.OTHER`, so the request stays in the waiting queue exactly as if SGLang's own budget were spent. If SGLang then declines or raises, the lease is returned. |
| `mem_cache.memory_pool.ReqToTokenPool.alloc` | before | Refuse to give a request a KV row unless it holds a lease. An admission path the adapter does not hook becomes a scheduler failure, not an ungoverned allocation. |
| `mem_cache.common.release_kv_cache` | after | Commit computed tokens and release the lease when the row is returned, for any reason. |

A lease therefore lives exactly as long as the request owns a KV row. A request
aborted while queued never had one. A retracted request returns its lease and
reserves again on re-admission. Leak-freedom follows from SGLang's own row
accounting rather than from enumerating its abort paths.

The shared-pool tier installs four more, and only when it is configured; in
the opaque tier these targets are neither wrapped nor looked up:

| Target | Hook | Purpose |
|---|---|---|
| `mem_cache.kv_cache_configurator.KVCacheConfigurator.config_from_budget` | before | Size the pool by the runtime's decision instead of free GPU memory: the byte budget is lifted, so `--max-total-tokens` alone fixes the address reservation, and the call after graph capture is capped to the tokens the runtime mapped. |
| `mem_cache.kv_vmm_backing.KvVmmBufferOwner._back_spans` | around | The one place SGLang backs its KV buffers. On the first call (while the pool is being created) the participant is registered and the runtime's segments are mapped. SGLang's own implementation, which creates memory, is never run. Later calls only check that SGLang asks for no more than is mapped. |
| `mem_cache.kv_vmm_backing.KvVmmArena.commit_range` | around | Refuse. If SGLang ever reaches its own memory-creating path, the scheduler fails instead of holding KV memory the runtime does not know about. |
| `managers.scheduler.Scheduler.get_next_batch_to_run` | before | Between scheduler steps: ask the runtime for a pending resize (at most once per poll interval) and apply it. |

Failure policy, both tiers:

- Allocation paths raise; SGLang's handler then terminates the server. This
  covers registration failure, any reserve error other than
  `capacity_exhausted`, a failed heartbeat while leases are live, and a row
  requested without a lease.
- Release paths never raise into SGLang's cleanup. A failure there is latched
  and raised at the next allocation.
- A capacity denial is not a failure. Further reserves are suppressed for
  `KAPSL_SGLANG_DENIAL_BACKOFF_MS` so a blocked queue does not flood the
  control socket.
- Shared pool: anything unexpected while attaching or resizing the pool is
  latched like a lost lease authority and raised from the scheduler step, so
  the server stops. See "Elastic shared pool".

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
  lines; the four opaque targets kept their names and signatures, but `main`
  already adds a `PrefillAdder` subclass. The adapter accepts only exact
  pinned versions, and the row guard is what turns an unhooked admission path
  into a hard failure.
- The shared-pool tier reaches further into SGLang than the opaque tier: a
  private method (`_back_spans`), private attributes of the VMM arena and its
  buffer specs, the allocator's free lists, and six capacity fields. Each is
  re-checked at startup where a check is possible, and the list has to be
  re-read on every SGLang version bump.
- Each admitted request costs three synchronous control round trips on the
  scheduler thread (reserve, commit, release). Phase 0 measured about 10%
  throughput at concurrency 8 and 16 for the opaque tier. The shared-pool
  tier adds one resize poll per `KAPSL_SGLANG_RESIZE_POLL_MS`; its cost is
  unmeasured.
- Termination on failure depends on SGLang's default `SIGQUIT` handling. A
  custom handler must preserve it.
- A denied request waits in SGLang's queue until capacity returns or a
  configured SGLang queue limit or timeout removes it. It is not failed fast.
- After a runtime restart the participant is unknown to the new runtime. The
  adapter does not re-register; the next reserve fails and the server stops.

## Elastic shared pool

### Layout

SGLang's `MHATokenToKVPool` keeps one K and one V buffer per layer, each a
tensor of `rows x heads x head dim` with one row per token slot. The KV ABI
(1.6) calls each such buffer a *stripe* of one binding. The adapter describes
the pool to the runtime like this:

- **Stripe count** is the number of K and V buffers, `2 x layers`. Stripe `s`
  is buffer `s` in SGLang's own order: K of layer `s` for `s < layers`, V of
  layer `s - layers` after that.
- **One block is one buffer row**, so block `j` sits at the same offset in
  every stripe. Bytes per block in one stripe are the row bytes
  (`heads x head dim x element size`) times the tokens per block, which is 1.
  `bytes_per_block` on the wire is that times the stripe count.
- **Block 0 is SGLang's padding slot.** A pool of `B` mapped blocks serves
  `B - 1` tokens. The reservation is `--max-total-tokens + 1` blocks.
- Every stripe's minimum, mapped and maximum size must be whole allocation
  granules. With row size `r` and granule `g`, block counts are multiples of
  `A = g / gcd(g, r)`, so **`--max-total-tokens` must be `k x A - 1`**. The
  adapter refuses anything else and says so.

For Qwen2.5-0.5B (24 layers, 2 KV heads of 64, bf16) and a 2 MiB granule: 48
stripes, 256 bytes per block in a stripe, 12,288 bytes per block, `A` = 8,192
blocks, and one growth step of `A` blocks maps 48 segments of 2 MiB (96 MiB).

### Startup

1. `kapsl-sglang-serve` checks the configuration and sets
   `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`. SGLang reads that variable in the
   launcher process, so it cannot be left to the scheduler subprocess.
2. In the scheduler subprocess, before the model loads: the version pin, all
   eight hooks, the variable above, `--tp-size`, `--pp-size` and `--dp-size`
   all 1, and `--max-total-tokens` present.
3. SGLang sizes the reservation. The adapter lifts the byte budget, so the
   reservation is exactly `--max-total-tokens`, however much GPU memory other
   replicas hold at that moment.
4. SGLang creates the pool and asks to back its first page. At that point the
   adapter reads the buffers, checks they fit the profile, registers a
   `shared_pool` participant (`cuda_vmm`, `live_pool_resize`,
   `striped_elastic_pool`, `participant_block_selection`,
   `external_pool_attachment`, and `provisioning_grant` when the runtime
   supplied one), and receives the binding and one OS handle per segment.
5. The binding is checked against the buffers: stripe count, bytes per block
   in a stripe, block count against the reservation, and allocation
   granularity against SGLang's arena. Any mismatch stops the start before
   anything is mapped.
6. Every segment is imported and mapped at its buffer's place in SGLang's
   reservation, and the received descriptors are closed. If a mapping fails,
   the ones already made are unmapped again and the start stops.
7. SGLang captures its graphs, then sizes the final pool. The adapter caps
   that to the tokens the runtime mapped.
8. After `Scheduler.__init__` returns, the scheduler is checked against the
   profile, all six capacity fields are read and must agree with the mapped
   size, the attachment is reported (one view per buffer, each naming its
   stripe), and the participant is activated. The runtime grants no lease
   before activation, so a scheduler that fails a check here never serves.

### The CUDA seam

Every CUDA call sits behind `shared_pool.VmmBackend` (`map`, `unmap`,
`synchronize`, in stripe-relative bytes). The implementation,
`sglang_pool.SglangVmmBackend`, uses SGLang's own primitives:
`import_peer_handle` to import the descriptor, `VmmReservation.map_existing`
to map it, `cuMemRelease` on the imported handle once mapped, and `cuMemUnmap`
on a whole mapping. It also keeps SGLang's arena bookkeeping (`backed_to`,
the per-offset commit table, the backed total) as SGLang's own `commit_range`
would. `torch` and `cuda_vmm_utils` are passed in, so the host tests run this
code against fakes.

### Resize

The control protocol sequences a resize through a worker stage and a scheduler
stage, each acknowledged by the actor that applied it. An SGLang server at
tensor-parallel size 1 runs both roles in one process, so the adapter answers
as both, in order, within one scheduler step where it can:

**Growth** (`map_workers`, then `activate_scheduler`):

1. map the new tail of every stripe from the handles that came with the
   operation; acknowledge;
2. write all six capacity fields, then append the new slots to the
   allocator's free list; acknowledge.

Nothing a running request uses moves, so growth does not wait for the server
to be idle and needs no pause.

**Idle retirement** (`retire_scheduler`, then `unmap_workers`):

1. refuse, before changing anything, a boundary that falls inside a mapping
   (CUDA unmaps whole mappings only);
2. wait until no request holds a lease and SGLang reports itself idle
   (`is_fully_idle(ignore_waiting=True)`: nothing running or awaiting its
   result, no queued request holding KV);
3. evict from the prefix cache, in the cache's own least-recently-used order,
   until every slot above the boundary is free; a tail that was never used
   costs no eviction;
4. drop the tail slots from the free lists, write all six capacity fields;
   acknowledge;
5. synchronize the device, unmap the whole mappings above the boundary in
   every stripe; acknowledge.

While a retirement waits (step 2 or 3), new requests are not admitted: they
stay in SGLang's queue, as on a capacity denial, so the server drains instead
of staying busy past the runtime's acknowledgement deadline. If the deadline
passes anyway, the runtime fences the participant and the adapter stops the
server.

A request's reservation is clamped to the pool's maximum, not its current
size, so a request that does not fit yet is denied by the runtime, which is
what makes the runtime grow the pool. After a denial the adapter polls for the
resize at the very next step.

**What stops the server.** A binding or operation that does not match the
attached pool; a mapping or unmapping that fails (a partial mapping is rolled
back first); capacity fields that disagree; stages out of order; a boundary
inside a mapping; the runtime rejecting a poll or an acknowledgement; no
answer to resize polls for two thirds of the lease TTL. Each is latched, so
every later admission and scheduler step fails as well.

**What does not.** A busy server when a shrink arrives (deferred, see above);
one unanswered poll; an acknowledgement whose reply was lost (the adapter asks
whether the stage is still on offer before repeating it).

### What a resize leaves alone

SGLang derives a few more values from the pool size once, at startup, and the
adapter does not rewrite them:

- `max_req_len` and `max_req_input_len`, the longest request SGLang accepts.
  This is why the pool minimum must hold one full context (see "Supported
  profile"): then both equal the context limit at every pool size.
- `max_running_requests`, which SGLang caps from the pool size it finalizes at
  startup, that is, from the initially mapped pool. Set
  `--max-running-requests` if the default matters.
- The capacity copies in `load_inquirer` and `kv_events_publisher`. SGLang's
  own load reports and KV events keep showing the startup capacity.

### Not implemented

- A fixed (CUDA IPC) shared pool, and `kapsl_vmm_conformance`. Both are
  refused in the configuration.
- Detaching at shutdown. The adapter returns its leases and exits; the runtime
  reclaims the pool when the server process is gone.
- Retirement while requests run. It needs mapping by page (design phase 6).
- A single acknowledgement per resize. The ABI has no combined stage, so the
  adapter sends the two acknowledgements the protocol defines.

## Supported profile

Startup fails unless the scheduler matches all of the following. Each limit
exists because the adapter's accounting would otherwise be wrong, not because
the configuration was tested and found broken.

Both tiers:

- SGLang exactly `0.5.23`.
- One scheduler rank: tensor, pipeline, and data parallel sizes all 1. With
  more ranks every rank schedules a broadcast copy of each request
  (`scheduler_components/request_receiver.py:163`) and a Kapsl decision would
  have to be made once and replicated.
- No PD disaggregation, speculative decoding, diffusion LLM, PD multiplexing,
  DP attention, HiSparse, unified memory, session radix cache, or hybrid
  sliding-window or recurrent-state pools.

Opaque tier:

- KV pool class `MHATokenToKVPool` or `MLATokenToKVPool`, without post-capture
  sizing.
- Device `cuda` with one `cuda` memory domain, or `cpu` with a `host` domain.

Shared-pool tier (narrower, and exactly what the design measured):

- KV pool class exactly `MHATokenToKVPool`, VMM-backed. MLA, sliding-window,
  hybrid and recurrent-state pools are refused.
- Allocator exactly `TokenToKVPoolAllocator` with page size 1, and the NHD
  layout (one row per token). Paged allocators and page-major layouts are
  refused.
- K and V rows of the same size, in fp16, bf16 or fp32. Quantized and FP8 KV
  are refused.
- Device `cuda`, one `cuda` memory domain.
- No KV shard widening, hierarchical cache or LMCache.
- `--max-total-tokens` set, and of the form `k x A - 1` (see "Layout").
- **The pool minimum must hold one full context**: SGLang fixes its longest
  request from the pool size at startup (`max_req_len`), so a pool that can
  shrink below `--context-length` would refuse or truncate long requests even
  after it had grown. The adapter refuses to start in that case.
- At most 253 buffers (126 layers), the stripe bound of one binding.

Known gap in both tiers: beam search is a per-request option, so it cannot be
refused at startup. Its member rows are allocated outside the leased request
row and are not accounted.

## Capacity model

One capacity group, `sglang.kv.0` in pool `sglang.pool.0`.

Opaque tier:

- `allocation_granularity_tokens` is SGLang's `page_size`;
- `max_allocations` is `max_total_num_tokens // page_size`;
- `bytes_per_allocation` is the pool's `get_kv_size_bytes()` total divided by
  those pages, rounded up, so a fully leased pool is charged at least its
  physical bytes including the padding page.

Shared-pool tier:

- `allocation_granularity_tokens` is 1;
- `max_allocations` is the reservation in blocks, `--max-total-tokens + 1`;
- `bytes_per_allocation` is the bytes of one block over all stripes, exactly.

A request reserves its prompt length plus `max_new_tokens`, clamped to the
pool (in the shared-pool tier, to the pool's maximum). Prefix-cache sharing is
not discounted. SGLang's radix cache keeps finished requests' tokens until
evicted; those are inside the pool and are not leased.

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
| `kapsl_kv_mode` | yes | `opaque`, or `shared_pool` |
| `kapsl_memory_domains` | yes | one domain, e.g. `[{"kind":"cuda","device_id":0}]`, in the runtime's device namespace |
| `kapsl_lease_ttl_ms` | yes | minimum 1000; must not exceed the runtime maximum |
| `kapsl_timeout_ms` | no | control round-trip timeout, default 2000 |
| `kapsl_live_resize` | shared pool: yes | must be `true` with `shared_pool`; refused with `opaque` |
| `kapsl_provisioning_grant` | no | the runtime's proof that it precharged the pool; presented at registration; refused with `opaque` |
| `kapsl_rank_device_map` | no | with `shared_pool`, may only map rank `0` to the device above |
| `kapsl_vmm_conformance` | no | not implemented; `true` is refused |

Other fields are ignored. `tests/fixtures/participant_config_*.json` are
verbatim copies of the `kapsl-kv-abi` fixtures.

Two adapter-local variables are not part of that document:
`KAPSL_SGLANG_DENIAL_BACKOFF_MS` (default 100) and, for the shared pool,
`KAPSL_SGLANG_RESIZE_POLL_MS` (default 100, minimum 1).

Configuration is environmental because SGLang has no per-plugin configuration
channel and its scheduler runs in a spawned subprocess.

Opaque tier:

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

Elastic shared pool (the runtime must provision striped pools and allowlist
this adapter's profile; no released runtime does yet):

```bash
export KAPSL_KV_PARTICIPANT_CONFIG='{
  "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
  "kapsl_participant_id": "sglang-0",
  "kapsl_model_fingerprint": "sha256:<model>",
  "kapsl_kv_mode": "shared_pool",
  "kapsl_live_resize": true,
  "kapsl_memory_domains": [{"kind": "cuda", "device_id": 0}],
  "kapsl_lease_ttl_ms": 30000
}'
# Qwen2.5-0.5B, 2 MiB granules: A = 8192, so the maximum is k * 8192 - 1.
kapsl-sglang-serve --model-path <model> --host 127.0.0.1 --port 30000 \
  --max-total-tokens 65535 --context-length 8191
```

The launcher sets `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1` itself. A server
started with `python -m sglang.launch_server` directly must set it, and a
launch that sets it to a false value is refused.

## What the engine needs

Opaque tier. The runtime's KV control listener accepts this adapter
unchanged: opaque registration has no backend allowlist, and each reservation
is charged against the domain's external-lease budget (exercised in phase 0).
Running SGLang as a *managed* backend needs generic pieces that on the
engine's `develop` exist only in vLLM-specific form (`ManagedVllm*` under
`kapsl-cli/src/runtime/managed/`); kapsl-engine #273, not merged, adds them:

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

Shared-pool tier. None of this exists in the engine yet:

1. `kapsl-kv-abi` 1.6 (kapsl-sdk #228). Against a runtime at 1.5 the
   registration is answered with a version mismatch and the server stops.
2. Striped provisioning: one segment per stripe per step, created, zeroed and
   exported by the runtime, with handle indices dense across stripes; the
   minimum and the headroom above it as separate segments, so the pool can
   shrink to its minimum on a mapping boundary.
3. An allowlisted profile, exactly: adapter `kapsl-sglang-adapter`, this
   adapter's version, backend version `0.5.23`, profile
   `sglang-v1-striped-cuda-vmm/mha-nhd-token`. Provisioning grants are checked
   against the literal backend `vllm` today.
4. Geometry planning before launch: stripe count, row bytes, the alignment
   unit `A`, a maximum of the form `k x A - 1` for `--max-total-tokens`, and a
   minimum of at least `--context-length + 1` blocks.
5. The allocation granularity the runtime uses for segments must equal the one
   SGLang's arena reports for the device. The adapter refuses a mismatch.
6. A resize policy for SGLang (growth on a denied reservation or on
   utilization, shrink when idle), and a load plan in which SGLang's static
   footprint no longer includes the KV pool. `--max-running-requests` should
   be set: SGLang otherwise sizes its request table from the reservation.

The engine pins this adapter by exact version. This branch is `0.3.0`; the
opaque tier is unchanged from `0.2.0` and still reports ABI 1.5 through
`kapsl_sglang_adapter.ABI_VERSION`. The shared-pool tier's version and profile
are `SHARED_POOL_ABI_VERSION` and `SHARED_POOL_PROFILE_ID`.

In both tiers, the engine's vLLM bridge does not pass its request ID
upstream, so an engine request cannot be matched to a lease's
`sequence.request_id`; the same would hold for SGLang, whose leases carry
SGLang's own `rid`.

## GPU checks still to run

None of these has been run. Setup as for phases 0 and 1 of the design: one
CUDA GPU, SGLang 0.5.23, flashinfer, Qwen2.5-0.5B-Instruct, greedy decoding,
8 fixed prompts of 48 output tokens compared token for token with an
unmodified server. They need a runtime that owns the memory and speaks ABI
1.6: the engine once it provisions striped pools, or a small stand-in that
creates, zeroes and exports CUDA VMM allocations and answers the control
socket.

Startup:

1. All eight hooks apply in the launcher and in the scheduler subprocess, and
   `KvVmmBufferOwner.__init__` reaches the hooked `_back_spans` through
   `ensure_prefix`. Pass: the participant registers; no `cuMemCreate` happens
   in the SGLang process for the KV pool.
2. `config_from_budget` is called exactly twice (first with `cap_tokens`
   unset), and with the byte budget lifted the reservation is exactly
   `--max-total-tokens`. Pass: `owner._reserved_num_tokens` equals it, on an
   empty GPU and on one where another process holds most of the memory.
3. The buffer facts are as read: names `k0..`, `v0..` in that order, one token
   per row, granule-aligned offsets, `store_dtype` `torch.bfloat16`, and an
   arena granularity equal to the runtime's. Pass: registration is sent and
   the binding accepted.
4. A descriptor exported by another process imports with `import_peer_handle`,
   maps with `map_existing`, and stays mapped after `cuMemRelease` of the
   imported handle. Pass: step 6 below.
5. Two segments per stripe arrive in one message at registration (96
   descriptors for this model). Pass: all are received and none is left open
   (`ls /proc/<pid>/fd`).
6. Graph capture and warm-up touch only mapped memory. Pass: startup completes
   and the 8 prompts match.
7. After startup the six capacity fields equal the mapped size minus one, and
   SGLang's idle leak check stays quiet for 60 s.
8. GPU memory used by the SGLang process excludes the unmapped part of the
   pool. Pass: `nvidia-smi` shows the pool's mapped bytes once, and they
   change only with resizes.
9. Mapped memory reads as zero before first use (the runtime zeroes it).

Growth:

10. Grow by one, two and four steps with 2, 4 and 8 requests running under
    the default overlap scheduling. Pass: every request completes, the 8
    prompts match afterwards, the idle leak check stays quiet, and the time
    per step is recorded.
11. A request longer than the mapped pool but within `--context-length` is
    accepted, waits, and runs after the pool grows. Pass: it is neither
    refused nor truncated.
12. Throughput at concurrency 1, 8 and 16 against the opaque tier and an
    unmanaged server with the same pool, at poll intervals 100 ms and 10 ms.

Retirement:

13. On an idle server whose tail was never used: retire to the minimum. Pass:
    no eviction, GPU memory drops by the unmapped bytes, the 8 prompts match.
14. With cached prefixes above and below the boundary: retire. Pass: a prompt
    whose prefix sat below the boundary still gets a cache hit, one above
    does not, and outputs match. This is new relative to phase 1, which
    flushed the whole cache.
15. `is_fully_idle(ignore_waiting=True)` is true on an idle server and false
    while a batch result is pending in overlap mode.
16. Start a retirement while 4 requests run and new ones arrive. Pass: new
    requests queue, the retirement is applied when the running ones finish,
    then the queue drains. Record how long that takes against the runtime's
    acknowledgement deadline (30 s in the engine today).
17. 50 cycles of grow to the maximum and retire to the minimum under light
    load. Pass: outputs match at the end, and the process holds the same
    number of descriptors and mappings as after the first cycle.

Failure policy:

18. A raise from the scheduler-step hook terminates the server (SGLang's
    scheduler exception path and `SIGQUIT`).
19. Stop the runtime while serving. Pass: the server stops within two thirds
    of the lease TTL.
20. Let a retirement miss the runtime's deadline. Pass: the runtime fences the
    participant and the server stops.
21. Each refusal on a real server: `--tp-size 2`, a paged-allocator backend or
    `--page-size 16`, an MLA model, a hybrid sliding-window model, FP8 KV, a
    maximum that is not `k x A - 1`, a context length above the minimum pool,
    no `--max-total-tokens`. Pass: startup fails with the matching reason and
    no lease is ever granted.

Not covered by this list: attention backends other than flashinfer, GPUs with
a fabric handle type, and models other than the one named.

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
- For the shared pool, the checks in "GPU checks still to run" as well.

## Code shared with `integrations/vllm`

`contract.py` and `client.py` overlap with the vLLM connector's modules of the
same names. `contract.py` is a deliberate, reduced copy holding only the
opaque subset, with wire-width and lease-against-request checks added.
Depending on `kapsl-vllm-connector` would pull a vLLM-named package into
SGLang environments, and nothing here should import vLLM code.

`tests/test_vllm_parity.py` loads the vLLM contract module by path and fails
if the two disagree on the overlapping opaque messages. `pool_contract.py` has
no counterpart there: the vLLM connector stays at ABI 1.5 and never registers
a striped pool. It is transcribed from the Rust contract in kapsl-sdk #228
(`shared_pool.rs`, `resize.rs`, `capabilities.rs`, `control.rs`) and has not
been checked against the Rust serialization. The proposed end state is a
dependency-free `kapsl-kv-client` Python package released from `kapsl-sdk`,
which already owns the contract and its fixtures, consumed by both adapters.

## Modules

| Module | Role |
|---|---|
| `contract.py`, `client.py` | wire contract (opaque, ABI 1.5) and the control client, including OS handle passing |
| `pool_contract.py` | wire contract of the striped elastic pool (ABI 1.6) |
| `participant_config.py`, `config.py` | the participant configuration document and tier selection |
| `capacity.py` | both profiles, and the opaque capacity model |
| `participant.py` | leases and the failure latch |
| `shared_pool.py` | pool geometry, binding checks, and the table of mapped segments; no SGLang, torch or CUDA |
| `pool_participant.py` | registration, attachment, activation and the resize sequence |
| `sglang_pool.py` | everything read or written inside SGLang for the shared pool, including the CUDA seam |
| `hooks.py`, `launch.py` | the hook functions and the fail-closed launcher |

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite needs no SGLang, torch or CUDA. `tests/fake_sglang.py` and
`tests/fake_vmm.py` stand in for SGLang's objects, torch, the CUDA driver and
the runtime's coordinator; they were written from reading SGLang 0.5.23 and
the engine's coordinator, and are not evidence of how either behaves. The
handle-passing tests use real Unix sockets and descriptors and are skipped on
platforms without `SCM_RIGHTS`.

`tests/fixtures/opaque_registration.json` is a verbatim copy of the
`kapsl-kv-abi` fixture the SDK checks against its Rust serialization.
`sglang_opaque_registration.json` and `sglang_striped_registration.json` are
this adapter's own registrations; they have not been checked by the SDK's Rust
tests. The control envelope shapes are transcribed from `kapsl-kv-abi`'s
`contract/control.rs`, for which the SDK publishes no fixtures.
