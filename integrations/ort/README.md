# Kapsl ORT adapter

This crate is the out-of-tree, in-process ONNX Runtime backend pack for Kapsl.
It exports the backend-neutral `kapsl_backend_v1` C function table from the
published `kapsl-backend-abi = "=0.2.0"` crate. It does not depend on a sibling
SDK checkout, a Cargo path patch, or the legacy `kapsl-backends` ORT module.

## Implemented phase

The current `0.2.0` adapter implements the stateless task pipeline and ONNX
generation across the CPU, CUDA 12, and TensorRT 10 profiles:

- strict ABI/config/host-table and signed-pack-root validation;
- one retained process-wide ORT environment plus clean model load, unload,
  reload, health, and shutdown lifecycle;
- bounded, prewarmed session pools with shape-bucket LRU behavior matching the
  embedded CPU path;
- contiguous borrowed host tensor inputs without JSON or Base64 payloads;
- `float16`, `float32`, `float64`, `int32`, `int64`, and `uint8` tensors;
- named multi-input models and one primary raw ONNX output;
- strict parsing and `EngineKind` validation of the published Kapsl manifest
  contract before an ORT session is created;
- raw forward, masked-mean/L2 embedding, classifier softmax, YOLO decode/NMS,
  and greedy CTC transcription output profiles;
- manifest-selected vision preprocessing from bounded JPEG/PNG bytes into
  normalized NCHW/NHWC tensors, including stretch and letterbox resize;
- manifest-selected audio preprocessing from finite float32 PCM into log-mel
  tensors, including Slaney/HTK filters, log compression, feature
  normalization, layouts, and optional derived frame-count inputs;
- autoregressive generation through the exact published
  `kapsl-llm = "=0.3.4"` crate, without a path patch or sibling checkout;
- bounded request-metadata decoding, UTF-8 prompt validation, request-scoped
  cancellation, continuous-batching policy, one-shot compatibility output, and
  repeated borrowed UTF-8 callbacks from the generation decode stream;
- request-coalescing batches that stack compatible tensors, perform one ORT
  run, split outputs in request order, and safely fall back for fixed-batch
  graphs;
- concurrent inference through the bounded session pool;
- adapter-owned result storage with explicit single and batch release ownership;
- planned/actual/request memory—including preprocessing resident and transient
  allocations—metrics, task-adjusted model info, and batching reports;
- pre/post-execution cancellation polling plus request-ID-scoped in-flight ORT
  run termination, with one shared termination handle for coalesced batches;
- live session-wait queue depth and cumulative pool wait count/duration in the
  standard engine metrics report;
- panic containment across every exported operation;
- real ORT identity-model tests for single, audio-preprocessed,
  task-postprocessed batch, concurrent, unload, and reload paths through the
  ABI v1 function table, plus a real tiny causal ONNX graph and tokenizer that
  prove two generation deltas, consumer cancellation, and one-shot collection.

Every profile advertises batching, streaming, concurrent inference, in-flight
cancellation, and memory reporting. Stateless tasks emit one borrowed stream
chunk; generation emits incremental UTF-8 chunks. The CPU binary claims only
CPU execution. Accelerator binaries additionally claim their exact provider
capabilities plus governed, scoped device allocation. No profile claims KV
participation.

## Accelerator implementation under host validation

The same crate now has three mutually exclusive build profiles:

- `profile-cpu` (the default) produces the `cpu` pack;
- `profile-cuda12` produces the `cuda12` pack and selects the CUDA provider;
- `profile-tensorrt10` produces the `tensorrt10` pack and selects TensorRT
  followed by CUDA for unsupported TensorRT nodes.

Each binary exports only its own execution and governed-allocation capability
bits. Initialization fails unless the signed pack profile, canonical provider,
accelerator class, governed-memory flag, and host callback table agree exactly.
Accelerator sessions require the environment allocator and disable implicit CPU
execution-provider fallback.

The accelerator allocator is environment-global per CUDA device, as required
by ORT, but registers every Kapsl model/replica host as a separate client. A
scoped session-build or inference call forwards an aligned allocation request
to that client's ABI callbacks and stores the exact returned identity for the
matching free. An allocation from an unscoped ORT provider thread fails closed
instead of being attributed to another model. Host-only tests exercise this
routing with aligned host-memory probes; they do not load a CUDA driver or
claim GPU certification.

Generation uses the published `kapsl-llm` device-allocation scope provider.
Model load, replica workspaces, individual requests, and request batches carry
explicit ABI scope IDs plus model/replica/request ownership into the same
governed allocator. Invalid, missing, or foreign scope ownership fails closed;
the adapter never substitutes the CPU provider.

No CUDA/TensorRT archive is published yet. Reproducible assembly and the exact
engine handoff contract now exist, but Vast provisioning and real GPU execution
remain deferred to the official stable-release gate.

The engine host calls the adapter's `cancel(request_id)` hook when its request
token fires. The adapter keeps each request registered from preprocessing
through postprocessing and attaches a private, non-preallocated ORT
`RunOptions` handle during graph execution. Cancellation before attachment is
remembered; cancellation during a run invokes ORT termination immediately;
unknown or already-completed IDs are treated idempotently to make completion
races harmless. Cancelling one request in a coalesced ABI batch cancels the
whole all-or-nothing batch result.

## Runtime topology

Moving this adapter changes source and release ownership, not the process
boundary:

```text
kapsl-engine
  -> signed native-pack loader
  -> kapsl_backend_v1 (this cdylib)
  -> ONNX Runtime session (same process)
```

The host passes the canonical signed-pack root and already-resolved per-model
ORT tuning in `options_json`. The adapter does not reinterpret global process
configuration. Tensor bytes cross as direct borrowed views and outputs are
copied once into the engine's existing tensor packet, matching the embedded
path's ownership model.

## Manifest task contract

The adapter resolves `format`, `model_type`, and `task` through the published
`kapsl-core = "=0.3.0"` contract. Every profile accepts `forward`, `embed`,
`classify`, `detect`, `transcribe`, and `generate` for `model_type: causal-lm`.
Task-specific knobs remain in `metadata.embed`, `metadata.classify`,
`metadata.detect`, and `metadata.transcribe`, matching the embedded
implementation.

Generation accepts exactly one host UTF-8 tensor named `input`. ABI request
metadata carries `session_id` and the published `RequestMetadata` sampling
fields. The adapter assigns an ABI-derived request ID when none is supplied,
uses the backend's internal continuous scheduler, and copies no output across
the C boundary until the engine host consumes each borrowed delta callback.
Generation packages require `tokenizer.json` beside the ONNX model or at the
package root; normal `generation_config.json`, `config.json`, and manifest LLM
metadata remain owned by the published LLM implementation.

Input preprocessing is optional. With no `metadata.preprocess`, the first ABI
tensor is sent directly to ORT. `kind: vision` accepts a uint8 packet containing
JPEG/PNG bytes and supports `width`, `height`, `resize`, `layout`, `scale`,
`mean`, `std`, and `pad`. It also accepts bounded-decoder controls
`max_decode_width`, `max_decode_height`, and `max_decode_bytes`.

`kind: audio` accepts finite float32 mono PCM and supports `sample_rate`,
`n_fft`, `hop_length`, `n_mels`, `f_min`, `f_max`, `mel_scale`, `norm`, `log`,
`power`, `center`, `normalize`, `normalize_eps`, and `layout`. When
`length_input` is set, the adapter derives that int32/int64 input from the
actual emitted frame count and replaces any stale client value with the same
name.

## Validation

```bash
cargo fmt --all -- --check
cargo clippy -p kapsl-backend-ort --all-targets --locked -- -D warnings
cargo test -p kapsl-backend-ort --locked
cargo test -p kapsl-backend-ort --no-default-features --features profile-cuda12 --locked
cargo test -p kapsl-backend-ort --no-default-features --features profile-tensorrt10 --locked
cargo build -p kapsl-backend-ort --no-default-features --features profile-cpu --release --locked
python3 -m unittest discover -s integrations/ort/conformance/tests -v
```

Pull requests run the CPU suite, compile and test both accelerator contracts
with host-memory probes, and verify that the CPU release library exports the
backend-neutral entrypoint. These checks neither install a GPU runtime nor
measure accelerator performance. Real GPU conformance is deliberately absent
from branch, pull-request, beta, and prerelease workflows.

The [`conformance/`](conformance/README.md) harness is the canonical CPU
embedded-versus-packaged retirement gate. It owns both runtime processes,
runs an ABBA sequence, and records correctness, latency, throughput, memory,
route-selection, process-identity, and teardown evidence. Engine CI consumes
it from the same exact integrations commit used to build the candidate pack.

The Linux x86_64 archives are assembled by the reproducible, fail-closed
workflow in [`packaging/`](packaging/README.md). It verifies the ABI symbol and
pack-local dynamic dependency closure, authenticates Microsoft's exact official
ORT distribution, includes complete Kapsl/ORT/Rust/NVIDIA notices and build
provenance, enforces the GLIBC 2.35 compatibility ceiling, emits engine manifest
templates, and can create detached domain-separated Ed25519 signatures without
ever placing the private key in a pack.

## Remaining migration gates

1. Run CPU embedded-versus-packaged generation parity and retain correctness,
   streaming, concurrency, cancellation, memory, and teardown evidence.
2. Rebuild the CUDA 12 and TensorRT 10 handoff in the pinned official-release
   environment and retain byte-for-byte reproducibility evidence.
3. Prove on a stable-release GPU run that ORT allocator callbacks remain in
   the scoped path, every device allocation belongs to the intended model and
   replica, implicit CPU fallback is disabled, and all memory returns on unload.
4. Exercise packaged accelerator unload/reload accounting and independent rebuild
   reproducibility as part of stable-release qualification.
5. Enable real GPU conformance only on an official stable release. The release
   must prove allocation ownership and unconditional ephemeral teardown.
6. Change the engine default and remove embedded ORT only after every required
   profile has a certified rollback and stable-release evidence.
