# Kapsl ORT adapter

This crate is the out-of-tree, in-process ONNX Runtime backend pack for Kapsl.
It exports the backend-neutral `kapsl_backend_v1` C function table from the
published `kapsl-backend-abi = "=0.1.0"` crate. It does not depend on a sibling
SDK checkout, a Cargo path patch, or the legacy `kapsl-backends` ORT module.

## Implemented phase

The current `0.1.0` adapter implements the stateless CPU task pipeline:

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
- explicit rejection of ONNX generation until its decode-loop profile is
  separately implemented and certified;
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
  ABI v1 function table.

The capability table advertises CPU execution, batching, concurrent inference,
in-flight cancellation, and memory reporting. It does not claim streaming, KV
participation, CUDA, TensorRT, or governed device allocation before those paths
are implemented and certified.

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
`kapsl-core = "=0.3.0"` contract. The stateless ONNX tasks are `forward`,
`embed`, `classify`, `detect`, and `transcribe`. Task-specific knobs remain in
`metadata.embed`, `metadata.classify`, `metadata.detect`, and
`metadata.transcribe`, matching the embedded implementation.

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
cargo build -p kapsl-backend-ort --release --locked
```

Pull requests run this CPU suite and verify that the release library exports
the backend-neutral entrypoint. Real GPU conformance is deliberately absent
from branch, pull-request, beta, and prerelease workflows.

The Linux x86_64 CPU archive is assembled by the reproducible, fail-closed
workflow in [`packaging/`](packaging/README.md). It verifies the ABI symbol and
dynamic dependency closure, includes complete Kapsl/ORT/Rust notices and build
provenance, emits the engine manifest template, and can create a detached
domain-separated Ed25519 artifact signature without ever placing the private
key in the pack.

## Remaining migration gates

1. Ingest the packaged CPU artifact into a locally signed engine backend index
   and complete CPU parity benchmarks against the embedded path.
2. Implement and separately certify the ONNX autoregressive generation
   profile.
3. Implement the custom `OrtAllocator` that forwards CUDA allocations to
   `KapslBackendHostV1`, then add separate CUDA and TensorRT artifacts.
4. Exercise packaged unload/reload accounting and independent rebuild
   reproducibility as part of stable-release qualification.
5. Enable real GPU conformance only on an official stable release. The release
   must prove allocation ownership and unconditional ephemeral teardown.
6. Change the engine default and remove embedded ORT only after every required
   profile has a certified rollback and stable-release evidence.
