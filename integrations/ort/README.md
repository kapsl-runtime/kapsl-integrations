# Kapsl ORT adapter

This crate is the out-of-tree, in-process ONNX Runtime backend pack for Kapsl.
It exports the backend-neutral `kapsl_backend_v1` C function table from the
published `kapsl-backend-abi = "=0.1.0"` crate. It does not depend on a sibling
SDK checkout, a Cargo path patch, or the legacy `kapsl-backends` ORT module.

## Implemented phase

The current `0.1.0` adapter implements the CPU forward-inference boundary:

- strict ABI/config/host-table and signed-pack-root validation;
- one retained process-wide ORT environment plus clean model load, unload,
  reload, health, and shutdown lifecycle;
- bounded, prewarmed session pools with shape-bucket LRU behavior matching the
  embedded CPU path;
- contiguous borrowed host tensor inputs without JSON or Base64 payloads;
- `float16`, `float32`, `float64`, `int32`, `int64`, and `uint8` tensors;
- named multi-input models and one primary raw ONNX output;
- request-coalescing batches that stack compatible tensors, perform one ORT
  run, split outputs in request order, and safely fall back for fixed-batch
  graphs;
- concurrent inference through the bounded session pool;
- adapter-owned result storage with explicit single and batch release ownership;
- planned/actual/request memory, metrics, model info, and batching reports;
- pre/post-execution cancellation polling and panic containment;
- real ORT identity-model tests for single, batched, concurrent, unload, and
  reload paths through the ABI v1 function table.

The capability table advertises CPU execution, batching, concurrent inference,
and memory reporting. It does not claim streaming, in-flight cancellation, KV
participation, CUDA, TensorRT, or governed device allocation before those paths
are implemented and certified.

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

## Remaining migration gates

1. Add in-flight ORT run termination and complete CPU parity benchmarks against
   the embedded path.
2. Move classification, embedding, detection, transcription, generation, and
   their preprocessing/postprocessing into explicit adapter profiles.
3. Implement the custom `OrtAllocator` that forwards CUDA allocations to
   `KapslBackendHostV1`, then add separate CUDA and TensorRT artifacts.
4. Package the adapter and its ORT dependency closure from this repository,
   certify parity against embedded ORT, and exercise unload/reload accounting.
5. Enable real GPU conformance only on an official stable release. The release
   must prove allocation ownership and unconditional ephemeral teardown.
6. Change the engine default and remove embedded ORT only after every required
   profile has a certified rollback and stable-release evidence.
