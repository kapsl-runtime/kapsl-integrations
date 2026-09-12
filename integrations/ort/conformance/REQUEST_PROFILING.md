# Manual request profiling

Set `KAPSL_REQUEST_PROFILING=1` before adapter initialization. Stateless CPU,
CUDA and TensorRT adapters collect at most 8,192 operation records per instance
across its complete lifetime, including reloads. Disabled profiling collects
nothing. Records contain model/replica IDs, ABI request IDs, operation sequence
and phase durations; tensor bytes, prompts, session IDs and metadata are omitted.

Inference performs no profiling log I/O. Unload or terminal shutdown emits
`KAPSL_REQUEST_PROFILE` JSON through the host logger. Unload each model before
terminating a diagnostic process, and retain its complete log. Abrupt process
termination can lose buffered samples. Normal qualification runs must leave
profiling disabled and retain their existing workload and gates.

`adapter.infer` measures input conversion, cancellation registration,
preprocessing, backend execution, postprocessing, metrics, output conversion and
return cleanup. Its nested `ort.infer` operation separates ownership/session
selection, session acquisition, run options, tensor conversion, ORT execution
and output conversion. The engine's companion instrumentation reports
`engine.dispatch` admission and `engine.native` ABI phases using the same record
schema. Native and adapter operations share the ABI request ID; nested durations
must not be added to enclosing durations. Engine dispatch records use ID zero;
their chronological sequence is useful for a serial diagnostic workload.

Run the adapter alone on CPU with the fixed MatMul graph from the bridge trial:

```sh
KAPSL_REQUEST_PROFILING=1 cargo test --release --locked \
  -p kapsl-backend-ort --no-default-features --features profile-cpu --lib \
  tests::profile_cpu_adapter_without_engine -- --exact --ignored --nocapture \
  > adapter-profile.log 2>&1
python3 integrations/ort/conformance/analyze_request_profile.py \
  adapter-profile.log --output adapter-profile.json
```

This ignored diagnostic verifies outputs, performs 40 warmups followed by three
1,000-request serial trials, and records latency without a performance threshold.
It calls the real adapter ABI directly, excluding engine scheduling, signature
verification and admission. It is not a signed-pack qualification. The graph's
SHA-256 is `6b551f269078dd6c0d690f4f1b57a35f188c23ac1e92f4d344827fbf4d5c9300`.

The analyzer groups each operation's first 40 records as warmup and subsequent
records into 1,000-request windows. Interrupted or concurrent workloads require
explicit request-ID correlation instead. Separate profiling runs from normal
performance captures: clocks, sample storage and additional synchronization can
affect measurements even though log I/O is deferred. PR CI runs correctness
checks only; this command does not provision hardware or publish a release.
