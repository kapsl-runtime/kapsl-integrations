# Kapsl ORT adapter

This directory will contain the in-process ONNX Runtime native backend pack.
The extraction must preserve the engine-owned allocation path through
`KapslBackendHostV1`; moving the source here must not introduce a process
boundary or tensor serialization.

Migration order:

1. CPU forward-inference parity using `kapsl-backend-abi = "=0.1.0"`.
2. Batching, load/unload, cancellation, and memory-report parity.
3. Task-specific tensor preparation and postprocessing.
4. Host-backed CUDA allocator registration before ORT session construction.
5. Separately identified CUDA, TensorRT, and generation profiles.

The embedded engine implementation remains the active rollback until the
packed adapter passes the documented parity and stable-release gates.
