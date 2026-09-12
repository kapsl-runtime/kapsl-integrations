# Governed ORT provider runtime

This integration owns the source patches and private allocator bridge for ORT
1.23.2. The engine continues to use `kapsl-backend-abi =0.2.0`; no engine types,
ORT options, or new SDK ABI cross that boundary.

Actual GPU execution is **not yet qualified**. CUDA forward inference passed the
manual engine/Vast checks, but generation cancellation exposed an unchecked null
allocation in ORT's C allocator wrapper. The failed candidates are unpublished,
and their runtime locks are cleared pending rebuild with the checked wrapper.
CPU packaging and its supported platforms are unchanged.
Do not publish an engine release from this work.

## Allocation and lifecycle

The adapter retains a private `KapslOrtAllocatorV1` table for each model replica.
Each TensorRT provider opens its own handle during session construction. That
handle captures the active engine allocation scope. After initialization it
revokes the load scope, then captures a new scope at each `OnRunStart`. Worker
threads use that captured scope; they do not select a process-wide owner.

The bridge rejects missing, foreign, inactive and overlapping ownership. The
engine receives the original model, replica, scope ID, request/batch IDs and
allocation class on every allocation, and remains authoritative for admission
and cancellation. Handles on the same model cannot free one another's pointers.
Frees synchronize through the host, validate recorded allocation identities and
retain failed frees for unload retry. Provider destruction also reclaims buffers
left by a failed constructor or inference operation. Sessions/providers must
drain before closing a handle and before releasing the adapter's allocator lease.

The core wrapper checks nonzero allocations returned by C callbacks, including
`Alloc`, `Reserve`, `AllocOnStream` and legacy fallback paths. Cancellation or
admission rejection must become an ORT exception before internal kernels receive
a null device pointer. Empty allocations retain their permitted null result.
Host tests compile the actual prepared wrapper methods with a fault-injecting C
callback table; they do not require CUDA or a GPU.

TensorRT's runtime and builder receive `IGpuAllocator` before model construction
or deserialization. Data-dependent outputs use the same allocator and release
their buffers at request completion. A failed release must succeed before a
later request can reuse the provider. CUDA graph capture is rejected because
this synchronous allocation contract does not support capture. Driver context
and library overhead still need to be measured during GPU qualification.

CUDA's capability patch preserves supported CUDA kernels when the existing
`session.disable_cpu_ep_fallback=1` option is present. Unsupported kernels still
fail. CPU fallback is never enabled to accommodate a placement failure.

## Runtime selection and artifacts

Accelerator adapters explicitly load their core library beside the verified
pack entrypoint. They do not link a shared `libonnxruntime` dependency or read
`ORT_DYLIB_PATH`. Private library names and shared-provider host symbols differ
between profiles and adapter versions. The build binds internal symbols locally. A
core build marker and provider version symbol reject stock/embedded runtime
substitution before execution. The actual mapped libraries and hashes must be
captured in the next engine/Vast test; host tests do not establish GPU isolation.

`source-lock.json` fixes the ORT commit, every patched source file, patch hashes,
and TensorRT's public headers. `prepare_source.py` requires a clean, dedicated
checkout at that commit. It applies the reviewed patch and records the prepared
sources. It never changes a published SDK or uses a Cargo path override.

On a Linux x86-64 build host with CUDA 12.8 development tools, cuDNN 9, CMake,
Git, a C++ compiler, `patchelf`, and (for TensorRT) the 10.9 SDK:

```sh
python3 integrations/ort/runtime/build_runtime.py \
  --source /build/ort-source-tensorrt10 \
  --build-dir /build/ort-build-tensorrt10 \
  --output-dir /artifacts/tensorrt10 \
  --profile tensorrt10 \
  --cuda-home /usr/local/cuda \
  --cudnn-home /opt/cudnn \
  --tensorrt-home /opt/tensorrt \
  --jobs 4
```

Use a separate source/build directory for `cuda12` and omit `--tensorrt-home`.
The command compiles code; it does not execute GPU inference, rent a machine,
sign artifacts or release anything. Source compilation still requires a Linux
validation run, including validation of the selected build-toolchain image.

The output includes namespaced libraries and `governed-runtime.json`, recording
source and recipe identities, tool versions/hashes, and library hashes. Review
and publish immutable integration runtime archives, then add their release URLs,
archive SHA-256, provenance SHA-256 and exact `files` maps to
`governed-runtimes.lock.json` through a PR. The release packager downloads and
checks these locks before collecting other dependencies. It signs packs only
after dependency closure, glibc compatibility and provenance checks pass.

The recipe normalizes compiler paths and verifies CMake's actual provider,
architecture and shared-library settings before and after compilation. A cache
reset that loses CUDA or TensorRT settings fails before compilation starts.
Configuration and build commands, plus the verified CMake settings, are recorded
in provenance. The artifact build excludes upstream unit-test executables;
integration conformance remains a separate required step.

Staging changes RUNPATH and SONAME in separate `patchelf` calls. Combining these
edits with Ubuntu 22.04's `patchelf` 0.14.3 can set the SONAME to `$ORIGIN`.
Staging verifies the resulting SONAME, RUNPATH and dependency names before
recording library hashes. Host-only Linux tests compile small C shared libraries
and load their staged dependency closure, without CUDA libraries or a GPU.
The shared-provider host can have no libc dependency or versioned glibc imports.
Such helper libraries record a null glibc requirement only when they have no
dynamic dependencies and no unresolved symbols beyond optional compiler CRT
hooks. Required unversioned imports and unsupported glibc versions are rejected.

## Host checks and remaining qualification

```sh
python3 integrations/ort/runtime/test_host.py
python3 -m unittest discover -s integrations/ort/packaging/tests -q
```

The first command authenticates the source fixtures, applies both profile
patches, and compiles/runs the allocator wrapper against TensorRT 10.9's actual
headers. Its CUDA header contains only opaque handle declarations. No driver,
GPU kernels or TensorRT runtime library are loaded. Rust accelerator-profile
tests exercise the private callback table with fake governed host allocations,
including worker threads, concurrent sessions, cross-owner frees, cancellation,
failed synchronization, failed frees and cleanup.

Complete the Linux runtime build and inspect the resulting ELF libraries.
Record immutable candidate artifact locks on the feature PR so the exact signed
candidate can be built and exercised through the engine on Vast. Qualification
must pass before those artifacts are promoted for a stable engine release. Prove CUDA/TensorRT placement, engine callbacks, model/replica isolation,
generation, batching, streaming, cancellation, unload/reload, memory reclamation,
output parity and existing performance gates. Keep the 1.5× startup threshold.
Poll at 30 seconds and verify instance/storage/runner teardown. PR CI remains
host-only. The final engine release remains blocked on the broader neutrality
migration and qualification.
