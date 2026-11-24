# Kapsl llama.cpp adapter

This crate is the in-process llama.cpp native backend pack for Kapsl. It
exports the llama.cpp-specific `kapsl_llama_cpp_backend_v1` C function table
(`KapslLlamaCppApiV1`, ABI v1, JSON wire format) from the published
`kapsl-backend-abi = "=0.2.0"` crate. Core sees only opaque handles, versioned
C structs, JSON wire messages, and explicit buffer ownership; Rust trait
objects never cross the library boundary.

The source moved here unchanged from
`kapsl-engine/kapsl-runtime/crates/kapsl-backend-llama-cpp` (engine `develop`
at `5f26753`, after the module split in kapsl-engine#267). The published backend
ABI keeps its llama.cpp v1 table during the transition to the backend-neutral
`kapsl_backend_v1` host. The patched allocator and request-lifecycle surface
stays explicit in Kapsl's llama.cpp fork crates until equivalent upstream
hooks exist.

## Build profiles

| Cargo features | Engine pack | Signed `kv_mode` | Capabilities |
| --- | --- | --- | --- |
| `cpu` (default) | `llama-cpp/cpu` | `native` | CPU, native KV, streaming, cancellation, memory reporting |
| `cuda12-shared-pool` | `llama-cpp/cuda12` | `shared_pool` | CUDA, Kapsl shared KV pool, streaming, cancellation, memory reporting |
| `cuda12` | separately signed native-KV CUDA rollback | `native` | CUDA, native KV, streaming, cancellation, memory reporting |

`cuda12-shared-pool` is `cuda12` plus `external-pool-contract`. That feature
links `kapsl-llama-cpp-sys-2` directly and enables
`kapsl-llm/gguf-external-kv`. The pack then wraps the runtime-owned
`GpuDevicePool` descriptor that core passes through the host callbacks inside
llama.cpp. It does not create another pool, CUDA process, or tensor-IPC path.
Each binary embeds a `KAPSL_LLAMA_CPP_KV_MODE=<mode>` marker, which the release
packager checks against the signed manifest.

## Dependencies

Only published crates from crates.io are used:

- `kapsl-backend-abi = "=0.2.0"`: the C ABI shared with the engine host;
- `kapsl-engine-api = "=0.3.0"`;
- `kapsl-llm = "=0.3.6"` with `gguf`, which provides the GGUF backend;
- `kapsl-llama-cpp-sys-2 = "=0.1.146-kapsl.1"`, Kapsl's llama.cpp fork
  bindings, which carry the external KV pool hooks.

Cargo resolves one `kapsl-llm 0.3.x` for the whole workspace. This pack and
the ORT adapter therefore move to a new `kapsl-llm` release together.

llama.cpp is compiled from the sources bundled in `kapsl-llama-cpp-sys-2` by
its build script. Building requires CMake, a C/C++ toolchain, and libclang for
bindgen. CUDA profiles also require the CUDA 12 toolkit.

## Validation

Pull-request CI covers the CPU profile and type-checks the shared-pool adapter
against the published external-pool contract without a GPU:

```bash
cargo fmt -p kapsl-backend-llama-cpp -- --check
cargo clippy -p kapsl-backend-llama-cpp --all-targets --locked -- -D warnings
cargo test -p kapsl-backend-llama-cpp --locked
cargo clippy -p kapsl-backend-llama-cpp --no-default-features \
  --features external-pool-contract --all-targets --locked -- -D warnings
```

CUDA builds (`cuda12`, `cuda12-shared-pool`) and real-GPU certification are not
part of pull-request CI. They run in the engine's release pipeline.

## Packaging and release

The engine still owns packaging and signing. The engine's
`.github/llama-cpp-integration.lock` pins an exact commit of this repository,
and the engine's release workflow checks out that commit.
`kapsl-engine/.github/scripts/package-linux-llama-cpp-backend-packs.sh`
verifies the checkout is clean and builds this crate with the committed
toolchain and `--locked`. It then assembles the `llama-cpp/cpu` and
`llama-cpp/cuda12` payloads: entrypoint, engine C header, licenses, the
non-driver CUDA dependency closure, and the manifest template. The engine's
Ed25519 backend-index publisher signs the result.

Still to move here, following `integrations/ort/packaging`:

- the payload layout, manifest template, and runtime-dependency closure checks;
- the llama.cpp license and redistribution notices;
- source provenance metadata that the engine handoff can verify.
