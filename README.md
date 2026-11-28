# Kapsl integrations

Backend adapters, packaging, and conformance tooling for
[Kapsl Runtime](https://github.com/kapsl-runtime/kapsl-engine).

This repository is the backend-specific edge of the Kapsl architecture:

- `kapsl-engine` owns routing, scheduling, model lifecycle, memory governance,
  signed-pack loading, and the host side of integration contracts.
- `kapsl-sdk` owns the public, versioned `kapsl-backend-abi` and
  `kapsl-kv-abi` contracts.
- `kapsl-integrations` owns adapters for ORT, vLLM, SGLang, llama.cpp, and the
  release/conformance logic tied to those backends.

## Repository layout

```text
integrations/
  vllm/       out-of-process vLLM KV participant
  ort/        in-process ORT native backend pack, packaging, and CPU conformance
  sglang/     SGLang adapter (planned)
  llama-cpp/  in-process llama.cpp native backend pack and CPU validation
```

The first migrated package is `integrations/vllm`. Its source and host tests
are carried unchanged from `kapsl-sdk` so repository and release plumbing can
be validated independently of behavior changes.

## Dependency policy

Integrations consume released SDK packages. Do not add sibling-checkout path
dependencies, Git submodules, or committed Cargo patches for Kapsl crates.
Development and CI must exercise the same published contract that a released
backend pack consumes.

Native adapters cross the engine boundary only through the versioned C ABI.
Python-native servers use the transport-neutral KV control protocol when they
participate in governed KV memory.

## ORT topology

Moving ORT here changes source and release ownership, not its runtime process
boundary. The governed adapter remains loaded in the Kapsl process:

```text
Kapsl GpuDevicePool
        |
        | KapslBackendHostV1 callbacks
        v
in-process ORT adapter -> OrtAllocator -> ORT execution provider
```

This preserves direct tensor views and the engine-owned device allocator. It
does not introduce RPC, CUDA IPC, Base64 tensors, or an additional copy.

## llama.cpp topology

The llama.cpp pack also moved without changing its runtime process boundary.
It stays loaded in the Kapsl process behind the llama.cpp-specific ABI v1
table:

```text
Kapsl GpuDevicePool
        |
        | KapslLlamaCppApiV1 host callbacks (shared-pool profile)
        v
in-process llama.cpp pack -> Kapsl llama.cpp fork external KV pool
```

The engine still packages and signs the llama.cpp packs. Its release workflow
builds this crate from an exact commit pinned in the engine's
`.github/llama-cpp-integration.lock`.

## Validation policy

Host tests, packaging checks, ABI checks, and CPU integration tests run on pull
requests. Real GPU conformance is reserved for an official stable release—never
for branch pushes, ordinary pull requests, beta tags, or prereleases. Stable
GPU jobs must run cleanup unconditionally and verify that their ephemeral
runner, VM, GPU-backed boot disk, and firewall resources are gone before a
release is published.

See each integration's README for its supported profiles and test commands.
