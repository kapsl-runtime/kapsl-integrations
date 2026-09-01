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
  ort/        in-process ORT native backend pack (CPU forward path implemented)
  sglang/     SGLang adapter (planned)
  llama-cpp/  llama.cpp native backend pack (planned migration)
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

## Validation policy

Host tests, packaging checks, ABI checks, and CPU integration tests run on pull
requests. Real GPU conformance is reserved for an official stable release—never
for branch pushes, ordinary pull requests, beta tags, or prereleases. Stable
GPU jobs must run cleanup unconditionally and verify that their ephemeral
runner, VM, GPU-backed boot disk, and firewall resources are gone before a
release is published.

See each integration's README for its supported profiles and test commands.
