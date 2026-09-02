# ORT pack assembly

This directory owns the reproducible, fail-closed release handoff for the Rust
ORT adapter. It packages the Linux x86_64 `cpu`, `cuda12`, and `tensorrt10`
profiles. The build entrypoints do not invoke a GPU or change the runtime's
backend index.

## Build from an exact source commit

Run the CPU pack from a clean checkout at the commit being certified:

```bash
KAPSL_VERSION=0.2.3 \
  integrations/ort/packaging/build_cpu_pack.sh
```

`KAPSL_VERSION` is the exact Kapsl Engine version that may load the resulting
pack. The CPU wrapper:

1. downloads Microsoft's official ONNX Runtime 1.23.2 Linux x64 CPU archive,
   verifies both the archive and extracted runtime SHA-256 values, and exposes
   only the exact runtime library to the linker;
2. builds `kapsl-backend-ort` with the explicit `profile-cpu` feature, Rust
   1.92, the committed Cargo lock, no incremental state, remapped source paths,
   no linker build ID, and a direct dependency on the pack-local ONNX Runtime
   SONAME;
3. generates notices from only normal dependencies reachable through the
   target-filtered locked Cargo graph;
4. downloads ONNX Runtime 1.23.2's official third-party notices from its exact
   tag and verifies the pinned SHA-256;
5. verifies both ELF64 x86_64 libraries, the adapter ABI export, the exact ORT
   SONAME, `$ORIGIN` resolution, the host-system dependency allowlist, no
   `__isoc23_*` imports, and no GLIBC requirement newer than 2.35;
6. writes a deterministic tar/gzip archive and matching engine manifest
   template, checksum, and provenance.

For accelerator assembly, provide the already-collected user-space runtime
closures from the pinned release image:

```bash
KAPSL_VERSION=0.2.3 \
KAPSL_CUDA_RUNTIME_ROOT=/absolute/cuda-runtime \
KAPSL_CUDA_RUNTIME_PROVENANCE=/absolute/cuda-runtime-source.json \
KAPSL_TENSORRT_RUNTIME_DIR=/absolute/tensorrt-runtime \
KAPSL_TENSORRT_LICENSE_DIR=/absolute/tensorrt-licenses \
KAPSL_TENSORRT_RUNTIME_PROVENANCE=/absolute/tensorrt-runtime-source.json \
  integrations/ort/packaging/build_accelerator_packs.sh
```

The accelerator wrapper downloads Microsoft's exact ONNX Runtime 1.23.2 GPU
archive and verifies the archive plus the core, shared, CUDA, and TensorRT
objects by size and SHA-256. It builds one adapter per mutually exclusive Cargo
profile, treats cuDNN's split libraries and TensorRT's loader-visible family as
dependency roots, follows every ELF `DT_NEEDED` edge, and rejects missing,
conflicting, or host-driver libraries. Every packaged object gets a deterministic
`$ORIGIN` runpath; provenance retains both its source hash and normalized pack
hash. The TensorRT pack contains CUDA as its explicit unsupported-node fallback.
Official release assembly obtains CUDA 12.8 and cuDNN 9 from a digest-pinned
NVIDIA runtime image, authenticates the complete TensorRT 10.9.0.34 wheel, and
extracts only its Linux inference, plugin, parser, and builder-resource objects.
The similarly named Windows builder resource is explicitly excluded.

The source commit and `SOURCE_DATE_EPOCH` are derived from `HEAD`. Packaging a
dirty checkout, a different stated commit, or a different timestamp fails.
Build inputs and output directories stay outside the checkout, and Python
bytecode writes are disabled, so packaging also leaves the certified source
tree clean.

The wrapper explicitly selects and verifies the toolchain named by the
committed `rust-toolchain.toml`; an outer workspace's `RUSTUP_TOOLCHAIN` cannot
silently change the compiler used for the adapter binary.

## Outputs

The default output directory is `dist/ort-cpu/`:

```text
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz.manifest.json
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz.sha256
```

`build_accelerator_packs.sh` defaults to `dist/ort-accelerator/` and emits the
same three-file handoff for both `cuda12` and `tensorrt10`.

The archive contains:

- `libkapsl_backend_ort.so`, linked only to the signed pack-local ORT runtime
  and the allowlisted host system libraries;
- `libonnxruntime.so.1`, extracted from Microsoft's exact official CPU release
  asset and covered by the manifest's installed-file digest map;
- `backend-pack.json`, explicitly marked with `adapter_abi:
  kapsl-backend-v1` and the profile's formats, tasks, provider requirements,
  capabilities, memory-ownership behavior, detached-signature contract, and
  provenance path, so it cannot be confused with legacy provider-only ONNX
  bundles;
- `provenance.json`, recording source, lock/toolchain, binary, ORT distribution,
  notice, and allowed dynamic-library digests/identities;
- Kapsl, ONNX Runtime, ONNX Runtime third-party, and linked Rust dependency
  license notices.

Accelerator archives additionally contain the exact official ORT provider
objects, only the resolved user-space CUDA/TensorRT closure (including the
image-authenticated zlib required by cuDNN), and the applicable NVIDIA, zlib,
and TensorRT redistribution notices. `libcuda` and `libnvidia-*` remain
host-driver owned and are forbidden in an archive.

The adjacent `.manifest.json` is intentionally unsigned and omits the archive
URL/digest/signature. The signed integration release catalog binds that handoff
to its immutable archive digest and transport parts. Kapsl Engine consumes the
catalog and archive; it does not rebuild backend-specific ORT code.

## Detached artifact signing

For a controlled release handoff, point to an Ed25519 private key and state the
expected raw public key explicitly:

```bash
KAPSL_VERSION=0.2.3 \
KAPSL_BACKEND_SIGNING_KEY=/secure/path/backend-ed25519.pem \
KAPSL_BACKEND_EXPECTED_PUBLIC_KEY='ed25519:<base64-raw-32-byte-key>' \
  integrations/ort/packaging/build_cpu_pack.sh
```

This adds `.tar.gz.sig`, an Ed25519 signature over:

```text
kapsl-backend-artifact-v1\0sha256:<archive-sha256>
```

That is the same domain-separated artifact message the engine verifies. The
key must match the stated public key or packaging fails. Never commit, copy
into the pack, echo, or upload the private key. Pull-request CI uses an
ephemeral test key only; official signing belongs in a protected stable-release
environment.

## Stable pack publication

`.github/workflows/publish-ort-packs.yml` is the host-only release owner. An
official release tag has the exact form
`kapsl-ort-packs-v<adapter-version>-kapsl-v<engine-version>`, contains stable
numeric versions only, and must point to a commit already merged into `main`.
The protected `ort-pack-release` environment supplies
`KAPSL_BACKEND_SIGNING_KEY_B64` and the matching
`KAPSL_BACKEND_SIGNING_PUBLIC_KEY`; neither value is stored in the repository.

Every profile is assembled twice from isolated scratch directories and must
produce the same archive digest, manifest, and deterministic Ed25519 signature.
Because GitHub release assets must remain below 2 GiB, the signed archive is
transported as ordered 1.9 GB-or-smaller parts. Per-part hashes, the reconstructed
archive hash and signature, source commit, exact engine compatibility, and asset
URLs are bound into signed per-profile catalogs and a signed top-level release
index. Publication remains a draft until all three profiles and the final index
have succeeded. This workflow never requests or probes a GPU; accelerator
qualification remains an engine stable-release responsibility.

## Reproducibility boundary

Archive order, paths, owners, modes, timestamps, JSON serialization, and gzip
metadata are deterministic. Pull-request CI builds the CPU adapter and assembles
it twice, requiring byte-identical archives, manifests, checksums, and
signatures. Stable accelerator qualification must perform the same independent
rebuild comparison inside the pinned CUDA/TensorRT build environment.

The adapter binary itself is reproducible only under the pinned
Rust/toolchain/target and linker inputs. `provenance.json` records those inputs,
the official ORT archive and library hashes, each ELF dependency closure, and
each library's highest required GLIBC version. Release policy rejects anything
above GLIBC 2.35, and CI performs a real `dlopen` on Ubuntu 22.04. A release
verifier can rebuild in a second isolated environment and compare both signed
library SHA-256 values.

## Certification handoff

After the engine release pipeline has accepted an archive into a locally signed
backend index, run the CPU ABBA harness or the corresponding stable-release GPU
suite from the same exact integrations commit. Preserve the archive, template,
signature/index, captures, logs, and teardown evidence together. Merely
assembling an accelerator archive is not GPU certification. Embedded ORT remains
the rollback until every required CPU task profile and accelerator gate passes.
