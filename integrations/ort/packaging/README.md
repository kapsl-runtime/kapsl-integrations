# ORT CPU pack assembly

This directory owns the reproducible, fail-closed handoff from the Rust ORT
adapter to Kapsl's signed backend-index publisher. It packages only the Linux
x86_64 CPU profile. It does not publish a release, invoke a GPU, or change the
runtime's backend index.

## Build from an exact source commit

Run from a clean checkout at the commit being certified:

```bash
KAPSL_VERSION=0.2.3 \
  integrations/ort/packaging/build_cpu_pack.sh
```

`KAPSL_VERSION` is the exact Kapsl Engine version that may load the resulting
pack. The wrapper:

1. downloads Microsoft's official ONNX Runtime 1.23.2 Linux x64 CPU archive,
   verifies both the archive and extracted runtime SHA-256 values, and exposes
   only the exact runtime library to the linker;
2. builds `kapsl-backend-ort` with Rust 1.92, the committed Cargo lock, no
   incremental state, remapped source paths, no linker build ID, and a direct
   dependency on the pack-local ONNX Runtime SONAME;
3. generates notices from only normal dependencies reachable through the
   target-filtered locked Cargo graph;
4. downloads ONNX Runtime 1.23.2's official third-party notices from its exact
   tag and verifies the pinned SHA-256;
5. verifies both ELF64 x86_64 libraries, the adapter ABI export, the exact ORT
   SONAME, `$ORIGIN` resolution, the host-system dependency allowlist, no
   `__isoc23_*` imports, and no GLIBC requirement newer than 2.35;
6. writes a deterministic tar/gzip archive and matching engine manifest
   template, checksum, and provenance.

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

The archive contains:

- `libkapsl_backend_ort.so`, linked only to the signed pack-local ORT runtime
  and the allowlisted host system libraries;
- `libonnxruntime.so.1`, extracted from Microsoft's exact official CPU release
  asset and covered by the manifest's installed-file digest map;
- the minimal `backend-pack.json` consumed after extraction, explicitly marked
  with `adapter_abi: kapsl-backend-v1` so it cannot be confused with legacy
  provider-only ONNX bundles;
- `provenance.json`, recording source, lock/toolchain, binary, ORT distribution,
  notice, and allowed dynamic-library digests/identities;
- Kapsl, ONNX Runtime, ONNX Runtime third-party, and linked Rust dependency
  license notices.

The adjacent `.manifest.json` is intentionally unsigned and omits the archive
URL/digest/signature. Kapsl Engine's official backend-index publisher adds
those fields only after independently validating this handoff artifact.

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

## Reproducibility boundary

Archive order, paths, owners, modes, timestamps, JSON serialization, and gzip
metadata are deterministic. CI builds one adapter and assembles it twice,
requiring byte-identical archives, manifests, checksums, and signatures.

The adapter binary itself is reproducible only under the pinned
Rust/toolchain/target and linker inputs. `provenance.json` records those inputs,
the official ORT archive and library hashes, each ELF dependency closure, and
each library's highest required GLIBC version. Release policy rejects anything
above GLIBC 2.35, and CI performs a real `dlopen` on Ubuntu 22.04. A release
verifier can rebuild in a second isolated environment and compare both signed
library SHA-256 values.

## Certification handoff

After the engine release pipeline has accepted the archive into a locally
signed backend index, run the CPU ABBA harness in `integrations/ort/conformance`
from the same exact integrations commit. Preserve the archive, template,
signature/index, parity captures, logs, and teardown evidence together.
Embedded ORT remains the rollback until every required CPU task profile passes
and the later CUDA/TensorRT gates are complete.
