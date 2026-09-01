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

1. builds `kapsl-backend-ort` with Rust 1.92, the committed Cargo lock, no
   incremental state, remapped source paths, and no linker build ID;
2. generates notices from only normal dependencies reachable through the
   target-filtered locked Cargo graph;
3. downloads ONNX Runtime 1.23.2's official third-party notices from its exact
   tag and verifies the pinned SHA-256;
4. verifies an ELF64 x86_64 entrypoint, the `kapsl_backend_v1` export, a
   pack-local runtime path, and an allowlist of host system libraries;
5. writes a deterministic tar/gzip archive and matching engine manifest
   template, checksum, and provenance.

The source commit and `SOURCE_DATE_EPOCH` are derived from `HEAD`. Packaging a
dirty checkout, a different stated commit, or a different timestamp fails.
Build inputs and output directories are ignored, so they do not weaken this
check.

## Outputs

The default output directory is `dist/ort-cpu/`:

```text
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz.manifest.json
kapsl-backend-onnx-cpu-<kapsl-version>-linux-x86_64.tar.gz.sha256
```

The archive contains:

- `libkapsl_backend_ort.so`, with ONNX Runtime statically linked;
- the minimal `backend-pack.json` consumed after extraction;
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

The binary itself is reproducible only under the pinned Rust/toolchain/target
and resolved ORT distribution inputs. `provenance.json` records those inputs so
a release verifier can rebuild in a second isolated environment and compare
the entrypoint SHA-256. The packager does not claim that two arbitrary Linux
distributions or linker versions produce the same binary.

## Certification handoff

After the engine release pipeline has accepted the archive into a locally
signed backend index, run the CPU ABBA harness in `kapsl-benchmarks`. Preserve
the archive, template, signature/index, parity captures, logs, and teardown
evidence together. Embedded ORT remains the rollback until every required CPU
task profile passes and the later CUDA/TensorRT gates are complete.
