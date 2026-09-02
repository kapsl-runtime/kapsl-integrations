# Embedded ORT versus native-pack CPU parity

This harness certifies that the out-of-tree `onnx/cpu` adapter preserves the
observable behavior of Kapsl's embedded ONNX Runtime path before embedded ORT
is removed. It is deliberately CPU-only. It never provisions a runner, VM, or
GPU and it does not publish an artifact.

The owned-process `certify` mode runs this fixed sequence:

```text
embedded baseline -> native-pack candidate -> native-pack candidate -> embedded baseline
```

This ABBA order reduces first-run, filesystem-cache, and thermal order bias.
Each process is stopped before the next starts. The harness always tears down
the process group it created, including when readiness or inference fails.

## What is certified

- Exact engine and integrations commits plus the SHA-256 of the `.aimod`
  artifact are recorded in every capture.
- The same model, request payloads, session tuning, and host are used on both
  routes.
- Baseline runs require `KAPSL_GENERIC_NATIVE_PACKS=0`; candidate runs require
  `KAPSL_GENERIC_NATIVE_PACKS=1`.
- Required and forbidden startup-log markers prove which route was activated.
- Tensor shape and dtype must match. Integer/string tensors are byte-exact;
  floating tensors use the configured combined absolute/relative tolerance.
- Warmup output and every measured response must remain byte-stable within a
  runtime session.
- `warmup_requests` is executed separately at every configured concurrency
  point immediately before its measured trials. This prepares batch-shaped
  session pools and worker paths without counting lazy initialization as
  steady-state latency.
- Throughput, p95/p99 latency, startup time, engine-reported model memory, and
  process peak RSS are evaluated against explicit gates.

This is a retirement gate, not proof for CUDA or TensorRT. GPU allocator
ownership, unload/reload, cancellation, and provider-specific performance need
their own conformance suites before those embedded implementations can move.

## Requirements

- Python 3.10 or newer; no third-party Python packages are required.
- One engine binary built from the exact commit in `identity.engine_commit`.
- A signed local `onnx/cpu` pack built from the exact integrations commit in
  `identity.integrations_commit` and installed through the normal pack path.
- An `.aimod` containing the ONNX model and representative tensor requests.
- A quiet CPU host with enough memory for four sequential runtime sessions.

## Use from an exact integrations checkout

The directory is also a composite GitHub Action. It runs the harness unit tests
and exposes the exact checked-out `parity.py` path to a consuming job:

```yaml
- uses: actions/checkout@v4
  with:
    repository: kapsl-runtime/kapsl-integrations
    ref: FULL_40_HEX_COMMIT
    path: kapsl-integrations

- id: ort-parity-harness
  uses: ./kapsl-integrations/integrations/ort/conformance

- run: >-
    python3 "${{ steps.ort-parity-harness.outputs.harness_path }}" certify
    --config "$CONFIG"
    --output-dir "$EVIDENCE"
```

Pin the public integrations checkout to a full commit and verify its resolved
commit before using the local action. This keeps the adapter source, packaging
logic, and adapter-specific conformance contract at one immutable revision.
The consuming engine workflow still owns the comparison inputs, thresholds,
signed-pack installation path, and retained evidence.

Do not put API tokens, signing keys, or other secrets in `command`. The harness
does not record raw arguments, but subprocess logs are retained as evidence.
Supply a reader token through `KAPSL_BENCHMARK_TOKEN` or `KAPSL_API_TOKEN`, and
keep signing material outside the artifact directory.

## Configure a certification

Copy [`example-config.json`](example-config.json) outside the repository and
replace every absolute placeholder. Keep the baseline and candidate commands
identical. Only route-selection environment variables should differ.

The command supports two placeholders:

- `{model_path}`: the resolved `identity.model_path`
- `{output_dir}`: the certification artifact directory

`identity.model_sha256` may be empty while preparing a configuration; the
harness computes it. For durable or reviewed runs, paste that digest into the
configuration so an artifact change fails before either runtime starts.

The sequence must contain one or more complete ABBA blocks. For a longer run,
repeat the four entries rather than inventing an unbalanced order.

The owned mode rejects different commands, working directories, endpoints, or
unapproved environment differences. Its default environment allowlist contains
only the signed-pack discovery and route switches shown in the example. Add a
name to `allowed_variant_env_differences` only when the difference is necessary
for route selection and cannot change runtime tuning.

## Run the owned-process certification

Use an empty or nonexistent output directory:

```bash
python3 integrations/ort/conformance/parity.py certify \
  --config /absolute/path/ort-cpu-parity.json \
  --output-dir /absolute/path/ort-cpu-parity-artifacts
```

Exit status `0` means every gate passed, `1` means the run completed but at
least one parity gate failed, and `2` means configuration or execution could
not produce a valid certification.

The output contains one log and JSON capture per process, plus `report.json`
and `REPORT.md`. The process record includes a binary hash and a hash of the
argument vector, but never the raw argument vector.

## Capture already-running endpoints

For infrastructure that owns runtime lifecycle, capture each endpoint
separately and then compare the artifacts:

```bash
python3 integrations/ort/conformance/parity.py capture \
  --config /absolute/path/ort-cpu-parity.json \
  --variant baseline \
  --log-file /absolute/path/baseline.log \
  --output /absolute/path/baseline.json

python3 integrations/ort/conformance/parity.py capture \
  --config /absolute/path/ort-cpu-parity.json \
  --variant candidate \
  --log-file /absolute/path/candidate.log \
  --output /absolute/path/candidate.json

python3 integrations/ort/conformance/parity.py compare \
  --config /absolute/path/ort-cpu-parity.json \
  --baseline /absolute/path/baseline.json \
  --candidate /absolute/path/candidate.json \
  --output /absolute/path/report.json \
  --markdown /absolute/path/REPORT.md
```

Repeat `--baseline` and `--candidate` to compare multiple captures. External
lifecycle mode cannot measure process RSS and cannot prove startup latency from
process launch. Set `require_process_identity`, `require_process_rss`, and
`require_startup_evidence` to `false` only when another trusted system captures
equivalent evidence.

## Interpreting the gate

Do not remove embedded ORT after a single convenient model. A retirement
decision should retain passing reports for every supported CPU task class,
including at least raw forward inference, classification postprocessing,
embeddings, vision preprocessing, audio preprocessing, multi-input models, and
the concurrency/session-pool profiles supported by the product.

If a gate fails, preserve all captures and logs. Change a threshold only with
a documented product-level reason; do not tune thresholds to hide a regression.
