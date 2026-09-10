# ORT CPU platform inputs

`fetch_ort_platform_runtime.py` prepares immutable upstream runtime inputs for
Linux x86-64/ARM64, macOS x86-64/ARM64 and Windows x86-64. Platform names and Rust
targets are explicit. It authenticates the archive and each selected regular
file before writing anything, rejects duplicate/link/missing members, and never
extracts paths supplied by an archive. Output directories must be empty.

```sh
python3 integrations/ort/packaging/fetch_ort_platform_runtime.py \
  --platform macos-aarch64 --output-dir /tmp/ort-macos-inputs
```

The archive digests were checked against the official Microsoft
[ONNX Runtime 1.23.2 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2),
then verified against the downloaded bytes. Per-library size/hash locks identify
the exact bytes selected from those archives. `--archive` accepts a local copy
under the same validation. `runtime-inputs.json` records platform, target, source
URL, archive hash and selected file hashes for subsequent pack provenance. The
Windows import library is marked as a build input; it is not a runtime sidecar.

These inputs are not installable signed backend packs. The current release
pipeline still publishes Linux x86-64 CPU/CUDA/TensorRT packs only. Before
retiring embedded ORT, the remaining platforms need adapter builds, platform
dependency/loader validation, complete signed archives, release-index coverage,
verified engine locks and CPU task/lifecycle tests through the installed packs.
Retained tasks include generation. This change does not narrow engine platform
support or claim platform qualification.

PR tests validate extraction and rejection on Linux, macOS and Windows with
small fake archives. They provision no GPU and enforce no performance threshold.
