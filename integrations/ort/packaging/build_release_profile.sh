#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 PROFILE KAPSL_VERSION SCRATCH_DIR OUTPUT_DIR" >&2
  exit 2
fi
: "${KAPSL_BACKEND_SIGNING_KEY:?KAPSL_BACKEND_SIGNING_KEY is required}"
: "${KAPSL_BACKEND_EXPECTED_PUBLIC_KEY:?KAPSL_BACKEND_EXPECTED_PUBLIC_KEY is required}"

profile="$1"
kapsl_version="$2"
scratch="$3"
output_dir="$4"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "ORT release profiles are built only on Linux x86_64." >&2
  exit 1
fi
if [ -e "$scratch" ] || [ -e "$output_dir" ]; then
  echo "ORT release scratch and output paths must not already exist." >&2
  exit 1
fi
mkdir -p "$scratch" "$(dirname "$output_dir")"

case "$profile" in
  cpu)
    KAPSL_VERSION="$kapsl_version" \
    KAPSL_ORT_PACK_OUTPUT_DIR="$output_dir" \
    KAPSL_ORT_PACK_BUILD_DIR="$scratch/build" \
      "$repo_root/integrations/ort/packaging/build_cpu_pack.sh"
    ;;
  cuda12 | tensorrt10)
    cuda_runtime="$scratch/cuda-runtime"
    cuda_provenance="$scratch/cuda-runtime-source.json"
    KAPSL_CUDA_IMAGE_CLEANUP=1 \
      "$repo_root/integrations/ort/packaging/collect_cuda_runtime.sh" \
      "$cuda_runtime" \
      "$cuda_provenance"

    accelerator_environment=(
      KAPSL_VERSION="$kapsl_version"
      KAPSL_ORT_PACK_PROFILES="$profile"
      KAPSL_ORT_PACK_CONSUME_INPUT_LIBRARIES=1
      KAPSL_ORT_PACK_OUTPUT_DIR="$output_dir"
      KAPSL_ORT_PACK_BUILD_DIR="$scratch/build"
      KAPSL_CUDA_RUNTIME_ROOT="$cuda_runtime"
      KAPSL_CUDA_RUNTIME_PROVENANCE="$cuda_provenance"
    )
    if [ "$profile" = "tensorrt10" ]; then
      tensorrt_runtime="$scratch/tensorrt-runtime"
      tensorrt_licenses="$scratch/tensorrt-licenses"
      tensorrt_provenance="$scratch/tensorrt-runtime-source.json"
      python3 "$repo_root/integrations/ort/packaging/fetch_tensorrt_runtime.py" \
        --runtime-dir "$tensorrt_runtime" \
        --license-dir "$tensorrt_licenses" \
        --provenance "$tensorrt_provenance"
      accelerator_environment+=(
        KAPSL_TENSORRT_RUNTIME_DIR="$tensorrt_runtime"
        KAPSL_TENSORRT_LICENSE_DIR="$tensorrt_licenses"
        KAPSL_TENSORRT_RUNTIME_PROVENANCE="$tensorrt_provenance"
      )
    fi
    env "${accelerator_environment[@]}" \
      "$repo_root/integrations/ort/packaging/build_accelerator_packs.sh"
    ;;
  *)
    echo "Unknown ORT release profile: $profile" >&2
    exit 1
    ;;
esac

artifact="kapsl-backend-onnx-${profile}-${kapsl_version}-linux-x86_64.tar.gz"
for required in \
  "$output_dir/$artifact" \
  "$output_dir/$artifact.manifest.json" \
  "$output_dir/$artifact.sha256" \
  "$output_dir/$artifact.sig"; do
  if [ ! -f "$required" ]; then
    echo "ORT release build did not emit $required" >&2
    exit 1
  fi
done
