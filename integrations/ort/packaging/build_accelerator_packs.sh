#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_CUDA_RUNTIME_ROOT:?KAPSL_CUDA_RUNTIME_ROOT is required}"
: "${KAPSL_TENSORRT_RUNTIME_DIR:?KAPSL_TENSORRT_RUNTIME_DIR is required}"
: "${KAPSL_TENSORRT_LICENSE_DIR:?KAPSL_TENSORRT_LICENSE_DIR is required}"

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "ORT accelerator packs are currently built only on Linux x86_64." >&2
  exit 1
fi
for command_name in cargo git nm openssl patchelf python3 readelf; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to build ORT accelerator packs." >&2
    exit 1
  fi
done
if [ -n "${RUSTFLAGS:-}" ] || [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  echo "Release packaging requires an unset RUSTFLAGS and CARGO_ENCODED_RUSTFLAGS." >&2
  exit 1
fi

packaging_toolchain="$({
  sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
    "$repo_root/rust-toolchain.toml" | head -n 1
})"
if [ "$packaging_toolchain" != "1.92.0" ]; then
  echo "ORT accelerator packaging requires the reviewed Rust 1.92.0 toolchain pin." >&2
  exit 1
fi
export RUSTUP_TOOLCHAIN="$packaging_toolchain"
actual_rustc="$(rustc --version)"
if [[ ! "$actual_rustc" =~ ^rustc[[:space:]]1\.92\.0[[:space:]] ]]; then
  echo "ORT accelerator packaging selected an unexpected compiler: $actual_rustc" >&2
  exit 1
fi

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
source_date_epoch="$(git -C "$repo_root" show -s --format=%ct HEAD)"
output_dir="${KAPSL_ORT_PACK_OUTPUT_DIR:-$repo_root/dist/ort-accelerator}"
build_root="${KAPSL_ORT_PACK_BUILD_DIR:-$repo_root/target/ort-accelerator-packaging}"
notices_dir="$build_root/notices"
runtime_dir="$build_root/onnxruntime-gpu"
nvidia_license="${KAPSL_NVIDIA_LICENSE_FILE:-$KAPSL_CUDA_RUNTIME_ROOT/NVIDIA-CONTAINER-LICENSE}"
mkdir -p "$notices_dir" "$runtime_dir"

for required in \
  "$KAPSL_CUDA_RUNTIME_ROOT" \
  "$KAPSL_TENSORRT_RUNTIME_DIR" \
  "$KAPSL_TENSORRT_LICENSE_DIR" \
  "$nvidia_license"; do
  if [ ! -e "$required" ]; then
    echo "Missing ORT accelerator packaging input: $required" >&2
    exit 1
  fi
done

export PYTHONDONTWRITEBYTECODE=1
python3 "$repo_root/integrations/ort/packaging/fetch_ort_gpu_runtime.py" \
  --output-dir "$runtime_dir"
ln -sfn libonnxruntime.so.1 "$runtime_dir/libonnxruntime.so"

python3 "$repo_root/integrations/ort/packaging/generate_cargo_notices.py" \
  --manifest-path "$repo_root/Cargo.toml" \
  --package kapsl-backend-ort \
  --target x86_64-unknown-linux-gnu \
  --workspace-license "$repo_root/LICENSE" \
  --output "$notices_dir/RUST-DEPENDENCY-NOTICES"

python3 "$repo_root/integrations/ort/packaging/fetch_ort_notices.py" \
  --output "$notices_dir/ONNX-RUNTIME-THIRD-PARTY-NOTICES"

signing_args=()
if [ -n "${KAPSL_BACKEND_SIGNING_KEY:-}" ]; then
  : "${KAPSL_BACKEND_EXPECTED_PUBLIC_KEY:?KAPSL_BACKEND_EXPECTED_PUBLIC_KEY is required when signing}"
  signing_args=(
    --signing-key "$KAPSL_BACKEND_SIGNING_KEY"
    --expected-public-key "$KAPSL_BACKEND_EXPECTED_PUBLIC_KEY"
  )
fi

separator=$'\x1f'
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=${repo_root}=.${separator}-C${separator}link-arg=-Wl,--build-id=none"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=symbols
export SOURCE_DATE_EPOCH="$source_date_epoch"

for profile in cuda12 tensorrt10; do
  feature="profile-${profile}"
  target_dir="$build_root/target-${profile}"
  ORT_LIB_LOCATION="$runtime_dir" \
  ORT_PREFER_DYNAMIC_LINK=1 \
  cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --package kapsl-backend-ort \
    --no-default-features \
    --features "$feature" \
    --release \
    --locked \
    --target x86_64-unknown-linux-gnu \
    --target-dir "$target_dir"

  profile_args=()
  if [ "$profile" = "tensorrt10" ]; then
    profile_args=(
      --tensorrt-runtime-dir "$KAPSL_TENSORRT_RUNTIME_DIR"
      --tensorrt-license-dir "$KAPSL_TENSORRT_LICENSE_DIR"
    )
  fi
  python3 "$repo_root/integrations/ort/packaging/package_accelerator.py" \
    --profile "$profile" \
    --library "$target_dir/x86_64-unknown-linux-gnu/release/libkapsl_backend_ort.so" \
    --ort-runtime-dir "$runtime_dir" \
    --cuda-runtime-dir "$KAPSL_CUDA_RUNTIME_ROOT" \
    --nvidia-license "$nvidia_license" \
    --output-dir "$output_dir" \
    --kapsl-version "$KAPSL_VERSION" \
    --source-commit "$source_commit" \
    --source-date-epoch "$source_date_epoch" \
    --repository-root "$repo_root" \
    --cargo-notices "$notices_dir/RUST-DEPENDENCY-NOTICES" \
    --ort-notices "$notices_dir/ONNX-RUNTIME-THIRD-PARTY-NOTICES" \
    "${profile_args[@]}" \
    "${signing_args[@]}"
done
