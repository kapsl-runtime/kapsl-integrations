#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "ORT CPU packs are currently built only on Linux x86_64." >&2
  exit 1
fi
for command_name in cargo git nm openssl python3 readelf; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to build an ORT CPU pack." >&2
    exit 1
  fi
done
if [ -n "${RUSTFLAGS:-}" ] || [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  echo "Release packaging requires an unset RUSTFLAGS and CARGO_ENCODED_RUSTFLAGS." >&2
  exit 1
fi

packaging_toolchain="$(
  sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
    "$repo_root/rust-toolchain.toml" | head -n 1
)"
if [ "$packaging_toolchain" != "1.92.0" ]; then
  echo "ORT CPU packaging requires the reviewed Rust 1.92.0 toolchain pin." >&2
  exit 1
fi
export RUSTUP_TOOLCHAIN="$packaging_toolchain"
actual_rustc="$(rustc --version)"
if [[ ! "$actual_rustc" =~ ^rustc[[:space:]]1\.92\.0[[:space:]] ]]; then
  echo "ORT CPU packaging selected an unexpected compiler: $actual_rustc" >&2
  exit 1
fi

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
source_date_epoch="$(git -C "$repo_root" show -s --format=%ct HEAD)"
output_dir="${KAPSL_ORT_PACK_OUTPUT_DIR:-$repo_root/dist/ort-cpu}"
build_root="${KAPSL_ORT_PACK_BUILD_DIR:-$repo_root/target/ort-cpu-packaging}"
notices_dir="$build_root/notices"
runtime_dir="$build_root/onnxruntime"
runtime_library="$runtime_dir/libonnxruntime.so.1"
mkdir -p "$notices_dir" "$runtime_dir"

export PYTHONDONTWRITEBYTECODE=1
python3 "$repo_root/integrations/ort/packaging/fetch_ort_runtime.py" \
  --output "$runtime_library"
ln -sfn "$(basename "$runtime_library")" "$runtime_dir/libonnxruntime.so"

separator=$'\x1f'
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=${repo_root}=.${separator}-C${separator}link-arg=-Wl,--build-id=none"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=symbols
export SOURCE_DATE_EPOCH="$source_date_epoch"

ORT_LIB_LOCATION="$runtime_dir" \
ORT_PREFER_DYNAMIC_LINK=1 \
cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --package kapsl-backend-ort \
  --no-default-features \
  --features profile-cpu \
  --release \
  --locked \
  --target x86_64-unknown-linux-gnu \
  --target-dir "$build_root/target"

python3 "$repo_root/integrations/ort/packaging/generate_cargo_notices.py" \
  --manifest-path "$repo_root/Cargo.toml" \
  --package kapsl-backend-ort \
  --target x86_64-unknown-linux-gnu \
  --workspace-license "$repo_root/LICENSE" \
  --supplemental-license-index \
    "$repo_root/integrations/ort/third_party/rust-license-supplements.json" \
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

python3 "$repo_root/integrations/ort/packaging/package_cpu.py" \
  --library "$build_root/target/x86_64-unknown-linux-gnu/release/libkapsl_backend_ort.so" \
  --runtime-library "$runtime_library" \
  --output-dir "$output_dir" \
  --kapsl-version "$KAPSL_VERSION" \
  --source-commit "$source_commit" \
  --source-date-epoch "$source_date_epoch" \
  --repository-root "$repo_root" \
  --cargo-notices "$notices_dir/RUST-DEPENDENCY-NOTICES" \
  --ort-notices "$notices_dir/ONNX-RUNTIME-THIRD-PARTY-NOTICES" \
  "${signing_args[@]}"
