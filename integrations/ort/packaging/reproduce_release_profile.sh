#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 PROFILE ADAPTER_VERSION KAPSL_VERSION RELEASE_TAG REPOSITORY SOURCE_COMMIT OUTPUT_DIR" >&2
  exit 2
fi
: "${KAPSL_BACKEND_SIGNING_KEY:?KAPSL_BACKEND_SIGNING_KEY is required}"
: "${KAPSL_BACKEND_EXPECTED_PUBLIC_KEY:?KAPSL_BACKEND_EXPECTED_PUBLIC_KEY is required}"

profile="$1"
adapter_version="$2"
kapsl_version="$3"
release_tag="$4"
repository="$5"
source_commit="$6"
output_dir="$7"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
artifact="kapsl-backend-onnx-${profile}-${kapsl_version}-linux-x86_64.tar.gz"
if [ -e "$output_dir" ]; then
  echo "ORT reproduced release output already exists: $output_dir" >&2
  exit 1
fi

release_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/kapsl-ort-release.XXXXXX")"
cleanup() {
  rm -rf "$release_root"
}
trap cleanup EXIT INT TERM
mkdir "$release_root/baseline"

for run in first second; do
  run_root="$release_root/$run"
  "$repo_root/integrations/ort/packaging/build_release_profile.sh" \
    "$profile" \
    "$kapsl_version" \
    "$run_root/scratch" \
    "$run_root/output"
  archive="$run_root/output/$artifact"
  digest="$(sha256sum "$archive" | awk '{print $1}')"
  if [ "$run" = "first" ]; then
    printf '%s\n' "$digest" > "$release_root/baseline/archive.sha256"
    cp "$archive.manifest.json" "$release_root/baseline/manifest.json"
    cp "$archive.sig" "$release_root/baseline/signature"
    rm -rf "$run_root"
  else
    if [ "$digest" != "$(cat "$release_root/baseline/archive.sha256")" ]; then
      echo "Independent ORT $profile release builds are not byte-identical." >&2
      exit 1
    fi
    cmp "$archive.manifest.json" "$release_root/baseline/manifest.json"
    cmp "$archive.sig" "$release_root/baseline/signature"
  fi
done

mv "$release_root/second/output" "$release_root/release-handoff"
rm -rf "$release_root/second"
python3 "$repo_root/integrations/ort/packaging/release.py" prepare-profile \
  --profile "$profile" \
  --adapter-version "$adapter_version" \
  --kapsl-version "$kapsl_version" \
  --release-tag "$release_tag" \
  --repository "$repository" \
  --source-commit "$source_commit" \
  --signing-key "$KAPSL_BACKEND_SIGNING_KEY" \
  --expected-public-key "$KAPSL_BACKEND_EXPECTED_PUBLIC_KEY" \
  --directory "$release_root/release-handoff" \
  --output-dir "$release_root/release-assets" \
  --consume-archive
mv "$release_root/release-assets" "$output_dir"
