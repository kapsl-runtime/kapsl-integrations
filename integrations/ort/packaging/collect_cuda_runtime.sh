#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 OUTPUT_DIR PROVENANCE_JSON" >&2
  exit 2
fi
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "CUDA runtime collection is supported only on Linux x86_64." >&2
  exit 1
fi
for command_name in docker python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to collect the CUDA runtime." >&2
    exit 1
  fi
done

output_dir="$1"
provenance_path="$2"
image="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010"
if [ -e "$output_dir" ] || [ -e "$provenance_path" ]; then
  echo "CUDA runtime output paths must not already exist." >&2
  exit 1
fi

scratch="$(mktemp -d "${RUNNER_TEMP:-/tmp}/kapsl-cuda-runtime.XXXXXX")"
cleanup() {
  rm -rf "$scratch"
  if [ "${KAPSL_CUDA_IMAGE_CLEANUP:-0}" = "1" ]; then
    docker image rm "$image" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM
mkdir "$scratch/runtime"

docker pull --platform linux/amd64 "$image"
docker run \
  --rm \
  --network none \
  --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --volume "$scratch/runtime:/kapsl-output" \
  --entrypoint /bin/bash \
  "$image" \
  -euo pipefail -c '
    shopt -s nullglob
    candidates=(
      /usr/local/cuda/targets/x86_64-linux/lib/lib*.so.*
      /usr/lib/x86_64-linux-gnu/libcudnn*.so.*
    )
    for source in "${candidates[@]}"; do
      name="$(basename "$source")"
      if [[ ! "$name" =~ ^lib[A-Za-z0-9_+-]+\.so\.[0-9]+$ ]]; then
        continue
      fi
      destination="/kapsl-output/$name"
      if [ -e "$destination" ]; then
        if ! cmp -s "$source" "$destination"; then
          echo "Conflicting CUDA runtime library basename: $name" >&2
          exit 1
        fi
        continue
      fi
      cp -L -- "$source" "$destination"
      chmod 0755 "$destination"
    done
    cp /NGC-DL-CONTAINER-LICENSE /kapsl-output/NVIDIA-CONTAINER-LICENSE
    chmod 0644 /kapsl-output/NVIDIA-CONTAINER-LICENSE
  '

for required in \
  libcudart.so.12 \
  libcublas.so.12 \
  libcublasLt.so.12 \
  libcudnn.so.9; do
  if [ ! -f "$scratch/runtime/$required" ] || [ -L "$scratch/runtime/$required" ]; then
    echo "Pinned CUDA image did not yield regular runtime library $required" >&2
    exit 1
  fi
done
if find "$scratch/runtime" -maxdepth 1 -type f \
  \( -name 'libcuda.so*' -o -name 'libnvidia-*.so*' \) -print -quit | grep -q .; then
  echo "CUDA runtime collection included a host NVIDIA driver library." >&2
  exit 1
fi

python3 - "$scratch/runtime" "$scratch/provenance.json" "$image" <<'PY'
import hashlib
import json
import pathlib
import sys

runtime = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
image = sys.argv[3]

files = {}
for path in sorted(item for item in runtime.iterdir() if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    files[path.name] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}

payload = {
    "schema_version": 1,
    "name": "NVIDIA CUDA 12.8 and cuDNN 9 Linux x86_64 runtime",
    "distribution": {"container_image": image},
    "files": files,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

mkdir -p "$(dirname "$output_dir")" "$(dirname "$provenance_path")"
mv "$scratch/runtime" "$output_dir"
mv "$scratch/provenance.json" "$provenance_path"
find "$output_dir" -maxdepth 1 -type f -print | sort
