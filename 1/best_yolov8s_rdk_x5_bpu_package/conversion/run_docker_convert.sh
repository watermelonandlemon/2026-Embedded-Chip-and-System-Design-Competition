#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE="${RDK_OPENEXPLORER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8}"

docker run --rm -it \
  --shm-size=15g \
  -v "$PACKAGE_DIR:/workspace" \
  -w /workspace/conversion \
  "$IMAGE" \
  bash -lc './convert_to_bin.sh'
