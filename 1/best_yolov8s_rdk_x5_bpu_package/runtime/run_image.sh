#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

python3 "$SCRIPT_DIR/infer_image.py" \
  --model "$PACKAGE_DIR/model/best_yolov8s_bpu_bayese_640x640_nv12.bin" \
  --image "${1:-$PACKAGE_DIR/cal_images/170.jpg}" \
  --output "${2:-$PACKAGE_DIR/result.jpg}" \
  --score 0.25 \
  --nms 0.70 \
  --bpu-cores 0
