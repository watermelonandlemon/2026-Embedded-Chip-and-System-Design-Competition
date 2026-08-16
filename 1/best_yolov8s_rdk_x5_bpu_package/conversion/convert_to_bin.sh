#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v hb_mapper >/dev/null 2>&1; then
  echo "错误：当前环境没有 hb_mapper。"
  echo "请先激活 D-Robotics OpenExplorer，或在 Ubuntu 22.04 / Python 3.10 中安装："
  echo "  pip install rdkx5-yolo-mapper"
  exit 1
fi

python3 "$SCRIPT_DIR/mapper.py" \
  --onnx "$PACKAGE_DIR/model/best_yolov8s_bpu.onnx" \
  --cal-images "$PACKAGE_DIR/cal_images" \
  --output-dir "$PACKAGE_DIR/model" \
  --workspace "$SCRIPT_DIR/.mapper_workspace" \
  --optimize-level O3 \
  --quantized int8

echo
printf '生成完成：%s\n' "$PACKAGE_DIR/model/best_yolov8s_bpu_bayese_640x640_nv12.bin"
