#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.sh"

fail() {
  echo "[错误] $*" >&2
  exit 1
}

if [[ "$ROOT_DIR" =~ [[:space:]] ]] || LC_ALL=C grep -q '[^ -~]' <<<"$ROOT_DIR"; then
  fail "工程路径不能含中文或空格。请移动到 ~/rdk_x5_converter 后重试。"
fi

if ! command -v conda >/dev/null 2>&1; then
  for p in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" /opt/conda/bin /root/miniconda3/bin /root/anaconda3/bin; do
    [[ -x "$p/conda" ]] && export PATH="$p:$PATH" && break
  done
fi
command -v conda >/dev/null 2>&1 || fail "找不到 Conda。"
eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" || fail "环境不存在，请先执行 bash setup_conda.sh"

[[ -f "$MODEL_PT" ]] || fail "没有找到 $MODEL_PT。请把权重重命名为 best.pt 后放入 model/。"
[[ -d "$CAL_IMAGES" ]] || fail "没有找到校准图片目录：$CAL_IMAGES"

IMAGE_COUNT=$(find "$CAL_IMAGES" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l | tr -d ' ')
if (( IMAGE_COUNT < MIN_CAL_IMAGES )); then
  cat >&2 <<MSG
[错误] 校准图片只有 $IMAGE_COUNT 张，至少需要 $MIN_CAL_IMAGES 张。
请放入 20～50 张真实场景图片，或者执行：
  bash select_calibration.sh /你的数据集/images/train 30
MSG
  exit 1
fi

echo "[信息] 模型：$MODEL_PT"
echo "[信息] 校准图片：$IMAGE_COUNT 张"
echo "[信息] 输出目录：$OUTPUT_DIR"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$ROOT_DIR/logs" "$ROOT_DIR/.ultralytics"
ONNX_PATH="${MODEL_PT%.*}.onnx"
rm -f "$ONNX_PATH"

run_py() {
  YOLO_CONFIG_DIR="$ROOT_DIR/.ultralytics" \
  conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

run_py "$ROOT_DIR/scripts/check_environment.py" | tee "$ROOT_DIR/logs/00_environment.log"

run_py "$ROOT_DIR/scripts/extract_labels.py" \
  --pt "$MODEL_PT" \
  --labels "$OUTPUT_DIR/labels.txt" \
  --metadata "$OUTPUT_DIR/model_metadata.json" \
  | tee "$ROOT_DIR/logs/01_labels.log"

run_py "$ROOT_DIR/scripts/export_monkey_patch.py" \
  --pt "$MODEL_PT" \
  --imgsz "$IMGSZ" \
  --opset "$OPSET" \
  | tee "$ROOT_DIR/logs/02_export_onnx.log"

[[ -f "$ONNX_PATH" ]] || fail "ONNX 导出结束但没有找到：$ONNX_PATH"

run_py "$ROOT_DIR/scripts/verify_onnx.py" \
  --onnx "$ONNX_PATH" \
  --expected-size "$IMGSZ" \
  --expected-opset "$OPSET" \
  --expected-output-count 6 \
  --report "$OUTPUT_DIR/onnx_report.json" \
  | tee "$ROOT_DIR/logs/03_verify_onnx.log"

MAPPER_ARGS=(
  --onnx "$ONNX_PATH"
  --cal-images "$CAL_IMAGES"
  --output-dir "$OUTPUT_DIR"
  --jobs "$JOBS"
  --optimize-level "$OPTIMIZE_LEVEL"
  --quantized "$QUANTIZED"
  --cal-sample-num "$CAL_SAMPLE_NUM"
  --resize-mode "$RESIZE_MODE"
  --skip-checker
)
(( KEEP_WORKSPACE == 1 )) && MAPPER_ARGS+=(--keep-workspace)

run_py "$ROOT_DIR/scripts/mapper.py" "${MAPPER_ARGS[@]}" \
  | tee "$ROOT_DIR/logs/04_mapper.log"

cp -f "$ROOT_DIR"/logs/*.log "$OUTPUT_DIR/" 2>/dev/null || true
run_py "$ROOT_DIR/scripts/create_manifest.py" \
  --output-dir "$OUTPUT_DIR" \
  --source-pt "$MODEL_PT" \
  --source-onnx "$ONNX_PATH"

BIN_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.bin' | sort | head -n 1 || true)
[[ -n "$BIN_FILE" ]] || fail "转换结束，但 output/ 中没有 .bin 文件。请查看 output/hb_mapper_makertbin_console.log。"

echo
echo "============================================================"
echo "转换成功"
echo "BPU 模型：$BIN_FILE"
echo "类别文件：$OUTPUT_DIR/labels.txt"
echo "日志目录：$OUTPUT_DIR"
echo "============================================================"
