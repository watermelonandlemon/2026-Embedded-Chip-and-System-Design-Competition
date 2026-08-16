#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.sh"

SOURCE_DIR="${1:-}"
COUNT="${2:-30}"
if [[ -z "$SOURCE_DIR" ]]; then
  echo "用法：bash select_calibration.sh /数据集/images/train [数量]" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  for p in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" /opt/conda/bin /root/miniconda3/bin /root/anaconda3/bin; do
    [[ -x "$p/conda" ]] && export PATH="$p:$PATH" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "[错误] 找不到 conda" >&2; exit 1; }

eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" || bash "$ROOT_DIR/setup_conda.sh"

conda run --no-capture-output -n "$ENV_NAME" \
  python "$ROOT_DIR/scripts/select_calibration_images.py" \
  --source "$SOURCE_DIR" \
  --dest "$ROOT_DIR/calibration_images" \
  --count "$COUNT" \
  --clear
