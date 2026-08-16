#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.sh"

CAL_SOURCE="${1:-}"

# 首次运行自动创建/检查 Conda 环境。
bash "$ROOT_DIR/setup_conda.sh"

# 可选：bash run_all.sh /path/to/images/train
if [[ -n "$CAL_SOURCE" ]]; then
  bash "$ROOT_DIR/select_calibration.sh" "$CAL_SOURCE" "$CAL_SAMPLE_NUM"
fi

bash "$ROOT_DIR/convert.sh"
