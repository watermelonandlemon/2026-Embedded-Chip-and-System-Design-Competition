#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.sh"

if ! command -v conda >/dev/null 2>&1; then
  for p in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" /opt/conda/bin /root/miniconda3/bin /root/anaconda3/bin; do
    [[ -x "$p/conda" ]] && export PATH="$p:$PATH" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "[错误] 找不到 conda" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda deactivate >/dev/null 2>&1 || true
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env remove -n "$ENV_NAME" -y
fi
exec bash "$ROOT_DIR/setup_conda.sh"
