#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.sh"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local candidate
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/root/miniconda3/bin/conda" \
    "/root/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      return 0
    fi
  done
  return 1
}

fail() {
  echo "[错误] $*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" ]] || fail "必须在 Linux 虚拟机中执行。"
[[ "$(uname -m)" == "x86_64" ]] || fail "必须使用 x86_64 Linux；当前架构：$(uname -m)"
find_conda || fail "没有找到 Conda。请先安装 Miniconda/Anaconda，或确认 conda 可执行文件路径。"

# D-Robotics 工具对路径较敏感，提前阻止中文和空格路径。
if [[ "$ROOT_DIR" =~ [[:space:]] ]] || LC_ALL=C grep -q '[^ -~]' <<<"$ROOT_DIR"; then
  cat >&2 <<MSG
[错误] 当前工程路径包含空格或中文：
  $ROOT_DIR
请把整个文件夹移动到纯英文路径，例如：
  mv "$ROOT_DIR" ~/rdk_x5_converter
  cd ~/rdk_x5_converter
MSG
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[信息] 已存在 Conda 环境：$ENV_NAME"
else
  echo "[信息] 创建 Conda 环境：$ENV_NAME（Python 3.10）"
  conda create -n "$ENV_NAME" python=3.10 pip -y
fi

run_py() {
  conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

run_py -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" wheel

# 先安装 CPU 版 PyTorch，避免虚拟机下载 CUDA 运行库。
run_py -m pip install \
  "torch==2.7.0" "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cpu

# 所有 mapper 与 Ultralytics 依赖放在同一次解析中，避免 NumPy/OpenCV 冲突。
run_py -m pip install \
  --only-binary=:all: \
  --upgrade-strategy only-if-needed \
  -c "$ROOT_DIR/constraints.txt" \
  -r "$ROOT_DIR/requirements.txt"

run_py -m pip check
YOLO_CONFIG_DIR="$ROOT_DIR/.ultralytics" \
  conda run --no-capture-output -n "$ENV_NAME" \
  python "$ROOT_DIR/scripts/check_environment.py"

echo
echo "[完成] Conda 环境配置成功：$ENV_NAME"
echo "下一步可以执行：bash convert.sh"
