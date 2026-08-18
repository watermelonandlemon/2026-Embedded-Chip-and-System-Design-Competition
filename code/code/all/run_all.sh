#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sunrise/code/all"
SRC="$ROOT/src"
LIB="$ROOT/lib"
LOG="$ROOT/log"
PYTHON="/usr/bin/python3"

if [[ "$EUID" -ne 0 ]]; then
  echo "请使用：sudo $ROOT/run_all.sh"
  exit 1
fi

for required in \
  "$SRC/node_count.py" \
  "$SRC/node_gpio.py" \
  "$SRC/node_web.py" \
  "$SRC/node_vision.py" \
  "$SRC/node_ui.py" \
  "$SRC/config.yaml"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] 缺少文件：$required"
    exit 2
  fi
done

mkdir -p "$LOG"
chmod 755 "$ROOT" "$SRC" "$LIB" "$LOG"
export PYTHONPATH="$LIB"

"$PYTHON" -c "import miniROS" >/dev/null 2>&1 || {
  echo "[ERROR] /usr/bin/python3 无法导入 miniROS"
  exit 2
}

"$PYTHON" -c "import cv2, numpy, hbm_runtime" >/dev/null 2>&1 || {
  echo "[ERROR] /usr/bin/python3 无法导入 cv2、numpy 或 hbm_runtime"
  exit 2
}

"$PYTHON" -c "import tkinter" >/dev/null 2>&1 || {
  echo "[ERROR] /usr/bin/python3 无法导入 tkinter，本地 UI 无法启动"
  echo "[ERROR] 可尝试安装：sudo apt install python3-tk"
  exit 2
}

DESKTOP_USER="${SUDO_USER:-sunrise}"
DESKTOP_HOME="$(
  getent passwd "$DESKTOP_USER" 2>/dev/null |
    cut -d: -f6 || true
)"
DESKTOP_HOME="${DESKTOP_HOME:-/home/sunrise}"

export DISPLAY="${SMARTBIN_DISPLAY:-:0}"
export XAUTHORITY="${SMARTBIN_XAUTHORITY:-$DESKTOP_HOME/.Xauthority}"

"$ROOT/stop_all.sh" >/dev/null 2>&1 || true

start_node() {
  local name="$1"
  shift

  : > "$LOG/$name.log"
  nohup "$@" >> "$LOG/$name.log" 2>&1 &

  local pid="$!"
  echo "$pid" > "$LOG/$name.pid"
  echo "[RUN] $name PID=$pid"
}

# 1. Broker 必须先启动。
start_node broker \
  "$PYTHON" -u -m miniROS \
  --host 0.0.0.0 \
  --port 8765
sleep 1

# 2. 业务节点。视觉节点只负责推理和写入预览帧，不再创建窗口。
start_node count \
  env PYTHONPATH="$LIB" \
  "$PYTHON" -u "$SRC/node_count.py"

start_node gpio \
  env PYTHONPATH="$LIB" \
  "$PYTHON" -u "$SRC/node_gpio.py"

start_node web \
  env PYTHONPATH="$LIB" \
  "$PYTHON" -u "$SRC/node_web.py"

start_node vision \
  env PYTHONPATH="$LIB" \
  "$PYTHON" -u "$SRC/node_vision.py"

# 3. UI 独立启动，仅 UI 进程需要 HDMI/X11 环境。
sleep 1
start_node ui \
  env \
  PYTHONPATH="$LIB" \
  DISPLAY="$DISPLAY" \
  XAUTHORITY="$XAUTHORITY" \
  "$PYTHON" -u "$SRC/node_ui.py"

sleep 1
failed=0
for name in broker count gpio web vision ui; do
  pid_file="$LOG/$name.pid"
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] $name 启动失败，请查看 $LOG/$name.log"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] 存在启动失败节点，正在回滚"
  "$ROOT/stop_all.sh"
  exit 3
fi

echo "[RUN] 全部启动完成"
echo "[RUN] UI显示：DISPLAY=$DISPLAY"
echo "[RUN] 日志目录：$LOG"
