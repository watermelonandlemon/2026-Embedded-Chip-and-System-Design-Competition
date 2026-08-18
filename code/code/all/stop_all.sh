#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sunrise/code/all"
LOG="$ROOT/log"

if [[ "$EUID" -ne 0 ]]; then
  echo "请使用：sudo $ROOT/stop_all.sh"
  exit 1
fi

stop_one() {
  local name="$1"
  local pid_file="$LOG/$name.pid"

  [[ -f "$pid_file" ]] || return 0

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true

    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done

    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi

    echo "[STOP] $name PID=$pid"
  fi

  rm -f "$pid_file"
}

# 先关闭 UI，再关闭摄像头、业务节点和 Broker。
for name in ui vision web gpio count broker; do
  stop_one "$name"
done

# 只清理本项目路径对应的残留进程。
pkill -TERM -f "/home/sunrise/code/all/src/node_" 2>/dev/null || true
pkill -TERM -f "python3 -u -m miniROS --host 0.0.0.0 --port 8765" 2>/dev/null || true

# 删除内存文件系统中的旧预览图，避免下次启动先显示旧画面。
rm -f /dev/shm/rdk_smartbin_latest.png
rm -f /dev/shm/.rdk_smartbin_latest.*.tmp.png

echo "[STOP] 已停止"
