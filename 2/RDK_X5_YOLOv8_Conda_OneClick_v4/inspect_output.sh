#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
BIN=$(find output -maxdepth 1 -type f -name '*.bin' | sort | head -n 1 || true)
[[ -n "$BIN" ]] || { echo "output/ 中没有 .bin 文件" >&2; exit 1; }
ls -lh "$BIN" output/labels.txt output/model_metadata.json output/manifest.json 2>/dev/null || true
sha256sum "$BIN"
echo
echo "板端检查命令："
echo "hrt_model_exec model_info --model_file $BIN"
echo "hrt_model_exec perf --model_file $BIN --thread_num 1"
