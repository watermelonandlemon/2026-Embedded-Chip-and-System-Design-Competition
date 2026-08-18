#!/usr/bin/env bash
set -euo pipefail

cd /home/sunrise/code/camera/

/usr/bin/python3 camera.py \
  --model /home/sunrise/code/camera/best_bayese_640x640_nv12.bin \
  --labels /home/sunrise/code/camera/classes.names \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --roi-x1 220 \
  --roi-y1 180 \
  --roi-x2 1100 \
  --roi-y2 680 \
  --score 0.50 \
  --nms 0.45 \
  --bpu-core 0
