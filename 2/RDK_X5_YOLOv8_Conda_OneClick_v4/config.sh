#!/usr/bin/env bash

# 使用全新的环境名，避免与之前失败的环境冲突。
ENV_NAME="rdkx5_convert_clean"

MODEL_PT="./model/best.pt"
CAL_IMAGES="./calibration_images"
OUTPUT_DIR="./output"

# 当前工程针对 Ultralytics YOLOv8 Detect。
IMGSZ=640
OPSET=11
JOBS=2
OPTIMIZE_LEVEL="O3"
QUANTIZED="int8"
CAL_SAMPLE_NUM=30
RESIZE_MODE="stretch"
MIN_CAL_IMAGES=20
KEEP_WORKSPACE=0
