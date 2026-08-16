#!/usr/bin/env python3
"""Prepare calibration data and compile an RDK X5 Bayes-e NV12 .bin model.

The generated mapper configuration follows D-Robotics RDK Model Zoo's
Apache-2.0 licensed Ultralytics YOLO conversion example.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("RDK_X5_MAPPER")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    LOGGER.info("执行：%s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"命令执行失败，返回码 {return_code}：{' '.join(command)}")


def evenly_sample(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indexes = np.linspace(0, len(paths) - 1, num=count, dtype=int)
    return [paths[int(index)] for index in indexes]


def letterbox_rgb(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    old_h, old_w = image_rgb.shape[:2]
    scale = min(width / old_w, height / old_h)
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def analyze_onnx(onnx_path: Path) -> tuple[str, int, int]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"模型输入数量为 {len(inputs)}，要求为 1")
    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise RuntimeError(f"模型输入类型为 {model_input.type}，要求 tensor(float)")
    shape = model_input.shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise RuntimeError(f"模型输入必须是静态 NCHW [1,3,H,W]，当前为 {shape}")
    if not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise RuntimeError(f"模型 H/W 必须是固定整数，当前为 {shape}")
    return model_input.name, int(shape[3]), int(shape[2])


def prepare_calibration(
    images: list[Path],
    destination: Path,
    width: int,
    height: int,
    resize_mode: str,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, image_path in enumerate(images):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            LOGGER.warning("跳过无法读取的图片：%s", image_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if resize_mode == "letterbox":
            rgb = letterbox_rgb(rgb, width, height)
        else:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        output_path = destination / f"{index:04d}_{image_path.stem}.rgbchw"
        tensor.tofile(output_path)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--cal-images", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantized", choices=["int8", "int16"], default="int8")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--optimize-level", choices=["O0", "O1", "O2", "O3"], default="O3")
    parser.add_argument("--cal-sample-num", type=int, default=30)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="stretch")
    parser.add_argument("--skip-checker", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    cal_images_dir = Path(args.cal_images).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("hb_mapper") is None:
        raise RuntimeError("没有找到 hb_mapper，请先运行 setup_conda.sh")
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not cal_images_dir.is_dir():
        raise NotADirectoryError(cal_images_dir)

    input_name, width, height = analyze_onnx(onnx_path)
    images = sorted(
        path for path in cal_images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"校准目录中没有有效图片：{cal_images_dir}")
    images = evenly_sample(images, max(1, args.cal_sample_num))
    if len(images) < 20:
        LOGGER.warning("只有 %d 张校准图片，推荐至少 20 张", len(images))

    workspace = output_dir / ".mapper_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    cal_data_dir = workspace / "calibration_data"
    bpu_output_dir = workspace / "bpu_model_output"
    workspace.mkdir(parents=True, exist_ok=True)

    written = prepare_calibration(
        images=images,
        destination=cal_data_dir,
        width=width,
        height=height,
        resize_mode=args.resize_mode,
    )
    if written == 0:
        raise RuntimeError("没有成功生成任何校准数据")
    LOGGER.info("已生成 %d 份校准数据", written)

    model_name = onnx_path.stem
    output_prefix = f"{model_name}_bayese_{width}x{height}_nv12"
    optimization = "set_Softmax_input_int8,set_Softmax_output_int8"
    if args.quantized == "int16":
        optimization += ",set_all_nodes_int16"

    config = {
        "model_parameters": {
            "onnx_model": str(onnx_path),
            "march": "bayes-e",
            "layer_out_dump": False,
            "working_dir": str(bpu_output_dir),
            "output_model_file_prefix": output_prefix,
        },
        "input_parameters": {
            "input_name": "",
            "input_type_rt": "nv12",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "norm_type": "data_scale",
            "scale_value": 1.0 / 255.0,
        },
        "calibration_parameters": {
            "cal_data_dir": str(cal_data_dir),
            "cal_data_type": "float32",
            "calibration_type": "default",
            "optimization": optimization,
        },
        "compiler_parameters": {
            "jobs": args.jobs,
            "compile_mode": "latency",
            "debug": True,
            "optimize_level": args.optimize_level,
        },
    }

    config_path = workspace / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(config_path, output_dir / "config_used.yaml")
    LOGGER.info("输入名称：%s，输入尺寸：%dx%d", input_name, width, height)
    LOGGER.info("Mapper 配置：%s", config_path)

    success = False
    try:
        if not args.skip_checker:
            run_command(
                ["hb_mapper", "checker", "--model-type", "onnx", "--config", "config.yaml"],
                cwd=workspace,
                log_path=output_dir / "hb_mapper_checker_console.log",
            )
        run_command(
            ["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"],
            cwd=workspace,
            log_path=output_dir / "hb_mapper_makertbin_console.log",
        )

        expected_bin = bpu_output_dir / f"{output_prefix}.bin"
        if not expected_bin.is_file():
            candidates = sorted(bpu_output_dir.rglob("*.bin")) if bpu_output_dir.exists() else []
            if len(candidates) == 1:
                expected_bin = candidates[0]
            else:
                raise FileNotFoundError(
                    f"未找到唯一的 .bin 产物。候选文件：{[str(item) for item in candidates]}"
                )
        final_bin = output_dir / expected_bin.name
        shutil.copy2(expected_bin, final_bin)

        for log_file in workspace.rglob("*.log"):
            target = output_dir / log_file.name
            if not target.exists():
                shutil.copy2(log_file, target)

        LOGGER.info("转换成功：%s", final_bin)
        success = True
    finally:
        if success and not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        elif workspace.exists():
            LOGGER.info("中间工作目录保留：%s", workspace)


if __name__ == "__main__":
    main()
