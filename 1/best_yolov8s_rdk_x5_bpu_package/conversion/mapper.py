#!/usr/bin/env python3
"""Compile a BPU-friendly Ultralytics YOLOv8 ONNX model for RDK X5.

Run this only in an x86 Linux environment that provides D-Robotics hb_mapper,
for example the OpenExplorer toolchain or the rdkx5-yolo-mapper Python package.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
LOG = logging.getLogger('RDK-X5-MAPPER')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='BPU-friendly ONNX path')
    parser.add_argument('--cal-images', required=True, help='Calibration image directory')
    parser.add_argument('--output-dir', required=True, help='Directory for the final .bin')
    parser.add_argument('--workspace', default='.mapper_workspace', help='Temporary workspace')
    parser.add_argument('--jobs', type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument('--optimize-level', choices=['O0', 'O1', 'O2', 'O3'], default='O3')
    parser.add_argument('--quantized', choices=['int8', 'int16'], default='int8')
    parser.add_argument('--keep-workspace', action='store_true')
    return parser.parse_args()


def check_hb_mapper() -> None:
    try:
        result = subprocess.run(
            ['hb_mapper', '--version'],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            'hb_mapper is unavailable. Activate the OpenExplorer environment '
            'or install rdkx5-yolo-mapper first.'
        ) from exc
    LOG.info('hb_mapper: %s', result.stdout.strip().splitlines()[0] if result.stdout.strip() else 'available')


def inspect_onnx(onnx_path: Path) -> tuple[str, int, int]:
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f'Expected one ONNX input, found {len(inputs)}')
    inp = inputs[0]
    if inp.type != 'tensor(float)':
        raise RuntimeError(f'Expected float32 ONNX input, found {inp.type}')
    shape = inp.shape
    if len(shape) != 4 or not all(isinstance(v, int) for v in shape):
        raise RuntimeError(f'Expected fixed 4-D input shape, found {shape}')
    if shape[1] == 3:  # NCHW
        height, width = int(shape[2]), int(shape[3])
    elif shape[-1] == 3:  # NHWC, supported only for defensive inspection
        height, width = int(shape[1]), int(shape[2])
    else:
        raise RuntimeError(f'Cannot identify RGB channel in ONNX shape {shape}')

    expected_outputs = [
        (1, height // 8, width // 8, 4),
        (1, height // 8, width // 8, 64),
        (1, height // 16, width // 16, 4),
        (1, height // 16, width // 16, 64),
        (1, height // 32, width // 32, 4),
        (1, height // 32, width // 32, 64),
    ]
    actual_outputs = [tuple(o.shape) for o in session.get_outputs()]
    if actual_outputs != expected_outputs:
        raise RuntimeError(
            'Unexpected ONNX output protocol.\n'
            f'Actual:   {actual_outputs}\n'
            f'Expected: {expected_outputs}'
        )
    return inp.name, width, height


def list_images(folder: Path) -> list[Path]:
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    )
    if not images:
        raise RuntimeError(f'No calibration images found in {folder}')
    if len(images) < 20:
        LOG.warning('Only %d calibration images were found; 20-50 is recommended.', len(images))
    if len(images) > 50:
        LOG.warning('%d calibration images were found; conversion may take longer.', len(images))
    return images[:50]


def make_calibration_data(images: list[Path], cal_dir: Path, width: int, height: int) -> None:
    cal_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to load calibration image: {path}')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        tensor.tofile(cal_dir / f'{index:03d}_{path.stem}.rgbchw')


def write_config(
    path: Path,
    onnx_path: Path,
    output_prefix: str,
    working_dir: Path,
    cal_dir: Path,
    jobs: int,
    optimize_level: str,
    quantized: str,
) -> None:
    int16_opt = ',set_all_nodes_int16' if quantized == 'int16' else ''
    content = f"""model_parameters:
  onnx_model: '{onnx_path}'
  march: 'bayes-e'
  layer_out_dump: False
  working_dir: '{working_dir}'
  output_model_file_prefix: '{output_prefix}'
input_parameters:
  input_name: ''
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{cal_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_opt}
compiler_parameters:
  jobs: {jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{optimize_level}'
"""
    path.write_text(content, encoding='utf-8')


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    cal_images = Path(args.cal_images).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not cal_images.is_dir():
        raise NotADirectoryError(cal_images)

    check_hb_mapper()
    input_name, width, height = inspect_onnx(onnx_path)
    LOG.info('ONNX input: %s, 1x3x%dx%d', input_name, height, width)

    if workspace.exists():
        shutil.rmtree(workspace)
    cal_dir = workspace / 'calibration_data'
    bpu_output_dir = workspace / 'bpu_model_output'
    workspace.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(cal_images)
    LOG.info('Preparing %d calibration samples...', len(images))
    make_calibration_data(images, cal_dir, width, height)

    output_prefix = f'{onnx_path.stem}_bayese_{width}x{height}_nv12'
    config_path = workspace / 'config.yaml'
    write_config(
        config_path,
        onnx_path,
        output_prefix,
        bpu_output_dir,
        cal_dir,
        args.jobs,
        args.optimize_level,
        args.quantized,
    )

    LOG.info('Starting hb_mapper compilation...')
    subprocess.run(
        ['hb_mapper', 'makertbin', '--config', str(config_path), '--model-type', 'onnx'],
        cwd=workspace,
        check=True,
    )

    generated = bpu_output_dir / f'{output_prefix}.bin'
    if not generated.is_file():
        candidates = sorted(str(p) for p in bpu_output_dir.glob('*')) if bpu_output_dir.exists() else []
        raise RuntimeError(f'Expected output not found: {generated}; generated files: {candidates}')

    final_path = output_dir / generated.name
    shutil.copy2(generated, final_path)
    log_src = workspace / 'hb_mapper_makertbin.log'
    if log_src.exists():
        shutil.copy2(log_src, output_dir / log_src.name)

    LOG.info('Conversion completed: %s', final_path)
    if not args.keep_workspace:
        shutil.rmtree(workspace)


if __name__ == '__main__':
    main()
