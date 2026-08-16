#!/usr/bin/env python3
"""Standalone YOLOv8 detection on RDK X5 using hbm_runtime and packed NV12."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import hbm_runtime

CLASSES = ['zhituan', 'pinggai', 'dianchi', 'jiaodai']
STRIDES = [8, 16, 32]
REG_MAX = 16

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOG = logging.getLogger('RDK-X5-YOLOV8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='RDK X5 .bin model path')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', default='result.jpg', help='Output image path')
    parser.add_argument('--score', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.70)
    parser.add_argument('--resize-type', type=int, choices=[0, 1], default=1,
                        help='0: direct resize, 1: letterbox')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', type=int, nargs='+', default=[0])
    return parser.parse_args()


def resize_image(image: np.ndarray, width: int, height: int, resize_type: int) -> np.ndarray:
    if resize_type == 0:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = width - new_w, height - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(127, 127, 127),
    )


def bgr_to_packed_nv12(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError('NV12 input width and height must be even.')
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y = yuv420p[:area]
    u = yuv420p[area:area + area // 4]
    v = yuv420p[area + area // 4:]
    uv = np.empty(area // 2, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return np.concatenate((y, uv)).astype(np.uint8, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_anchors(grid_h: int, grid_w: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(grid_h, dtype=np.float32) + 0.5,
        np.arange(grid_w, dtype=np.float32) + 0.5,
        indexing='ij',
    )
    return np.stack((x, y), axis=-1).reshape(-1, 2)


def decode_level(
    cls_output: np.ndarray,
    box_output: np.ndarray,
    stride: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls = np.asarray(cls_output).reshape(-1, len(CLASSES)).astype(np.float32)
    raw_threshold = -np.log(1.0 / score_threshold - 1.0)
    max_logits = np.max(cls, axis=1)
    valid = np.flatnonzero(max_logits >= raw_threshold)
    if valid.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    class_ids = np.argmax(cls[valid], axis=1).astype(np.int32)
    scores = sigmoid(max_logits[valid]).astype(np.float32)

    shape = np.asarray(cls_output).shape
    if len(shape) != 4:
        raise RuntimeError(f'Unexpected classification output shape: {shape}')
    grid_h, grid_w = int(shape[1]), int(shape[2])

    boxes = np.asarray(box_output).reshape(-1, 4, REG_MAX).astype(np.float32)[valid]
    distribution = softmax(boxes, axis=2)
    weights = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX)
    ltrb = np.sum(distribution * weights, axis=2)

    anchors = make_anchors(grid_h, grid_w)[valid]
    x1y1 = anchors - ltrb[:, :2]
    x2y2 = anchors + ltrb[:, 2:]
    decoded = np.concatenate((x1y1, x2y2), axis=1) * float(stride)
    return decoded.astype(np.float32), scores, class_ids


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < threshold]
    return keep


def scale_boxes_back(
    boxes: np.ndarray,
    original_w: int,
    original_h: int,
    input_w: int,
    input_h: int,
    resize_type: int,
) -> np.ndarray:
    boxes = boxes.copy()
    if resize_type == 0:
        boxes[:, [0, 2]] *= original_w / input_w
        boxes[:, [1, 3]] *= original_h / input_h
    else:
        scale = min(input_w / original_w, input_h / original_h)
        pad_w = (input_w - original_w * scale) / 2.0
        pad_h = (input_h - original_h * scale) / 2.0
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
    return boxes


def postprocess(
    raw_outputs: dict[str, np.ndarray],
    output_names: list[str],
    original_w: int,
    original_h: int,
    input_w: int,
    input_h: int,
    score_threshold: float,
    nms_threshold: float,
    resize_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(output_names) != 6:
        raise RuntimeError(f'Expected six outputs [cls, box] x 3, found {len(output_names)}')

    all_boxes, all_scores, all_classes = [], [], []
    for level, stride in enumerate(STRIDES):
        cls_output = raw_outputs[output_names[level * 2]]
        box_output = raw_outputs[output_names[level * 2 + 1]]
        boxes, scores, class_ids = decode_level(cls_output, box_output, stride, score_threshold)
        if boxes.size:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(class_ids)

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_classes)

    kept_boxes, kept_scores, kept_classes = [], [], []
    for class_id in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == class_id)
        keep_local = nms(boxes[idx], scores[idx], nms_threshold)
        if not keep_local:
            continue
        selected = idx[np.asarray(keep_local, dtype=np.int64)]
        kept_boxes.append(boxes[selected])
        kept_scores.append(scores[selected])
        kept_classes.append(class_ids[selected])

    if not kept_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.concatenate(kept_boxes)
    scores = np.concatenate(kept_scores)
    class_ids = np.concatenate(kept_classes)
    boxes = scale_boxes_back(
        boxes, original_w, original_h, input_w, input_h, resize_type,
    )
    order = scores.argsort()[::-1]
    return boxes[order], scores[order], class_ids[order]


def draw_results(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        label = f'{CLASSES[int(class_id)]} {score:.2f}'
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(result, (x1, top), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(result, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Failed to load image: {image_path}')

    t0 = time.perf_counter()
    runtime = hbm_runtime.HB_HBMRuntime(str(model_path))
    model_name = runtime.model_names[0]
    input_name = runtime.input_names[model_name][0]
    output_names = list(runtime.output_names[model_name])
    input_shape = runtime.input_shapes[model_name][input_name]
    if input_shape[1] == 3:
        input_h, input_w = int(input_shape[2]), int(input_shape[3])
    else:
        input_h, input_w = int(input_shape[1]), int(input_shape[2])

    runtime.set_scheduling_params(
        priority={model_name: args.priority},
        bpu_cores={model_name: args.bpu_cores},
    )
    load_ms = (time.perf_counter() - t0) * 1000

    original_h, original_w = image.shape[:2]
    t1 = time.perf_counter()
    resized = resize_image(image, input_w, input_h, args.resize_type)
    packed = bgr_to_packed_nv12(resized)
    preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    outputs = runtime.run({model_name: {input_name: packed}})
    forward_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    boxes, scores, class_ids = postprocess(
        outputs[model_name], output_names,
        original_w, original_h, input_w, input_h,
        args.score, args.nms, args.resize_type,
    )
    post_ms = (time.perf_counter() - t3) * 1000

    result = draw_results(image, boxes, scores, class_ids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f'Failed to save result: {output_path}')

    LOG.info('model=%s, input=%dx%d, detections=%d', model_name, input_w, input_h, len(boxes))
    LOG.info('load=%.2f ms, preprocess=%.2f ms, BPU=%.2f ms, postprocess=%.2f ms',
             load_ms, preprocess_ms, forward_ms, post_ms)
    for box, score, class_id in zip(boxes, scores, class_ids):
        LOG.info('%s score=%.4f box=%s', CLASSES[int(class_id)], float(score),
                 np.rint(box).astype(int).tolist())
    LOG.info('saved: %s', output_path)


if __name__ == '__main__':
    main()
