#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK X5 YOLOv8 六输出 BPU 推理底层。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:
    import hbm_runtime
except ImportError as exc:
    raise RuntimeError(
        "没有找到 hbm_runtime，请在 RDK X5 系统 Python 中运行"
    ) from exc


@dataclass(frozen=True)
class LetterboxMeta:
    roi_w: int
    roi_h: int
    resized_w: int
    resized_h: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    input_w: int
    input_h: int


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    score: float
    class_id: int


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，不原地修改 BPU 输出数组。"""
    values = np.asarray(x, dtype=np.float32)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / np.maximum(denominator, 1e-12)


def classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    threshold: float,
) -> List[int]:
    """按类别执行 NMS，返回保留检测在原数组中的索引。"""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("NMS 阈值必须位于 [0, 1]")

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes 必须为 Nx4，当前 shape={boxes.shape}")
    if not (boxes.shape[0] == scores.size == class_ids.size):
        raise ValueError("boxes、scores、class_ids 数量不一致")

    keep: List[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        cls_boxes = boxes[indices]
        cls_scores = scores[indices]

        x1, y1, x2, y2 = (cls_boxes[:, i] for i in range(4))
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(cls_scores)[::-1]

        while order.size > 0:
            current = int(order[0])
            keep.append(int(indices[current]))
            if order.size == 1:
                break

            remain = order[1:]
            xx1 = np.maximum(x1[current], x1[remain])
            yy1 = np.maximum(y1[current], y1[remain])
            xx2 = np.minimum(x2[current], x2[remain])
            yy2 = np.minimum(y2[current], y2[remain])
            intersection = (
                np.maximum(0.0, xx2 - xx1)
                * np.maximum(0.0, yy2 - yy1)
            )
            union = areas[current] + areas[remain] - intersection
            iou = intersection / np.maximum(union, 1e-9)
            order = remain[iou < threshold]

    return keep


def letterbox(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    fill_value: int = 0,
) -> Tuple[np.ndarray, LetterboxMeta]:
    """等比例缩放并填充到模型输入尺寸。"""
    if not isinstance(image, np.ndarray):
        raise TypeError("image 必须是 numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image 必须是 BGR 三通道图像，当前 {image.shape}")

    src_h, src_w = image.shape[:2]
    target_w = int(target_w)
    target_h = int(target_h)
    fill_value = int(fill_value)

    if src_w <= 0 or src_h <= 0:
        raise ValueError("输入图像尺寸无效")
    if target_w <= 0 or target_h <= 0:
        raise ValueError("目标尺寸必须大于 0")
    if not 0 <= fill_value <= 255:
        raise ValueError("letterbox 填充值必须位于 [0, 255]")

    ratio = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, min(target_w, int(round(src_w * ratio))))
    resized_h = max(1, min(target_h, int(round(src_h * ratio))))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=interpolation,
    )

    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    output = np.full(
        (target_h, target_w, 3),
        fill_value,
        dtype=np.uint8,
    )
    output[
        pad_top:pad_top + resized_h,
        pad_left:pad_left + resized_w,
    ] = resized

    return output, LetterboxMeta(
        roi_w=src_w,
        roi_h=src_h,
        resized_w=resized_w,
        resized_h=resized_h,
        scale_x=resized_w / src_w,
        scale_y=resized_h / src_h,
        pad_left=pad_left,
        pad_top=pad_top,
        input_w=target_w,
        input_h=target_h,
    )


def bgr_to_packed_nv12(image: np.ndarray) -> np.ndarray:
    """BGR 转换为连续的一维 NV12 数据。"""
    if not isinstance(image, np.ndarray):
        raise TypeError("image 必须是 numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"NV12 输入必须是 BGR 三通道图像：{image.shape}")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("NV12 输入图像尺寸无效")
    if height % 2 or width % 2:
        raise ValueError(f"NV12 要求偶数宽高，当前 {width}x{height}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    area = height * width
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_plane = i420[:area]
    u_plane = i420[area:area + area // 4]
    v_plane = i420[area + area // 4:]

    uv_plane = np.empty(area // 2, dtype=np.uint8)
    uv_plane[0::2] = u_plane
    uv_plane[1::2] = v_plane
    return np.ascontiguousarray(
        np.concatenate((y_plane, uv_plane)),
        dtype=np.uint8,
    )


def map_box_to_full_frame(
    model_box: Sequence[float],
    meta: LetterboxMeta,
    roi_x1: int,
    roi_y1: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """把模型输入坐标还原到完整摄像头画面坐标。"""
    if len(model_box) != 4:
        raise ValueError("model_box 必须包含 x1,y1,x2,y2")
    if meta.scale_x <= 0 or meta.scale_y <= 0:
        raise ValueError("letterbox 缩放比例无效")
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("完整画面尺寸无效")

    x1, y1, x2, y2 = map(float, model_box)
    x1 = (x1 - meta.pad_left) / meta.scale_x
    x2 = (x2 - meta.pad_left) / meta.scale_x
    y1 = (y1 - meta.pad_top) / meta.scale_y
    y2 = (y2 - meta.pad_top) / meta.scale_y

    x1 = float(np.clip(x1, 0, meta.roi_w - 1))
    x2 = float(np.clip(x2, 0, meta.roi_w - 1))
    y1 = float(np.clip(y1, 0, meta.roi_h - 1))
    y2 = float(np.clip(y2, 0, meta.roi_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    x1 = float(np.clip(x1 + roi_x1, 0, frame_w - 1))
    x2 = float(np.clip(x2 + roi_x1, 0, frame_w - 1))
    y1 = float(np.clip(y1 + roi_y1, 0, frame_h - 1))
    y2 = float(np.clip(y2 + roi_y1, 0, frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


class YOLOv8BPU:
    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        bpu_core: int = 0,
        priority: int = 0,
        strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"没有找到模型：{model_path}")

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.strides = tuple(int(value) for value in strides)
        self.reg_max = int(reg_max)

        if not 0.0 < self.score_threshold < 1.0:
            raise ValueError("score_threshold 必须位于 (0, 1)")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold 必须位于 [0, 1]")
        if not self.strides or any(value <= 0 for value in self.strides):
            raise ValueError("strides 必须是非空正整数序列")
        if self.reg_max <= 0:
            raise ValueError("reg_max 必须大于 0")

        self.raw_score_threshold = float(
            np.log(self.score_threshold / (1.0 - self.score_threshold))
        )
        self.dfl_weights = np.arange(self.reg_max, dtype=np.float32)

        print(f"[VISION] 加载模型：{model_path}", flush=True)
        started = time.perf_counter()
        self.runtime = hbm_runtime.HB_HBMRuntime(model_path)

        model_names = list(self.runtime.model_names)
        if not model_names:
            raise RuntimeError("BPU 模型中没有可用 graph")
        self.model_name = model_names[0]

        self.input_names = list(self.runtime.input_names[self.model_name])
        self.output_names = list(self.runtime.output_names[self.model_name])
        if len(self.input_names) != 1:
            raise RuntimeError(
                f"当前仅支持单输入模型，实际输入：{self.input_names}"
            )

        input_shape = list(
            self.runtime.input_shapes[self.model_name][self.input_names[0]]
        )
        if len(input_shape) != 4:
            raise RuntimeError(f"不支持的输入 shape：{input_shape}")

        if int(input_shape[1]) == 3:  # NCHW
            self.input_h = int(input_shape[2])
            self.input_w = int(input_shape[3])
        elif int(input_shape[3]) == 3:  # NHWC
            self.input_h = int(input_shape[1])
            self.input_w = int(input_shape[2])
        else:
            # 部分 NV12 模型的 shape 元数据不是标准 3 通道布局；
            # 沿用原项目可工作的 H/W 解析方式。
            self.input_h = int(input_shape[1])
            self.input_w = int(input_shape[2])

        if self.input_w <= 0 or self.input_h <= 0:
            raise RuntimeError(f"模型输入尺寸无效：{input_shape}")
        if self.input_w % 2 or self.input_h % 2:
            raise RuntimeError(
                f"NV12 模型输入宽高必须为偶数：{self.input_w}x{self.input_h}"
            )

        expected_outputs = len(self.strides) * 2
        if len(self.output_names) != expected_outputs:
            raise RuntimeError(
                f"需要 YOLOv8 {expected_outputs} 输出模型，"
                f"实际输出：{self.output_names}"
            )

        self.output_pairs = self._resolve_output_pairs()
        try:
            self.runtime.set_scheduling_params(
                priority={self.model_name: int(priority)},
                bpu_cores={self.model_name: [int(bpu_core)]},
            )
        except Exception as exc:
            print(f"[VISION] BPU 调度参数未设置：{exc}", flush=True)

        self.actual_classes: int | None = None
        print(
            f"[VISION] 模型输入：{self.input_w}x{self.input_h}，"
            f"输出={len(self.output_names)}",
            flush=True,
        )
        print(
            f"[VISION] 模型就绪，用时 "
            f"{(time.perf_counter() - started) * 1000:.1f}ms",
            flush=True,
        )

    def _resolve_output_pairs(self) -> List[Tuple[int, str, str]]:
        pairs: List[Tuple[int, str, str]] = []
        lower_names = {name.lower(): name for name in self.output_names}

        for level, stride in enumerate(self.strides):
            cls_name = next(
                (
                    original
                    for lower, original in lower_names.items()
                    if f"s{stride}_cls" in lower
                ),
                None,
            )
            box_name = next(
                (
                    original
                    for lower, original in lower_names.items()
                    if f"s{stride}_box" in lower
                ),
                None,
            )

            if cls_name is None or box_name is None:
                start = level * 2
                cls_name, box_name = self.output_names[start:start + 2]
            pairs.append((stride, cls_name, box_name))

        used_names = [name for _, cls, box in pairs for name in (cls, box)]
        if len(set(used_names)) != len(used_names):
            raise RuntimeError(f"模型输出配对重复：{pairs}")
        return pairs

    @staticmethod
    def _flatten(
        tensor: np.ndarray,
        grid_h: int,
        grid_w: int,
        expected_channels: int | None = None,
    ) -> np.ndarray:
        arr = np.asarray(tensor)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 3 and arr.shape[:2] == (grid_h, grid_w):
            result = arr.reshape(-1, arr.shape[2])
        elif arr.ndim == 3 and arr.shape[1:] == (grid_h, grid_w):
            result = np.transpose(arr, (1, 2, 0)).reshape(-1, arr.shape[0])
        else:
            grid_count = grid_h * grid_w
            if grid_count <= 0 or arr.size % grid_count:
                raise RuntimeError(f"无法解析输出 shape：{arr.shape}")
            result = arr.reshape(grid_count, arr.size // grid_count)

        if (
            expected_channels is not None
            and result.shape[1] != expected_channels
        ):
            raise RuntimeError(
                f"输出通道={result.shape[1]}，预期={expected_channels}"
            )
        return np.ascontiguousarray(result, dtype=np.float32)

    @staticmethod
    def _anchors(grid_h: int, grid_w: int) -> np.ndarray:
        y, x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32) + 0.5,
            np.arange(grid_w, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        return np.stack((x, y), axis=-1).reshape(-1, 2)

    def _decode(
        self,
        cls_tensor: np.ndarray,
        box_tensor: np.ndarray,
        stride: int,
    ) -> List[Detection]:
        grid_h = self.input_h // stride
        grid_w = self.input_w // stride
        if grid_h <= 0 or grid_w <= 0:
            raise RuntimeError(f"stride={stride} 与模型输入尺寸不匹配")

        cls_output = self._flatten(cls_tensor, grid_h, grid_w)
        box_output = self._flatten(
            box_tensor,
            grid_h,
            grid_w,
            4 * self.reg_max,
        )

        classes_num = int(cls_output.shape[1])
        if classes_num <= 0:
            raise RuntimeError("分类输出通道数无效")
        if self.actual_classes is None:
            self.actual_classes = classes_num
            print(f"[VISION] 模型类别数：{classes_num}", flush=True)
        elif classes_num != self.actual_classes:
            raise RuntimeError(
                "不同输出层的类别数不一致："
                f"{self.actual_classes} 与 {classes_num}"
            )

        max_logits = np.max(cls_output, axis=1)
        valid = np.flatnonzero(max_logits >= self.raw_score_threshold)
        if valid.size == 0:
            return []

        class_ids = np.argmax(cls_output[valid], axis=1).astype(np.int32)
        scores = sigmoid(max_logits[valid]).astype(np.float32)
        selected = box_output[valid].reshape(-1, 4, self.reg_max)
        probabilities = softmax(selected, axis=2)
        distances = np.sum(
            probabilities * self.dfl_weights.reshape(1, 1, -1),
            axis=2,
        )
        anchors = self._anchors(grid_h, grid_w)[valid]
        boxes = np.concatenate(
            (
                anchors - distances[:, :2],
                anchors + distances[:, 2:],
            ),
            axis=1,
        ) * float(stride)

        return [
            Detection(
                box=boxes[index],
                score=float(scores[index]),
                class_id=int(class_ids[index]),
            )
            for index in range(boxes.shape[0])
        ]

    def infer(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        if image.shape[:2] != (self.input_h, self.input_w):
            raise ValueError(
                "模型输入图像尺寸错误："
                f"当前 {image.shape[1]}x{image.shape[0]}，"
                f"需要 {self.input_w}x{self.input_h}"
            )

        packed = bgr_to_packed_nv12(image)
        started = time.perf_counter()
        outputs = self.runtime.run(
            {self.model_name: {self.input_names[0]: packed}}
        )
        infer_ms = (time.perf_counter() - started) * 1000.0

        if self.model_name not in outputs:
            raise RuntimeError(
                f"BPU 输出中缺少模型 {self.model_name}"
            )
        raw = outputs[self.model_name]

        detections: List[Detection] = []
        for stride, cls_name, box_name in self.output_pairs:
            if cls_name not in raw or box_name not in raw:
                raise RuntimeError(
                    f"BPU 输出缺少 {cls_name} 或 {box_name}"
                )
            detections.extend(
                self._decode(raw[cls_name], raw[box_name], stride)
            )

        if not detections:
            return [], infer_ms

        boxes = np.stack([item.box for item in detections]).astype(np.float32)
        scores = np.asarray(
            [item.score for item in detections],
            dtype=np.float32,
        )
        classes = np.asarray(
            [item.class_id for item in detections],
            dtype=np.int32,
        )
        keep = classwise_nms(
            boxes,
            scores,
            classes,
            self.nms_threshold,
        )
        final = [detections[index] for index in keep]
        final.sort(key=lambda item: item.score, reverse=True)
        return final, infer_ms


def _fourcc_text(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    text = "".join(chars).strip("\x00")
    return text or "UNKNOWN"


class LatestFrameCamera:
    """后台持续采集，只保留最新一帧，防止摄像头缓存堆积。"""

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
        use_mjpg: bool = True,
        failure_timeout_s: float = 3.0,
    ) -> None:
        self.index = int(index)
        self.failure_timeout_s = max(0.5, float(failure_timeout_s))
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 /dev/video{self.index}")

        if use_mjpg:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        actual_fourcc = _fourcc_text(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(
            "[VISION] 摄像头请求："
            f"{int(width)}x{int(height)}@{int(fps)} "
            f"{'MJPG' if use_mjpg else '默认格式'}",
            flush=True,
        )
        print(
            "[VISION] 摄像头实际："
            f"{actual_width}x{actual_height}@{actual_fps:.1f} "
            f"FOURCC={actual_fourcc}",
            flush=True,
        )

        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.running = True
        self.last_error = ""
        self.thread = threading.Thread(
            target=self._loop,
            name="camera-latest",
            daemon=True,
        )
        self.thread.start()

    def _loop(self) -> None:
        failure_started: float | None = None
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if failure_started is None:
                    failure_started = time.monotonic()
                elif time.monotonic() - failure_started >= self.failure_timeout_s:
                    self.last_error = (
                        f"摄像头连续读取失败超过 {self.failure_timeout_s:.1f}s"
                    )
                    self.running = False
                    break
                time.sleep(0.02)
                continue

            failure_started = None
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read_latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def read_latest_with_id(self) -> tuple[int, np.ndarray | None]:
        """可选接口；原有 read_latest() 行为保持不变。"""
        with self.lock:
            return (
                self.frame_id,
                None if self.frame is None else self.frame.copy(),
            )

    def release(self) -> None:
        self.running = False
        try:
            self.cap.release()
        finally:
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
