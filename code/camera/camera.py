#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDK X5 USB camera + YOLOv8 BPU detection.

Behavior:
1. Display the complete camera frame.
2. Only crop the configured ROI and send that ROI to the BPU.
3. Letterbox the ROI to the model input size without distortion.
4. Decode YOLOv8 six-head outputs.
5. Undo letterbox scaling and map boxes back into the complete camera frame.

Keys:
    q / ESC : quit
    s       : save the current complete result frame
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import hbm_runtime
except ImportError as exc:
    raise SystemExit(
        "Cannot import hbm_runtime.\n"
        "Run this script with the RDK X5 system Python, normally:\n"
        "  /usr/bin/python3 camera_roi_bpu.py\n"
        f"Original error: {exc}"
    )


DEFAULT_MODEL = "/home/sunrise/code/camera/best_bayese_640x640_nv12.bin"
DEFAULT_LABELS = "/home/sunrise/code/camera/classes.names"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDK X5: full-frame display, ROI-only BPU detection"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RDK X5 .bin model")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="One class name per line")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, /dev/videoN")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")

    # Latest project ROI: vertical region x=180..1060, complete image height.
    parser.add_argument("--roi-x1", type=int, default=180)
    parser.add_argument("--roi-y1", type=int, default=0)
    parser.add_argument("--roi-x2", type=int, default=1060)
    parser.add_argument(
        "--roi-y2",
        type=int,
        default=-1,
        help="-1 means the bottom edge of the camera frame",
    )

    parser.add_argument("--score", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="Class-wise NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum boxes per frame")
    parser.add_argument("--bpu-core", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--hide-roi",
        action="store_true",
        help="Do not draw the ROI boundary on the displayed full frame",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="/home/sunrise/code/camera/snapshots",
        help="Directory used when the s key is pressed",
    )
    return parser.parse_args()


def load_labels(path: str, class_count: int) -> List[str]:
    label_path = Path(path)
    labels: List[str] = []

    if label_path.is_file():
        labels = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if len(labels) < class_count:
        labels.extend(f"class_{i}" for i in range(len(labels), class_count))

    return labels[:class_count]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def bgr_to_packed_nv12(image: np.ndarray) -> np.ndarray:
    """Convert an even-sized BGR image to one packed NV12 uint8 buffer."""
    height, width = image.shape[:2]
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(f"NV12 input width/height must be even, got {width}x{height}")

    area = width * height
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)

    y = yuv420p[:area]
    u = yuv420p[area : area + area // 4]
    v = yuv420p[area + area // 4 :]

    uv = np.empty(area // 2, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v

    return np.concatenate((y, uv)).astype(np.uint8, copy=False)


def letterbox(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize without distortion and pad to the model input size.

    Returns:
        padded image, scale, left padding, top padding
    """
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)

    new_w = max(2, int(round(src_w * scale)))
    new_h = max(2, int(round(src_h * scale)))

    # NV12 conversion needs even input dimensions; final target is normally 640x640.
    new_w = min(target_w, new_w)
    new_h = min(target_h, new_h)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(127, 127, 127),
    )
    return padded, scale, left, top


def make_anchors(grid_h: int, grid_w: int) -> np.ndarray:
    grid_y, grid_x = np.meshgrid(
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5), axis=1)


def nms_classwise(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int,
) -> np.ndarray:
    kept: List[int] = []

    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]

        while order.size > 0:
            current = int(order[0])
            kept.append(current)

            if len(kept) >= max_det or order.size == 1:
                break

            rest = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[current, 3], boxes[rest, 3])

            inter_w = np.maximum(0.0, x2 - x1)
            inter_h = np.maximum(0.0, y2 - y1)
            intersection = inter_w * inter_h

            area_current = max(
                0.0,
                (boxes[current, 2] - boxes[current, 0])
                * (boxes[current, 3] - boxes[current, 1]),
            )
            area_rest = np.maximum(
                0.0,
                (boxes[rest, 2] - boxes[rest, 0])
                * (boxes[rest, 3] - boxes[rest, 1]),
            )

            union = area_current + area_rest - intersection + 1e-9
            iou = intersection / union
            order = rest[iou < iou_threshold]

        if len(kept) >= max_det:
            break

    if not kept:
        return np.empty((0,), dtype=np.int32)

    kept_array = np.asarray(kept, dtype=np.int32)
    kept_array = kept_array[np.argsort(scores[kept_array])[::-1]]
    return kept_array[:max_det]


class YOLOv8BPU:
    """Standalone decoder for the RDK X5 Ultralytics YOLO six-output protocol."""

    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        max_det: int,
        bpu_core: int,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model does not exist: {model_file}")

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_det = int(max_det)
        self.strides = (8, 16, 32)

        print(f"[INFO] Loading model: {model_file}")
        load_start = time.perf_counter()
        self.runtime = hbm_runtime.HB_HBMRuntime(str(model_file))
        print(f"[INFO] Model loaded in {(time.perf_counter() - load_start) * 1000:.1f} ms")

        self.model_name = self.runtime.model_names[0]
        self.input_names = self.runtime.input_names[self.model_name]
        self.output_names = self.runtime.output_names[self.model_name]
        self.input_shapes: Dict[str, Tuple[int, ...]] = self.runtime.input_shapes[self.model_name]
        self.output_shapes: Dict[str, Tuple[int, ...]] = self.runtime.output_shapes[self.model_name]

        if len(self.output_names) != 6:
            raise RuntimeError(
                "This program expects the standard Ultralytics YOLO six-output model, "
                f"but the model reports {len(self.output_names)} outputs:\n"
                + "\n".join(self.output_names)
            )

        input_shape = tuple(self.input_shapes[self.input_names[0]])
        if len(input_shape) != 4:
            raise RuntimeError(f"Unexpected input shape: {input_shape}")

        # Official X5 models may report logical NCHW even though the input is packed NV12.
        if input_shape[1] == 3:
            self.input_h = int(input_shape[2])
            self.input_w = int(input_shape[3])
        else:
            self.input_h = int(input_shape[1])
            self.input_w = int(input_shape[2])

        first_cls_shape = tuple(self.output_shapes[self.output_names[0]])
        first_box_shape = tuple(self.output_shapes[self.output_names[1]])
        self.class_count = int(first_cls_shape[-1])
        self.reg = int(first_box_shape[-1]) // 4

        if self.class_count <= 0:
            raise RuntimeError(f"Cannot infer class count from output shape {first_cls_shape}")
        if self.reg <= 0 or int(first_box_shape[-1]) != self.reg * 4:
            raise RuntimeError(f"Cannot infer DFL reg count from output shape {first_box_shape}")

        # Model Zoo order is cls8, box8, cls16, box16, cls32, box32.
        for level, stride in enumerate(self.strides):
            cls_shape = tuple(self.output_shapes[self.output_names[level * 2]])
            box_shape = tuple(self.output_shapes[self.output_names[level * 2 + 1]])
            if int(cls_shape[-1]) != self.class_count:
                raise RuntimeError(
                    f"Class output channel mismatch at stride {stride}: {cls_shape}"
                )
            if int(box_shape[-1]) != self.reg * 4:
                raise RuntimeError(
                    f"Box output channel mismatch at stride {stride}: {box_shape}"
                )

        try:
            self.runtime.set_scheduling_params(
                priority={self.model_name: 0},
                bpu_cores={self.model_name: [bpu_core]},
            )
        except Exception as exc:
            print(f"[WARN] Could not set BPU core explicitly: {exc}")

        print(f"[INFO] Model name : {self.model_name}")
        print(f"[INFO] Input      : {self.input_names[0]} {input_shape}")
        print(f"[INFO] Input size : {self.input_w}x{self.input_h}")
        print(f"[INFO] Classes    : {self.class_count}")
        print(f"[INFO] DFL reg    : {self.reg}")
        for index, name in enumerate(self.output_names):
            print(f"[INFO] Output {index}   : {name} {tuple(self.output_shapes[name])}")

        score = min(max(self.score_threshold, 1e-6), 1.0 - 1e-6)
        self.raw_score_threshold = float(np.log(score / (1.0 - score)))
        self.dfl_weights = np.arange(self.reg, dtype=np.float32).reshape(1, 1, -1)

    def infer(
        self,
        roi_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Run detection on one ROI.

        Returned boxes are in ROI coordinates, not full-frame coordinates.
        """
        model_image, scale, pad_left, pad_top = letterbox(
            roi_bgr, self.input_w, self.input_h
        )
        packed_nv12 = bgr_to_packed_nv12(model_image)
        input_dict = {
            self.model_name: {
                self.input_names[0]: packed_nv12,
            }
        }

        infer_start = time.perf_counter()
        outputs = self.runtime.run(input_dict)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        raw = outputs[self.model_name]

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_class_ids: List[np.ndarray] = []

        for level, stride in enumerate(self.strides):
            cls_tensor = np.asarray(raw[self.output_names[level * 2]])
            box_tensor = np.asarray(raw[self.output_names[level * 2 + 1]])

            cls_shape = tuple(cls_tensor.shape)
            box_shape = tuple(box_tensor.shape)

            if cls_shape[-1] != self.class_count or box_shape[-1] != self.reg * 4:
                raise RuntimeError(
                    "Unexpected runtime output layout. Expected NHWC outputs.\n"
                    f"cls shape={cls_shape}, box shape={box_shape}"
                )

            grid_h = int(cls_shape[-3])
            grid_w = int(cls_shape[-2])

            cls_flat = cls_tensor.reshape(-1, self.class_count).astype(np.float32)
            max_logits = np.max(cls_flat, axis=1)
            valid = np.flatnonzero(max_logits >= self.raw_score_threshold)
            if valid.size == 0:
                continue

            class_ids = np.argmax(cls_flat[valid], axis=1).astype(np.int32)
            scores = sigmoid(max_logits[valid]).astype(np.float32)

            box_flat = box_tensor.reshape(-1, self.reg * 4).astype(np.float32)
            distributions = box_flat[valid].reshape(-1, 4, self.reg)
            distances = np.sum(
                softmax_last(distributions) * self.dfl_weights,
                axis=2,
            )

            anchors = make_anchors(grid_h, grid_w)[valid]
            xy1 = anchors - distances[:, 0:2]
            xy2 = anchors + distances[:, 2:4]
            boxes = np.concatenate((xy1, xy2), axis=1) * float(stride)

            all_boxes.append(boxes.astype(np.float32))
            all_scores.append(scores)
            all_class_ids.append(class_ids)

        if not all_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                infer_ms,
            )

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        keep = nms_classwise(
            boxes,
            scores,
            class_ids,
            self.nms_threshold,
            self.max_det,
        )
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Exact inverse of the ROI -> 640x640 letterbox transformation.
        roi_h, roi_w = roi_bgr.shape[:2]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(pad_left)) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(pad_top)) / scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, roi_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, roi_h - 1)

        valid_box = (
            (boxes[:, 2] - boxes[:, 0] >= 2)
            & (boxes[:, 3] - boxes[:, 1] >= 2)
        )
        return boxes[valid_box], scores[valid_box], class_ids[valid_box], infer_ms


def resolve_roi(
    frame_width: int,
    frame_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[int, int, int, int]:
    x2 = frame_width if x2 <= 0 else x2
    y2 = frame_height if y2 <= 0 else y2

    x1 = int(np.clip(x1, 0, max(0, frame_width - 2)))
    y1 = int(np.clip(y1, 0, max(0, frame_height - 2)))
    x2 = int(np.clip(x2, x1 + 2, frame_width))
    y2 = int(np.clip(y2, y1 + 2, frame_height))
    return x1, y1, x2, y2


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    # Deterministic OpenCV BGR color, without external dependencies.
    value = (class_id * 67 + 29) % 255
    return (
        int((value * 3 + 80) % 255),
        int((value * 7 + 140) % 255),
        int((value * 11 + 200) % 255),
    )


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    labels: List[str],
    roi_origin: Tuple[int, int],
) -> None:
    roi_x1, roi_y1 = roi_origin

    for box, score, class_id in zip(boxes, scores, class_ids):
        # Add ROI origin to restore coordinates into the complete camera frame.
        x1 = int(round(box[0] + roi_x1))
        y1 = int(round(box[1] + roi_y1))
        x2 = int(round(box[2] + roi_x1))
        y2 = int(round(box[3] + roi_y1))

        color = color_for_class(int(class_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        class_name = labels[int(class_id)] if int(class_id) < len(labels) else f"class_{class_id}"
        text = f"{class_name} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )

        text_y = max(text_h + baseline + 2, y1)
        cv2.rectangle(
            frame,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 6, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 3, text_y - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {index}. Check: ls -l /dev/video*"
        )

    # MJPG normally allows the USB camera to reach 1280x720@30 more reliably.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    args = parse_args()

    detector = YOLOv8BPU(
        model_path=args.model,
        score_threshold=args.score,
        nms_threshold=args.nms,
        max_det=args.max_det,
        bpu_core=args.bpu_core,
    )
    labels = load_labels(args.labels, detector.class_count)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

    print(
        f"[INFO] Camera /dev/video{args.camera}: "
        f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS, FOURCC={fourcc!r}"
    )
    print("[INFO] Complete frame is displayed; only the ROI is sent to the BPU.")
    print("[INFO] Keys: q/ESC quit, s save snapshot")

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("RDK X5 ROI BPU", cv2.WINDOW_NORMAL)

    smooth_fps = 0.0
    previous_time = time.perf_counter()
    roi_printed = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed")
                continue

            frame_h, frame_w = frame.shape[:2]
            roi_x1, roi_y1, roi_x2, roi_y2 = resolve_roi(
                frame_w,
                frame_h,
                args.roi_x1,
                args.roi_y1,
                args.roi_x2,
                args.roi_y2,
            )
            if not roi_printed:
                print(
                    f"[INFO] Effective ROI: "
                    f"x={roi_x1}:{roi_x2}, y={roi_y1}:{roi_y2}, "
                    f"size={roi_x2 - roi_x1}x{roi_y2 - roi_y1}"
                )
                roi_printed = True

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            boxes, scores, class_ids, infer_ms = detector.infer(roi)

            display = frame.copy()
            draw_detections(
                display,
                boxes,
                scores,
                class_ids,
                labels,
                (roi_x1, roi_y1),
            )

            if not args.hide_roi:
                cv2.rectangle(
                    display,
                    (roi_x1, roi_y1),
                    (roi_x2 - 1, roi_y2 - 1),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "BPU ROI",
                    (roi_x1 + 6, max(24, roi_y1 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            smooth_fps = instant_fps if smooth_fps == 0.0 else 0.9 * smooth_fps + 0.1 * instant_fps

            status = (
                f"FPS {smooth_fps:.1f} | BPU {infer_ms:.1f} ms | "
                f"DET {len(boxes)} | full {frame_w}x{frame_h}"
            )
            cv2.putText(
                display,
                status,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # This is the complete original camera frame with restored boxes.
            cv2.imshow("RDK X5 ROI BPU", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = snapshot_dir / f"result_{timestamp}.jpg"
                if cv2.imwrite(str(save_path), display):
                    print(f"[SAVED] {save_path}")
                else:
                    print(f"[WARN] Could not save {save_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
