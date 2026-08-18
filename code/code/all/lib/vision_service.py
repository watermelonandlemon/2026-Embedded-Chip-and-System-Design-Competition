#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉节点：BPU 检测、区域判定、消息发布和 UI 预览帧输出。

职责：
1. 从摄像头读取最新帧，仅对 ROI 区域执行 BPU 推理；
2. 将 YOLO 类别映射为全局垃圾类型 a；
3. 在完整原图坐标系中绘制四个触发区域和全部 YOLO 检测框；
4. 只有垃圾类别与所在区域匹配时，才向 GPIO 节点发布真实 a；
5. 不匹配、未到达目标区域或没有目标时发布 a=-1；
6. 可回收垃圾进入第三区域后，从 ROI 右边界离开时判定进入第四区域；
7. 将完整预览画面原子写入 latest_frame，供独立 UI 节点读取。

区域默认定义（完整原图坐标）：
- 区域 1：x=100~500，其他垃圾；
- 区域 2：x=500~850，有害垃圾；
- 区域 3：x=850~1100，厨余垃圾；
- 区域 4：x>1100，可回收垃圾。

本模块不直接创建 OpenCV 窗口，也不读取或修改 count.yaml。
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2
import numpy as np

from core import (
    DEFAULT_CONFIG_PATH,
    create_node,
    garbage_maps,
    load_config,
    resolve_path,
    yolo_to_garbage,
)
from vision_bpu import (
    LatestFrameCamera,
    YOLOv8BPU,
    letterbox,
    map_box_to_full_frame,
)


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """在检测框上方绘制英文类别与置信度。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        0.55,
        1,
    )

    text_x = max(0, int(x))
    text_y = max(text_h + baseline + 3, int(y))
    right = min(image.shape[1] - 1, text_x + text_w + 7)
    top = max(0, text_y - text_h - baseline - 5)
    bottom = min(image.shape[0] - 1, text_y + 3)

    cv2.rectangle(
        image,
        (text_x, top),
        (right, bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (text_x + 3, text_y - baseline - 1),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_region(
    image: np.ndarray,
    region_id: int,
    name: str,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    """在完整原图上绘制一个竖向触发区域。"""
    frame_h, frame_w = image.shape[:2]
    left = max(0, min(frame_w - 1, int(x1)))
    right = max(left + 1, min(frame_w, int(x2)))
    top = max(0, min(frame_h - 1, int(y1)))
    bottom = max(top + 1, min(frame_h, int(y2)))

    cv2.rectangle(
        image,
        (left, top),
        (right - 1, bottom - 1),
        color,
        2,
    )
    _draw_label(
        image,
        f"R{region_id} {name}",
        left + 4,
        top + 24,
        color,
    )


def _color(class_id: int) -> tuple[int, int, int]:
    """为不同 YOLO 类别返回固定 BGR 颜色。"""
    colors = (
        (56, 56, 255),
        (49, 210, 207),
        (10, 249, 72),
        (255, 194, 0),
    )
    return colors[int(class_id) % len(colors)]


def _atomic_write_frame(path: Path, frame: np.ndarray) -> None:
    """先写临时图片，再原子替换，避免 UI 读到半张图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(
            f"latest_frame 仅支持 .png/.jpg/.jpeg，当前为：{path}"
        )

    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    )

    if suffix == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 2]
    else:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 88]

    try:
        success = cv2.imwrite(
            str(temporary),
            frame,
            encode_params,
        )
        if not success:
            raise RuntimeError(f"无法写入预览帧：{temporary}")

        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_labels(
    labels_path: Path,
    yolo_map: Dict[int, Dict[str, Any]],
) -> None:
    """检查 classes.names 顺序是否与 config.yaml 一致。"""
    if not labels_path.is_file():
        print(
            f"[VISION] 警告：未找到 labels 文件 {labels_path}",
            flush=True,
        )
        return

    labels = [
        line.strip()
        for line in labels_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    expected = [
        yolo_map[index]["name"]
        for index in sorted(yolo_map)
    ]

    if labels[: len(expected)] != expected:
        print(
            "[VISION] 警告：labels 顺序 "
            f"{labels[:len(expected)]} 与配置 {expected} 不一致",
            flush=True,
        )


def _field_to_a(
    by_a: Mapping[int, Mapping[str, Any]],
) -> Dict[str, int]:
    """建立垃圾字段名到全局 a 的映射。"""
    result: Dict[str, int] = {}
    for a, item in by_a.items():
        result[str(item["field"])] = int(a)
    required = {"other", "hazardous", "kitchen", "recyclable"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"garbage 配置缺少字段：{sorted(missing)}")
    return result


def _region_for_x(
    center_x: int,
    region1_x1: int,
    region1_x2: int,
    region2_x2: int,
    region3_x2: int,
) -> int:
    """根据完整原图中心点 x 返回区域编号；区域外返回 0。"""
    x = int(center_x)
    if region1_x1 <= x < region1_x2:
        return 1
    if region1_x2 <= x < region2_x2:
        return 2
    if region2_x2 <= x < region3_x2:
        return 3
    if x >= region3_x2:
        return 4
    return 0


def run_vision_node(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    """运行视觉节点。"""
    config = load_config(config_path)
    vision = config["vision"]
    topics = config["miniros"]["topics"]
    yolo_map = yolo_to_garbage(config)
    by_a, _ = garbage_maps(config)
    field_to_a = _field_to_a(by_a)

    labels_path = resolve_path(
        vision.get("labels_path", "")
    )
    _validate_labels(labels_path, yolo_map)

    model = YOLOv8BPU(
        model_path=str(resolve_path(vision["model_path"])),
        score_threshold=float(
            vision.get("score_threshold", 0.35)
        ),
        nms_threshold=float(
            vision.get("nms_threshold", 0.45)
        ),
        bpu_core=int(vision.get("bpu_core", 0)),
        priority=int(vision.get("bpu_priority", 0)),
    )

    camera = LatestFrameCamera(
        int(vision.get("camera_index", 0)),
        int(vision.get("width", 1280)),
        int(vision.get("height", 720)),
        int(vision.get("fps", 30)),
        bool(vision.get("use_mjpg", True)),
    )

    node = create_node("node_vision", config)
    vision_topic = node.register_publisher(
        topics["vision_result"]
    )

    # 原有的 3 秒总限频继续保留，避免 GPIO 在短时间内重复动作。
    target_cooldown = max(
        3.0,
        float(vision.get("target_cooldown_s", 3.0)),
    )
    no_action_interval = max(
        0.2,
        float(vision.get("no_target_publish_s", 1.0)),
    )
    preview_interval = max(
        0.1,
        float(vision.get("preview_interval_ms", 200))
        / 1000.0,
    )
    fill_value = int(vision.get("letterbox_value", 0))

    # 可选配置；config.yaml 不增加这些字段时，直接使用用户指定的坐标。
    region_cfg = vision.get("trigger_regions", {})
    if not isinstance(region_cfg, Mapping):
        raise ValueError("vision.trigger_regions 必须是映射")

    region1_x1 = int(region_cfg.get("region1_x1", 400))
    region1_x2 = int(region_cfg.get("region1_x2", 700))
    region2_x2 = int(region_cfg.get("region2_x2", 950))
    region3_x2 = int(region_cfg.get("region3_x2", 1100))

    if not (
        region1_x1 < region1_x2 < region2_x2 < region3_x2
    ):
        raise ValueError(
            "触发区域坐标必须满足："
            "region1_x1 < region1_x2 < region2_x2 < region3_x2"
        )

    # 目标离开指定区域后经过该时间，才允许下一件垃圾重新触发。
    region_rearm_s = max(
        0.1,
        float(region_cfg.get("rearm_s", 0.5)),
    )

    # 可回收目标从第三区域消失后，等待数帧确认不是偶发漏检。
    recyclable_exit_confirm_s = max(
        0.1,
        float(region_cfg.get("recyclable_exit_confirm_s", 0.35)),
    )

    expected_a_by_region = {
        1: field_to_a["other"],
        2: field_to_a["hazardous"],
        3: field_to_a["kitchen"],
        4: field_to_a["recyclable"],
    }

    region_names = {
        1: "OTHER",
        2: "HAZARDOUS",
        3: "KITCHEN",
        4: "RECYCLABLE",
    }
    region_colors = {
        1: (160, 160, 160),
        2: (70, 70, 255),
        3: (0, 190, 255),
        4: (70, 220, 120),
    }

    latest_frame_value = config.get("paths", {}).get(
        "latest_frame",
        "/dev/shm/rdk_smartbin_latest.png",
    )
    latest_frame_path = resolve_path(latest_frame_value)

    last_target_sent = -1e9
    last_no_action_sent = -1e9
    last_preview_saved = -1e9
    stop_requested = False

    # 区域 1~3 使用进入区域触发；目标停留在区域内不会反复发送。
    region_latched = {1: False, 2: False, 3: False}
    region_last_seen = {1: -1e9, 2: -1e9, 3: -1e9}

    # 第四区域位于推理 ROI 之外：先在第三区域看到可回收垃圾，
    # 随后其从右侧离开并连续消失一段时间，才判定进入第四区域。
    recyclable_armed = False
    recyclable_last_seen_region3 = -1e9
    recyclable_last_candidate: Dict[str, Any] | None = None
    recyclable_exit_pending = False

    def shutdown(*_: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(
        f"[VISION] UI预览帧：{latest_frame_path}，"
        f"刷新间隔={preview_interval:.3f}s",
        flush=True,
    )
    print(
        "[VISION] 触发区域："
        f"R1={region1_x1}~{region1_x2} 其他，"
        f"R2={region1_x2}~{region2_x2} 有害，"
        f"R3={region2_x2}~{region3_x2} 厨余，"
        f"R4=>={region3_x2} 可回收",
        flush=True,
    )
    print("[VISION] 等待摄像头首帧", flush=True)

    frame = None
    for _ in range(300):
        frame = camera.read_latest()
        if frame is not None:
            break
        if not camera.running:
            break
        time.sleep(0.01)

    if frame is None:
        camera.release()
        node.close()
        raise RuntimeError("摄像头已打开但没有读取到图像")

    try:
        while not stop_requested and camera.running:
            frame = camera.read_latest()
            if frame is None:
                time.sleep(0.002)
                continue

            frame_h, frame_w = frame.shape[:2]
            roi_cfg = vision["roi"]

            x1 = max(
                0,
                min(frame_w - 1, int(roi_cfg["x1"])),
            )
            y1 = max(
                0,
                min(frame_h - 1, int(roi_cfg["y1"])),
            )
            x2 = max(
                x1 + 1,
                min(frame_w, int(roi_cfg["x2"])),
            )
            y2 = max(
                y1 + 1,
                min(frame_h, int(roi_cfg["y2"])),
            )

            roi = frame[y1:y2, x1:x2]
            model_input, meta = letterbox(
                roi,
                model.input_w,
                model.input_h,
                fill_value,
            )
            detections, _infer_ms = model.infer(model_input)

            # 全部检测框始终绘制；是否触发 GPIO 与是否绘框完全分离。
            display = frame.copy()

            # R1~R3 位于 ROI 内；R4 从 x=1100 延伸到完整画面右边界。
            _draw_region(
                display,
                1,
                region_names[1],
                region1_x1,
                region1_x2,
                y1,
                y2,
                region_colors[1],
            )
            _draw_region(
                display,
                2,
                region_names[2],
                region1_x2,
                region2_x2,
                y1,
                y2,
                region_colors[2],
            )
            _draw_region(
                display,
                3,
                region_names[3],
                region2_x2,
                region3_x2,
                y1,
                y2,
                region_colors[3],
            )
            if frame_w > region3_x2:
                _draw_region(
                    display,
                    4,
                    region_names[4],
                    region3_x2,
                    frame_w,
                    y1,
                    y2,
                    region_colors[4],
                )

            best_visible: Dict[str, Any] | None = None
            best_match_by_region: Dict[int, Dict[str, Any]] = {}
            best_recyclable: Dict[str, Any] | None = None

            for detection in detections:
                full_box = map_box_to_full_frame(
                    detection.box,
                    meta,
                    x1,
                    y1,
                    frame_w,
                    frame_h,
                )
                if full_box is None:
                    continue

                bx1, by1, bx2, by2 = np.rint(
                    full_box
                ).astype(np.int32)

                class_info = yolo_map.get(
                    int(detection.class_id)
                )
                class_name = (
                    str(class_info["name"])
                    if class_info is not None
                    else f"class_{detection.class_id}"
                )
                color = _color(int(detection.class_id))

                # 即使类别和当前区域不匹配，YOLO 框也照常绘制。
                cv2.rectangle(
                    display,
                    (int(bx1), int(by1)),
                    (int(bx2), int(by2)),
                    color,
                    2,
                )
                _draw_label(
                    display,
                    f"{class_name} {detection.score:.2f}",
                    int(bx1),
                    max(2, int(by1)),
                    color,
                )

                # 未在 config.yaml 中映射的类别只显示，不参与控制。
                if class_info is None:
                    continue

                center_x = int(round((bx1 + bx2) / 2))
                center_y = int(round((by1 + by2) / 2))
                region_id = _region_for_x(
                    center_x,
                    region1_x1,
                    region1_x2,
                    region2_x2,
                    region3_x2,
                )

                candidate: Dict[str, Any] = {
                    "a": int(class_info["a"]),
                    "x": center_x,
                    "y": center_y,
                    "score": float(detection.score),
                    "yolo_class_id": int(detection.class_id),
                    "yolo_class_name": class_name,
                    "region_id": region_id,
                    "region_name": region_names.get(region_id, "OUTSIDE"),
                    "box": [
                        int(bx1),
                        int(by1),
                        int(bx2),
                        int(by2),
                    ],
                }

                if (
                    best_visible is None
                    or candidate["score"] > best_visible["score"]
                ):
                    best_visible = candidate

                if candidate["a"] == expected_a_by_region[4]:
                    if (
                        best_recyclable is None
                        or candidate["score"] > best_recyclable["score"]
                    ):
                        best_recyclable = candidate

                # 区域 1~3：类别与区域完全匹配才成为 GPIO 候选。
                if region_id in (1, 2, 3):
                    if candidate["a"] == expected_a_by_region[region_id]:
                        previous = best_match_by_region.get(region_id)
                        if (
                            previous is None
                            or candidate["score"] > previous["score"]
                        ):
                            best_match_by_region[region_id] = candidate

            now = time.monotonic()

            # 更新区域 1~3 的占用状态；只有从“未占用”进入“占用”才触发一次。
            for region_id in (1, 2, 3):
                candidate = best_match_by_region.get(region_id)
                if candidate is not None:
                    region_last_seen[region_id] = now
                elif (
                    region_latched[region_id]
                    and now - region_last_seen[region_id] >= region_rearm_s
                ):
                    region_latched[region_id] = False

            # 可回收垃圾的第四区域判定。
            if best_recyclable is not None:
                recyclable_region = int(best_recyclable["region_id"])
                if recyclable_region == 3:
                    recyclable_armed = True
                    recyclable_exit_pending = False
                    recyclable_last_seen_region3 = now
                    recyclable_last_candidate = dict(best_recyclable)
                elif recyclable_armed and recyclable_region in (1, 2):
                    # 目标向左返回，不认为进入第四区域。
                    recyclable_armed = False
                    recyclable_exit_pending = False
                    recyclable_last_candidate = None
                elif recyclable_region == 4:
                    # 兼容未来扩大推理范围后直接检测到第四区域的情况。
                    recyclable_exit_pending = True
                    recyclable_armed = False
                    recyclable_last_candidate = dict(best_recyclable)
            elif (
                recyclable_armed
                and recyclable_last_candidate is not None
                and now - recyclable_last_seen_region3
                >= recyclable_exit_confirm_s
            ):
                # ROI 只推理到 x=1100。可回收垃圾在 R3 出现后从右侧消失，
                # 连续消失达到确认时间，即认为中心已进入 R4。
                recyclable_exit_pending = True
                recyclable_armed = False

            action: Dict[str, Any] | None = None

            # 优先处理当前帧内区域 1~3 的有效进入事件。
            for region_id in (1, 2, 3):
                candidate = best_match_by_region.get(region_id)
                if candidate is None or region_latched[region_id]:
                    continue
                if action is None or candidate["score"] > action["score"]:
                    action = dict(candidate)

            # 没有 R1~R3 动作时，再处理可回收进入 R4 的离开事件。
            if (
                action is None
                and recyclable_exit_pending
                and recyclable_last_candidate is not None
            ):
                action = dict(recyclable_last_candidate)
                action["a"] = expected_a_by_region[4]
                action["x"] = region3_x2
                action["region_id"] = 4
                action["region_name"] = region_names[4]
                action["score"] = float(
                    recyclable_last_candidate.get("score", 0.0)
                )

            if (
                action is not None
                and now - last_target_sent >= target_cooldown
            ):
                region_id = int(action["region_id"])
                payload = {
                    **action,
                    "region_match": True,
                    "source": "vision",
                    "timestamp": time.time(),
                    "request_id": (
                        f"{time.time_ns()}-vision-r{region_id}-a{action['a']}"
                    ),
                }
                # box 只用于视觉节点内部调试，不向 GPIO 发送 numpy 或多余结构。
                payload.pop("box", None)
                node.publish(vision_topic, payload)
                last_target_sent = now

                if region_id in region_latched:
                    region_latched[region_id] = True
                    region_last_seen[region_id] = now
                if region_id == 4:
                    recyclable_exit_pending = False
                    recyclable_last_candidate = None

                print(
                    "[VISION] 区域匹配，发送 "
                    f"R{region_id} (a,x,y)="
                    f"({payload['a']},{payload['x']},{payload['y']}) "
                    f"score={payload['score']:.3f}",
                    flush=True,
                )
            elif now - last_no_action_sent >= no_action_interval:
                # 有识别框但类别与区域不匹配时，同样发布 a=-1；
                # GPIO 节点会直接忽略，但 UI 预览中的 YOLO 框仍保留。
                if best_visible is not None:
                    no_action_payload = {
                        "a": -1,
                        "x": int(best_visible["x"]),
                        "y": int(best_visible["y"]),
                        "detected_a": int(best_visible["a"]),
                        "score": float(best_visible["score"]),
                        "yolo_class_id": int(
                            best_visible["yolo_class_id"]
                        ),
                        "yolo_class_name": str(
                            best_visible["yolo_class_name"]
                        ),
                        "region_id": int(best_visible["region_id"]),
                        "region_name": str(best_visible["region_name"]),
                        "region_match": False,
                        "source": "vision",
                        "timestamp": time.time(),
                        "request_id": (
                            f"{time.time_ns()}-vision-no-action"
                        ),
                    }
                else:
                    no_action_payload = {
                        "a": -1,
                        "x": -1,
                        "y": -1,
                        "detected_a": -1,
                        "score": 0.0,
                        "region_id": 0,
                        "region_name": "NONE",
                        "region_match": False,
                        "source": "vision",
                        "timestamp": time.time(),
                        "request_id": (
                            f"{time.time_ns()}-vision-none"
                        ),
                    }

                node.publish(vision_topic, no_action_payload)
                last_no_action_sent = now

            if now - last_preview_saved >= preview_interval:
                try:
                    _atomic_write_frame(
                        latest_frame_path,
                        display,
                    )
                    last_preview_saved = now
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[VISION] 写入 UI 预览帧失败：{exc}",
                        flush=True,
                    )
                    last_preview_saved = now

    finally:
        camera.release()
        node.close()
        print("[VISION] 已退出", flush=True)
