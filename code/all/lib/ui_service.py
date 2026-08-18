#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立 HDMI UI：显示视觉预览、四类垃圾数量和当前识别状态。

数据来源：
1. 视频：读取视觉节点原子写入的 latest_frame 图片；
2. 计数：优先订阅 miniROS /counter/data，定期读取 count.yaml 兜底；
3. 识别状态：订阅 miniROS /vision/result。

本模块只读，不直接修改 count.yaml，也不打开摄像头。
"""

from __future__ import annotations

import signal
import threading
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
)
from counting import CountStore


COUNT_FIELDS = (
    "recyclable",
    "kitchen",
    "hazardous",
    "other",
)

CARD_COLORS = {
    "recyclable": "#36c2a3",
    "kitchen": "#f0b35a",
    "hazardous": "#ef6b73",
    "other": "#6f8ff4",
}


class UIState:
    """在 miniROS 回调线程与 Tk 主线程之间安全共享状态。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {field: 0 for field in COUNT_FIELDS}
        self._total = 0
        self._revision = -1
        self._count_timestamp = 0.0
        self._vision: Dict[str, Any] = {
            "a": -1,
            "x": -1,
            "y": -1,
            "score": 0.0,
            "timestamp": 0.0,
        }

    def update_counts(self, payload: Any) -> None:
        normalized = _normalize_counts(payload)
        if normalized is None:
            return

        counts, total, revision, timestamp = normalized

        with self._lock:
            # 有 revision 时优先按 revision 判断；无 revision 时按时间戳判断。
            if revision >= 0 and revision < self._revision:
                return
            if revision < 0 and timestamp > 0 and timestamp < self._count_timestamp:
                return

            self._counts = counts
            self._total = total
            self._revision = max(self._revision, revision)
            self._count_timestamp = max(self._count_timestamp, timestamp)

    def update_vision(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        try:
            a = int(payload.get("a", -1))
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
            score = float(payload.get("score", 0.0) or 0.0)
            timestamp = float(payload.get("timestamp", time.time()) or time.time())
        except (TypeError, ValueError):
            return

        with self._lock:
            self._vision = {
                "a": a,
                "x": x,
                "y": y,
                "score": score,
                "timestamp": timestamp,
            }

    def snapshot(self) -> tuple[Dict[str, int], int, int, Dict[str, Any]]:
        with self._lock:
            return (
                dict(self._counts),
                int(self._total),
                int(self._revision),
                dict(self._vision),
            )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_counts(
    payload: Any,
) -> tuple[Dict[str, int], int, int, float] | None:
    """兼容 count.yaml 快照和标准 /counter/data 消息。"""
    if not isinstance(payload, dict):
        return None

    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, dict):
        # 兼容直接把四个字段放在顶层的旧格式。
        raw_counts = payload

    counts = {
        field: _safe_int(raw_counts.get(field, 0))
        for field in COUNT_FIELDS
    }
    total = sum(counts.values())

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    revision = _safe_int(
        payload.get("revision", metadata.get("revision", -1)),
        default=-1,
    )
    timestamp = _safe_float(payload.get("timestamp", 0.0), 0.0)

    return counts, total, revision, timestamp


def _fit_frame(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """等比例缩放并居中，保持视频画面不变形。"""
    width = max(2, int(width))
    height = max(2, int(height))

    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("视频帧尺寸无效")

    scale = min(width / frame_w, height / frame_h)
    scaled_w = max(1, int(round(frame_w * scale)))
    scaled_h = max(1, int(round(frame_h * scale)))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        frame,
        (scaled_w, scaled_h),
        interpolation=interpolation,
    )

    canvas = np.full(
        (height, width, 3),
        (10, 20, 32),
        dtype=np.uint8,
    )
    offset_x = (width - scaled_w) // 2
    offset_y = (height - scaled_h) // 2
    canvas[
        offset_y:offset_y + scaled_h,
        offset_x:offset_x + scaled_w,
    ] = resized
    return canvas


def _to_tk_photo(tk: Any, frame: np.ndarray) -> Any:
    """将 OpenCV BGR 图像转换为 Tk PhotoImage，不依赖 Pillow。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    ppm = (
        f"P6\n{width} {height}\n255\n".encode("ascii")
        + rgb.tobytes()
    )
    return tk.PhotoImage(data=ppm, format="PPM")


def run_ui_node(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    """运行独立本地 UI。"""
    config = load_config(config_path)
    ui_config = config.get("ui", {})
    topics = config["miniros"]["topics"]
    by_a, _ = garbage_maps(config)

    frame_path = resolve_path(
        config.get("paths", {}).get(
            "latest_frame",
            "/dev/shm/rdk_smartbin_latest.png",
        )
    )
    store = CountStore(config)
    state = UIState()

    try:
        state.update_counts(store.read())
    except Exception as exc:
        print(
            f"[UI] 初次读取 count.yaml 失败：{exc}",
            flush=True,
        )

    node = create_node("node_ui", config)
    counter_topic = topics.get("counter_data", "/counter/data")
    vision_topic = topics.get("vision_result", "/vision/result")

    node.register_subscriber(
        counter_topic,
        state.update_counts,
    )
    node.register_subscriber(
        vision_topic,
        state.update_vision,
    )

    spin_thread = threading.Thread(
        target=node.spin,
        name="ui-miniros-spin",
        daemon=True,
    )
    spin_thread.start()

    import tkinter as tk

    display = str(ui_config.get("display", ":0"))
    try:
        root = tk.Tk(screenName=display)
    except tk.TclError as exc:
        node.close()
        raise RuntimeError(
            f"无法连接 HDMI 图形显示 {display}。"
            "请确认桌面已登录，并通过 run_all.sh 启动。"
        ) from exc

    title_text = str(
        ui_config.get(
            "window_title",
            ui_config.get(
                "window_name",
                "RDK X5 智能垃圾分类系统",
            ),
        )
    )
    window_width = max(960, int(ui_config.get("width", 1280)))
    window_height = max(600, int(ui_config.get("height", 720)))
    fullscreen = bool(ui_config.get("fullscreen", False))
    refresh_ms = max(100, int(ui_config.get("refresh_interval_ms", 200)))
    fallback_ms = max(
        500,
        int(ui_config.get("count_fallback_interval_ms", 1000)),
    )

    root.title(title_text)
    root.configure(bg="#07131f")
    root.minsize(960, 600)
    root.geometry(f"{window_width}x{window_height}")
    if fullscreen:
        root.attributes("-fullscreen", True)

    font_family = str(
        ui_config.get("font_family", "Noto Sans CJK SC")
    )
    bg = "#07131f"
    panel = "#0d2032"
    panel_inner = "#102a41"
    primary = "#f3f8ff"
    secondary = "#8ca5ba"
    cyan = "#5bd6ff"

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # 顶部标题栏。
    header = tk.Frame(root, bg=bg, height=82)
    header.grid(row=0, column=0, sticky="ew", padx=26, pady=(14, 6))
    header.grid_propagate(False)
    header.grid_columnconfigure(0, weight=1)

    tk.Label(
        header,
        text="RDK X5 智能垃圾分类系统",
        font=(font_family, 27, "bold"),
        fg=primary,
        bg=bg,
    ).grid(row=0, column=0, sticky="w")

    tk.Label(
        header,
        text="AI 视觉识别 · 自动分拣 · 云端管理",
        font=(font_family, 12),
        fg=secondary,
        bg=bg,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    connection_dot = tk.Label(
        header,
        text="●",
        font=("DejaVu Sans", 13, "bold"),
        fg="#f0b35a",
        bg=bg,
    )
    connection_dot.grid(row=0, column=1, rowspan=2, padx=(0, 8))

    connection_label = tk.Label(
        header,
        text="miniROS 连接中",
        font=(font_family, 12, "bold"),
        fg=secondary,
        bg=bg,
    )
    connection_label.grid(row=0, column=2, rowspan=2, padx=(0, 24))

    clock_label = tk.Label(
        header,
        text="",
        font=("DejaVu Sans", 15, "bold"),
        fg=primary,
        bg=bg,
    )
    clock_label.grid(row=0, column=3, rowspan=2, sticky="e")

    # 主体区域：左视频、右计数。
    body = tk.Frame(root, bg=bg)
    body.grid(row=1, column=0, sticky="nsew", padx=26, pady=(4, 14))
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(0, weight=1)

    video_card = tk.Frame(
        body,
        bg=panel,
        highlightthickness=1,
        highlightbackground="#17364f",
    )
    video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    video_card.grid_rowconfigure(1, weight=1)
    video_card.grid_columnconfigure(0, weight=1)

    video_header = tk.Frame(video_card, bg=panel, height=48)
    video_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(8, 0))
    video_header.grid_propagate(False)

    tk.Label(
        video_header,
        text="实时视觉画面",
        font=(font_family, 15, "bold"),
        fg=primary,
        bg=panel,
    ).pack(side="left", pady=8)

    preview_status = tk.Label(
        video_header,
        text="等待视觉节点",
        font=(font_family, 10),
        fg="#f0b35a",
        bg=panel,
    )
    preview_status.pack(side="right", pady=10)

    video_label = tk.Label(
        video_card,
        text="视觉预览等待中",
        font=(font_family, 18),
        fg=secondary,
        bg="#0a1420",
    )
    video_label.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=14,
        pady=(6, 14),
    )

    side = tk.Frame(
        body,
        bg=panel,
        width=340,
        highlightthickness=1,
        highlightbackground="#17364f",
    )
    side.grid(row=0, column=1, sticky="ns")
    side.grid_propagate(False)
    side.grid_columnconfigure(0, weight=1)

    tk.Label(
        side,
        text="累计投放",
        font=(font_family, 14, "bold"),
        fg=secondary,
        bg=panel,
    ).grid(row=0, column=0, pady=(22, 0))

    total_label = tk.Label(
        side,
        text="0",
        font=("DejaVu Sans", 48, "bold"),
        fg=cyan,
        bg=panel,
    )
    total_label.grid(row=1, column=0, pady=(0, 10))

    count_labels: Dict[str, Any] = {}
    ordered_items = [by_a[index] for index in sorted(by_a)]

    for row_index, item in enumerate(ordered_items, start=2):
        field = str(item["field"])
        card = tk.Frame(
            side,
            bg=panel_inner,
            height=70,
            highlightthickness=1,
            highlightbackground="#193c59",
        )
        card.grid(
            row=row_index,
            column=0,
            sticky="ew",
            padx=16,
            pady=6,
        )
        card.grid_propagate(False)
        card.grid_columnconfigure(1, weight=1)

        accent = tk.Frame(
            card,
            bg=CARD_COLORS.get(field, cyan),
            width=5,
        )
        accent.grid(row=0, column=0, sticky="ns")

        tk.Label(
            card,
            text=str(item.get("name", field)),
            font=(font_family, 14, "bold"),
            fg=primary,
            bg=panel_inner,
        ).grid(row=0, column=1, sticky="w", padx=14)

        value = tk.Label(
            card,
            text="0",
            font=("DejaVu Sans", 27, "bold"),
            fg=CARD_COLORS.get(field, cyan),
            bg=panel_inner,
        )
        value.grid(row=0, column=2, sticky="e", padx=16)
        count_labels[field] = value

    recognition_card = tk.Frame(
        side,
        bg="#0b1a29",
        highlightthickness=1,
        highlightbackground="#193c59",
    )
    recognition_card.grid(
        row=6,
        column=0,
        sticky="ew",
        padx=16,
        pady=(14, 16),
    )

    tk.Label(
        recognition_card,
        text="当前识别",
        font=(font_family, 11),
        fg=secondary,
        bg="#0b1a29",
    ).pack(anchor="w", padx=14, pady=(10, 0))

    recognition_label = tk.Label(
        recognition_card,
        text="等待目标",
        font=(font_family, 16, "bold"),
        fg=primary,
        bg="#0b1a29",
    )
    recognition_label.pack(anchor="w", padx=14, pady=(3, 0))

    coordinate_label = tk.Label(
        recognition_card,
        text="中心点：--",
        font=("DejaVu Sans", 11),
        fg=secondary,
        bg="#0b1a29",
    )
    coordinate_label.pack(anchor="w", padx=14, pady=(3, 10))

    # 底部状态栏。
    footer = tk.Frame(root, bg="#091a29", height=42)
    footer.grid(row=2, column=0, sticky="ew")
    footer.grid_propagate(False)

    revision_label = tk.Label(
        footer,
        text="数据版本：0",
        font=(font_family, 10),
        fg=secondary,
        bg="#091a29",
    )
    revision_label.pack(side="left", padx=26, pady=10)

    tk.Label(
        footer,
        text="Esc 退出界面",
        font=(font_family, 10),
        fg=secondary,
        bg="#091a29",
    ).pack(side="right", padx=26, pady=10)

    photo_reference: Dict[str, Any] = {"image": None}
    last_frame_mtime = -1
    last_render_size = (0, 0)
    last_fallback = 0.0
    closing = False

    def close_ui(*_: Any) -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        try:
            root.after(0, root.destroy)
        except Exception:
            pass

    root.bind("<Escape>", close_ui)
    root.protocol("WM_DELETE_WINDOW", close_ui)
    signal.signal(signal.SIGINT, close_ui)
    signal.signal(signal.SIGTERM, close_ui)

    def refresh() -> None:
        nonlocal last_frame_mtime, last_render_size, last_fallback

        if closing:
            return

        now = time.monotonic()

        # miniROS 数据为主，count.yaml 每秒兜底一次。
        if now - last_fallback >= fallback_ms / 1000.0:
            try:
                state.update_counts(store.read())
            except Exception as exc:
                print(
                    f"[UI] 读取 count.yaml 失败：{exc}",
                    flush=True,
                )
            last_fallback = now

        counts, total, revision, vision = state.snapshot()
        for field, label in count_labels.items():
            label.config(text=str(counts.get(field, 0)))
        total_label.config(text=str(total))
        revision_label.config(text=f"数据版本：{max(0, revision)}")

        a = int(vision.get("a", -1))
        if a in by_a:
            item = by_a[a]
            recognition_label.config(
                text=str(item.get("name", f"a={a}")),
                fg=CARD_COLORS.get(str(item["field"]), primary),
            )
            coordinate_label.config(
                text=(
                    f"中心点：({int(vision.get('x', -1))}, "
                    f"{int(vision.get('y', -1))})"
                )
            )
        else:
            recognition_label.config(text="等待目标", fg=primary)
            coordinate_label.config(text="中心点：--")

        connected = bool(getattr(node, "is_connected", False))
        connection_dot.config(
            fg="#55e6a5" if connected else "#f0b35a"
        )
        connection_label.config(
            text="miniROS 已连接" if connected else "miniROS 重连中",
            fg=primary if connected else secondary,
        )
        clock_label.config(text=time.strftime("%Y-%m-%d  %H:%M:%S"))

        # 仅在预览文件或显示区域发生变化时重新解码和缩放。
        try:
            stat = frame_path.stat()
            mtime = stat.st_mtime_ns
            target_width = max(2, video_label.winfo_width())
            target_height = max(2, video_label.winfo_height())
            render_size = (target_width, target_height)

            if (
                mtime != last_frame_mtime
                or render_size != last_render_size
            ):
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    fitted = _fit_frame(
                        frame,
                        target_width,
                        target_height,
                    )
                    photo = _to_tk_photo(tk, fitted)
                    photo_reference["image"] = photo
                    video_label.config(image=photo, text="")
                    preview_status.config(
                        text="视觉节点在线",
                        fg="#55e6a5",
                    )
                    last_frame_mtime = mtime
                    last_render_size = render_size
        except FileNotFoundError:
            preview_status.config(
                text="等待视觉节点",
                fg="#f0b35a",
            )
        except Exception as exc:
            preview_status.config(
                text="预览读取异常",
                fg="#ef6b73",
            )
            print(f"[UI] 预览刷新失败：{exc}", flush=True)

        root.after(refresh_ms, refresh)

    print(
        f"[UI] 计数话题：{counter_topic}",
        flush=True,
    )
    print(
        f"[UI] 识别话题：{vision_topic}",
        flush=True,
    )
    print(
        f"[UI] 预览文件：{frame_path}",
        flush=True,
    )

    root.after(50, refresh)

    try:
        root.mainloop()
    finally:
        closing = True
        try:
            node.close()
        except Exception:
            pass
        spin_thread.join(timeout=1.0)
        print("[UI] 已退出", flush=True)
