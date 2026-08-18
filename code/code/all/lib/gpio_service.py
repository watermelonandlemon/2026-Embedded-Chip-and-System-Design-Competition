#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPIO 执行服务。"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
from collections import OrderedDict
from itertools import count
from pathlib import Path
from typing import Any, Dict, Mapping

from core import DEFAULT_CONFIG_PATH, as_bool, create_node, garbage_maps, load_config


class GPIOController:
    def __init__(self, by_a: Mapping[int, Mapping[str, Any]], dry_run: bool) -> None:
        self.by_a = by_a
        self.dry_run = dry_run
        self.GPIO = None
        if dry_run:
            return
        try:
            import Hobot.GPIO as GPIO
        except ImportError as exc:
            raise RuntimeError("找不到 Hobot.GPIO，请使用 RDK X5 系统 Python") from exc
        self.GPIO = GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        self.release_all()

    def pull_low(self, pin: int) -> None:
        if self.dry_run:
            print(f"[GPIO-DRY] GPIO{pin} -> LOW", flush=True)
            return
        self.GPIO.setup(pin, self.GPIO.OUT, initial=self.GPIO.LOW)
        self.GPIO.output(pin, self.GPIO.LOW)

    def release(self, pin: int) -> None:
        if self.dry_run:
            print(f"[GPIO-DRY] GPIO{pin} -> 高阻态", flush=True)
            return
        self.GPIO.setup(pin, self.GPIO.IN)

    def release_all(self) -> None:
        for item in self.by_a.values():
            self.release(int(item["gpio"]))

    def close(self) -> None:
        self.release_all()
        if not self.dry_run and self.GPIO is not None:
            self.GPIO.cleanup()


def _normalize(message: Any, by_code: Mapping[str, int]) -> Dict[str, Any] | None:
    if isinstance(message, (list, tuple)) and len(message) >= 3:
        message = {"a": message[0], "x": message[1], "y": message[2], "source": "vision"}
    if not isinstance(message, dict):
        return None
    raw_a = message.get("a")
    if raw_a is None and message.get("code") is not None:
        raw_a = by_code.get(str(message.get("code")))
    try:
        a = int(raw_a)
    except (TypeError, ValueError):
        return None
    result = dict(message)
    result["a"] = a
    try:
        result["x"] = int(message.get("x", -1))
        result["y"] = int(message.get("y", -1))
    except (TypeError, ValueError):
        result["x"] = -1
        result["y"] = -1
    result.setdefault("source", "unknown")
    result.setdefault("timestamp", time.time())
    result.setdefault("request_id", f"{time.time_ns()}-{result['source']}-a{a}")
    return result


def run_gpio_node(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    by_a, by_code = garbage_maps(config)
    topics = config["miniros"]["topics"]
    control = config["control"]
    dry_run = as_bool(os.getenv("SMARTBIN_GPIO_DRY_RUN"), bool(control.get("dry_run", False)))
    active_s = max(0.05, float(control.get("active_low_seconds", 1.0)))
    cooldown_s = max(0.0, float(control.get("post_action_cooldown_s", 2.0)))
    gpio = GPIOController(by_a, dry_run)
    node = create_node("node_gpio", config)
    result_topic = node.register_publisher(topics["actuator_executed"])
    work_queue: queue.PriorityQueue[tuple[int, int, Dict[str, Any]]] = queue.PriorityQueue(
        maxsize=int(control.get("queue_size", 16))
    )
    sequence = count()
    stop_event = threading.Event()
    state_lock = threading.Lock()
    pending: set[str] = set()
    cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    cache_limit = int(control.get("result_cache_size", 200))

    def request_key(message: Mapping[str, Any]) -> str:
        return str(message.get("cloud_command_id") or message.get("request_id") or "")

    def publish_result(message: Mapping[str, Any], success: bool, error: str = "") -> Dict[str, Any]:
        a = int(message.get("a", -1))
        item = by_a.get(a, {})
        result = {
            "a": a,
            "x": int(message.get("x", -1)),
            "y": int(message.get("y", -1)),
            "code": item.get("code"),
            "gpio": item.get("gpio"),
            "success": bool(success),
            "error": error,
            "source": message.get("source", "unknown"),
            "request_id": message.get("request_id"),
            "cloud_command_id": message.get("cloud_command_id"),
            "event_id": f"{time.time_ns()}-a{a}",
            "finished_at": time.time(),
        }
        node.publish(result_topic, result)
        print(f"[GPIO] 结果 {result}", flush=True)
        return result

    def enqueue(raw: Any, manual: bool) -> None:
        message = _normalize(raw, by_code)
        if message is None:
            print(f"[GPIO] 忽略非法消息：{raw!r}", flush=True)
            return
        a = int(message["a"])
        if a == -1:
            return
        if a not in by_a:
            publish_result(message, False, f"非法垃圾 a={a}")
            return
        key = request_key(message)
        with state_lock:
            if key and key in cache:
                replay = dict(cache[key])
                node.publish(result_topic, replay)
                print(f"[GPIO] 重复命令复用结果：{key}", flush=True)
                return
            if key and key in pending:
                return
            if key:
                pending.add(key)
        try:
            work_queue.put_nowait((0 if manual else 1, next(sequence), message))
        except queue.Full:
            with state_lock:
                pending.discard(key)
            publish_result(message, False, "GPIO 命令队列已满")

    def on_vision(message: Any) -> None:
        enqueue(message, manual=False)

    def on_manual(message: Any) -> None:
        enqueue(message, manual=True)

    def worker() -> None:
        while not stop_event.is_set():
            try:
                _, _, message = work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            a = int(message["a"])
            item = by_a[a]
            pin = int(item["gpio"])
            key = request_key(message)
            success = False
            error = ""
            try:
                gpio.pull_low(pin)
                print(
                    f"[GPIO] a={a} {item['name']} -> GPIO{pin} 拉低 {active_s:.2f}s",
                    flush=True,
                )
                if stop_event.wait(active_s):
                    raise RuntimeError("程序停止导致动作中断")
                gpio.release(pin)
                success = True
            except Exception as exc:
                error = str(exc)
                try:
                    gpio.release(pin)
                except Exception:
                    pass
            result = publish_result(message, success, error)
            with state_lock:
                pending.discard(key)
                if key:
                    cache[key] = result
                    while len(cache) > cache_limit:
                        cache.popitem(last=False)
            work_queue.task_done()
            if success and cooldown_s > 0:
                stop_event.wait(cooldown_s)

    node.register_subscriber(topics["vision_result"], on_vision)
    node.register_subscriber(topics["manual_request"], on_manual)
    thread = threading.Thread(target=worker, name="gpio-worker", daemon=True)

    def shutdown(*_: Any) -> None:
        stop_event.set()
        try:
            node.close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"[GPIO] dry_run={dry_run}，动作={active_s}s，动作后冷却={cooldown_s}s", flush=True)
    thread.start()
    try:
        node.spin()
    finally:
        shutdown()
        thread.join(timeout=2.0)
        gpio.close()
