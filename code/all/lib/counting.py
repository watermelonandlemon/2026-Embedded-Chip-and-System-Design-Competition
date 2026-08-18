#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""count.yaml 存储与计数节点。

职责：
1. node_count 是 count.yaml 的唯一写入者；
2. 仅在 GPIO 成功执行后增加对应垃圾数量；
3. 响应 Web 节点发来的单类或全部清零命令；
4. 通过 miniROS 周期发布标准计数消息；
5. 使用文件锁和原子替换，避免并发读写造成文件损坏。
"""

from __future__ import annotations

import copy
import fcntl
import os
import signal
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from core import (
    DEFAULT_CONFIG_PATH,
    create_node,
    dump_yaml,
    garbage_maps,
    load_config,
    load_yaml,
    resolve_path,
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CountStore:
    """count.yaml 的进程安全存储层。

    ``allow_repair=False`` 时严格只读：文件格式错误只抛出异常，不移动、
    不覆盖文件。UI 和 Web 节点应使用默认值。

    ``allow_repair=True`` 仅供计数节点使用：格式错误时先把异常文件备份，
    然后抛出异常并停止启动，避免在不知情的情况下清空历史计数。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        allow_repair: bool = False,
    ) -> None:
        self.path = resolve_path(config["paths"]["count_yaml"])
        self.lock_path = resolve_path(config["paths"]["count_lock"])
        self.allow_repair = bool(allow_repair)
        self.by_a, _ = garbage_maps(config)
        self.fields = tuple(
            self.by_a[index]["field"]
            for index in sorted(self.by_a)
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "counts": {field: 0 for field in self.fields},
            "total": 0,
            "metadata": {
                "revision": 0,
                "updated_at": None,
                "updated_by": "bootstrap",
                "last_action": "initialize",
                "last_event_id": None,
            },
        }

    def _normalize(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("count.yaml 顶层必须是映射")

        result = self._default()

        schema_version = raw.get("schema_version", 1)
        if isinstance(schema_version, bool):
            raise ValueError("schema_version 不能是布尔值")
        result["schema_version"] = max(1, int(schema_version))

        counts = raw.get("counts", {})
        if not isinstance(counts, dict):
            raise ValueError("count.yaml 的 counts 必须是映射")

        for field in self.fields:
            value = counts.get(field, 0)
            if isinstance(value, bool):
                raise ValueError(f"counts.{field} 不能是布尔值")
            parsed = int(value)
            if parsed < 0:
                raise ValueError(f"counts.{field} 不能小于 0")
            result["counts"][field] = parsed

        result["total"] = sum(result["counts"].values())

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("count.yaml 的 metadata 必须是映射")

        revision = metadata.get("revision", 0)
        if isinstance(revision, bool):
            raise ValueError("metadata.revision 不能是布尔值")

        result["metadata"]["revision"] = max(0, int(revision))
        result["metadata"]["updated_at"] = metadata.get("updated_at")
        result["metadata"]["updated_by"] = str(
            metadata.get("updated_by", "unknown")
        )
        result["metadata"]["last_action"] = str(
            metadata.get("last_action", "unknown")
        )

        event_id = metadata.get("last_event_id")
        result["metadata"]["last_event_id"] = (
            None if event_id in (None, "") else str(event_id)
        )
        return result

    def _backup_broken_file(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = self.path.with_name(
            f"{self.path.name}.broken.{timestamp}.{os.getpid()}"
        )
        os.replace(self.path, backup)
        return backup

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default()

        try:
            return self._normalize(load_yaml(self.path))
        except Exception as exc:
            if not self.allow_repair:
                raise RuntimeError(
                    f"count.yaml 格式错误，已保持原文件不变：{exc}"
                ) from exc

            try:
                backup = self._backup_broken_file()
            except OSError as backup_exc:
                raise RuntimeError(
                    f"count.yaml 格式错误，并且备份失败：{backup_exc}；"
                    f"原始错误：{exc}"
                ) from exc

            raise RuntimeError(
                f"count.yaml 格式错误，已备份为 {backup.name}：{exc}"
            ) from exc

    def _write_unlocked(self, data: Mapping[str, Any]) -> None:
        content = dump_yaml(self._normalize(dict(data)))
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def initialize(self) -> Dict[str, Any]:
        """读取现有计数；文件不存在时创建标准 count.yaml。"""
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                data = self._read_unlocked()
                self._write_unlocked(data)
                return copy.deepcopy(data)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read(self) -> Dict[str, Any]:
        """读取当前计数快照；只读模式不会移动或覆盖异常文件。"""
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return copy.deepcopy(self._read_unlocked())
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def increment(
        self,
        a: int,
        updated_by: str,
        event_id: str | None,
    ) -> Dict[str, Any]:
        """指定全局垃圾类型 ``a`` 增加 1。"""
        if a not in self.by_a:
            raise ValueError(f"非法垃圾 a={a}")

        field = self.by_a[a]["field"]
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                data = self._read_unlocked()
                data["counts"][field] += 1
                data["total"] = sum(data["counts"].values())

                metadata = data["metadata"]
                metadata["revision"] += 1
                metadata["updated_at"] = _now_iso()
                metadata["updated_by"] = str(updated_by)
                metadata["last_action"] = f"increment:a={a}:{field}"
                metadata["last_event_id"] = event_id

                self._write_unlocked(data)
                return copy.deepcopy(data)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _resolve_reset_field(self, target: str | int) -> str | None:
        normalized = str(target).strip().lower()
        if normalized in {"all", "*", "全部", "全部清零", "all_counts"}:
            return None

        # 必须先匹配四位状态码。否则 int("0001") == 1，
        # 会把可回收垃圾错误解析成 a=1（厨余垃圾）。
        for item in self.by_a.values():
            if normalized == str(item["code"]).strip().lower():
                return str(item["field"])

        # 再匹配字段名和中文名称。
        for item in self.by_a.values():
            aliases = {
                str(item["field"]).strip().lower(),
                str(item["name"]).strip().lower(),
            }
            if normalized in aliases:
                return str(item["field"])

        # 最后把 0、1、2、3 解释为全局垃圾类型 a。
        try:
            a = int(normalized)
        except ValueError as exc:
            raise ValueError(f"非法清零目标：{target}") from exc

        if a not in self.by_a:
            raise ValueError(f"非法清零目标：{target}")
        return str(self.by_a[a]["field"])

    def reset(
        self,
        target: str | int,
        updated_by: str,
    ) -> Dict[str, Any]:
        """按 all、a、字段名、状态码或中文名称清零。"""
        field = self._resolve_reset_field(target)

        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                data = self._read_unlocked()

                if field is None:
                    for name in self.fields:
                        data["counts"][name] = 0
                    action = "reset:all"
                else:
                    data["counts"][field] = 0
                    action = f"reset:{field}"

                data["total"] = sum(data["counts"].values())
                metadata = data["metadata"]
                metadata["revision"] += 1
                metadata["updated_at"] = _now_iso()
                metadata["updated_by"] = str(updated_by)
                metadata["last_action"] = action
                metadata["last_event_id"] = None

                self._write_unlocked(data)
                return copy.deepcopy(data)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _counter_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    raw_counts = data.get("counts", {})
    metadata = data.get("metadata", {})
    counts = {
        "recyclable": int(raw_counts.get("recyclable", 0)),
        "kitchen": int(raw_counts.get("kitchen", 0)),
        "hazardous": int(raw_counts.get("hazardous", 0)),
        "other": int(raw_counts.get("other", 0)),
    }
    return {
        "source": "node_count",
        "timestamp": time.time(),
        "revision": int(metadata.get("revision", 0)),
        "counts": counts,
        "total": sum(counts.values()),
        "updated_at": metadata.get("updated_at"),
        "last_action": str(metadata.get("last_action", "")),
    }


def _print_counts(prefix: str, data: Mapping[str, Any]) -> None:
    counts = data["counts"]
    print(
        f"{prefix} "
        f"可回收={counts['recyclable']} "
        f"厨余={counts['kitchen']} "
        f"有害={counts['hazardous']} "
        f"其他={counts['other']} "
        f"总计={data['total']}",
        flush=True,
    )


def run_count_node(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    config = load_config(config_path)
    topics = config["miniros"]["topics"]
    counter_config = config.get("counter", {})

    # 只有 node_count 允许在异常时备份损坏文件；UI/Web 均为严格只读。
    store = CountStore(config, allow_repair=True)
    initial = store.initialize()

    node = create_node("node_count", config)
    counter_data_topic = node.register_publisher(
        topics.get("counter_data", "/counter/data")
    )
    reset_result_topic = node.register_publisher(
        topics["counter_reset_result"]
    )

    stop_event = threading.Event()
    publish_lock = threading.Lock()

    recent_order: deque[str] = deque()
    recent_set: set[str] = set()
    in_flight: set[str] = set()
    recent_lock = threading.Lock()
    recent_limit = max(
        1,
        int(counter_config.get("recent_event_limit", 2000)),
    )
    publish_interval = max(
        0.2,
        float(counter_config.get("publish_interval_s", 0.5)),
    )

    def publish_counts(data: Mapping[str, Any] | None = None) -> None:
        snapshot = dict(data) if data is not None else store.read()
        payload = _counter_payload(snapshot)
        with publish_lock:
            node.publish(counter_data_topic, payload)

    def remember_completed(event_id: str) -> None:
        if not event_id:
            return
        if event_id in recent_set:
            return
        recent_set.add(event_id)
        recent_order.append(event_id)
        while len(recent_order) > recent_limit:
            recent_set.discard(recent_order.popleft())

    def claim_event(event_id: str) -> bool:
        """原子占用事件，防止并发回调对同一事件重复计数。"""
        if not event_id:
            return True
        with recent_lock:
            if event_id in recent_set or event_id in in_flight:
                return False
            in_flight.add(event_id)
            return True

    def finish_event(event_id: str, success: bool) -> None:
        if not event_id:
            return
        with recent_lock:
            in_flight.discard(event_id)
            if success:
                remember_completed(event_id)

    initial_event = str(
        initial.get("metadata", {}).get("last_event_id") or ""
    )
    if initial_event:
        with recent_lock:
            remember_completed(initial_event)

    def on_executed(message: Any) -> None:
        if not isinstance(message, dict):
            return
        if not bool(message.get("success", False)):
            return

        try:
            a = int(message.get("a", -1))
        except (TypeError, ValueError):
            return

        event_id = str(
            message.get("event_id")
            or message.get("request_id")
            or message.get("cloud_command_id")
            or ""
        )

        if not claim_event(event_id):
            print(f"[COUNT] 忽略重复事件 {event_id}", flush=True)
            return

        success = False
        try:
            data = store.increment(
                a,
                updated_by=f"gpio:{message.get('source', 'unknown')}",
                event_id=event_id or None,
            )
            success = True
            _print_counts(f"[COUNT] a={a} +1 ->", data)
            publish_counts(data)
        except Exception as exc:
            print(f"[COUNT] 计数失败：{exc}", flush=True)
        finally:
            # 写入失败时释放占用，允许同一事件重新投递。
            finish_event(event_id, success)

    def on_reset(message: Any) -> None:
        payload = message if isinstance(message, dict) else {}
        target = payload.get("target", "all")
        command_id = (
            payload.get("cloud_command_id")
            or payload.get("command_id")
        )

        try:
            data = store.reset(
                target,
                updated_by=str(payload.get("source", "web")),
            )
            result = {
                "success": True,
                "target": target,
                "cloud_command_id": command_id,
                "request_id": payload.get("request_id"),
                "revision": data["metadata"]["revision"],
                "counts": dict(data["counts"]),
                "total": int(data["total"]),
                "error": "",
            }
            _print_counts(f"[COUNT] 清零 {target} ->", data)
            publish_counts(data)
        except Exception as exc:
            result = {
                "success": False,
                "target": target,
                "cloud_command_id": command_id,
                "request_id": payload.get("request_id"),
                "error": str(exc),
            }
            print(f"[COUNT] 清零失败：{exc}", flush=True)

        with publish_lock:
            node.publish(reset_result_topic, result)

    node.register_subscriber(
        topics["actuator_executed"],
        on_executed,
    )
    node.register_subscriber(
        topics["counter_reset"],
        on_reset,
    )

    def heartbeat() -> None:
        while not stop_event.wait(publish_interval):
            try:
                publish_counts()
            except Exception as exc:
                if not stop_event.is_set():
                    print(
                        f"[COUNT] 发布计数心跳失败：{exc}",
                        flush=True,
                    )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="count-heartbeat",
        daemon=True,
    )

    def shutdown(*_: Any) -> None:
        stop_event.set()
        try:
            node.close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"[COUNT] 唯一写入文件：{store.path}", flush=True)
    print(
        f"[COUNT] 计数话题：{counter_data_topic}，"
        f"发布周期={publish_interval:.2f}s",
        flush=True,
    )
    _print_counts("[COUNT] 启动读取 ->", initial)

    try:
        publish_counts(initial)
    except Exception as exc:
        print(f"[COUNT] 首次发布计数失败：{exc}", flush=True)

    heartbeat_thread.start()
    try:
        node.spin()
    finally:
        shutdown()
        heartbeat_thread.join(timeout=1.0)
