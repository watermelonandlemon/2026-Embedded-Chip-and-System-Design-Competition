#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公网 Web API 与 Web 节点。

云端地址、设备编号和设备密钥只从 ``src/config.yaml`` 读取。
Web 节点负责：计数上报、管理员手动控制、远程清零和云端状态发布。
"""

from __future__ import annotations

import json
import signal
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from core import (
    DEFAULT_CONFIG_PATH,
    as_bool,
    create_node,
    garbage_maps,
    load_config,
)
from counting import CountStore


class CloudConfigError(RuntimeError):
    pass


class CloudRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class CloudSettings:
    enabled: bool
    server: str
    device_id: str
    device_key: str
    verify_tls: bool
    timeout_s: float
    report_interval_s: float
    command_poll_interval_s: float
    execution_wait_s: float
    firmware: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CloudSettings":
        cloud = config.get("cloud", {})
        if not isinstance(cloud, Mapping):
            raise CloudConfigError("config.yaml 中的 cloud 必须是映射")

        return cls(
            enabled=as_bool(cloud.get("enabled"), True),
            server=str(cloud.get("base_url", "")).strip().rstrip("/"),
            device_id=str(cloud.get("device_id", "")).strip(),
            device_key=str(cloud.get("device_key", "")).strip(),
            verify_tls=as_bool(cloud.get("verify_tls", True), True),
            timeout_s=float(cloud.get("timeout_s", 5.0)),
            report_interval_s=float(cloud.get("report_interval_s", 1.0)),
            command_poll_interval_s=float(
                cloud.get("command_poll_interval_s", 1.0)
            ),
            execution_wait_s=float(cloud.get("execution_wait_s", 10.0)),
            firmware=str(
                cloud.get("firmware", "rdk-x5-smartbin")
            ).strip(),
        )

    def validate(self) -> None:
        for name, value in (
            ("timeout_s", self.timeout_s),
            ("report_interval_s", self.report_interval_s),
            ("command_poll_interval_s", self.command_poll_interval_s),
            ("execution_wait_s", self.execution_wait_s),
        ):
            if value <= 0:
                raise CloudConfigError(f"cloud.{name} 必须大于 0")

        if not self.enabled:
            return

        placeholders = ("YOUR_", "CHANGE_ME", "PLACEHOLDER")
        upper_server = self.server.upper()
        upper_id = self.device_id.upper()
        upper_key = self.device_key.upper()

        if (
            not self.server
            or any(token in upper_server for token in placeholders)
            or not self.server.startswith(("http://", "https://"))
        ):
            raise CloudConfigError(
                "config.yaml 中的 cloud.base_url 无效"
            )

        if (
            not self.device_id
            or any(token in upper_id for token in placeholders)
        ):
            raise CloudConfigError(
                "config.yaml 中的 cloud.device_id 无效"
            )

        if (
            not self.device_key
            or any(token in upper_key for token in placeholders)
        ):
            raise CloudConfigError(
                "config.yaml 中的 cloud.device_key 无效"
            )


class SmartBinClient:
    def __init__(self, settings: CloudSettings) -> None:
        settings.validate()
        self.settings = settings
        self._ssl_context = self._build_context()

    def _build_context(self) -> ssl.SSLContext | None:
        if not self.settings.server.startswith("https://"):
            return None
        if self.settings.verify_tls:
            return ssl.create_default_context()
        return ssl._create_unverified_context()  # noqa: SLF001

    def request(
        self,
        method: str,
        path: str,
        body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not path.startswith("/"):
            raise ValueError("云端 API path 必须以 / 开头")

        data = (
            None
            if body is None
            else json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        request = urllib.request.Request(
            self.settings.server + path,
            method=method.upper(),
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "X-Device-ID": self.settings.device_id,
                "X-Device-Key": self.settings.device_key,
                "User-Agent": "RDK-X5-SmartBin/4.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_s,
                context=self._ssl_context,
            ) as response:
                raw = response.read()
                if not raw:
                    return {}
                decoded = json.loads(raw.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise CloudRequestError("云端响应顶层必须是 JSON 对象")
                return decoded
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CloudRequestError(
                f"HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CloudRequestError(f"网络错误：{exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise CloudRequestError(f"云端返回的 JSON 无效：{exc}") from exc
        except (TimeoutError, OSError, ValueError) as exc:
            raise CloudRequestError(str(exc)) from exc

    def report(self, counts: Mapping[str, int]) -> Dict[str, Any]:
        return self.request(
            "POST",
            "/api/iot/v1/report",
            {
                "recyclable": int(counts.get("recyclable", 0)),
                "kitchen": int(counts.get("kitchen", 0)),
                "hazardous": int(counts.get("hazardous", 0)),
                "other": int(counts.get("other", 0)),
                "firmware": self.settings.firmware,
                "local_status": "running",
            },
        )

    def get_commands(
        self,
        command_type: str,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "command_type": str(command_type),
                "limit": max(1, min(int(limit), 10)),
            }
        )
        result = self.request(
            "GET",
            f"/api/iot/v1/commands?{query}",
        )
        commands = result.get("commands", [])
        if not isinstance(commands, list):
            raise CloudRequestError("云端 commands 字段必须是列表")
        return [item for item in commands if isinstance(item, dict)]

    def ack(
        self,
        command_id: str,
        success: bool,
        message: str,
    ) -> Dict[str, Any]:
        quoted = urllib.parse.quote(str(command_id), safe="")
        return self.request(
            "POST",
            f"/api/iot/v1/commands/{quoted}/ack",
            {
                "success": bool(success),
                "message": str(message)[:500],
            },
        )


def run_web_node(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    config = load_config(config_path)
    settings = CloudSettings.from_config(config)
    settings.validate()

    topics = config["miniros"]["topics"]
    by_a, by_code = garbage_maps(config)
    node = create_node("node_web", config)

    publish_lock = threading.Lock()
    log_lock = threading.Lock()
    stop_event = threading.Event()

    status_topic = node.register_publisher(
        topics.get("cloud_status", "/cloud/status")
    )

    def log(message: str) -> None:
        with log_lock:
            print(message, flush=True)

    status_lock = threading.Lock()
    status_online: bool | None = None
    status_error = ""
    last_success_at: float | None = None

    def publish_message(topic: str, payload: Mapping[str, Any]) -> None:
        with publish_lock:
            node.publish(topic, dict(payload))

    def publish_cloud_status(
        online: bool,
        error: str = "",
        *,
        force: bool = False,
    ) -> bool:
        """发布云端状态；返回连接状态是否发生变化。"""
        nonlocal status_online, status_error, last_success_at

        now = time.time()
        error = str(error)[:300]
        with status_lock:
            changed = (
                status_online is None
                or bool(online) != status_online
                or error != status_error
            )
            status_online = bool(online)
            status_error = error
            if online:
                last_success_at = now

            payload = {
                "source": "node_web",
                "enabled": settings.enabled,
                "online": bool(online),
                "timestamp": now,
                "last_success_at": last_success_at,
                "last_error": error,
            }

        if force or changed:
            try:
                publish_message(status_topic, payload)
            except Exception as exc:
                log(f"[WEB] 发布云端状态失败：{exc}")
        return changed

    execution_condition = threading.Condition()
    reset_condition = threading.Condition()
    execution_results: Dict[str, Dict[str, Any]] = {}
    reset_results: Dict[str, Dict[str, Any]] = {}
    waiting_execution: set[str] = set()
    waiting_reset: set[str] = set()

    def on_executed(message: Any) -> None:
        if not isinstance(message, dict):
            return
        command_id = str(message.get("cloud_command_id", ""))
        if not command_id:
            return
        with execution_condition:
            if command_id not in waiting_execution:
                return
            execution_results[command_id] = dict(message)
            execution_condition.notify_all()

    def on_reset_result(message: Any) -> None:
        if not isinstance(message, dict):
            return
        command_id = str(message.get("cloud_command_id", ""))
        if not command_id:
            return
        with reset_condition:
            if command_id not in waiting_reset:
                return
            reset_results[command_id] = dict(message)
            reset_condition.notify_all()

    node.register_subscriber(
        topics["actuator_executed"],
        on_executed,
    )
    node.register_subscriber(
        topics["counter_reset_result"],
        on_reset_result,
    )

    def wait_result(
        condition: threading.Condition,
        table: Dict[str, Dict[str, Any]],
        waiting: set[str],
        command_id: str,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + settings.execution_wait_s
        with condition:
            try:
                while not stop_event.is_set():
                    result = table.pop(command_id, None)
                    if result is not None:
                        return result
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    condition.wait(timeout=min(remaining, 0.5))
                return None
            finally:
                waiting.discard(command_id)
                table.pop(command_id, None)

    def shutdown(*_: Any) -> None:
        stop_event.set()
        with execution_condition:
            execution_condition.notify_all()
        with reset_condition:
            reset_condition.notify_all()
        try:
            node.close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not settings.enabled:
        def disabled_status_loop() -> None:
            while not stop_event.is_set():
                publish_cloud_status(
                    False,
                    "云端功能已在 config.yaml 中禁用",
                    force=True,
                )
                stop_event.wait(2.0)

        thread = threading.Thread(
            target=disabled_status_loop,
            name="web-disabled-status",
            daemon=True,
        )
        log("[WEB] 云端功能已禁用，节点仅保持本地状态发布")
        thread.start()
        try:
            node.spin()
        finally:
            shutdown()
            thread.join(timeout=1.0)
        return

    client = SmartBinClient(settings)
    # Web 节点只读 count.yaml，遇到错误时绝不移动或覆盖文件。
    store = CountStore(config, allow_repair=False)
    manual_topic = node.register_publisher(topics["manual_request"])
    reset_topic = node.register_publisher(topics["counter_reset"])

    if settings.server.startswith("http://"):
        log(
            "[WEB] 警告：当前使用 HTTP，设备密钥和控制命令未加密；"
            "比赛网络允许时建议改为 HTTPS"
        )

    def mark_request_success() -> None:
        changed = publish_cloud_status(True, "", force=True)
        if changed:
            log("[WEB] 云端连接已恢复")

    def mark_request_failure(prefix: str, exc: Exception) -> None:
        message = str(exc)
        changed = publish_cloud_status(False, message, force=True)
        if changed:
            log(f"[WEB] {prefix}：{message}")

    def report_loop() -> None:
        last_logged_revision = -1
        last_heartbeat_log = 0.0

        while not stop_event.is_set():
            started = time.monotonic()
            try:
                data = store.read()
                client.report(data["counts"])
                mark_request_success()

                revision = int(data.get("metadata", {}).get("revision", 0))
                now = time.monotonic()
                if (
                    revision != last_logged_revision
                    or now - last_heartbeat_log >= 30.0
                ):
                    counts = data["counts"]
                    log(
                        "[WEB] 上报成功 "
                        f"可回收={counts['recyclable']} "
                        f"厨余={counts['kitchen']} "
                        f"有害={counts['hazardous']} "
                        f"其他={counts['other']}"
                    )
                    last_logged_revision = revision
                    last_heartbeat_log = now
            except Exception as exc:
                mark_request_failure("上报失败", exc)

            elapsed = time.monotonic() - started
            stop_event.wait(
                max(0.05, settings.report_interval_s - elapsed)
            )

    def manual_loop() -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                commands = client.get_commands("manual", limit=1)
                mark_request_success()

                for command in commands:
                    command_id = str(command.get("command_id", "")).strip()
                    if not command_id:
                        continue

                    payload = command.get("payload") or {}
                    if not isinstance(payload, Mapping):
                        client.ack(command_id, False, "manual payload 必须是对象")
                        continue

                    code = str(payload.get("code", "")).strip()
                    raw_a = payload.get("a")
                    a = by_code.get(code)
                    if a is None and raw_a is not None:
                        try:
                            a = int(raw_a)
                        except (TypeError, ValueError):
                            a = None

                    if a not in by_a:
                        client.ack(
                            command_id,
                            False,
                            f"非法 manual 参数：code={code}, a={raw_a}",
                        )
                        continue

                    request = {
                        "a": int(a),
                        "x": -1,
                        "y": -1,
                        "code": by_a[a]["code"],
                        "source": "cloud_admin",
                        "timestamp": time.time(),
                        "request_id": f"{time.time_ns()}-cloud-a{a}",
                        "cloud_command_id": command_id,
                    }

                    with execution_condition:
                        execution_results.pop(command_id, None)
                        waiting_execution.add(command_id)
                    publish_message(manual_topic, request)
                    log(f"[WEB] 手动命令 -> a={a} id={command_id}")

                    result = wait_result(
                        execution_condition,
                        execution_results,
                        waiting_execution,
                        command_id,
                    )
                    if result is None:
                        client.ack(command_id, False, "GPIO 执行等待超时")
                        log(f"[WEB] 手动命令超时 ACK id={command_id}")
                        continue

                    success = bool(result.get("success", False))
                    error = str(result.get("error", ""))
                    message = (
                        f"a={a} GPIO{result.get('gpio')} 执行成功"
                        if success
                        else f"a={a} 执行失败：{error}"
                    )
                    client.ack(command_id, success, message)
                    log(
                        f"[WEB] 手动命令 ACK success={success} "
                        f"id={command_id}"
                    )
                    mark_request_success()
            except Exception as exc:
                mark_request_failure("手动命令轮询失败", exc)

            elapsed = time.monotonic() - started
            stop_event.wait(
                max(0.05, settings.command_poll_interval_s - elapsed)
            )

    def reset_loop() -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                commands = client.get_commands("reset", limit=1)
                mark_request_success()

                for command in commands:
                    command_id = str(command.get("command_id", "")).strip()
                    if not command_id:
                        continue

                    payload = command.get("payload") or {}
                    if not isinstance(payload, Mapping):
                        client.ack(command_id, False, "reset payload 必须是对象")
                        continue

                    if payload.get("target") is not None:
                        target = payload.get("target")
                    elif payload.get("code") is not None:
                        target = payload.get("code")
                    elif payload.get("a") is not None:
                        target = payload.get("a")
                    else:
                        target = "all"

                    request = {
                        "target": target,
                        "source": "cloud_admin",
                        "timestamp": time.time(),
                        "request_id": f"{time.time_ns()}-cloud-reset",
                        "cloud_command_id": command_id,
                    }

                    with reset_condition:
                        reset_results.pop(command_id, None)
                        waiting_reset.add(command_id)
                    publish_message(reset_topic, request)
                    log(f"[WEB] 清零命令 -> target={target} id={command_id}")

                    result = wait_result(
                        reset_condition,
                        reset_results,
                        waiting_reset,
                        command_id,
                    )
                    if result is None:
                        client.ack(command_id, False, "计数清零等待超时")
                        log(f"[WEB] 清零命令超时 ACK id={command_id}")
                        continue

                    success = bool(result.get("success", False))
                    message = (
                        "清零完成"
                        if success
                        else str(result.get("error", "清零失败"))
                    )
                    client.ack(command_id, success, message)
                    log(
                        f"[WEB] 清零 ACK success={success} "
                        f"id={command_id}"
                    )
                    mark_request_success()
            except Exception as exc:
                mark_request_failure("清零命令轮询失败", exc)

            elapsed = time.monotonic() - started
            stop_event.wait(
                max(0.05, settings.command_poll_interval_s - elapsed)
            )

    threads = [
        threading.Thread(
            target=report_loop,
            name="web-report",
            daemon=True,
        ),
        threading.Thread(
            target=manual_loop,
            name="web-manual",
            daemon=True,
        ),
        threading.Thread(
            target=reset_loop,
            name="web-reset",
            daemon=True,
        ),
    ]

    log(f"[WEB] 云端={settings.server} 设备={settings.device_id}")
    publish_cloud_status(False, "等待首次连接", force=True)

    for thread in threads:
        thread.start()

    try:
        node.spin()
    finally:
        shutdown()
        for thread in threads:
            thread.join(timeout=settings.timeout_s + 0.5)
