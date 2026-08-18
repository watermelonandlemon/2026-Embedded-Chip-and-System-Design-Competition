#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目公共能力：路径、简化 YAML、配置、miniROS 与全局类别映射。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
LOG_DIR = ROOT_DIR / "log"
DEFAULT_CONFIG_PATH = SRC_DIR / "config.yaml"


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """读取本项目使用的纯映射 YAML；不依赖 PyYAML。"""
    target = Path(path).expanduser().resolve()
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-2, root)]
    text = target.read_text(encoding="utf-8")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        if "\t" in leading:
            raise ValueError(f"{target}: 第 {line_number} 行不能使用 Tab 缩进")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"{target}: 第 {line_number} 行必须使用偶数空格缩进")
        content = raw_line.strip()
        if ":" not in content:
            raise ValueError(f"{target}: 第 {line_number} 行缺少冒号")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{target}: 第 {line_number} 行键名为空")

        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        raw_value = raw_value.strip()
        if not raw_value:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml(data: Mapping[str, Any], indent: int = 0) -> str:
    """将嵌套映射写成稳定、易读的 YAML 文本。"""
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ValueError("YAML 键必须是非空字符串")
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.append(dump_yaml(value, indent + 2).rstrip("\n"))
        elif isinstance(value, (list, tuple, set)):
            raise ValueError("本项目 YAML 不允许列表类型")
        else:
            lines.append(f"{prefix}{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    config["_root_dir"] = str(ROOT_DIR)
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def create_node(name: str, config: Mapping[str, Any]):
    try:
        from miniROS import Node
    except ImportError as exc:
        raise RuntimeError("miniROS 未安装到 /usr/bin/python3 环境") from exc

    mini = config["miniros"]
    return Node(
        name,
        host=str(mini.get("host", "127.0.0.1")),
        port=int(mini.get("port", 8765)),
        connect_timeout=float(mini.get("connect_timeout_s", 5.0)),
        auto_reconnect=True,
        reconnect_interval=float(mini.get("reconnect_interval_s", 1.0)),
    )


def garbage_maps(config: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    by_a: dict[int, dict[str, Any]] = {}
    by_code: dict[str, int] = {}
    for field, raw in config["garbage"].items():
        item = dict(raw)
        item["field"] = field
        item["a"] = int(item["a"])
        item["gpio"] = int(item["gpio"])
        item["code"] = str(item["code"])
        by_a[item["a"]] = item
        by_code[item["code"]] = item["a"]
    expected = {-1, 0, 1, 2, 3}
    if set(by_a) != expected - {-1}:
        raise ValueError(f"全局垃圾 a 必须完整定义为 0,1,2,3，当前为 {sorted(by_a)}")
    return by_a, by_code


def yolo_to_garbage(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for key, raw in config["vision"]["yolo_classes"].items():
        class_id = int(str(key).replace("class_", ""))
        result[class_id] = {
            "name": str(raw["name"]),
            "a": int(raw["a"]),
        }
    return result
