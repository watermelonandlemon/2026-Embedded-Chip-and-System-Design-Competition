#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "未安装"


def main() -> int:
    errors: list[str] = []
    print("===== RDK X5 转换环境检查 =====")
    print(f"系统: {platform.platform()}")
    print(f"架构: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Conda 环境: {os.environ.get('CONDA_DEFAULT_ENV', '未激活')}")

    if platform.system() != "Linux":
        errors.append("必须在 Linux 中运行。")
    if platform.machine() not in {"x86_64", "AMD64"}:
        errors.append("必须使用 x86_64 Linux。")
    if sys.version_info[:2] != (3, 10):
        errors.append("rdkx5-yolo-mapper 要求 Python 3.10。")
    if not os.environ.get("CONDA_PREFIX"):
        errors.append("当前没有激活 Conda 环境。")

    expected_versions = {
        "rdkx5-yolo-mapper": "1.0.0",
        "numpy": "1.23.0",
        "opencv-python": "4.6.0.66",
        "onnx": "1.15.0",
        "onnxruntime": "1.16.2",
        "protobuf": "3.20.3",
        "scikit-image": "0.19.0",
    }

    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("protobuf", "google.protobuf"),
        ("scikit-image", "skimage"),
        ("rdkx5-yolo-mapper", None),
    ]
    for dist_name, module_name in packages:
        installed_version = package_version(dist_name)
        print(f"{dist_name}: {installed_version}")
        expected = expected_versions.get(dist_name)
        if expected is not None and installed_version != expected:
            errors.append(
                f"{dist_name} 版本错误：当前 {installed_version}，要求 {expected}"
            )
        if module_name:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"无法导入 {module_name}: {exc}")

    hb_mapper = shutil.which("hb_mapper")
    print(f"hb_mapper: {hb_mapper or '未找到'}")
    if not hb_mapper:
        errors.append("没有找到 hb_mapper，请安装 rdkx5-yolo-mapper。")
    else:
        result = subprocess.run(
            [hb_mapper, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout.strip() or f"hb_mapper 返回码: {result.returncode}")
        if result.returncode != 0:
            errors.append("hb_mapper --version 执行失败。")

    if errors:
        print("\n===== 检查失败 =====", file=sys.stderr)
        for item in errors:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("===== 环境检查通过 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
