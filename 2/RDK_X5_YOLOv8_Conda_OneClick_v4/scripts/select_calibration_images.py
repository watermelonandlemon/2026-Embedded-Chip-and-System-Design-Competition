#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="原始图片目录，可递归搜索")
    parser.add_argument("--dest", default="./calibration_images")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--clear", action="store_true", help="先清空目标目录中的旧图片")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if args.count < 1:
        raise ValueError("count 必须大于 0")

    candidates = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    valid = [path for path in candidates if cv2.imread(str(path), cv2.IMREAD_COLOR) is not None]
    if not valid:
        raise RuntimeError(f"没有找到可读取的图片：{source}")

    if args.clear and dest.exists():
        for old_file in dest.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in IMAGE_SUFFIXES:
                old_file.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    count = min(args.count, len(valid))
    indexes = np.linspace(0, len(valid) - 1, num=count, dtype=int)
    selected = [valid[int(index)] for index in indexes]

    for index, source_path in enumerate(selected, start=1):
        target = dest / f"{index:03d}_{source_path.name}"
        shutil.copy2(source_path, target)
        print(f"{source_path} -> {target}")

    print(f"已选择 {len(selected)} 张图片，保存到：{dest}")


if __name__ == "__main__":
    main()
