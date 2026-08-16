#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    raise TypeError(f"无法识别的类别名称类型：{type(names)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    pt_path = Path(args.pt).resolve()
    labels_path = Path(args.labels).resolve()
    metadata_path = Path(args.metadata).resolve()

    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    model = YOLO(str(pt_path))
    names = normalize_names(model.names)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    metadata = {
        "source_pt": str(pt_path),
        "source_pt_sha256": sha256_file(pt_path),
        "task": getattr(model, "task", "unknown"),
        "class_count": len(names),
        "class_names": names,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"任务类型: {metadata['task']}")
    print(f"类别数量: {len(names)}")
    for index, name in enumerate(names):
        print(f"  {index}: {name}")
    print(f"类别文件: {labels_path}")


if __name__ == "__main__":
    main()
