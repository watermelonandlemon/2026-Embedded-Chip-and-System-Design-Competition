#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip()


def file_record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-pt", required=True)
    parser.add_argument("--source-onnx", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    pt_path = Path(args.source_pt).resolve()
    onnx_path = Path(args.source_onnx).resolve()

    artifacts = [
        file_record(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "ultralytics": package_version("ultralytics"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "rdkx5-yolo-mapper": package_version("rdkx5-yolo-mapper"),
            "hb_mapper": command_output(["hb_mapper", "--version"]),
        },
        "sources": {
            "pt": file_record(pt_path),
            "onnx": file_record(onnx_path),
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"转换清单：{manifest_path}")


if __name__ == "__main__":
    main()
