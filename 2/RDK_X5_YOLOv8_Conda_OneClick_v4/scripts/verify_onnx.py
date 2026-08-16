#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def dim_value(dim) -> int | None:
    return int(dim.dim_value) if dim.HasField("dim_value") else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--expected-size", type=int, default=640)
    parser.add_argument("--expected-opset", type=int, default=11)
    parser.add_argument("--expected-output-count", type=int, default=6)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    report_path = Path(args.report).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    default_opset = opsets.get("ai.onnx")
    if default_opset != args.expected_opset:
        raise RuntimeError(f"ONNX opset={default_opset}，预期={args.expected_opset}")

    if len(model.graph.input) != 1:
        raise RuntimeError(f"模型输入数量为 {len(model.graph.input)}，预期为 1")

    graph_input = model.graph.input[0]
    shape = [dim_value(dim) for dim in graph_input.type.tensor_type.shape.dim]
    if shape != [1, 3, args.expected_size, args.expected_size]:
        raise RuntimeError(
            f"ONNX 输入 shape={shape}，预期=[1, 3, {args.expected_size}, {args.expected_size}]"
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, {input_meta.name: dummy})

    if len(outputs) != args.expected_output_count:
        raise RuntimeError(
            f"ONNX 输出数量={len(outputs)}，YOLOv8 Detect RDK 导出预期={args.expected_output_count}。"
            "请确认使用了 export_monkey_patch.py。"
        )

    output_info = []
    print(f"ONNX: {onnx_path}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset: {opsets}")
    print(f"Input: name={input_meta.name}, shape={shape}, type={input_meta.type}")
    for index, array in enumerate(outputs):
        info = {
            "index": index,
            "name": session.get_outputs()[index].name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()),
            "max": float(array.max()),
        }
        output_info.append(info)
        print(
            f"Output[{index}]: name={info['name']}, shape={info['shape']}, "
            f"dtype={info['dtype']}"
        )

    # YOLOv8 Detect 的 Monkey Patch 输出顺序应为 [cls, box] * 3，且使用 NHWC。
    for index, info in enumerate(output_info):
        shape_i = info["shape"]
        if len(shape_i) != 4:
            raise RuntimeError(f"Output[{index}] 不是 4 维张量：{shape_i}")
        if index % 2 == 1 and shape_i[-1] != 64:
            raise RuntimeError(f"Output[{index}] 应为 DFL box 64 通道，实际：{shape_i[-1]}")

    inferred_classes = output_info[0]["shape"][-1]
    report = {
        "onnx": str(onnx_path),
        "ir_version": int(model.ir_version),
        "opsets": opsets,
        "input": {
            "name": input_meta.name,
            "shape": shape,
            "type": input_meta.type,
        },
        "output_count": len(output_info),
        "inferred_class_count": inferred_classes,
        "outputs": output_info,
        "status": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"推断类别数: {inferred_classes}")
    print(f"ONNX 检查通过，报告：{report_path}")


if __name__ == "__main__":
    main()
