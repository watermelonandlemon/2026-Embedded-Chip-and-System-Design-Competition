#!/usr/bin/env python3
"""Export Ultralytics YOLO models to RDK X5-friendly static ONNX.

Core head-output transformations are derived from D-Robotics RDK Model Zoo's
Apache-2.0 licensed ultralytics_yolo conversion example.
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Classify, Detect, OBB, Pose, Segment

try:
    from ultralytics.nn.modules.head import v10Detect
except ImportError:  # Older Ultralytics versions
    v10Detect = None  # type: ignore[assignment]

try:
    from ultralytics.nn.modules.block import AAttn, Attention
except ImportError:  # Models without these modules do not need the patch.
    AAttn = None  # type: ignore[assignment]
    Attention = None  # type: ignore[assignment]


def classify_forward(self, x):
    x = torch.cat(x, 1) if isinstance(x, list) else x
    return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


def detect_forward(self, x):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.cv3[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[index](x[index]).permute(0, 2, 3, 1).contiguous())
    return outputs


def v10_detect_forward(self, x):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.one2one_cv3[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.one2one_cv2[index](x[index]).permute(0, 2, 3, 1).contiguous())
    return outputs


def segment_forward(self, x):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.cv3[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[index](x[index]).permute(0, 2, 3, 1).contiguous())
    outputs.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())
    return outputs


def pose_forward(self, x):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.cv3[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[index](x[index]).permute(0, 2, 3, 1).contiguous())
    return outputs


def obb_forward(self, x):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.cv3[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[index](x[index]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[index](x[index]).permute(0, 2, 3, 1).contiguous())
    return outputs


def attention_forward(self, x):
    batch, channels, height, width = x.shape
    positions = height * width
    qkv = self.qkv(x)
    q, k, v = qkv.view(
        batch,
        self.num_heads,
        self.key_dim * 2 + self.head_dim,
        positions,
    ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.permute(0, 3, 1, 2).contiguous()
    max_attn = attn.max(dim=1, keepdim=True).values
    exp_attn = torch.exp(attn - max_attn)
    attn = exp_attn / exp_attn.sum(dim=1, keepdim=True)
    attn = attn.permute(0, 2, 3, 1).contiguous()
    out = (v @ attn.transpose(-2, -1)).view(batch, channels, height, width)
    out = out + self.pe(v.reshape(batch, channels, height, width))
    return self.proj(out)


def aattn_forward(self, x):
    batch, channels, height, width = x.shape
    positions = height * width
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(batch * self.area, positions // self.area, channels * 3)
    work_batch, work_positions, _ = qkv.shape
    q, k, v = (
        qkv.view(work_batch, work_positions, self.num_heads, self.head_dim * 3)
        .permute(0, 2, 3, 1)
        .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
    )
    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.permute(0, 3, 1, 2).contiguous()
    max_attn = attn.max(dim=1, keepdim=True).values
    exp_attn = torch.exp(attn - max_attn)
    attn = exp_attn / exp_attn.sum(dim=1, keepdim=True)
    attn = attn.permute(0, 2, 3, 1).contiguous()
    out = v @ attn.transpose(-2, -1)
    out = out.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    if self.area > 1:
        out = out.reshape(batch, positions, channels)
        v = v.reshape(batch, positions, channels)
    out = out.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
    return self.proj(out + self.pe(v))


def patch_model(module) -> int:
    patched = 0
    for name, child in module.named_children():
        replacement = None
        if type(child) is Classify:
            replacement = classify_forward
        elif type(child) is Detect:
            replacement = detect_forward
        elif v10Detect is not None and type(child) is v10Detect:
            replacement = v10_detect_forward
        elif type(child) is Segment:
            replacement = segment_forward
        elif type(child) is Pose:
            replacement = pose_forward
        elif type(child) is OBB:
            replacement = obb_forward
        elif AAttn is not None and type(child) is AAttn:
            replacement = aattn_forward
        elif Attention is not None and type(child) is Attention:
            replacement = attention_forward

        if replacement is not None:
            child.forward = types.MethodType(replacement, child)
            patched += 1
            print(f"[Patch] {name}: {type(child).__name__}")

        patched += patch_model(child)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Ultralytics .pt model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    pt_path = Path(args.pt).resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)
    if args.opset != 11:
        print(f"[警告] RDK X5 推荐 opset 11，当前设置为 {args.opset}")

    model = YOLO(str(pt_path))
    if getattr(model, "task", None) != "detect":
        print(f"[警告] 当前任务是 {getattr(model, 'task', 'unknown')}，一键脚本默认按 Detect 验证。")

    patched = patch_model(model.model.model)
    if patched == 0:
        raise RuntimeError("没有找到可替换的 Ultralytics Head，可能是版本或模型结构不兼容。")

    result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=False,
        dynamic=False,
        batch=1,
        nms=False,
        device="cpu",
    )

    onnx_path = Path(str(result)).resolve()
    if not onnx_path.is_file():
        expected = pt_path.with_suffix(".onnx")
        if expected.is_file():
            onnx_path = expected
        else:
            raise FileNotFoundError(f"导出结束，但没有找到 ONNX：{result}")

    print(f"已替换模块数量: {patched}")
    print(f"ONNX 输出: {onnx_path}")


if __name__ == "__main__":
    main()
