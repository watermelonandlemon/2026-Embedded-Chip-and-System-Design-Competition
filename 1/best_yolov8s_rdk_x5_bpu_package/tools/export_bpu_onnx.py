from __future__ import annotations
import sys, types, os
from pathlib import Path
import torch
from torch import nn

# -----------------------------------------------------------------------------
# Minimal Ultralytics-compatible class definitions required to unpickle the
# uploaded YOLOv8 checkpoint. The checkpoint already contains the full module
# tree and parameters; these classes provide the forward implementations.
# -----------------------------------------------------------------------------

# Create module hierarchy before torch.load().
for name in [
    'ultralytics', 'ultralytics.nn', 'ultralytics.nn.tasks',
    'ultralytics.nn.modules', 'ultralytics.nn.modules.conv',
    'ultralytics.nn.modules.block', 'ultralytics.nn.modules.head',
]:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod


class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        return self.cv2(torch.cat((y0, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def forward(self, x):
        # Included for checkpoint compatibility. It is bypassed by the BPU
        # optimized Detect forward used below.
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    def forward(self, x):
        # D-Robotics Model Zoo BPU-friendly protocol: [cls, box] * 3,
        # channel-last outputs for stride 8/16/32.
        result = []
        for i in range(self.nl):
            result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
            result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        return tuple(result)


class DetectionModel(nn.Module):
    def forward(self, x):
        y = []
        for m in self.model:
            f = getattr(m, 'f', -1)
            if f != -1:
                if isinstance(f, int):
                    x = y[f]
                else:
                    x = [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x if getattr(m, 'i', -1) in self.save else None)
        return x


# Bind classes to their original qualified module paths for pickle loading.
DetectionModel.__module__ = 'ultralytics.nn.tasks'
Conv.__module__ = 'ultralytics.nn.modules.conv'
Concat.__module__ = 'ultralytics.nn.modules.conv'
C2f.__module__ = 'ultralytics.nn.modules.block'
Bottleneck.__module__ = 'ultralytics.nn.modules.block'
SPPF.__module__ = 'ultralytics.nn.modules.block'
DFL.__module__ = 'ultralytics.nn.modules.block'
Detect.__module__ = 'ultralytics.nn.modules.head'

sys.modules['ultralytics.nn.tasks'].DetectionModel = DetectionModel
sys.modules['ultralytics.nn.modules.conv'].Conv = Conv
sys.modules['ultralytics.nn.modules.conv'].Concat = Concat
sys.modules['ultralytics.nn.modules.block'].C2f = C2f
sys.modules['ultralytics.nn.modules.block'].Bottleneck = Bottleneck
sys.modules['ultralytics.nn.modules.block'].SPPF = SPPF
sys.modules['ultralytics.nn.modules.block'].DFL = DFL
sys.modules['ultralytics.nn.modules.head'].Detect = Detect


def load_model(pt_path: str | os.PathLike) -> DetectionModel:
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    if model is None:
        raise RuntimeError('Checkpoint does not contain model/ema.')
    model = model.float().eval()
    return model


def export_onnx(pt_path: str | os.PathLike, onnx_path: str | os.PathLike) -> None:
    model = load_model(pt_path)
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(dummy)
    expected = [
        (1, 80, 80, 4), (1, 80, 80, 64),
        (1, 40, 40, 4), (1, 40, 40, 64),
        (1, 20, 20, 4), (1, 20, 20, 64),
    ]
    actual = [tuple(o.shape) for o in outputs]
    if actual != expected:
        raise RuntimeError(f'Unexpected model outputs: {actual}; expected: {expected}')

    # The legacy PyTorch exporter already produces a valid ONNX protobuf in C++.
    # It normally imports onnx only to append custom ONNX-script functions. This
    # model has none, so bypass that optional step in this offline environment.
    from torch.onnx._internal.torchscript_exporter import onnx_proto_utils
    onnx_proto_utils._add_onnxscript_fn = lambda model_bytes, custom_opsets: model_bytes

    output_names = [
        's8_cls', 's8_box', 's16_cls', 's16_box', 's32_cls', 's32_box'
    ]
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=['images'],
        output_names=output_names,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        export_params=True,
        dynamo=False,
    )

    out = Path(onnx_path)
    if not out.exists() or out.stat().st_size < 1_000_000:
        raise RuntimeError('ONNX export failed or produced an unexpectedly small file.')

    print('MODEL_NAMES=', model.names)
    print('MODEL_STRIDE=', model.stride.tolist())
    print('OUTPUT_SHAPES=', actual)
    print('ONNX_PATH=', out)
    print('ONNX_SIZE=', out.stat().st_size)


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else '/mnt/data/best(2).pt'
    dst = sys.argv[2] if len(sys.argv) > 2 else '/mnt/data/best_yolov8s_bpu.onnx'
    export_onnx(src, dst)
