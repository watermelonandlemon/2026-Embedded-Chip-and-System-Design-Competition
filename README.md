# 2026 嵌入式系统与芯片设计竞赛 — YOLOv8s BPU 模型转换

本分支保存 **2026 嵌入式系统与芯片设计竞赛** 项目中 YOLO 模型面向 **D-Robotics RDK X5 BPU** 的模型转换、量化编译与部署测试程序。

本项目使用的原始模型为：

```text
YOLOv8s
```

训练完成后得到 PyTorch 权重：

```text
best.pt
```

随后将模型转换为适用于 RDK X5 Bayes-e BPU 的：

```text
.pt
 ↓
ONNX
 ↓
INT8 Calibration
 ↓
hb_mapper
 ↓
.bin
```

最终 `.bin` 模型可在 RDK X5 上通过 `hbm_runtime` 调用 BPU 进行硬件加速推理。

> Branch: `BPU_Part_Code`
> Source Model: YOLOv8s
> Input Size: `640 × 640`
> Target: RDK X5 / Bayes-e BPU
> Runtime Input: NV12
> Quantization: INT8

---

## 分支用途

本分支主要解决：

* YOLOv8s `.pt` 模型导出 ONNX
* 将 Ultralytics YOLOv8 检测头修改为 BPU 友好结构
* 生成 RDK X5 支持的静态输入 ONNX
* 使用校准图片执行 INT8 量化
* 使用 `hb_mapper` 编译 Bayes-e BPU `.bin`
* 在 RDK X5 上使用 `hbm_runtime` 进行 BPU 推理
* 对比两种 YOLOv8 → BPU 转换方案
* 为不同 `.bin` 输出结构提供对应后处理方法

---

# 目录结构

本分支目前包含两套独立的模型转换方案：

```text
BPU_Part_Code/
│
├── README.md
│
├── 1/
│   ├── readme.md
│   ├── best_yolov8s_rdk_x5_bpu_package.zip
│   │
│   └── best_yolov8s_rdk_x5_bpu_package/
│       ├── source/
│       │   └── best_yolov8s.pt
│       │
│       ├── model/
│       │   ├── best_yolov8s_bpu.onnx
│       │   └── best_yolov8s_bpu_bayese_640x640_nv12.bin
│       │
│       ├── calibration_images/
│       ├── conversion/
│       ├── runtime/
│       ├── tools/
│       └── metadata/
│
└── 2/
    ├── readme.md
    ├── RDK_X5_YOLOv8_Conda_OneClick_v4.zip
    │
    └── RDK_X5_YOLOv8_Conda_OneClick_v4/
        ├── model/
        │   ├── best.pt
        │   └── best.onnx
        │
        ├── calibration_images/
        ├── scripts/
        ├── output/
        │   └── best_bayese_640x640_nv12.bin
        │
        ├── setup_conda.sh
        ├── convert.sh
        ├── run_all.sh
        └── config.sh
```

---

# 两套转换方案

## 方案 1：定制 BPU-Friendly YOLOv8s 导出

目录：

```text
1/best_yolov8s_rdk_x5_bpu_package/
```

方案 1 不直接依赖完整 Ultralytics 推理流程，而是在导出脚本中重新构造模型所需要的最小 Ultralytics 兼容模块，并修改 YOLOv8 `Detect` 检测头的 `forward()`。

核心思想为：

```text
YOLOv8s
   │
   ├── Backbone
   ├── Neck
   │
   └── Detect
         │
         ├── Stride 8  → cls + DFL box
         ├── Stride 16 → cls + DFL box
         └── Stride 32 → cls + DFL box
```

不再让 Ultralytics 在模型内部执行完整的：

```text
DFL Decode
Anchor Decode
Sigmoid
NMS
```

而是直接将三个尺度的分类和回归结果输出给 CPU 后处理。

因此转换后的模型非常适合 RDK X5：

```text
BPU
 └── CNN 前向计算

CPU
 ├── Sigmoid
 ├── DFL Softmax
 ├── Bounding Box Decode
 └── NMS
```

### 方案 1 输出

模型共包含 6 个输出：

| 输出        | Stride | 含义        | Shape          |
| --------- | -----: | --------- | -------------- |
| `s8_cls`  |      8 | 分类 Logits | `[1,80,80,4]`  |
| `s8_box`  |      8 | DFL Box   | `[1,80,80,64]` |
| `s16_cls` |     16 | 分类 Logits | `[1,40,40,4]`  |
| `s16_box` |     16 | DFL Box   | `[1,40,40,64]` |
| `s32_cls` |     32 | 分类 Logits | `[1,20,20,4]`  |
| `s32_box` |     32 | DFL Box   | `[1,20,20,64]` |

其中：

```text
4
```

表示当前模型共有 4 个目标类别。

而：

```text
64 = 4 × 16
```

表示 YOLOv8 的 DFL：

```text
Left
Top
Right
Bottom
```

四个方向，每个方向使用：

```text
reg_max = 16
```

个离散概率值进行边界框回归。

---

# 方案 2：Ultralytics + Monkey Patch 一键转换

目录：

```text
2/RDK_X5_YOLOv8_Conda_OneClick_v4/
```

方案 2 直接安装并使用 Ultralytics YOLO 环境，然后在模型导出阶段动态替换：

```text
Detect.forward()
```

使其生成 RDK X5 更适合处理的检测头结构。

同时该工具不仅考虑 YOLOv8 Detect，还预留了对：

```text
Detect
Segment
Pose
OBB
Classify
```

等 Ultralytics 网络头结构的修改能力，因此相比方案 1 更通用。

整体流程为：

```text
best.pt
   │
   ▼
Ultralytics YOLO
   │
Monkey Patch
   │
   ▼
best.onnx
   │
INT8 Calibration
   │
hb_mapper
   │
   ▼
best_bayese_640x640_nv12.bin
```

方案 2 提供完整的一键转换流程：

```bash
bash run_all.sh
```

脚本负责：

```text
检查环境
    ↓
建立 Conda 环境
    ↓
读取 best.pt
    ↓
获取类别
    ↓
导出 ONNX
    ↓
检查 ONNX
    ↓
准备 Calibration 数据
    ↓
运行 hb_mapper
    ↓
生成 .bin
```

---

# 两个 `.bin` 的结构区别

两个压缩包中已经分别包含转换完成的 BPU 模型：

### 方案 1

```text
best_yolov8s_bpu_bayese_640x640_nv12.bin
```

文件大小：

```text
11,866,889 Bytes
```

SHA256：

```text
70c36131a916045596e37e8c814cb798b3f163e6c4397ea0598d60dc2c152047
```

### 方案 2

```text
best_bayese_640x640_nv12.bin
```

文件大小：

```text
11,866,811 Bytes
```

SHA256：

```text
1191d0c351130b19e44aa8328060986cd397882bd345420b882029a73b122aa8
```

因此两个 `.bin` **并不是同一个二进制模型文件**。

---

## 但是二者的外部 I/O 协议基本一致

两个模型都使用：

```text
Input:
NV12 640 × 640

Outputs:
Stride 8  Classification
Stride 8  DFL Box

Stride 16 Classification
Stride 16 DFL Box

Stride 32 Classification
Stride 32 DFL Box
```

也就是：

```text
6 Outputs
```

其 Shape 都是：

```text
[1,80,80,4]
[1,80,80,64]

[1,40,40,4]
[1,40,40,64]

[1,20,20,4]
[1,20,20,64]
```

因此：

> 两套模型的区别主要不是最终检测结果的数据协议，而是 ONNX/BPU 内部计算图的构造方式不同。

---

# 核心结构差异：Slice 与 Split

这是两个转换方案最明显的内部结构差别之一。

## 方案 1

方案 1 自己定义了 YOLOv8 `C2f.forward()`：

```python
y = list(self.cv1(x).chunk(2, 1))
```

因此导出 ONNX 后，通道拆分操作会被展开为类似：

```text
Slice
Slice
Concat
```

在 `hb_mapper` 编译日志中可以看到大量：

```text
Slice
Slice_1
Concat
```

因此方案 1 的 BPU 图内部更接近：

```text
Conv
  │
  ├── Slice ──► ...
  │
  └── Slice ──► ...
         │
       Concat
```

---

## 方案 2

方案 2依赖实际安装的 Ultralytics 网络结构。

当前转换环境中的 C2f 使用的拆分方式会生成：

```text
Split
```

因此 BPU 编译日志中可以看到：

```text
Split
Concat
```

对应的内部结构更接近：

```text
Conv
  │
 Split
 ├──────► ...
 └──────► ...
     │
   Concat
```

---

## 因此两者虽然数学功能等价，但计算图并不完全相同

可以简单理解为：

```text
方案 1
C2f:
Conv → Slice + Slice → Bottleneck → Concat

方案 2
C2f:
Conv → Split → Bottleneck → Concat
```

两种结构最终实现的 YOLOv8 C2f 功能是一致的，但经过 ONNX 导出和 `hb_mapper` 编译以后：

* 节点类型不同
* 节点名称不同
* 部分 Tensor 名称不同
* BPU Graph 编排不同
* 最终 `.bin` 二进制内容不同

所以不能通过文件名判断二者是完全相同的模型。

---

# 输出 Tensor 名称的区别

方案 1 在导出时主动指定了具有实际含义的输出名称：

```text
s8_cls
s8_box
s16_cls
s16_box
s32_cls
s32_box
```

因此模型结构更容易人工查看和调试。

---

方案 2 的 ONNX 输出名称则主要来自 Ultralytics/PyTorch 导出图，例如：

```text
output0
326
334
342
350
358
```

虽然名称不同，但它们按照顺序分别对应：

```text
output[0] → stride 8 cls
output[1] → stride 8 box

output[2] → stride 16 cls
output[3] → stride 16 box

output[4] → stride 32 cls
output[5] → stride 32 box
```

因此方案 2 的运行程序**不建议硬编码 Tensor 名称**。

建议使用：

```python
output_names = runtime.output_names[model_name]
```

动态获取输出，并按照输出顺序解析。

---

# 两种模型结构对比

| 项目             | 方案 1             | 方案 2                     |
| -------------- | ---------------- | ------------------------ |
| 原始模型           | YOLOv8s          | YOLOv8s                  |
| 输入             | 640×640          | 640×640                  |
| BPU 输入         | NV12             | NV12                     |
| Target         | Bayes-e          | Bayes-e                  |
| Quantization   | INT8             | INT8                     |
| 输出数量           | 6                | 6                        |
| 检测头格式          | cls + DFL box    | cls + DFL box            |
| 输出排列           | NHWC             | NHWC                     |
| C2f 拆分         | `Slice`          | `Split`                  |
| 输出名称           | 语义名称             | 自动生成名称                   |
| 导出方式           | 自定义兼容模型          | Ultralytics Monkey Patch |
| Ultralytics 依赖 | 较弱               | 较强                       |
| 自动化程度          | 较高               | 更高                       |
| 通用性            | 针对 YOLOv8 Detect | 支持扩展多类 YOLO Head         |
| `.bin`         | 11,866,889 B     | 11,866,811 B             |

---

# 为什么不直接使用 YOLOv8 默认 ONNX 输出

Ultralytics 默认 YOLOv8 导出通常会包含更多后处理逻辑，或者形成类似：

```text
[1, 84, 8400]
```

类型的组合输出。

这种结构虽然适用于 CPU / CUDA / ONNX Runtime，但对于 RDK X5 BPU 并不是最理想的形式。

本项目采用：

```text
Classification Branch
+
DFL Regression Branch
```

分开输出。

让：

```text
卷积计算 → BPU
后处理   → CPU
```

这样可以减少不适合 BPU 的算子进入模型图，并且更符合 RDK Model Zoo 的 YOLO 部署方式。

---

# BPU 后处理原理

以 Stride 8 为例：

```text
cls:
[1,80,80,4]

box:
[1,80,80,64]
```

分类分支执行：

```text
logit
  ↓
Sigmoid
  ↓
Class Probability
```

回归分支执行：

```text
64
 ↓
reshape
 ↓
4 × 16
 ↓
Softmax
 ↓
DFL Expectation
 ↓
Left / Top / Right / Bottom
```

然后根据 Grid Anchor：

```text
anchor = (x + 0.5, y + 0.5)
```

计算：

```text
x1 = anchor_x - left
y1 = anchor_y - top
x2 = anchor_x + right
y2 = anchor_y + bottom
```

最后乘以对应：

```text
stride = 8 / 16 / 32
```

恢复成 640×640 输入图上的 Bounding Box。

---

# BPU 程序示例 — 方案 1

方案 1已经提供：

```text
1/best_yolov8s_rdk_x5_bpu_package/runtime/infer_image.py
```

RDK X5 上可以直接加载：

```python
import hbm_runtime

MODEL_PATH = "best_yolov8s_bpu_bayese_640x640_nv12.bin"

runtime = hbm_runtime.HB_HBMRuntime(MODEL_PATH)

model_name = runtime.model_names[0]

input_name = runtime.input_names[model_name][0]

output_names = list(runtime.output_names[model_name])

print("Model:", model_name)
print("Input:", input_name)
print("Outputs:", output_names)
```

输入准备完成以后：

```python
outputs = runtime.run({
    model_name: {
        input_name: nv12_data
    }
})
```

方案 1 的六个输出按照：

```python
s8_cls  = outputs[model_name][output_names[0]]
s8_box  = outputs[model_name][output_names[1]]

s16_cls = outputs[model_name][output_names[2]]
s16_box = outputs[model_name][output_names[3]]

s32_cls = outputs[model_name][output_names[4]]
s32_box = outputs[model_name][output_names[5]]
```

进行解析。

推荐不要完全依赖固定 Tensor 字符串，而是同时检查：

```python
print(runtime.output_names[model_name])
```

以实际 `.bin` 中保存的输出名称为准。

---

# BPU 程序示例 — 方案 2

方案 2的 ONNX 输出名称可能是：

```text
output0
326
334
342
350
358
```

因此程序不要写：

```python
outputs["s8_cls"]
```

而应该通过输出顺序处理：

```python
import hbm_runtime

MODEL_PATH = "best_bayese_640x640_nv12.bin"

runtime = hbm_runtime.HB_HBMRuntime(MODEL_PATH)

model_name = runtime.model_names[0]
input_name = runtime.input_names[model_name][0]
output_names = list(runtime.output_names[model_name])

print("Output names:")
for i, name in enumerate(output_names):
    print(i, name)

outputs = runtime.run({
    model_name: {
        input_name: nv12_data
    }
})

raw = outputs[model_name]

s8_cls  = raw[output_names[0]]
s8_box  = raw[output_names[1]]

s16_cls = raw[output_names[2]]
s16_box = raw[output_names[3]]

s32_cls = raw[output_names[4]]
s32_box = raw[output_names[5]]
```

后面的 DFL Decode 和 NMS 与方案 1 完全可以共用。

因此程序架构建议写成：

```text
                BPU Runtime
                     │
              6 Raw Outputs
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Stride 8   Stride 16  Stride 32
          │          │          │
      cls + box   cls + box   cls + box
          └──────────┼──────────┘
                     ▼
                 DFL Decode
                     ▼
                  Sigmoid
                     ▼
                    NMS
                     ▼
                Final Boxes
```

---

# 通用 DFL 解码示例

两种 `.bin` 均可以使用类似下面的后处理函数：

```python
import numpy as np

REG_MAX = 16

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def make_anchors(h, w):
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32) + 0.5,
        np.arange(w, dtype=np.float32) + 0.5,
        indexing="ij"
    )

    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


def decode_level(cls_output, box_output, stride):
    h = cls_output.shape[1]
    w = cls_output.shape[2]

    cls = cls_output.reshape(-1, 4)
    box = box_output.reshape(-1, 4, REG_MAX)

    scores = sigmoid(cls)

    distribution = softmax(box, axis=2)

    weights = np.arange(REG_MAX, dtype=np.float32)

    ltrb = np.sum(
        distribution * weights.reshape(1, 1, REG_MAX),
        axis=2
    )

    anchors = make_anchors(h, w)

    xy1 = anchors - ltrb[:, 0:2]
    xy2 = anchors + ltrb[:, 2:4]

    boxes = np.concatenate(
        [xy1, xy2],
        axis=1
    )

    boxes *= stride

    return boxes, scores
```

分别执行：

```python
boxes8, scores8 = decode_level(
    s8_cls,
    s8_box,
    8
)

boxes16, scores16 = decode_level(
    s16_cls,
    s16_box,
    16
)

boxes32, scores32 = decode_level(
    s32_cls,
    s32_box,
    32
)
```

之后合并：

```python
boxes = np.concatenate(
    [boxes8, boxes16, boxes32],
    axis=0
)

scores = np.concatenate(
    [scores8, scores16, scores32],
    axis=0
)
```

再执行置信度筛选与 NMS 即可得到最终检测结果。

---

# 模型输入

两套 `.bin` 均配置为：

```text
Runtime Input:
NV12

Resolution:
640 × 640

Batch:
1
```

训练阶段 ONNX 输入为：

```text
RGB
NCHW
1 × 3 × 640 × 640
float32
```

`hb_mapper` 编译后运行时则使用：

```text
NV12
```

因此 RDK X5 摄像头图像进入 BPU 前需要完成：

```text
Camera / BGR
     │
     ▼
Resize / Letterbox
     │
     ▼
BGR → NV12
     │
     ▼
hbm_runtime
     │
     ▼
BPU
```

---

# 模型类别

当前 YOLOv8s 模型共有 4 类：

| Class ID | Name      |
| -------: | --------- |
|      `0` | `zhituan` |
|      `1` | `pinggai` |
|      `2` | `dianchi` |
|      `3` | `jiaodai` |

即：

```text
纸团
瓶盖
电池
胶带
```

---

# 方案 1 转换方法

进入：

```bash
cd best_yolov8s_rdk_x5_bpu_package
```

创建环境：

```bash
conda create -n rdk_x5_mapper python=3.10 -y
conda activate rdk_x5_mapper
```

安装依赖：

```bash
pip install -r conversion/requirements-convert.txt
```

执行：

```bash
chmod +x conversion/convert_to_bin.sh
./conversion/convert_to_bin.sh
```

生成：

```text
model/
└── best_yolov8s_bpu_bayese_640x640_nv12.bin
```

---

# 方案 2 转换方法

首先将：

```text
best.pt
```

放入：

```text
model/
```

然后准备约：

```text
20 ～ 50
```

张有代表性的训练/测试图片放入：

```text
calibration_images/
```

推荐约：

```text
30
```

张。

进入目录：

```bash
cd RDK_X5_YOLOv8_Conda_OneClick_v4
```

直接执行：

```bash
bash run_all.sh
```

转换完成后生成：

```text
output/
└── best_bayese_640x640_nv12.bin
```

---

# 转换环境

模型转换并不需要在 RDK X5 开发板上完成。

推荐使用：

```text
x86_64 Linux
Ubuntu
Conda
Python
D-Robotics hb_mapper
```

可以部署在：

```text
本地 Linux 电脑
```

也可以部署在：

```text
Linux 云服务器
```

本项目实际使用香港 Linux 云服务器完成相关模型转换流程。

在具备正常国际网络访问能力的香港云服务器环境下，一般可以直接访问 Python、Conda、Ultralytics 等相关软件源和上游资源，不需要额外配置本地网络代理。

因此推荐的开发流程为：

```text
Windows / Training PC
        │
        │ best.pt
        ▼
香港 Linux 云服务器
        │
        │ ONNX + hb_mapper
        ▼
RDK X5 .bin
        │
        ▼
     RDK X5
        │
        ▼
    BPU Inference
```

也可以完全在自己的 Linux 电脑中完成：

```text
Linux PC
   │
   ├── Conda
   ├── Ultralytics
   ├── hb_mapper
   │
   ▼
.bin
   │
   ▼
RDK X5
```

---

# 为什么需要 Calibration Images

RDK X5 BPU 模型转换过程中使用 INT8 Quantization。

FP32 模型：

```text
FP32
```

转换为：

```text
INT8
```

时，需要使用一批真实图片统计网络各层 Tensor 的数值分布。

因此：

```text
Calibration Images
        │
        ▼
Tensor Distribution
        │
        ▼
Quantization Scale
        │
        ▼
INT8 Model
```

校准图片应尽量覆盖实际运行场景，包括：

* 不同垃圾类别
* 不同距离
* 不同角度
* 不同光照
* 不同背景
* 不同目标尺寸

不推荐使用完全无关的图片进行量化校准。

---

# RDK X5 上运行

将 `.bin` 复制到 RDK X5。

确认系统能够：

```python
import hbm_runtime
import cv2
import numpy
```

然后加载模型：

```python
import hbm_runtime

runtime = hbm_runtime.HB_HBMRuntime(
    "best_bayese_640x640_nv12.bin"
)

print(runtime.model_names)
```

可以继续查看：

```python
model_name = runtime.model_names[0]

print(
    runtime.input_names[model_name]
)

print(
    runtime.output_names[model_name]
)

print(
    runtime.input_shapes[model_name]
)

print(
    runtime.output_shapes[model_name]
)
```

在编写正式检测程序前，推荐首先执行以上代码确认实际 `.bin` 的 Tensor 信息。

---

# 两套方案应该选择哪一个

如果只是针对当前：

```text
YOLOv8s Detect
```

模型，并希望结构简单、输出名称清晰，可以使用：

```text
方案 1
```

方案 1 对当前模型针对性更强：

```text
s8_cls
s8_box
s16_cls
s16_box
s32_cls
s32_box
```

结构直观，适合研究 BPU 后处理。

---

如果希望以后：

* 更换新的 `.pt`
* 重新训练模型后快速转换
* 自动创建环境
* 自动检查 ONNX
* 自动提取 Classes
* 自动量化
* 自动运行 hb_mapper
* 扩展其他 Ultralytics 模型

则推荐：

```text
方案 2
```

它更适合作为持续使用的“一键模型转换工具”。

---

# 重要说明

两个 `.bin`：

```text
best_yolov8s_bpu_bayese_640x640_nv12.bin

best_bayese_640x640_nv12.bin
```

虽然：

* 输入相同
* 类别相同
* 输出数量相同
* 输出 Shape 相同
* DFL 结构相同

但不能认为它们是完全相同的模型文件。

两种导出方式造成了：

```text
PyTorch Graph
      ↓
ONNX Graph
      ↓
Quantized Graph
      ↓
BPU Graph
```

中的部分节点结构差异。

例如：

```text
方案 1：Slice

方案 2：Split
```

因此最终生成的 `.bin`：

* 文件大小不同
* SHA256 不同
* 内部 Node Graph 不完全一致
* Tensor Name 不完全一致

但是由于二者最终都遵循：

```text
[cls, box] × 3
```

这一 YOLOv8 BPU 检测头协议，所以 DFL Decode、置信度筛选和 NMS 代码可以共用。

---

# 推荐程序设计

RDK X5 正式程序中不要依赖某个固定 `.bin` 文件的 Tensor 名称。

推荐程序启动时动态读取：

```python
runtime.model_names
runtime.input_names
runtime.output_names
runtime.input_shapes
runtime.output_shapes
```

然后验证：

```text
Output Count == 6
```

再按照：

```text
0 → s8 cls
1 → s8 box
2 → s16 cls
3 → s16 box
4 → s32 cls
5 → s32 box
```

进行解析。

这样同一套 BPU 检测代码即可兼容本分支中的两种 `.bin`。

---

# 本分支在完整系统中的位置

整体项目可以理解为：

```text
YOLOv8s Training
       │
       │ best.pt
       ▼
BPU_Part_Code
       │
       │ model conversion
       ▼
RDK X5 .bin
       │
       ▼
RDK X5 Device Program
       │
       ├── Camera
       ├── BPU Inference
       ├── Object Detection
       ├── Coordinate Calculation
       └── Device Control
```

本分支主要维护：

```text
PT → ONNX → INT8 → BPU BIN
```

相关程序。

RDK X5 上的完整业务逻辑、摄像头、机械控制以及云服务器通信程序由其他对应分支维护。

---

# 项目信息

**项目：** 2026 嵌入式系统与芯片设计竞赛
**模块：** YOLOv8s BPU 模型转换
**Source Model：** Ultralytics YOLOv8s
**Target Device：** D-Robotics RDK X5
**BPU Architecture：** Bayes-e
**Input：** NV12 / 640×640
**Quantization：** INT8
**Compiler：** `hb_mapper`
**Runtime：** `hbm_runtime`
**Model Format：** `.bin`
