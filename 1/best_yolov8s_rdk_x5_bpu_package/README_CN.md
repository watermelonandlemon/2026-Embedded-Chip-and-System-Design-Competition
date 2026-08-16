# YOLOv8s → RDK X5 BPU 部署包

## 已完成内容

- 已从 `best_yolov8s.pt` 导出固定输入 `1×3×640×640` 的 BPU 友好 ONNX。
- 检测头已改为 RDK Model Zoo 的六输出协议：
  `s8_cls, s8_box, s16_cls, s16_box, s32_cls, s32_box`。
- 已配置 4 个类别：`zhituan`、`pinggai`、`dianchi`、`jiaodai`。
- 已整理 32 张 INT8 校准图（16 张原图 + 16 张水平翻转图）。
- 已提供 RDK X5 上可直接使用的 `hbm_runtime` + NV12 推理程序。

> `model/best_yolov8s_bpu.onnx` 已经生成并校验通过。最终 RDK X5 使用的
> `.bin` 必须由 D-Robotics 的 `hb_mapper` 编译器生成。该编译器只在
> OpenExplorer/rdkx5-yolo-mapper 环境中提供。

## 目录结构

```text
model/
  best_yolov8s_bpu.onnx                 已完成的 BPU 友好 ONNX
source/
  best_yolov8s.pt                       原始训练权重
cal_images/                             32 张量化校准图
conversion/
  convert_to_bin.sh                     一键生成 RDK X5 .bin
  run_docker_convert.sh                 OpenExplorer Docker 方式
  mapper.py                             转换主程序
runtime/
  infer_image.py                        RDK X5 单图推理
  run_image.sh                          单图推理快捷脚本
tools/
  export_bpu_onnx.py                    PT 重新导出 ONNX 的脚本
metadata/
  model_info.json
  sha256sums.txt
```

## 方法一：在 Conda 虚拟机中生成 `.bin`

建议使用 Ubuntu 22.04 和 Python 3.10：

```bash
conda create -n rdk_x5_mapper python=3.10 -y
conda activate rdk_x5_mapper
pip install -r conversion/requirements-convert.txt
chmod +x conversion/convert_to_bin.sh
./conversion/convert_to_bin.sh
```

转换成功后得到：

```text
model/best_yolov8s_bpu_bayese_640x640_nv12.bin
```

## 方法二：使用 OpenExplorer Docker

先安装 Docker，并准备 X5 OpenExplorer 镜像，然后运行：

```bash
chmod +x conversion/run_docker_convert.sh conversion/convert_to_bin.sh
./conversion/run_docker_convert.sh
```

若你的镜像标签不同：

```bash
RDK_OPENEXPLORER_IMAGE=你的镜像名 ./conversion/run_docker_convert.sh
```

## 在 RDK X5 上测试

将整个目录复制到 RDK X5。RDK 系统需能导入 `hbm_runtime`、`cv2` 和
`numpy`。然后执行：

```bash
chmod +x runtime/run_image.sh
./runtime/run_image.sh cal_images/170.jpg result.jpg
```

或直接运行：

```bash
python3 runtime/infer_image.py \
  --model model/best_yolov8s_bpu_bayese_640x640_nv12.bin \
  --image cal_images/170.jpg \
  --output result.jpg \
  --score 0.25 \
  --nms 0.70 \
  --bpu-cores 0
```

## 模型接口

### 输入

- 运行时格式：NV12
- 分辨率：640×640
- batch：1
- 预处理：保持比例 letterbox，灰色填充 127，随后打包 NV12

### 输出

| 序号 | 步长 | 内容 | 形状 |
|---:|---:|---|---|
| 0 | 8 | 分类 logits | `[1,80,80,4]` |
| 1 | 8 | DFL 框 | `[1,80,80,64]` |
| 2 | 16 | 分类 logits | `[1,40,40,4]` |
| 3 | 16 | DFL 框 | `[1,40,40,64]` |
| 4 | 32 | 分类 logits | `[1,20,20,4]` |
| 5 | 32 | DFL 框 | `[1,20,20,64]` |

## 重新导出 ONNX

仅在替换 `.pt` 后需要执行：

```bash
python3 tools/export_bpu_onnx.py \
  source/best_yolov8s.pt \
  model/best_yolov8s_bpu.onnx
```
