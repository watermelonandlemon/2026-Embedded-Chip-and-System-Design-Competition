# RDK X5 YOLOv8 Conda 一键转换包 v4

用途：在 **Windows 宿主机中的 Ubuntu 22.04 x86_64 虚拟机**内，将 Ultralytics YOLOv8 Detect 的 `best.pt` 转换为 RDK X5 BPU 可部署的 NV12 `.bin` 模型。

转换链路：

```text
best.pt → RDK 友好 ONNX（opset 11）→ Bayes-e INT8/NV12 .bin
```

本工程只使用 **Conda**，不会创建 venv、virtualenv 或 Docker 环境。

## 重要限制

可靠的 INT8 量化不能只使用 `best.pt`。还必须准备 **20～50 张真实场景校准图片**。这些图片不需要标签，但应覆盖目标、背景、光照、角度和距离。

## 推荐虚拟机配置

- Ubuntu 22.04 x86_64
- Conda 已安装
- CPU 4 核以上
- 内存 8 GB 以上，推荐 16 GB
- 磁盘剩余 15 GB 以上
- 虚拟机可访问 PyPI、PyTorch 下载源

## 0. 移动到纯英文路径

不要在 `/home/用户名/下载/` 这种含中文的路径里转换。解压后把整个目录移动到：

```bash
mv ~/下载/RDK_X5_YOLOv8_Conda_OneClick_v4 ~/rdk_x5_converter
cd ~/rdk_x5_converter
```

文件夹实际位置不同就按实际路径修改。最终建议路径：

```text
/home/你的用户名/rdk_x5_converter
```

## 1. 放入模型

将训练权重复制为：

```text
model/best.pt
```

例如：

```bash
cp /你的模型路径/best.pt ./model/best.pt
```

## 2. 放入校准图片

### 方法 A：直接复制

把 20～50 张 JPG/PNG 图片放入：

```text
calibration_images/
```

### 方法 B：从训练集自动均匀抽取 30 张

```bash
bash setup_conda.sh
bash select_calibration.sh /你的数据集/images/train 30
```

## 3. 一条命令完成安装和转换

模型与校准图片放好后执行：

```bash
bash run_all.sh
```

也可以把训练图片目录作为参数，让脚本自动抽取校准图片：

```bash
bash run_all.sh /你的数据集/images/train
```

首次执行会创建全新的 Conda 环境：

```text
rdkx5_convert_clean
```

并安装：

- Python 3.10
- CPU 版 PyTorch 2.7.0
- torchvision 0.22.0
- Ultralytics 8.3.183
- rdkx5-yolo-mapper 1.0.0
- 工具链要求的 NumPy 1.23.0、OpenCV 4.6.0.66、ONNX 1.15.0 等

此前出错的环境名是 `rdk_x5_convert`，新包使用不同环境名，不会继承原来的冲突。

## 4. 分步执行方式

```bash
bash setup_conda.sh
bash convert.sh
```

环境损坏时彻底重建：

```bash
bash reset_conda_env.sh
```

## 5. 转换结果

成功后主要文件位于：

```text
output/best_bayese_640x640_nv12.bin
output/labels.txt
output/model_metadata.json
output/onnx_report.json
output/manifest.json
```

复制到 RDK X5 的核心文件是：

```text
best_bayese_640x640_nv12.bin
```

检查输出：

```bash
bash inspect_output.sh
```

在 RDK X5 板端检查：

```bash
hrt_model_exec model_info --model_file best_bayese_640x640_nv12.bin
hrt_model_exec perf --model_file best_bayese_640x640_nv12.bin --thread_num 1
```

## 6. 常见问题

### `ResolutionImpossible`

不要手动改 NumPy 或 OpenCV。执行：

```bash
bash reset_conda_env.sh
```

新包在同一次 pip 解析中安装 mapper 与 Ultralytics，并使用 mapper 官方精确依赖，不再声明冲突的 `numpy>=1.23.5` 或 `opencv>=4.8`。

### 虚拟机内存不足或进程被杀死

编辑 `config.sh`：

```bash
JOBS=1
OPTIMIZE_LEVEL="O2"
```

然后重新执行：

```bash
bash convert.sh
```

### `libGL.so.1` 缺失

Ubuntu 中执行：

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

这只是系统动态库，不是另一种 Python 虚拟环境。

### ONNX 输出数量不是 6

本包面向 YOLOv8 Detect。若 `best.pt` 是分割、姿态、分类、OBB 或自行修改过检测头的模型，需要使用对应导出与后处理流程。
