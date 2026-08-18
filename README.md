# 2026 嵌入式系统与芯片设计竞赛 — RDK X5 设备端程序

本分支保存 **2026 嵌入式系统与芯片设计竞赛** 项目中的 **RDK X5 设备端程序**，负责摄像头图像采集、YOLOv8 BPU 推理、垃圾类别判定、GPIO 执行机构控制、本地计数、HDMI 可视化界面，以及与云服务器之间的数据上报和远程控制。

本分支对应整个系统中的 **边缘设备端（Edge Device）**。

> Branch: `RDK_Part_Code`
> Device: `D-Robotics RDK X5`
> AI Accelerator: `Bayes-e BPU`
> Vision Model: `YOLOv8s`
> Runtime Model: `640 × 640 NV12 .bin`
> Communication: `miniROS + HTTP REST API`
> Local UI: `Tkinter + HDMI/X11`

---

## 分支用途

本分支主要实现：

* RDK X5 USB 摄像头实时图像采集
* YOLOv8s `.bin` 模型在 RDK X5 BPU 上进行硬件加速推理
* ROI 区域检测与完整画面坐标还原
* YOLOv8 六输出检测头的 DFL 解码与 NMS 后处理
* 四类垃圾状态统一映射
* miniROS 节点间消息通信
* GPIO 分类执行机构控制
* 垃圾分类数量本地持久化
* HDMI 本地全屏 UI
* 与云服务器进行垃圾计数上报
* 接收云端手动分类命令
* 接收云端计数清零命令
* 云端命令执行结果 ACK
* 一键启动、停止与日志管理

---

# 系统整体架构

```text
                         ┌──────────────────────────────┐
                         │        Cloud Server          │
                         │ FastAPI + SQLite + Web UI    │
                         └──────────────┬───────────────┘
                                        │
                                  HTTP REST API
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                           RDK X5                                 │
│                                                                  │
│  USB Camera                                                      │
│      │                                                           │
│      ▼                                                           │
│  node_vision                                                     │
│  YOLOv8 + BPU                                                    │
│      │                                                           │
│      │ /vision/result                                            │
│      ▼                                                           │
│  ┌──────────────┐       miniROS       ┌──────────────┐           │
│  │ node_gpio    │◄───────────────────►│ node_count   │           │
│  │ GPIO 执行机构 │                    │ 本地计数      │           │
│  └──────┬───────┘                     └──────┬───────┘           │
│         │                                    │                   │
│         │                              count.yaml                │
│         │                                    │                   │
│         ▼                                    ▼                   │
│   分类机械机构                           node_ui                  │
│                                        HDMI UI                   │
│                                                                  │
│                         node_web                                 │
│                    云端通信 / 命令处理                            │
└──────────────────────────────────────────────────────────────────┘
```

RDK X5 负责 **视觉识别 + 本地控制 + 本地显示 + 云端通信**，云服务器主要负责设备管理、数据存储和远程控制。

---

# 目录结构

```text
RDK_Part_Code/
│
├── README.md
│
└── code/
    │
    ├── miniros-0.1.0-py3-none-any.whl
    │   └── MiniROS 本地安装包
    │
    ├── camera/
    │   ├── best_bayese_640x640_nv12.bin
    │   ├── camera.py
    │   ├── classes.names
    │   ├── readme.md
    │   └── run_camera.sh
    │
    └── all/
        ├── README.md
        ├── lib/
        ├── src/
        ├── log/
        ├── run_all.sh
        └── stop_all.sh
```

---

# `camera/`：独立视觉检测程序

目录：

```text
code/camera/
```

该目录用于 **单独测试摄像头与 BPU 模型**，不需要启动 miniROS、GPIO、云端通信和完整系统。

| 文件                             | 作用                      |
| ------------------------------ | ----------------------- |
| `best_bayese_640x640_nv12.bin` | RDK X5 Bayes-e BPU 模型   |
| `camera.py`                    | 摄像头读取、ROI 裁剪、BPU 推理和后处理 |
| `classes.names`                | YOLO 类别名称               |
| `run_camera.sh`                | 独立视觉程序启动脚本              |
| `readme.md`                    | camera 子模块说明            |

模型输入：

```text
640 × 640
NV12
```

当前 BPU 模型采用 YOLOv8 的三尺度分类 / DFL 回归分离输出：

```text
Stride 8  → Classification + DFL Box
Stride 16 → Classification + DFL Box
Stride 32 → Classification + DFL Box
```

共 6 个输出。

模型转换方法及 BPU 结构说明见：

```text
BPU_Part_Code
```

分支。

---

# `all/`：比赛完整设备端程序

目录：

```text
code/all/
```

这是比赛运行时使用的完整 RDK X5 程序。

程序采用 **miniROS 多节点架构**，将视觉识别、GPIO 控制、计数、云端通信和本地 UI 分离。

## 节点说明

| 节点             | 入口文件                 | 主要职责              |
| -------------- | -------------------- | ----------------- |
| miniROS Broker | `python3 -m miniROS` | 节点连接和话题转发         |
| 视觉节点           | `src/node_vision.py` | 摄像头采集、BPU 推理、目标输出 |
| GPIO 节点        | `src/node_gpio.py`   | 控制四路 GPIO 分类执行机构  |
| 计数节点           | `src/node_count.py`  | 分类成功后计数及持久化       |
| 云端节点           | `src/node_web.py`    | 数据上报、远程控制、命令 ACK  |
| UI 节点          | `src/node_ui.py`     | HDMI 本地界面显示       |

---

# miniROS 通信

默认 Broker：

```text
127.0.0.1:8765
```

主要话题：

| Topic                      | 发布者           | 订阅者                     | 用途        |
| -------------------------- | ------------- | ----------------------- | --------- |
| `/vision/result`           | `node_vision` | `node_gpio`、`node_ui`   | 视觉检测结果    |
| `/actuator/request/manual` | `node_web`    | `node_gpio`             | 云端手动分类请求  |
| `/actuator/executed`       | `node_gpio`   | `node_count`、`node_web` | GPIO 执行结果 |
| `/counter/data`            | `node_count`  | `node_ui`               | 当前垃圾计数    |
| `/counter/reset`           | `node_web`    | `node_count`            | 云端清零请求    |
| `/counter/reset/result`    | `node_count`  | `node_web`              | 清零执行结果    |
| `/cloud/status`            | `node_web`    | 预留                      | 云端连接状态    |

---

# 自动垃圾分类流程

```text
USB Camera
    │
    ▼
node_vision
    │
    │ BPU YOLOv8 推理
    ▼
/vision/result
    │
    ▼
node_gpio
    │
    │ GPIO 动作
    ▼
分类执行机构
    │
    ▼
/actuator/executed
    │
    ▼
node_count
    │
    ├── 更新 count.yaml
    └── 发布 /counter/data
              │
              ├──► node_ui
              └──► node_web → Cloud Server
```

---

# 垃圾类别定义

设备端使用统一字段 `a` 表示垃圾分类结果：

|  `a` | 垃圾类别  | 状态码    |   GPIO | 当前 YOLO 类别          |
| ---: | ----- | ------ | -----: | ------------------- |
| `-1` | 无目标   | —      |      — | 未检测到目标              |
|  `0` | 可回收垃圾 | `0001` | GPIO17 | `zhituan`、`pinggai` |
|  `1` | 厨余垃圾  | `0010` | GPIO27 | 当前视觉模型未配置           |
|  `2` | 有害垃圾  | `0100` | GPIO23 | `dianchi`           |
|  `3` | 其他垃圾  | `1000` | GPIO24 | `jiaodai`           |

当前 `classes.names`：

```text
zhituan
pinggai
dianchi
jiaodai
```

即：

```text
纸团 → 可回收垃圾
瓶盖 → 可回收垃圾
电池 → 有害垃圾
胶带 → 其他垃圾
```

---

# BPU 模型

默认模型路径：

```text
/home/sunrise/code/camera/best_bayese_640x640_nv12.bin
```

模型链路：

```text
YOLOv8s.pt
     │
     ▼
ONNX
     │
     ▼
INT8 Calibration
     │
     ▼
hb_mapper
     │
     ▼
Bayes-e BPU .bin
```

模型转换程序及模型结构分析请查看：

```text
BPU_Part_Code
```

分支。

---

# 云端通信

RDK X5 通过 HTTP REST API 主动连接云服务器。

云服务器程序见：

```text
Web_Part_Code
```

分支。

设备端主要负责：

* 定时上报四类垃圾数量
* 轮询云端控制命令
* 执行手动垃圾分类
* 执行垃圾计数清零
* 返回命令执行状态

---

# 运行环境

| 项目          | 配置                             |
| ----------- | ------------------------------ |
| 开发板         | RDK X5                         |
| Python      | `/usr/bin/python3`，Python 3.10 |
| BPU Runtime | `hbm_runtime`                  |
| GPIO        | `Hobot.GPIO`                   |
| 图像处理        | OpenCV + NumPy                 |
| 节点通信        | miniROS                        |
| UI          | Tkinter + X11/HDMI             |
| 模型          | YOLOv8s Bayes-e BPU `.bin`     |
| 摄像头         | USB Camera `/dev/video0`       |

---

# 安装 MiniROS

仓库中已提供：

```text
code/miniros-0.1.0-py3-none-any.whl
```

安装：

```bash
cd /home/sunrise/code
sudo /usr/bin/python3 -m pip install ./miniros-0.1.0-py3-none-any.whl
```

---

# RDK X5 部署

程序默认部署目录：

```text
/home/sunrise/code/
├── all/
├── camera/
└── miniros-0.1.0-py3-none-any.whl
```

增加启动权限：

```bash
sudo chmod +x /home/sunrise/code/all/run_all.sh
sudo chmod +x /home/sunrise/code/all/stop_all.sh
sudo chmod +x /home/sunrise/code/camera/run_camera.sh
```

如需 HDMI UI：

```bash
sudo apt update
sudo apt install -y python3-tk fonts-noto-cjk
```

---

# 独立测试摄像头与 BPU

```bash
cd /home/sunrise/code/camera
./run_camera.sh
```

程序链路：

```text
Camera
  ↓
ROI
  ↓
BPU YOLOv8
  ↓
Bounding Box
  ↓
Full Frame Display
```

运行过程中：

```text
q / ESC → 退出
s       → 保存当前检测画面
```

---

# 启动完整比赛程序

```bash
sudo /home/sunrise/code/all/run_all.sh
```

启动顺序：

```text
1. miniROS Broker
2. node_count
3. node_gpio
4. node_web
5. node_vision
6. node_ui
```

---

# 停止完整系统

```bash
sudo /home/sunrise/code/all/stop_all.sh
```

---

# 日志查看

日志目录：

```text
/home/sunrise/code/all/log/
```

例如：

```bash
tail -f /home/sunrise/code/all/log/vision.log
tail -f /home/sunrise/code/all/log/gpio.log
tail -f /home/sunrise/code/all/log/web.log
```

---

# 与其他分支的关系

| Branch          | 作用                        |
| --------------- | ------------------------- |
| `main`          | 项目入口 / 总体说明               |
| `BPU_Part_Code` | YOLOv8s → RDK X5 BPU 模型转换 |
| `RDK_Part_Code` | RDK X5 设备端完整程序            |
| `Web_Part_Code` | 云服务器与 Web 数据管理程序          |

整体关系：

```text
BPU_Part_Code
     │
     │ 生成 .bin
     ▼
RDK_Part_Code
     │
     │ HTTP REST API
     ▼
Web_Part_Code
```

即：

```text
BPU_Part_Code = 模型怎么转换

RDK_Part_Code = 模型怎么在开发板上运行并控制设备

Web_Part_Code = 云服务器怎么管理设备和数据
```

---

# 注意事项

1. `hbm_runtime` 和 `Hobot.GPIO` 为 RDK X5 板端组件，普通 PC Python 环境通常无法直接运行。
2. BPU 模型、`classes.names` 和 `vision.yolo_classes` 的类别顺序必须保持一致。
3. GPIO 接线前应再次核对程序配置中的 GPIO 编号。
4. 当前部分路径固定为 `/home/sunrise/code/...`，修改部署目录时需要同步修改配置文件和启动脚本。
5. HDMI UI 依赖 X11 / Tkinter，纯 SSH 环境下可能无法正常显示。
6. 公开仓库中不要提交真实 `device_key`、Token、密码或其他认证信息。

---

# 项目链路总结

```text
YOLOv8s.pt
    │
    │ BPU_Part_Code
    ▼
RDK X5 .bin
    │
    │ RDK_Part_Code
    ▼
Camera → BPU → miniROS → GPIO → Count → HDMI UI
                         │
                         ▼
                    Cloud API
                         │
                         │ Web_Part_Code
                         ▼
                Cloud Server + Web UI
```

本分支负责其中的 **RDK X5 边缘设备端**，覆盖从视觉感知、BPU 推理、节点通信、执行控制、本地显示到云端交互的完整设备侧程序。
