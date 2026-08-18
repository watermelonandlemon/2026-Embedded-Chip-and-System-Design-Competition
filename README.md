# 2026 嵌入式系统与芯片设计竞赛

本仓库用于保存 **2026 嵌入式系统与芯片设计竞赛** 项目的完整软件代码与部署资料。

项目以 **D-Robotics RDK X5** 为边缘计算核心，完成摄像头图像采集、YOLOv8 垃圾识别、BPU 硬件加速推理、GPIO 分类执行、本地数据统计，并通过云服务器实现设备管理、数据上报、远程控制和 Web 数据可视化。

> Repository: `2026-Embedded-Chip-and-System-Design-Competition`
> Edge Device: `D-Robotics RDK X5`
> Vision Model: `YOLOv8s`
> AI Accelerator: `Bayes-e BPU`
> Cloud: `FastAPI + SQLite + Nginx`
> Device Communication: `miniROS + HTTP REST API`

---

## 仓库说明

本仓库按照功能模块使用 **独立 Git 分支** 管理代码。

`main` 分支仅作为项目总入口和说明页，具体程序分别保存在对应功能分支中，避免设备端、模型转换端和云端代码混在一起。

| Branch                                                                                                                       | 模块         | 主要内容                                        |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------- |
| [`main`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/main)                   | 项目总览       | 项目架构、分支导航、整体说明                              |
| [`BPU_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/BPU_Part_Code) | BPU 模型转换   | YOLOv8s `.pt` → ONNX → INT8 → RDK X5 `.bin` |
| [`RDK_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/RDK_Part_Code) | RDK X5 设备端 | BPU 推理、摄像头、miniROS、GPIO、计数、HDMI UI、云端通信     |
| [`Web_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/Web_Part_Code) | 云服务器       | FastAPI、SQLite、设备管理、数据统计、远程控制、Web 数据大屏      |

---

# 项目整体架构

```text
                           ┌─────────────────────┐
                           │      YOLOv8s.pt     │
                           └──────────┬──────────┘
                                      │
                                      │ BPU_Part_Code
                                      ▼
                         ┌─────────────────────────┐
                         │ RDK X5 Bayes-e BPU .bin │
                         └────────────┬────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                            RDK X5                                 │
│                                                                   │
│   USB Camera                                                      │
│       │                                                           │
│       ▼                                                           │
│   YOLOv8 BPU 推理                                                 │
│       │                                                           │
│       ▼                                                           │
│   miniROS 节点通信                                                │
│       │                                                           │
│       ├────────► GPIO 分类执行机构                                │
│       │                                                           │
│       ├────────► 本地垃圾计数                                    │
│       │                                                           │
│       ├────────► HDMI 本地 UI                                    │
│       │                                                           │
│       └────────► Cloud Communication                             │
└──────────────────────────────┬────────────────────────────────────┘
                               │
                               │ HTTP REST API
                               ▼
                    ┌─────────────────────────┐
                    │      Cloud Server       │
                    │                         │
                    │ FastAPI + SQLite        │
                    │ Uvicorn + Nginx         │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
          ┌───────────────┐             ┌───────────────┐
          │ Device Manager│             │   Web Browser │
          │ Remote Control│             │   Data Screen │
          └───────────────┘             └───────────────┘
```

整个系统可以概括为：

```text
模型训练 / 转换
      ↓
RDK X5 边缘识别
      ↓
GPIO 分类执行
      ↓
本地统计与显示
      ↓
云端数据管理
      ↓
Web 可视化与远程控制
```

---

# 1. BPU 模型转换

对应分支：

[`BPU_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/BPU_Part_Code)

该分支负责将训练完成的 YOLOv8s PyTorch 模型转换为 RDK X5 Bayes-e BPU 可以直接运行的模型。

模型转换链路：

```text
YOLOv8s.pt
     │
     ▼
ONNX
     │
     ▼
BPU-Friendly Detect Head
     │
     ▼
INT8 Calibration
     │
     ▼
hb_mapper
     │
     ▼
RDK X5 Bayes-e .bin
```

主要内容包括：

* YOLOv8s `.pt` 导出 ONNX
* YOLOv8 Detect 检测头 BPU 友好化
* INT8 校准与量化
* `hb_mapper` 编译
* 生成 RDK X5 `.bin` 模型
* 两种模型转换方案对比
* Slice / Split 等 BPU 图结构分析
* YOLOv8 DFL 后处理示例

当前模型主要参数：

```text
Model: YOLOv8s
Input: 640 × 640
BPU Input: NV12
Quantization: INT8
Target: Bayes-e
Outputs: 6
```

详细说明请查看该分支中的 `README.md`。

---

# 2. RDK X5 设备端

对应分支：

[`RDK_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/RDK_Part_Code)

该分支负责比赛设备端的主要运行逻辑。

核心功能：

* USB 摄像头实时图像采集
* YOLOv8 BPU 硬件加速推理
* ROI 图像处理
* DFL Decode + NMS 后处理
* miniROS 多节点通信
* GPIO 分类执行机构控制
* 垃圾分类计数
* 本地数据持久化
* HDMI 全屏 UI
* 云端数据上报
* 云端手动控制
* 云端计数清零
* 命令执行 ACK

设备端主要节点：

```text
miniROS Broker
      │
      ├── node_vision   → 摄像头 + BPU 视觉识别
      ├── node_gpio     → GPIO 分类机构控制
      ├── node_count    → 垃圾数量统计
      ├── node_web      → 云服务器通信
      └── node_ui       → HDMI 本地界面
```

详细安装、GPIO、摄像头、模型以及启动方法请查看该分支中的 `README.md`。

---

# 3. 云服务器与 Web

对应分支：

[`Web_Part_Code`](https://github.com/watermelonandlemon/2026-Embedded-Chip-and-System-Design-Competition/tree/Web_Part_Code)

云服务器作为系统的数据与远程控制中心，主要负责：

* RDK X5 设备注册与身份认证
* 四类垃圾数量实时上报
* 设备在线 / 离线状态监测
* 历史数据保存
* 垃圾分类趋势统计
* Web 数据大屏
* 设备新增、查询和删除
* 云端垃圾计数清零
* 云端手动垃圾分类控制
* 设备拉取云端命令
* 命令执行结果 ACK
* SQLite 数据持久化
* Nginx + Uvicorn 公网部署
* Docker / Docker Compose 部署

主要技术栈：

```text
FastAPI
SQLite
Uvicorn
Nginx
HTML / CSS / JavaScript
Docker
```

RDK X5 主动通过 HTTP REST API 与服务器通信，因此设备位于局域网、校园网或移动网络环境时，不需要服务器主动连接 RDK X5。

---

# 四类垃圾协议

系统统一使用以下四类垃圾状态：

| 状态码    | 垃圾类别  | API 字段       |
| ------ | ----- | ------------ |
| `0001` | 可回收垃圾 | `recyclable` |
| `0010` | 厨余垃圾  | `kitchen`    |
| `0100` | 有害垃圾  | `hazardous`  |
| `1000` | 其他垃圾  | `other`      |

当前设备端视觉模型中主要使用的 YOLO 类别包括：

```text
zhituan  → 纸团
pinggai  → 瓶盖
dianchi  → 电池
jiaodai  → 胶带
```

设备端会进一步将 YOLO 类别映射为四类垃圾状态，再执行相应 GPIO 分类动作并同步计数结果。

---

# 软件链路

## 自动分类

```text
Camera
  ↓
YOLOv8s
  ↓
RDK X5 BPU
  ↓
Detection Result
  ↓
miniROS
  ↓
GPIO
  ↓
Classification Mechanism
  ↓
Counter
  ↓
Cloud Server
  ↓
Web Data Screen
```

## 云端手动控制

```text
Web / Cloud Server
       │
       ▼
RDK X5 node_web
       │
       ▼
miniROS
       │
       ▼
node_gpio
       │
       ▼
GPIO Classification
       │
       ▼
Command ACK
       │
       ▼
Cloud Server
```

---

# 技术栈

| 模块          | 技术                                 |
| ----------- | ---------------------------------- |
| AI 模型       | YOLOv8s                            |
| 模型训练        | PyTorch / Ultralytics              |
| 模型交换        | ONNX                               |
| BPU 编译      | D-Robotics Toolchain / `hb_mapper` |
| 边缘计算        | RDK X5 / Bayes-e BPU               |
| BPU Runtime | `hbm_runtime`                      |
| 图像处理        | OpenCV / NumPy                     |
| 设备内通信       | miniROS                            |
| 硬件控制        | Hobot.GPIO                         |
| 本地界面        | Tkinter / HDMI / X11               |
| 云服务器        | FastAPI / Uvicorn                  |
| 数据库         | SQLite                             |
| 反向代理        | Nginx                              |
| Web 前端      | HTML / CSS / JavaScript            |
| 容器部署        | Docker / Docker Compose            |
| 设备云通信       | HTTP REST API                      |

---

# 推荐阅读顺序

如果第一次查看本项目，建议按照以下顺序了解：

```text
1. main
   │
   │ 项目整体架构
   ▼
2. BPU_Part_Code
   │
   │ YOLOv8s 如何转换为 RDK X5 BPU 模型
   ▼
3. RDK_Part_Code
   │
   │ RDK X5 如何识别并控制实际设备
   ▼
4. Web_Part_Code
   │
   │ RDK X5 如何与云服务器和 Web 端交互
   ▼
完整系统
```

---

# 分支之间的关系

```text
┌───────────────────────┐
│     BPU_Part_Code     │
│                       │
│ YOLOv8s → BPU .bin    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     RDK_Part_Code     │
│                       │
│ Vision + GPIO + UI    │
└───────────┬───────────┘
            │
            │ REST API
            ▼
┌───────────────────────┐
│     Web_Part_Code     │
│                       │
│ Cloud + Database + Web│
└───────────────────────┘
```

因此三个代码分支分别解决：

```text
BPU_Part_Code → 模型怎么转换
RDK_Part_Code → 模型怎么在开发板上运行并控制设备
Web_Part_Code → 设备怎么接入云服务器并进行数据管理
```

---

# 注意事项

1. `main` 分支用于项目总览，具体程序请进入对应功能分支查看。
2. BPU 模型必须与 RDK X5 / Bayes-e 运行环境及后处理程序保持一致。
3. `hbm_runtime`、`Hobot.GPIO` 等组件需要在 RDK X5 官方系统环境中运行。
4. 云服务器部署前应修改设备认证信息和服务器配置。
5. 不要在公开仓库中提交真实的 `Device Key`、Token、密码或其他敏感认证信息。
6. 如果密钥曾经公开提交到 GitHub，应在服务端重新生成或轮换，单纯删除文件中的明文并不能使旧密钥失效。

---

# 项目总结

本项目形成了从 **AI 模型转换、RDK X5 边缘推理、硬件执行控制到云服务器数据管理与 Web 可视化** 的完整软件链路：

```text
YOLOv8s
   ↓
BPU Model Conversion
   ↓
RDK X5 Edge AI
   ↓
GPIO Classification
   ↓
Local Counter / HDMI UI
   ↓
Cloud Server
   ↓
Web Visualization / Remote Control
```

三个功能分支相互独立，同时通过统一模型格式、垃圾分类状态和通信协议组成完整系统。
