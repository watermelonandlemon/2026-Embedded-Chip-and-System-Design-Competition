# 2026 嵌入式系统与芯片设计竞赛 — 云服务器程序

本分支保存 **2026 嵌入式系统与芯片设计竞赛** 项目中的**云端程序**，主要负责 RDK X5 设备接入、垃圾分类数据管理、设备状态监测、远程控制以及 Web 数据大屏展示。

> Branch: `Web_Part_Code`
> Cloud Server: FastAPI + SQLite + Nginx
> Device: RDK X5
> API Version: `/api/iot/v1`

---

## 项目功能

云服务器作为整个系统的云端数据与控制中心，主要实现：

* RDK X5 设备注册与身份认证
* 四类垃圾分类数量实时上报
* 设备在线 / 离线状态监测
* 垃圾分类历史数据存储与趋势统计
* Web 数据大屏实时展示
* 管理员新增、查询和删除设备
* 云端下发垃圾计数清零命令
* 云端下发手动垃圾分类控制命令
* RDK X5 拉取云端命令并返回执行结果
* SQLite 数据持久化
* Ubuntu + systemd + Nginx 公网部署
* Docker / Docker Compose 部署

---

## 系统架构

```text
                        Internet
                           │
                           ▼
                  ┌─────────────────┐
                  │   Cloud Server  │
                  │                 │
                  │ FastAPI + SQLite│
                  │ Nginx + Uvicorn │
                  └────────┬────────┘
                           │
                    REST / HTTP API
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ┌───────────┐            ┌─────────────┐
        │  RDK X5   │            │ Web Browser │
        │ Device End│            │ Data Screen │
        └───────────┘            └─────────────┘
              │
              ▼
       垃圾分类与执行机构
```

RDK X5 通过自定义轻量 REST 接口主动连接云服务器，不需要云服务器主动访问位于局域网中的设备。

---

## 四类垃圾状态码

| 状态码    | 垃圾类别  | API 字段       |
| ------ | ----- | ------------ |
| `0001` | 可回收垃圾 | `recyclable` |
| `0010` | 厨余垃圾  | `kitchen`    |
| `0100` | 有害垃圾  | `hazardous`  |
| `1000` | 其他垃圾  | `other`      |

设备端具体 GPIO、运动机构或分类执行逻辑可由 RDK X5 程序完成，云服务器主要负责数据管理与控制命令传输。

---

## 目录结构

```text
smartbin_cloud_server/
├── app/
│   ├── main.py               # FastAPI 主程序与 API 路由
│   ├── config.py             # 环境变量及服务器配置
│   ├── db.py                 # SQLite 数据库操作
│   ├── schemas.py            # API 数据模型
│   └── static/
│       ├── index.html        # Web 数据大屏
│       ├── app.js
│       └── style.css
│
├── data/
│   └── .gitkeep              # 数据库数据目录
│
├── deploy/
│   ├── install_ubuntu.sh     # Ubuntu 自动部署脚本
│   ├── nginx-smartbin.conf   # Nginx 配置
│   └── smartbin-cloud.service# systemd 服务
│
├── tests/
│   └── test_api.py           # API 测试
│
├── tools/
│   └── device_simulator.py   # RDK 设备模拟器
│
├── .env.example              # 环境变量配置模板
├── API_PROTOCOL.md           # RDK X5 ↔ 云服务器通信协议
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── VERSION.txt
```

---

## 云端与 RDK X5 通信

设备使用独立的 `Device ID` 和 `Device Key` 进行身份认证：

```text
X-Device-ID: 设备编号
X-Device-Key: 设备密钥
```

主要设备接口：

```text
POST /api/iot/v1/report
GET  /api/iot/v1/commands
POST /api/iot/v1/commands/{command_id}/ack
```

其中：

* `report`：RDK X5 向服务器上传四类垃圾累计数量及设备状态
* `commands`：RDK X5 从云端获取待执行控制命令
* `ack`：设备执行完成后向服务器返回执行结果

云端命令采用**租约机制**，如果设备获取命令后未正常回执，租约超时后服务器可以重新下发该命令。

详细通信协议见：

**[`smartbin_cloud_server/API_PROTOCOL.md`](smartbin_cloud_server/API_PROTOCOL.md)**

---

## Web 数据大屏

服务器内置 Web 前端，可展示：

* 全国垃圾分类物联网节点概况
* 哈尔滨真实 RDK X5 设备
* 四类垃圾实时分类数量
* 分类数量变化趋势
* 设备在线 / 离线状态
* 设备编号及固件信息
* 待执行云端命令状态

管理员登录后还可以：

* 新增垃圾桶设备
* 生成设备编号与设备密钥
* 删除设备
* 清零设备垃圾计数
* 手动下发 `0001 / 0010 / 0100 / 1000` 分类控制命令
* 查看设备命令执行记录

---

## 数据存储

服务器使用 **SQLite** 保存设备、历史数据和控制命令。

主要数据包括：

```text
devices     设备信息与当前垃圾分类数量
reports     垃圾分类历史数据
commands    云端控制命令及执行状态
```

默认数据库部署位置：

```text
/opt/smartbin-cloud/data/smartbin.db
```

数据库启用 WAL 模式，以提高服务器运行过程中读写操作的稳定性。

---

## 部署方式

项目支持两种主要部署方案：

### Ubuntu + systemd + Nginx

适用于正式公网服务器部署。

项目提供：

```text
deploy/install_ubuntu.sh
deploy/smartbin-cloud.service
deploy/nginx-smartbin.conf
```

可以将 FastAPI 服务安装到：

```text
/opt/smartbin-cloud
```

并通过 systemd 保持服务持续运行，再由 Nginx 提供公网访问。

### Docker

同时提供：

```text
Dockerfile
docker-compose.yml
```

可直接使用 Docker Compose 部署。

完整服务器安装、环境变量配置和 Nginx 配置方法见：

**[`smartbin_cloud_server/README.md`](smartbin_cloud_server/README.md)**

---

## 安全说明

真实部署时请勿将以下内容提交至 GitHub：

```text
.env
ADMIN_TOKEN
APP_SECRET
Device Key
数据库文件 *.db
```

仓库中的：

```text
.env.example
```

仅作为配置模板使用。

设备密钥在创建设备时只返回一次，服务器数据库仅保存经过摘要处理后的设备密钥。

---

## 相关分支

本仓库按照系统组成划分不同程序。

```text
main
└── 项目总体说明

Web_Part_Code
└── 云服务器、Web 数据大屏及 IoT API

RDK X5 相关分支
└── RDK X5 设备端程序
```

本分支仅维护**云服务器及 Web 端相关程序**，RDK X5 本地视觉识别、设备控制及执行机构程序请查看对应设备端分支。

---

## 项目信息

**项目：** 2026 嵌入式系统与芯片设计竞赛
**模块：** 云服务器 / Web 数据大屏 / IoT 通信接口
**设备端：** RDK X5
**后端：** Python / FastAPI / Uvicorn
**数据库：** SQLite
**部署：** Ubuntu / systemd / Nginx / Docker
**API：** RESTful API
