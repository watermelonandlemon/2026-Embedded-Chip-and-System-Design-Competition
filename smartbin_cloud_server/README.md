# 智能垃圾分类公网 IoT 服务器与数据大屏

本目录部署在具有公网 IP 或域名的 Ubuntu 云服务器上。RDK X5 不运行本网站；RDK 使用配套设备端压缩包，通过自定义轻量 REST 接口主动上报计数、领取清零和手动控制命令。

## 1. 功能

- 全国模拟节点数据大屏；
- 哈尔滨 RDK X5 真实设备数据；
- 四类垃圾实时计数与趋势；
- 真实设备在线/离线判断；
- 新增垃圾桶并生成设备编号与一次性设备密钥；
- 对指定设备下发清零命令；
- 对指定设备下发 `0001/0010/0100/1000` 手动命令；
- 命令租约、设备执行回执和命令状态；
- SQLite 持久化；
- Nginx、systemd、Docker 部署文件。

## 2. 目录

```text
smartbin_cloud_server/
├── app/
│   ├── main.py
│   ├── db.py
│   ├── schemas.py
│   └── static/
├── data/
├── deploy/
│   ├── install_ubuntu.sh
│   ├── smartbin-cloud.service
│   └── nginx-smartbin.conf
├── tools/
│   └── device_simulator.py
├── tests/
├── API_PROTOCOL.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py
```

## 3. Ubuntu 一键安装

适用于 Ubuntu 22.04/24.04。将整个目录上传到服务器，例如：

```text
/root/smartbin_cloud_server
```

执行：

```bash
cd /root/smartbin_cloud_server
sudo bash deploy/install_ubuntu.sh
```

脚本会：

1. 安装 Python、venv、Nginx、OpenSSL 和 rsync；
2. 创建 `/opt/smartbin-cloud`；
3. 创建独立系统用户 `smartbin`；
4. 安装 FastAPI/Uvicorn；
5. 自动生成 `ADMIN_TOKEN` 和 `APP_SECRET`；
6. 启动 `smartbin-cloud.service`；
7. 将数据库保存到 `/opt/smartbin-cloud/data/smartbin.db`。

查看管理员令牌：

```bash
sudo cat /opt/smartbin-cloud/credentials.txt
```

检查服务：

```bash
sudo systemctl status smartbin-cloud
curl http://127.0.0.1:8000/health
sudo journalctl -u smartbin-cloud -f
```

## 4. 配置 Nginx 与公网访问

```bash
sudo cp /opt/smartbin-cloud/deploy/nginx-smartbin.conf \
  /etc/nginx/sites-available/smartbin-cloud
sudo nano /etc/nginx/sites-available/smartbin-cloud
```

将：

```nginx
server_name YOUR_DOMAIN_OR_SERVER_IP;
```

改为你的公网 IP 或域名，然后：

```bash
sudo ln -sf /etc/nginx/sites-available/smartbin-cloud \
  /etc/nginx/sites-enabled/smartbin-cloud
sudo nginx -t
sudo systemctl reload nginx
```

云厂商安全组至少放行 TCP 80。使用 HTTPS 时再放行 443，并在 Nginx 中配置证书。

浏览器访问：

```text
http://你的公网IP/
```

## 5. 网页首次使用

1. 打开数据大屏；
2. 点击右上角“管理员登录”；
3. 输入 `/opt/smartbin-cloud/credentials.txt` 中的 `ADMIN_TOKEN`；
4. 点击“新增垃圾桶”；
5. 城市填写“哈尔滨”；
6. 立即复制设备编号和设备密钥；
7. 使用配套 RDK 端的 `configure_cloud.sh` 写入设备凭据。

设备密钥只在创建时返回一次，数据库只保存摘要。密钥遗失后应删除设备并重新创建。

## 6. Docker 部署

```bash
cd /root/smartbin_cloud_server
ADMIN_TOKEN=$(openssl rand -hex 24)
APP_SECRET=$(openssl rand -hex 32)
printf 'ADMIN_TOKEN=%s\nAPP_SECRET=%s\n' "$ADMIN_TOKEN" "$APP_SECRET" > .env
docker compose up -d --build
```

直接访问：

```text
http://服务器IP:8000/
```

正式公网部署仍建议在前面增加 Nginx 和 HTTPS。

## 7. 快速模拟设备

先在网页中创建设备，然后：

```bash
export SMARTBIN_SERVER=http://你的公网IP
export SMARTBIN_DEVICE_ID=网页生成的设备编号
export SMARTBIN_DEVICE_KEY=网页生成的设备密钥
python3 tools/device_simulator.py
```

网页中的真实设备计数将每秒变化，可用于先验证服务器、安全组、Nginx 和设备认证。

## 8. 本地开发和测试

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
python run.py
```

访问：

```text
http://127.0.0.1:8000/
```

运行接口测试：

```bash
pip install pytest httpx
PYTHONPATH=. pytest -q
```

## 9. 关键接口

设备认证请求头：

```text
X-Device-ID: 设备编号
X-Device-Key: 设备密钥
```

设备接口：

```text
POST /api/iot/v1/report
GET  /api/iot/v1/commands?command_type=reset&limit=1
GET  /api/iot/v1/commands?command_type=manual&limit=1
POST /api/iot/v1/commands/{command_id}/ack
```

管理员接口使用：

```text
X-Admin-Token: ADMIN_TOKEN
```

详细数据结构见 `API_PROTOCOL.md`。
