#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "请使用 root 执行：sudo bash deploy/install_ubuntu.sh"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=/opt/smartbin-cloud
SERVICE_USER=smartbin

apt-get update
apt-get install -y python3 python3-venv nginx openssl rsync

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --home "$TARGET" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

mkdir -p "$TARGET" "$TARGET/data"
rsync -a --delete \
  --exclude '.venv' --exclude '.env' --exclude 'data/*.db*' \
  "$PROJECT_DIR/" "$TARGET/"

python3 -m venv "$TARGET/.venv"
"$TARGET/.venv/bin/python" -m pip install --upgrade pip
"$TARGET/.venv/bin/pip" install -r "$TARGET/requirements.txt"

if [[ ! -f "$TARGET/.env" ]]; then
  ADMIN_TOKEN="$(openssl rand -hex 24)"
  APP_SECRET="$(openssl rand -hex 32)"
  cat > "$TARGET/.env" <<EOF
APP_NAME=智能垃圾分类物联网平台
DATABASE_PATH=$TARGET/data/smartbin.db
ADMIN_TOKEN=$ADMIN_TOKEN
APP_SECRET=$APP_SECRET
DEVICE_OFFLINE_SECONDS=8
HISTORY_SAMPLE_SECONDS=5
HISTORY_RETENTION_DAYS=7
PUBLIC_DASHBOARD=true
DOCS_ENABLED=false
EOF
  cat > "$TARGET/credentials.txt" <<EOF
管理员令牌（ADMIN_TOKEN）：$ADMIN_TOKEN
请妥善保存，网页管理员登录时使用。
EOF
  chmod 600 "$TARGET/.env" "$TARGET/credentials.txt"
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$TARGET"
chmod 750 "$TARGET" "$TARGET/data"

cp "$TARGET/deploy/smartbin-cloud.service" /etc/systemd/system/smartbin-cloud.service
systemctl daemon-reload
systemctl enable --now smartbin-cloud.service

cat <<EOF

服务已安装：
  systemctl status smartbin-cloud
  journalctl -u smartbin-cloud -f

管理员令牌：
  sudo cat $TARGET/credentials.txt

下一步：
1. 修改 $TARGET/deploy/nginx-smartbin.conf 中的 server_name。
2. 复制到 /etc/nginx/sites-available/smartbin-cloud。
3. 建立 sites-enabled 软链接并重载 Nginx。
EOF
