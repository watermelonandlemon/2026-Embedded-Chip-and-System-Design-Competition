from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DATABASE_PATH", "/tmp/smartbin-import-test.db")
os.environ.setdefault("ADMIN_TOKEN", "import-test-admin")
os.environ.setdefault("APP_SECRET", "import-test-secret")

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        app_name="test",
        database_path=tmp_path / "test.db",
        admin_token="admin-test-token",
        app_secret="app-secret-for-test",
        device_offline_seconds=8,
        history_sample_seconds=1,
        history_retention_days=1,
        public_dashboard=True,
        docs_enabled=False,
    )
    return TestClient(create_app(settings))


def test_device_lifecycle(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    admin = {"X-Admin-Token": "admin-test-token"}

    created = client.post(
        "/api/admin/devices",
        headers=admin,
        json={"name": "测试桶", "city": "哈尔滨", "device_id": "HB-TEST-01"},
    )
    assert created.status_code == 200, created.text
    credentials = created.json()
    device_headers = {
        "X-Device-ID": credentials["device_id"],
        "X-Device-Key": credentials["device_key"],
    }

    report = client.post(
        "/api/iot/v1/report",
        headers=device_headers,
        json={"recyclable": 3, "kitchen": 2, "hazardous": 1, "other": 4},
    )
    assert report.status_code == 200, report.text

    dashboard = client.get("/api/public/dashboard")
    assert dashboard.status_code == 200
    data = dashboard.json()
    assert data["overview"]["real_devices"] == 1
    assert data["real_totals"]["other"] == 4

    command = client.post(
        "/api/admin/devices/HB-TEST-01/reset",
        headers=admin,
        json={},
    )
    assert command.status_code == 200, command.text

    pulled = client.get(
        "/api/iot/v1/commands?command_type=reset",
        headers=device_headers,
    )
    assert pulled.status_code == 200
    command_id = pulled.json()["commands"][0]["command_id"]

    ack = client.post(
        f"/api/iot/v1/commands/{command_id}/ack",
        headers=device_headers,
        json={"success": True, "message": "done"},
    )
    assert ack.status_code == 200
    assert client.get("/api/public/dashboard").json()["overview"]["real_classified_total"] == 0
