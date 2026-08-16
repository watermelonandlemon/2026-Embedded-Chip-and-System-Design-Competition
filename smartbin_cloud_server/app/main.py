from __future__ import annotations

import hmac
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .db import Database
from .schemas import CommandAck, DeviceCreate, DeviceReport, ManualCommand


STATIC_DIR = Path(__file__).resolve().parent / "static"

FAKE_CITIES = [
    {"city": "北京", "x": 69, "y": 28, "devices": 126, "classified": 318_420},
    {"city": "上海", "x": 78, "y": 50, "devices": 98, "classified": 274_880},
    {"city": "广州", "x": 64, "y": 78, "devices": 86, "classified": 231_760},
    {"city": "深圳", "x": 69, "y": 82, "devices": 72, "classified": 205_140},
    {"city": "成都", "x": 43, "y": 59, "devices": 61, "classified": 168_330},
    {"city": "武汉", "x": 59, "y": 56, "devices": 55, "classified": 151_790},
    {"city": "西安", "x": 49, "y": 45, "devices": 48, "classified": 137_020},
    {"city": "杭州", "x": 73, "y": 55, "devices": 44, "classified": 126_900},
    {"city": "南京", "x": 69, "y": 49, "devices": 39, "classified": 117_340},
    {"city": "重庆", "x": 47, "y": 63, "devices": 42, "classified": 120_510},
]

CATEGORY_MAP = {
    "0001": "可回收垃圾",
    "0010": "厨余垃圾",
    "0100": "有害垃圾",
    "1000": "其他垃圾",
}


def iso_time(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(
        path=settings.database_path,
        app_secret=settings.app_secret,
        history_sample_seconds=settings.history_sample_seconds,
        history_retention_days=settings.history_retention_days,
    )
    db.initialize()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.db = db
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def require_admin(
        authorization: Annotated[str | None, Header()] = None,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> None:
        token = x_admin_token
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token or not hmac.compare_digest(token, settings.admin_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="管理员令牌无效",
            )

    def device_auth(
        x_device_id: Annotated[str | None, Header()] = None,
        x_device_key: Annotated[str | None, Header()] = None,
    ) -> str:
        if not x_device_id or not x_device_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="缺少设备认证信息",
            )
        if not db.authenticate_device(x_device_id, x_device_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="设备编号或密钥错误",
            )
        return x_device_id

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": settings.app_name,
            "server_time": iso_time(time.time()),
        }

    @app.get("/api/public/dashboard")
    async def public_dashboard() -> dict[str, Any]:
        if not settings.public_dashboard:
            raise HTTPException(status_code=403, detail="公开数据大屏已关闭")

        now = time.time()
        raw_devices = db.list_devices()
        devices: list[dict[str, Any]] = []
        # 数据大屏的四类实时统计只汇总“哈尔滨”真实节点。
        harbin_totals = {"recyclable": 0, "kitchen": 0, "hazardous": 0, "other": 0}
        harbin_online_count = 0
        all_online_count = 0

        for row in raw_devices:
            last_seen = row.get("last_seen")
            online = bool(last_seen and now - float(last_seen) <= settings.device_offline_seconds)
            if online:
                all_online_count += 1
                if row["city"] == "哈尔滨":
                    harbin_online_count += 1
            counts = {
                "recyclable": int(row["recyclable"]),
                "kitchen": int(row["kitchen"]),
                "hazardous": int(row["hazardous"]),
                "other": int(row["other"]),
            }
            if row["city"] == "哈尔滨":
                for key, value in counts.items():
                    harbin_totals[key] += value
            devices.append(
                {
                    "device_id": row["device_id"],
                    "name": row["name"],
                    "city": row["city"],
                    "longitude": row["longitude"],
                    "latitude": row["latitude"],
                    "online": online,
                    "last_seen": iso_time(last_seen),
                    "last_seen_epoch": last_seen,
                    "counts": counts,
                    "total": sum(counts.values()),
                    "firmware": row.get("firmware"),
                    "pending_commands": int(row.get("pending_commands", 0)),
                }
            )

        harbin_ids = [d["device_id"] for d in devices if d["city"] == "哈尔滨"]
        history_rows = db.recent_history(harbin_ids or [d["device_id"] for d in devices], 180)
        history = [
            {
                "time": iso_time(row["reported_at"]),
                "epoch": row["reported_at"],
                "recyclable": row["recyclable"],
                "kitchen": row["kitchen"],
                "hazardous": row["hazardous"],
                "other": row["other"],
                "total": row["recyclable"] + row["kitchen"] + row["hazardous"] + row["other"],
            }
            for row in history_rows
        ]

        fake_device_count = sum(city["devices"] for city in FAKE_CITIES)
        fake_classified = sum(city["classified"] for city in FAKE_CITIES)
        harbin_classified = sum(harbin_totals.values())
        harbin_device_count = len(harbin_ids)

        return {
            "generated_at": iso_time(now),
            "overview": {
                "simulated_devices": fake_device_count,
                "real_devices": harbin_device_count,
                "real_online_devices": harbin_online_count,
                "all_real_devices": len(devices),
                "all_real_online_devices": all_online_count,
                "network_devices": fake_device_count + len(devices),
                "classified_total": fake_classified + harbin_classified,
                "real_classified_total": harbin_classified,
            },
            "real_totals": harbin_totals,
            "fake_cities": FAKE_CITIES,
            "devices": devices,
            "history": history,
            "category_codes": CATEGORY_MAP,
        }

    @app.post("/api/admin/devices", dependencies=[Depends(require_admin)])
    async def create_device(payload: DeviceCreate) -> dict[str, Any]:
        try:
            result = db.create_device(
                name=payload.name,
                city=payload.city,
                longitude=payload.longitude,
                latitude=payload.latitude,
                note=payload.note,
                requested_device_id=payload.device_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            **result,
            "warning": "设备密钥只在本次创建时返回，请立即保存。",
        }

    @app.get("/api/admin/devices", dependencies=[Depends(require_admin)])
    async def admin_list_devices() -> dict[str, Any]:
        now = time.time()
        rows = db.list_devices()
        return {
            "devices": [
                {
                    **row,
                    "created_at": iso_time(row["created_at"]),
                    "last_seen": iso_time(row["last_seen"]),
                    "online": bool(
                        row["last_seen"]
                        and now - float(row["last_seen"]) <= settings.device_offline_seconds
                    ),
                    "key_digest": None,
                }
                for row in rows
            ]
        }

    @app.delete("/api/admin/devices/{device_id}", dependencies=[Depends(require_admin)])
    async def delete_device(device_id: str) -> dict[str, bool]:
        if not db.delete_device(device_id):
            raise HTTPException(status_code=404, detail="设备不存在")
        return {"deleted": True}

    @app.post("/api/admin/devices/{device_id}/reset", dependencies=[Depends(require_admin)])
    async def reset_device(device_id: str) -> dict[str, Any]:
        try:
            return db.create_command(
                device_id=device_id,
                command_type="reset",
                payload={"scope": "all_counts"},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="设备不存在") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/admin/devices/{device_id}/manual", dependencies=[Depends(require_admin)])
    async def manual_device(device_id: str, payload: ManualCommand) -> dict[str, Any]:
        try:
            return db.create_command(
                device_id=device_id,
                command_type="manual",
                payload={
                    "code": payload.code,
                    "category": CATEGORY_MAP[payload.code],
                },
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="设备不存在") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get(
        "/api/admin/devices/{device_id}/commands",
        dependencies=[Depends(require_admin)],
    )
    async def command_history(device_id: str) -> dict[str, Any]:
        if not db.get_device(device_id):
            raise HTTPException(status_code=404, detail="设备不存在")
        rows = db.command_history(device_id)
        for row in rows:
            row["created_at"] = iso_time(row["created_at"])
            row["ack_at"] = iso_time(row["ack_at"])
            row.pop("payload_json", None)
        return {"commands": rows}

    @app.post("/api/iot/v1/report")
    async def device_report(
        payload: DeviceReport,
        request: Request,
        device_id: str = Depends(device_auth),
    ) -> dict[str, Any]:
        forwarded = request.headers.get("x-forwarded-for")
        ip_address = forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else None
        return db.update_report(
            device_id=device_id,
            counts={
                "recyclable": payload.recyclable,
                "kitchen": payload.kitchen,
                "hazardous": payload.hazardous,
                "other": payload.other,
            },
            firmware=payload.firmware,
            ip_address=ip_address,
        )

    @app.get("/api/iot/v1/commands")
    async def pull_commands(
        device_id: str = Depends(device_auth),
        command_type: str | None = Query(default=None, pattern="^(reset|manual)$"),
        limit: int = Query(default=5, ge=1, le=20),
    ) -> dict[str, Any]:
        return {
            "commands": db.lease_commands(
                device_id=device_id,
                command_type=command_type,
                limit=limit,
            )
        }

    @app.post("/api/iot/v1/commands/{command_id}/ack")
    async def ack_command(
        command_id: str,
        payload: CommandAck,
        device_id: str = Depends(device_auth),
    ) -> dict[str, bool]:
        ok = db.ack_command(
            device_id=device_id,
            command_id=command_id,
            success=payload.success,
            message=payload.message,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="命令不存在")
        return {"acknowledged": True}

    return app


app = create_app()
