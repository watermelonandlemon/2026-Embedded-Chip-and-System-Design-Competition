from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    database_path: Path
    admin_token: str
    app_secret: str
    device_offline_seconds: int
    history_sample_seconds: int
    history_retention_days: int
    public_dashboard: bool
    docs_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        database_path = Path(
            os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "smartbin.db"))
        ).expanduser()
        database_path.parent.mkdir(parents=True, exist_ok=True)

        admin_token = os.getenv("ADMIN_TOKEN", "change-this-admin-token")
        app_secret = os.getenv("APP_SECRET", "change-this-app-secret")

        return cls(
            app_name=os.getenv("APP_NAME", "智能垃圾分类物联网平台"),
            database_path=database_path,
            admin_token=admin_token,
            app_secret=app_secret,
            device_offline_seconds=max(
                3, int(os.getenv("DEVICE_OFFLINE_SECONDS", "8"))
            ),
            history_sample_seconds=max(
                1, int(os.getenv("HISTORY_SAMPLE_SECONDS", "5"))
            ),
            history_retention_days=max(
                1, int(os.getenv("HISTORY_RETENTION_DAYS", "7"))
            ),
            public_dashboard=_as_bool(os.getenv("PUBLIC_DASHBOARD"), True),
            docs_enabled=_as_bool(os.getenv("DOCS_ENABLED"), False),
        )
