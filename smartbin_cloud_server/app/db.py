from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import string
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


CITY_PREFIX = {
    "哈尔滨": "HB",
    "北京": "BJ",
    "上海": "SH",
    "广州": "GZ",
    "深圳": "SZ",
    "成都": "CD",
    "武汉": "WH",
    "西安": "XA",
    "杭州": "HZ",
    "南京": "NJ",
    "重庆": "CQ",
}


class Database:
    def __init__(
        self,
        path: Path,
        app_secret: str,
        history_sample_seconds: int,
        history_retention_days: int,
    ) -> None:
        self.path = Path(path)
        self.app_secret = app_secret.encode("utf-8")
        self.history_sample_seconds = history_sample_seconds
        self.history_retention_days = history_retention_days
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL UNIQUE,
            key_digest TEXT NOT NULL,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            longitude REAL,
            latitude REAL,
            created_at REAL NOT NULL,
            last_seen REAL,
            recyclable INTEGER NOT NULL DEFAULT 0,
            kitchen INTEGER NOT NULL DEFAULT 0,
            hazardous INTEGER NOT NULL DEFAULT 0,
            other INTEGER NOT NULL DEFAULT 0,
            firmware TEXT,
            ip_address TEXT,
            note TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            reported_at REAL NOT NULL,
            recyclable INTEGER NOT NULL,
            kitchen INTEGER NOT NULL,
            hazardous INTEGER NOT NULL,
            other INTEGER NOT NULL,
            FOREIGN KEY(device_id) REFERENCES devices(device_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_reports_device_time
            ON reports(device_id, reported_at DESC);

        CREATE TABLE IF NOT EXISTS commands (
            command_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            command_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            lease_until REAL,
            ack_at REAL,
            result_message TEXT,
            FOREIGN KEY(device_id) REFERENCES devices(device_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_commands_device_status
            ON commands(device_id, status, created_at);
        """
        with self.connect() as conn:
            conn.executescript(schema)

    def _digest_key(self, key: str) -> str:
        return hmac.new(
            self.app_secret,
            key.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def authenticate_device(self, device_id: str, device_key: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT key_digest FROM devices WHERE device_id=?",
                (device_id,),
            ).fetchone()
        if not row:
            return False
        return hmac.compare_digest(row["key_digest"], self._digest_key(device_key))

    def create_device(
        self,
        name: str,
        city: str,
        longitude: float | None,
        latitude: float | None,
        note: str = "",
        requested_device_id: str | None = None,
    ) -> dict[str, Any]:
        device_key = secrets.token_urlsafe(24)
        key_digest = self._digest_key(device_key)
        now = time.time()

        for _ in range(20):
            device_id = requested_device_id or self._generate_device_id(city)
            try:
                with self.connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO devices(
                            device_id, key_digest, name, city, longitude, latitude,
                            created_at, note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            device_id,
                            key_digest,
                            name.strip(),
                            city.strip(),
                            longitude,
                            latitude,
                            now,
                            note.strip(),
                        ),
                    )
                return {
                    "device_id": device_id,
                    "device_key": device_key,
                    "name": name,
                    "city": city,
                    "created_at": now,
                }
            except sqlite3.IntegrityError:
                if requested_device_id:
                    raise ValueError("设备编号已存在")
        raise RuntimeError("无法生成唯一设备编号")

    def _generate_device_id(self, city: str) -> str:
        prefix = CITY_PREFIX.get(city.strip(), "BIN")
        suffix = "".join(secrets.choice(string.digits) for _ in range(6))
        return f"{prefix}-{suffix}"

    def list_devices(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM commands c
                     WHERE c.device_id=d.device_id
                       AND c.status IN ('pending','processing')) AS pending_commands
                FROM devices d
                ORDER BY CASE WHEN d.city='哈尔滨' THEN 0 ELSE 1 END,
                         d.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM devices WHERE device_id=?",
                (device_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_device(self, device_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM devices WHERE device_id=?",
                (device_id,),
            )
        return cur.rowcount > 0

    def update_report(
        self,
        device_id: str,
        counts: dict[str, int],
        firmware: str | None,
        ip_address: str | None,
    ) -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            current = conn.execute(
                "SELECT last_seen FROM devices WHERE device_id=?",
                (device_id,),
            ).fetchone()
            if current is None:
                raise KeyError(device_id)

            conn.execute(
                """
                UPDATE devices SET
                    last_seen=?, recyclable=?, kitchen=?, hazardous=?, other=?,
                    firmware=COALESCE(?, firmware), ip_address=?
                WHERE device_id=?
                """,
                (
                    now,
                    counts["recyclable"],
                    counts["kitchen"],
                    counts["hazardous"],
                    counts["other"],
                    firmware,
                    ip_address,
                    device_id,
                ),
            )

            last_history = conn.execute(
                """
                SELECT reported_at FROM reports
                WHERE device_id=?
                ORDER BY reported_at DESC LIMIT 1
                """,
                (device_id,),
            ).fetchone()
            if (
                last_history is None
                or now - float(last_history["reported_at"])
                >= self.history_sample_seconds
            ):
                conn.execute(
                    """
                    INSERT INTO reports(
                        device_id, reported_at, recyclable, kitchen, hazardous, other
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device_id,
                        now,
                        counts["recyclable"],
                        counts["kitchen"],
                        counts["hazardous"],
                        counts["other"],
                    ),
                )

            # 轻量清理：每次报告最多删除一次超期数据。
            cutoff = now - self.history_retention_days * 86400
            conn.execute(
                "DELETE FROM reports WHERE reported_at < ?",
                (cutoff,),
            )

        return {"accepted": True, "server_time": now}

    def create_command(
        self,
        device_id: str,
        command_type: str,
        payload: dict[str, Any],
        manual_cooldown_seconds: int = 5,
    ) -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM devices WHERE device_id=?",
                (device_id,),
            ).fetchone():
                raise KeyError(device_id)

            if command_type == "manual":
                active = conn.execute(
                    """
                    SELECT command_id FROM commands
                    WHERE device_id=? AND command_type='manual'
                      AND status IN ('pending','processing')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (device_id,),
                ).fetchone()
                if active:
                    raise RuntimeError("已有待执行的手动控制命令")

                last = conn.execute(
                    """
                    SELECT created_at FROM commands
                    WHERE device_id=? AND command_type='manual'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (device_id,),
                ).fetchone()
                if last and now - float(last["created_at"]) < manual_cooldown_seconds:
                    remaining = manual_cooldown_seconds - (now - float(last["created_at"]))
                    raise RuntimeError(f"手动控制冷却中，请等待 {remaining:.1f} 秒")

            if command_type == "reset":
                active = conn.execute(
                    """
                    SELECT command_id FROM commands
                    WHERE device_id=? AND command_type='reset'
                      AND status IN ('pending','processing')
                    LIMIT 1
                    """,
                    (device_id,),
                ).fetchone()
                if active:
                    raise RuntimeError("该设备已有待执行的清零命令")

            command_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO commands(
                    command_id, device_id, command_type, payload_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (
                    command_id,
                    device_id,
                    command_type,
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )

        return {
            "command_id": command_id,
            "device_id": device_id,
            "command_type": command_type,
            "payload": payload,
            "status": "pending",
            "created_at": now,
        }

    def lease_commands(
        self,
        device_id: str,
        command_type: str | None,
        limit: int,
        lease_seconds: int = 12,
    ) -> list[dict[str, Any]]:
        now = time.time()
        lease_until = now + lease_seconds
        with self.connect() as conn:
            where = "device_id=? AND (status='pending' OR (status='processing' AND lease_until<?))"
            params: list[Any] = [device_id, now]
            if command_type:
                where += " AND command_type=?"
                params.append(command_type)

            rows = conn.execute(
                f"""
                SELECT * FROM commands
                WHERE {where}
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()

            result: list[dict[str, Any]] = []
            for row in rows:
                conn.execute(
                    """
                    UPDATE commands
                    SET status='processing', lease_until=?
                    WHERE command_id=?
                    """,
                    (lease_until, row["command_id"]),
                )
                result.append(
                    {
                        "command_id": row["command_id"],
                        "command_type": row["command_type"],
                        "payload": json.loads(row["payload_json"]),
                        "created_at": row["created_at"],
                        "lease_until": lease_until,
                    }
                )
        return result

    def ack_command(
        self,
        device_id: str,
        command_id: str,
        success: bool,
        message: str,
    ) -> bool:
        now = time.time()
        status = "done" if success else "failed"
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT command_type FROM commands
                WHERE command_id=? AND device_id=?
                """,
                (command_id, device_id),
            ).fetchone()
            if not row:
                return False

            conn.execute(
                """
                UPDATE commands
                SET status=?, ack_at=?, result_message=?, lease_until=NULL
                WHERE command_id=? AND device_id=?
                """,
                (status, now, message[:500], command_id, device_id),
            )

            if success and row["command_type"] == "reset":
                conn.execute(
                    """
                    UPDATE devices SET recyclable=0, kitchen=0, hazardous=0, other=0
                    WHERE device_id=?
                    """,
                    (device_id,),
                )
                conn.execute(
                    """
                    INSERT INTO reports(
                        device_id, reported_at, recyclable, kitchen, hazardous, other
                    ) VALUES (?, ?, 0, 0, 0, 0)
                    """,
                    (device_id, now),
                )
        return True

    def recent_history(self, device_ids: list[str], limit: int = 180) -> list[dict[str, Any]]:
        if not device_ids:
            return []
        placeholders = ",".join("?" for _ in device_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT reported_at, recyclable, kitchen, hazardous, other
                FROM reports
                WHERE device_id IN ({placeholders})
                ORDER BY reported_at DESC
                LIMIT ?
                """,
                (*device_ids, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def command_history(self, device_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT command_id, command_type, payload_json, status,
                       created_at, ack_at, result_message
                FROM commands
                WHERE device_id=?
                ORDER BY created_at DESC LIMIT ?
                """,
                (device_id, limit),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]
