from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Counts(BaseModel):
    recyclable: int = Field(ge=0, le=1_000_000_000)
    kitchen: int = Field(ge=0, le=1_000_000_000)
    hazardous: int = Field(ge=0, le=1_000_000_000)
    other: int = Field(ge=0, le=1_000_000_000)


class DeviceReport(Counts):
    firmware: str | None = Field(default=None, max_length=80)
    local_status: str | None = Field(default=None, max_length=100)


class DeviceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    city: str = Field(min_length=1, max_length=40)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    note: str = Field(default="", max_length=300)
    device_id: str | None = Field(default=None, min_length=3, max_length=64)

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        if any(ch not in allowed for ch in value):
            raise ValueError("设备编号只能包含字母、数字、连字符和下划线")
        return value


class ManualCommand(BaseModel):
    code: Literal["0001", "0010", "0100", "1000"]


class CommandAck(BaseModel):
    success: bool
    message: str = Field(default="", max_length=500)
