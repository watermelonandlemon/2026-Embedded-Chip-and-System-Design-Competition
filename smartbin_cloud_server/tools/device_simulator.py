#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在电脑或服务器上模拟 RDK 每秒上报，便于先验证网页。"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request

SERVER = os.getenv("SMARTBIN_SERVER", "http://127.0.0.1:8000").rstrip("/")
DEVICE_ID = os.getenv("SMARTBIN_DEVICE_ID", "")
DEVICE_KEY = os.getenv("SMARTBIN_DEVICE_KEY", "")
counts = {"recyclable": 0, "kitchen": 0, "hazardous": 0, "other": 0}

while True:
    category = random.choice(list(counts))
    counts[category] += 1
    request = urllib.request.Request(
        SERVER + "/api/iot/v1/report",
        method="POST",
        data=json.dumps({**counts, "firmware": "simulator-1.0"}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Device-ID": DEVICE_ID,
            "X-Device-Key": DEVICE_KEY,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            print(category, counts, response.status)
    except Exception as exc:  # noqa: BLE001
        print("ERROR", exc)
    time.sleep(1)
