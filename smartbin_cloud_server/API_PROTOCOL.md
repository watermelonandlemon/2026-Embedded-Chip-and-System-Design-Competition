# 自定义轻量 IoT 接口协议

## 1. 设备认证

每个设备使用两项凭据：

```text
X-Device-ID: HB-123456
X-Device-Key: 创建设备时返回的密钥
```

设备密钥只在创建时返回一次，服务器数据库只保存摘要。

## 2. 四类垃圾与状态码

| 状态码 | 类别 | JSON 字段 |
|---|---|---|
| `0001` | 可回收垃圾 | `recyclable` |
| `0010` | 厨余垃圾 | `kitchen` |
| `0100` | 有害垃圾 | `hazardous` |
| `1000` | 其他垃圾 | `other` |

如你的实际 GPIO/气缸顺序不同，只修改 RDK 端的状态码映射，不需要修改服务器数据库结构。

## 3. 设备每秒上报

`POST /api/iot/v1/report`

```json
{
  "recyclable": 12,
  "kitchen": 8,
  "hazardous": 3,
  "other": 9,
  "firmware": "rdk-x5-node6-1.0",
  "local_status": "running"
}
```

计数必须是非负整数。服务器更新当前值，并按 `HISTORY_SAMPLE_SECONDS` 对趋势数据降采样。

## 4. 拉取云端命令

清零节点：

```text
GET /api/iot/v1/commands?command_type=reset&limit=3
```

手动控制节点：

```text
GET /api/iot/v1/commands?command_type=manual&limit=1
```

返回示例：

```json
{
  "commands": [
    {
      "command_id": "UUID",
      "command_type": "manual",
      "payload": {"code": "0001", "category": "可回收垃圾"},
      "created_at": 1785000000.0,
      "lease_until": 1785000012.0
    }
  ]
}
```

命令采用租约机制。RDK 未回执时，租约到期后服务器会重新下发。

## 5. 命令回执

`POST /api/iot/v1/commands/{command_id}/ack`

```json
{
  "success": true,
  "message": "命令已投递给运动执行节点"
}
```

## 6. 管理接口

管理接口统一使用：

```text
X-Admin-Token: 服务器 .env 中的 ADMIN_TOKEN
```

主要接口：

- `POST /api/admin/devices`：新增垃圾桶并返回编号、密钥。
- `GET /api/admin/devices`：查询真实设备。
- `POST /api/admin/devices/{device_id}/reset`：下发清零命令。
- `POST /api/admin/devices/{device_id}/manual`：下发四选一手动状态码。
- `DELETE /api/admin/devices/{device_id}`：删除设备。
