# ResearchKit EMG 丢包字段复核

复核日期：2026-10-06。对象为工作区 ResearchKit SDK 1.0.0、CPython 3.10 Windows 原生模块。本次直接导入 `.pyd`，检查 dataclass、属性、协议枚举和未打开硬件的 BLEManager 实例，并对照已有协议抓包分析。未启动设备采集。

## 运行时结果

`SensorData` 的实际 dataclass 字段只有 `sensor_type`、`timestamp`、`device_timestamp`、`data`、`sample_rate`。没有设备包号、累计样本号、设备丢包计数或溢出状态。

构造 12×9 数据矩阵得到 `sample_count=12`、`duration_ms=12`（1000 Hz）；将矩阵截为 7×9 后得到 `sample_count=7`、`duration_ms=7`。这证明它们是根据 SDK 输出矩阵和配置采样率计算的属性，不是设备累计计数。

`BLEFrame` 只有 `service_id`、`command_id`、`tlv_nodes`。UPLOAD_DATA 枚举只有 TIMESTAMP 和 EMG/IMU/PPG/EDA 数据块，未发现实时序号 TLV。初始化后的 BLEManager 属性未发现收包累计序号或丢包计数。解析方法的局部变量 `packet_count` 与协议载荷首字节的包内样本数相对应，不能仅凭变量名认作累计包号。

## 候选字段及适用范围

| 字段/信息 | 来源与含义 | 丢包判断能力 |
|---|---|---|
| `device_timestamp` | 设备上传帧 TIMESTAMP TLV，uint64 小端原值 | 可作为疑似时间缺口证据；不能保证缺失包数或样本数 |
| DATA_EMG 首字节 N | 设备给出的本包样本数；后接 N×9 个 int16 值，按样本排列 | 说明本包长度，不提供跨包连续性 |
| `SensorData.sample_count` | SDK 输出矩阵行数 | 不是设备累计样本号；不能识别完全缺失的数据包 |
| 数据矩阵行号 | 包内样本偏移 0..N-1 | 可以定位包内样本，不能识别包间缺口 |
| 帧 CRC16 | 帧完整性校验 | 可检出收到的损坏帧，无法检出一整帧完全未到达；SDK 当前未向业务回调提供 CRC 丢弃计数 |
| OFFLINE_FILE_INDEX / OFFSET / SIZE / CRC | 离线文件编号、下载偏移、文件大小及文件校验 | 用于离线文件传输完整性；不是实时 EMG 包号，相关设备行为尚未实测 |
| 连接状态/断开事件 | BLE 连接状态 | 可报告连接中断，无法给出中断期间本应产生的样本总数 |

协议载荷布局与历史逐位核对证据见 [RSK_PROTOCOL_SPEC.md](../../reverse_engineering/RSK_PROTOCOL_SPEC.md) §6.4。SDK 时间来源及历史量化统计见 [SDK_DEVICE_TIME_UNIX_TIME_RE.md](../../reverse_engineering/SDK_DEVICE_TIME_UNIX_TIME_RE.md)。

## 当前判断边界

现有实时接口没有能严格证明包连续性的设备序号。设备时间与包内样本数可以联合生成 `possible_gap` 诊断，但应保留计时量化、采样时钟偏差、设备暂停及重置等解释。历史报告测得约 9.72–9.74 ms 的计时量化尺度；这不是当前所有设备/固件都已确认的保证，不能用固定“三个步长”规则断言恰好丢失一个包。

精确识别设备端产生后丢失的包，需要设备提供每包递增序号或包首累计采样号，最好同时提供采样溢出/丢弃计数。仅增加主机接收序号不能补足这项证据。

## 本次被检查二进制的 SHA-256

| 模块 | SHA-256 |
|---|---|
| `models` | `5ad7401af758fd5428ab7d380885615816356197caaf10e155604b145797fc45` |
| `ble_manager` | `d4c19e20537bb482fbb2b2d442ddb0e3a2c94bcbab01f400a7fcc46f49df4ad0` |
| `rsk_protocol` | `8e61319f825dbe9a43291b76832b0be5967362abfee6e6370ec2c97bc0063f7e` |
| `tlv_parse` | `0bb56bddd8a212ed5fb4be36b3021de8050de589161d80bf02b5f34fcd25a11a` |
