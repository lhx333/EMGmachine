# 五模态时间戳与帧索引审计（2026-10-12）

审计对象：`data/session_20261012_195040_a2fd91/raw/` 下的 EMG、IMU、NOKOV、RealSense、TouchGlove 原始 H5。TouchGlove Windows DLL 使用 Ghidra 12.1.3 反编译，并与 `reverse_engineering/TACTILE_DEVICE_TIMESTAMP_RE.md` 的静态/动态逆向结论交叉核对。

## 结论

这些字段并不都是设备时钟：

| 模态 | 原始时标 | 来源与单位 | 可用帧/包索引 | 当前 H5 的索引证据 | 对齐意义 |
| --- | --- | --- | --- | --- | --- |
| TouchGlove | `frames/device_timestamp_us` | Windows `touch_glove.dll` 在主机解析串口帧时调用 `GetSystemTimePreciseAsFileTime`，转换为 Unix epoch 微秒；不是设备 RTC | `frames/seq_id`，SDK 按通道递增 | ch1–5 分别 2202、2240、2255、2091、2090 帧；每通道相邻 `seq_id` 全部 +1，没有已记录序号缺口 | `timestamp_us` 是主机墙钟，可作为近似主机接收时刻；同一台 PC 上可与 RealSense `system_time` 对照。它不是设备时钟同步证据 |
| EMG | `record_json[].device_timestamp_raw` | ResearchKit `UPLOAD_DATA` 中 TIMESTAMP TLV（type=1）的 64 位小端原值，设备相对毫秒计数；无 epoch 换算 | 没有观察到设备包序号；H5 的数组行号只是主机记录顺序 | 8722 包，每包 12 个样本，共 104664 样本；时标从 512493 到 618190，未回退、未重复 | 可做整段线性映射到 QPC，并按配置的 1000 Hz 在包内展开；时间跳变可提示间断，但没有 packet sequence 不能单独证明丢包 |
| IMU | `record_json[].device_timestamp_raw` | 同一 ResearchKit TIMESTAMP TLV，设备相对毫秒计数 | 没有观察到设备包序号；H5 数组行号是主机记录顺序 | 526 包，每包 10 个样本，共 5260 样本；时标从 512386 到 617976，未回退、未重复 | 可整段映射到 QPC，并按 50 Hz 展开；包时标间隔不等于每个样本的独立设备时间 |
| NOKOV | `device_timestamp_raw` / SDK `iTimeStamp` | SDK 文档单位为毫秒；Ghidra 显示 SDK 从接收的数据包解析并复制该字段到帧结构，SDK 未在回调处用本机时间生成它。本次数值接近 Unix epoch 毫秒；字段由 NOKOV 发送端携带，但具体时钟源仍未确认 | `device_frame_index` / SDK `iFrame` | 9710 帧，54260 到 63969；每个相邻值 +1，未见重复、回退或缺号 | `iFrame` 用于检查帧间断，也用于将连续相同 `iTimeStamp` 的帧均匀放入相邻映射锚点间。NOKOV 时间轴使用首末 `(iTimeStamp, QPC)` 建立单条线性映射 |
| RealSense RGB | `camera_timestamp_ms` | SDK 帧时标，单位毫秒；本次 domain 为 `timestamp_domain.system_time`，即 SDK 报告的 system-time 域 | `camera_frame_number` | 3106 帧，700 到 3816；共发现 11 个缺失帧号（8 个间断位置），H5 `frame_index` 本身只是连续写入行号 | 相机帧号可检查丢帧。当前 QPC 是 `wait_for_frames()` 返回后的主机观察时刻；相机时标与 QPC 可在本机数据上拟合，但它们不是同一个原始时钟字段 |

所有采集适配器另有 `host_receive_mono_ns`，是 Windows QPC 单调时标；它是跨模态共同的主机时钟基准，但采样边界不同：EMG/IMU 在 SDK 数据回调中、NOKOV 在 SDK 回调入口、RealSense 在 `wait_for_frames()` 返回后、TouchGlove 在 `poll()` 返回后。它们描述主机观察时刻，不等价于传感器曝光/采样时刻。

## TouchGlove Windows DLL 逆向证据

当前 SDK DLL 的 SHA-256 为 `7a12932a912d24b85e5f576461e0ca1e92925a4d7aac40a0ead87105f2148e4e`，与既有 Windows DLL 逆向材料中的文件指纹前缀一致。Ghidra 导入了该 DLL 的 DWARF 信息；导出符号包括 `touch_glove_poll_read` 和 `touch_glove_send_rtc_sync`。后者确实构造并发送设备 RTC 的 BCD 同步命令，但采集 Python 路径不调用它。

既有定点反编译报告已追到 `poll_read` 的帧记录写入：`seq_id` 来自按通道递增的 SDK 计数器；对外 `timestamp_us` 的唯一写入来自 `GetSystemTimePreciseAsFileTime` 转 Unix 微秒。DLL 解析了串口帧中的 8 字节 BCD RTC，但没有把这个设备 RTC 写入对外 `Frame.timestamp_us`。因此 raw H5 中 `device_timestamp_us` 这个名字会误导：Windows 版本实际记录的是主机墙钟。

本次 H5 给出独立交叉验证：把 TouchGlove `timestamp_us` 转成毫秒后，与 RealSense 的 `system_time` 帧时标按其 QPC 观测时刻拟合，残差分位数约为 −9.03、−0.34、+0.79、+1.24、+1.35 ms（0、1、50、99、100 分位）。这与两者都来自同一台 PC 的 system-time 时钟一致。它验证的是主机墙钟关联，不能把 TouchGlove 字段重新解释成设备 RTC。

RTC 可从原始串口图像帧读取：协议字段位于帧字节 `[7:15]`，按 `YY MM DD hh mm ss ms/100 ms%100` 编码为 8 字节 BCD。Windows DLL 的公开 `Frame` 没有该字段，解析后对外只返回主机生成的 `timestamp_us`，所以已有 DLL 模式 H5 无法还原已丢弃的 RTC。Python 原始串口解析模式保存 `device_rtc_raw`、解码 `device_rtc_timestamp_us` 与 `device_rtc_valid`，并保留 `device_timestamp_us` 兼容字段。已在真实手套上验证：单通道帧的 RTC 相邻值未重复。集成采集现默认使用 Python 后端；时间轴按每个有数据的通道分别取首尾 RTC/QPC 锚点，建立一次线性映射。DLL 模式旧 H5 不包含 RTC，不能用新算法重建；既有派生 v1 数据仍保留。

## NOKOV `iTimeStamp` 的赋值链

本地 Python SDK 手册只明确 `FrameOfMocapData.iTimeStamp` 的单位是毫秒，没有说明时钟源。绑定把它声明为 `c_longlong`，并将它作为帧结构字段交给回调。

对 `full_system/vendor/nokov/nokov/lib/nokov_sdk.dll` 的 Ghidra 反编译追到：接收线程调用 `recvfrom` 收到 UDP 数据；数据进入分包/重组函数，再按协议命令解析完整帧。完整帧解析器将网络帧尾部连续字段复制到输出帧结构，其中 8 字节字段被原样复制到与 `iTimeStamp` 对应的位置；没有看到调用本机时钟函数后写入该帧字段的路径。SDK 的 `GetLastFrameTimeStamp` 则读取已缓存的最新帧信息。由此可以确认：**`iTimeStamp` 是 NOKOV 数据流中携带的帧字段，SDK 负责解析/复制，不是 SDK 在本机回调时临时打的时间戳。**

这仍不能证明最初由相机、NOKOV 服务端还是更上游的采集组件生成，也不能证明它属于相机硬件曝光时钟。SDK 文档和公开 FAQ 未给出该时钟的纪元/来源定义。H5 的数值接近 Unix epoch 毫秒、与 PC 的 system-time 速率近似一致，但相对回调观测约晚 9–20 ms（少数样本有更大延迟）；这是“服务端传来的墙钟型帧时标”的佐证，不是来源的最终证明。

复核入口：Ghidra 项目 `reverse_engineering/ghidra_nokov/nokov_sdk`；关键函数为 UDP 接收包装 `FUN_18000bd60`、分片重组 `FUN_180005d30`、完整帧解析 `FUN_1800018d0`、缓存帧处理 `FUN_1800082e0`。绑定字段声明位于 `full_system/vendor/nokov/nokov/nokovsdk.py` 的 `FrameOfMocapData`。

同一完整帧解析器把网络帧开头的 4 字节复制到输出结构开头，对应 `FrameOfMocapData.iFrame`；未发现 SDK 在收包后自增生成 `iFrame` 的路径。因此它是 NOKOV 发送端随每帧广播的帧编号，SDK 只是解析/转交。其编号应理解为发送端帧序列，而不是相机硬件提供的传感器包号；具体计数器何时复位由 NOKOV 软件/流会话行为决定，本地文档没有进一步说明。本次 H5 从 54260 开始且连续到 63969，说明采集开始时发送端计数器已在运行；本次接收记录范围内无帧号缺口。

当前实现以首、末记录的 `(iTimeStamp, host_receive_mono_ns)` 建立单条线性映射，并将映射结果减去 session QPC 起点。连续相同 `iTimeStamp` 的帧组保持首帧在该时间戳映射锚点，再按组内 `iFrame` 在该锚点与下一不同时间戳锚点之间插值；末尾重复组则以最后帧为锚点，从前一不同时间戳锚点反向均匀铺开。原始 `iTimeStamp`、`iFrame` 和每帧 QPC 都写入派生 H5 供审计。此规则沿用 EMG“重复设备时标按包内/组内序号展开”的思路，但时间戳来源未知，线性速率假设仍须通过实际采集验证。

## 本次数据的索引和时间规律

- TouchGlove 每通道记录数为 2090–2255。每通道内部 `seq_id` 全部连续，能证明 SDK 已交给采集循环的帧没有在已保存记录中跳号；它不能证明设备到串口、或串口到 DLL 解析之间没有丢帧，因为此计数器不是设备发出的帧号。不同通道的帧会由 SDK 分队列返回，整张 H5 的行顺序不是全局采样顺序。本次全局行序中有 742 次 `device_timestamp_us` 下降；每个通道各自的时标则都递增。这解释了当前 `force_timeline` 统计出的回退，不能据此认定设备时钟回退。
- NOKOV `iFrame` 与本次 callback 行数严格一一递增；`iTimeStamp` 每帧递增 11 或 12 ms，符合约 90 Hz 的节奏。帧号足以检查本次回调序列中的缺号；它不能指出缺帧发生在相机、NOKOV 服务端、网络还是本机 SDK 回调。
- RealSense `camera_frame_number` 有 11 个缺失编号；这比连续的本地主机 `frame_index` 更有意义。`camera_timestamp_ms` 单调递增，步长中位数约 33.35 ms。它和 QPC 都在同一电脑上观测，但实际输出图像时刻与 `wait_for_frames()` 返回时刻之间仍有队列延迟。
- EMG/IMU 两个数据流的 device-time 均单调递增且无重复。协议逆向确认 EMG 每个上传包含 12 个样本、IMU 每个上传包含 10 个样本；当前采集配置分别是 1000 Hz 与 50 Hz。协议中没有发现可用于逐包判缺的设备序号，所以时标间隔只能用于异常提示，不能等同于确证丢包数。

## 对现有实现的直接影响

1. 原 TouchGlove DLL raw H5 的 `device_timestamp_us` 实际是主机墙钟 Unix 微秒。原 H5 保留原样作为采集证据；当前 Python 后端继续保留该兼容字段，同时单独记录并使用 `device_rtc_timestamp_us`。
2. 已废弃的 `force_session_endpoint_map_v1` 把 DLL 的主机墙钟时间按全局首末行映射。新 `force_channel_rtc_endpoint_map_v2` 使用 Python 解析得到的 BCD RTC，并按通道独立建立 RTC→QPC 首尾一次函数，保留原始行顺序和帧引用；不把不同通道间相同的 RTC 当作重复帧。
3. 丢帧检查优先使用真实帧号：NOKOV `iFrame`、RealSense `camera_frame_number`。TouchGlove `seq_id` 只报告 SDK 接收/缓存之后的帧连续性。EMG/IMU 当前没有可靠设备包序号。
4. 所有时标和索引都应原样保存。为对齐生成的 `time_ns` 是派生值；不要覆盖原始值，也不要把推算出来的样本索引标成设备提供的 sequence number。

## 证据位置

- TouchGlove 逆向报告：`reverse_engineering/TACTILE_DEVICE_TIMESTAMP_RE.md`，尤其 Windows DLL 与 `sequence_id` 两节。
- ResearchKit 协议逆向：`reverse_engineering/SDK_DEVICE_TIME_UNIX_TIME_RE.md`、`reverse_engineering/RSK_PROTOCOL_SPEC.md`。
- 本次原始数据：`full_system/data/session_20261012_195040_a2fd91/raw/{force,emg,imu,nokov,realsense_rgb}.h5`；会话时钟与采样率见同目录上层 `manifest.json`。
- H5 中的 `record_json` 保留应用层收到的 SDK 字段；其中 JSON 数组行号不是额外的设备侧帧号。
