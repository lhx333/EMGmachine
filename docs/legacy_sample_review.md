# `example_sample` 旧采集样例

该目录用于验证旧格式识别和迁移报告。导入器会按原相对路径复制文件，计算 SHA-256，并记录 CSV 列名、行数、HDF5 属性/数据集形状，以及可解析时间列的单调性。原始文件不会被改写。

已确认的主要结构：

| 来源文件 | 可见字段/数据集 | 导入处理 |
|---|---|---|
| `windows/01_emg_aligned.csv` | `time`、`processed_time`、`system_timestamp_ns`、`host_perf_ns`、`Unix_Time`、`Device_Time`、`EMG_CH1`…`EMG_CH9` | 保留全部列和值；不把 `time` 自动认作可信采样时间 |
| `windows/05_imu_aligned.csv` | 同类时间字段及 `AX/AY/AZ/GX/GY/GZ` | 保留设备单位与时间语义的不确定性 |
| `windows/02_nokov_human_markers_aligned.csv` | `time`、`system_timestamp_ns`、`host_perf_ns`、各 marker XYZ（mm） | 坐标按旧值原样保留；缺失/哨兵值后续由专用解析器识别 |
| `windows/03_realsense_timestamps.csv` | `frame_index`、`camera_frame_number`、系统/单调/设备时间 | 和 MP4 一起原样保留；不以 MP4 帧率推导采样时间 |
| `windows/03_realsense_rgb.mp4` | RGB 视频流 | 原文件复制，当前不转码 |
| `ubuntu/tactile.h5` | `force`、`disp`、`sequence_id`、`device_timestamp_us`、`system_timestamp_ns`、`monotonic_timestamp_ns`、`valid_channel`、`time` | 原 HDF5 文件原样复制；全量 dense 场不转换成新采集的 XYZ 汇总格式 |
| `ubuntu/tactile_gui_processed.csv` | 每通道 `force_x/force_y/force_z/resultant` 与 `time` | 作为旧派生结果保留，不能替代原始 HDF5 |

样例的既有 `validation_report.json` 显示 EMG、IMU 的系统/单调时标并非严格单调；其他设备时标通过了该旧报告里的单调检查。旧 `alignment_report.json` 的最大首帧残差约 7.26 ms，只是首帧选择差异，不代表全时段同步精度。导入报告会重新检查文件字段与时间列，但保留这些旧报告作为来源证据。

旧样例跨 Ubuntu 与 Windows 采集，文件里的 `monotonic_timestamp_ns` 不具备同一 Windows QPC 时钟域保证。因此导入时只作为旧字段保存，不生成新的 `time_ns`，也不建立伪造的设备间精确时间映射。
