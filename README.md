# Full System

Windows acquisition host being rebuilt from `../architecture_design.md`.

## Start

```powershell
cd E:\reshape\full_system
uv venv --python 3.10
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m acquisition.app
```

Python 3.10 is selected because the bundled Windows ResearchKit SDK contains CPython 3.10 native modules.

The Windows UI connects devices one at a time. For EMG, select the ResearchKit USB receiver, scan nearby bands, choose the returned MAC, then connect; connect NOKOV, RealSense, and TouchGlove with their own controls. Each connected device streams a live preview, while the formal recording button stays disabled until all four device groups are connected. Preview data is not written to the session. Once started, all five streams record from one future Windows QPC deadline. Stopping builds aligned timelines and runs TouchGlove DirectML inference after raw image capture closes.

All acquisition SDK files used by the application live under `vendor/`: ResearchKit (`vendor/researchkit`), NOKOV (`vendor/nokov`), and TouchGlove (`vendor/touchglove`). Native SDK binaries and the TouchGlove model are stored with Git LFS; install Git LFS and run `git lfs pull` after cloning if the binaries were not downloaded automatically. Python package dependencies are declared in `requirements.txt` and `pyproject.toml` and installed into the project-local `.venv`; RealSense still requires its Windows device driver. TouchGlove DirectML provider is checked when that device is connected. Set `EMG_PORT` only when you need to override the detected ResearchKit adapter.

To check device connection without starting a recording, run `python tools/emg_preflight.py`. A short live write/readback check that uses a temporary session is available as `python tools/emg_smoke_capture.py --seconds 3`; the temporary capture is deleted when the command exits. To check the camera independently, run `python tools/realsense_smoke_capture.py --seconds 2`; it briefly records RGB, reads frame metadata back from HDF5, and deletes its temporary output.

The UI also has **导入旧采集目录**. It copies a legacy folder to a new `IMPORTED_LEGACY` session, preserves the original files under `raw/legacy_source/`, computes file hashes, inventories CSV/HDF5 fields, and reports non-monotonic time columns. It does not rewrite old timestamps as QPC time. The supplied sample can also be imported from PowerShell:

```powershell
full-system-import-legacy ..\example_sample --output .\data
```

Each session is created under `data/<session_id>/` and is never overwritten. `raw/<stream>.h5` contains append-only JSON records plus QPC observation times. `manifest.json` and `events.jsonl` record session state and control events. Each device writes its own HDF5 record stream; RealSense RGB video is stored alongside its frame metadata.

## EMG record fields

`raw/emg.h5` contains one JSON record per SDK callback in `record_json` and a parallel `host_receive_mono_ns` array. Each record includes `sensor_type`, `device_timestamp_raw`, `sample_count`, `sample_rate_hz`, `data` (the SDK packet matrix), `host_receive_mono_ns`, `timestamp_stage`, and `clock_id`. The SDK-generated Unix timestamp is deliberately not recorded; `manifest.json` records that policy.

`device_timestamp_raw` is the raw uint64 little-endian TIMESTAMP TLV; the SDK documents milliseconds, and legacy recordings show a device-relative counter with about 9.72–9.74 ms per counter step. The acquisition path does not call `SET_TIMESTAMP`, so this counter is not Unix time. `host_receive_mono_ns` is our Windows QPC reading in the SDK callback. It is the main host monotonic observation time, not the exact hardware arrival or per-sample time. The exact packet anchor (first/last sample) and device counter reset/wrap behavior remain unknown. These semantics and limits are recorded in the EMG stream descriptor in `manifest.json`. `events.jsonl` records prepare, arm, actual recording start, errors, and stop.

The observed `DATA_EMG` payload begins with a one-byte sample count, followed by sample-major rows (`N × 9` int16 values). This provides an offset `0..N-1` inside a packet, but the protocol has no per-sample timestamp/index or packet sequence number. Row order is preserved in `data`; it is not a device index and cannot reveal samples lost before the SDK callback. A host-generated packet counter can identify callback order, but it would not preserve device-side gaps.

Connect and prepare both devices in the UI before arming. PREPARE confirms the handband connection and enabled sensor configuration, camera profile, and first RGB frame; recording begins at the shared future QPC deadline. Stopping and sealing the session finalizes the RGB MP4 and automatically builds separate estimated EMG and IMU sample timelines described below.

## IMU record and aligned output

`raw/imu.h5` contains one JSON record per IMU callback, with the same raw `device_timestamp_raw`, QPC `host_receive_mono_ns`, `timestamp_stage`, and `clock_id` fields as EMG, plus `sample_count`, configured/SDK `sample_rate_hz`, and the six-channel packet matrix in SDK order. Values remain the SDK `int16` readings: no unit conversion or channel-axis interpretation is applied. There is no identified per-sample or packet sequence number.

After stop, `derived/imu_session_endpoint_map_v1_<unique_id>/imu.h5` and `report.json` contain the aligned timeline. It uses the same rule as EMG: the first and last IMU device timestamps map to their own QPC callback observations over the session, and only consecutive equal-timestamp packets expand rows at the configured 50 Hz interval. The IMU mapping is calculated from IMU packet endpoints, independently of the EMG mapping; both are relative to the same planned QPC start. This preserves the current alignment convention without assuming both sensor streams have identical callback endpoints.

## RealSense RGB record

PREPARE requires an attached RealSense camera and confirms its serial number, firmware, active RGB profile, intrinsics, and a received color frame. The SDK worker records the 640×480 BGR RGB stream at 30 FPS to `raw/realsense_rgb.mp4` (`mp4v`); depth and IMU are not enabled. The MP4 is lossy compressed video, not original per-frame pixel data.

`raw/realsense_rgb.h5` stores one record per RGB frame: `frame_index`, SDK `camera_frame_number`, raw `camera_timestamp_ms`, its `camera_timestamp_domain`, QPC `host_receive_mono_ns` observed after `wait_for_frames`, `timestamp_stage`, and `clock_id`. The stream descriptor in `manifest.json` records the actual profile, camera identity, RGB intrinsics, codec, video path, frame count, and observed gaps in delivered SDK frame numbers. Such gaps describe frame-number discontinuities visible to the host; they do not prove where or why a frame was lost. After stop, `derived/realsense_host_qpc_v1_<id>/realsense_rgb.h5` gives each frame `time_ns = host_receive_mono_ns - planned_start_mono_ns`; the SDK timestamp and domain remain alongside it but are not converted to QPC. This is the host's frame-observation time, not exposure time.

## NOKOV motion-capture record

The XING Python SDK 2.4.0.5428 binding and 64-bit Windows runtime are bundled under `vendor/nokov/`; no separately installed `nokovpy` package is needed. The UI's **NOKOV server** field defaults to the address used by the legacy Windows acquisition code. PREPARE initializes the SDK at that address and fails clearly if the XINGYING streaming server is not reachable. Check the address against the XINGYING machine and start its motion-capture stream before preparing.

`raw/nokov.h5` contains one JSON record per SDK callback. Each row preserves `device_frame_index` (`iFrame`), `device_timestamp_raw` (`iTimeStamp`), timecode, raw frame parameters and latency, callback-entry QPC `host_receive_mono_ns`, marker sets with set name/index and marker index/XYZ, labeled markers with SDK IDs, rigid-body IDs and pose, skeleton/segment IDs and pose, unidentified marker coordinates, and analog values. XYZ uses the NOKOV sample SDK's millimeter convention; rigid-body quaternion order is `qx,qy,qz,qw`. Coordinates, `params`, and SDK timestamp values are retained as returned. The bundled NOKOV SDK manual defines `iTimeStamp` in milliseconds. A live 6 s test advanced 6011 ms over 541 frame intervals and kept an approximately 39 ms offset from sampled PC Unix milliseconds, but did not establish whether the clock originates in the camera/device or XINGYING host. The legacy Windows sample encodes missing human marker XYZ as `9999999.0`; raw values remain unchanged in storage.

After a session is sealed, **结果可视化预览** opens the latest result or a selected historical Full_system session. It provides a shared-time playback slider for the aligned EMG/IMU, per-channel TouchGlove force components/resultant, selectable NOKOV XYZ tracks, and RealSense frames. EMG uses the old analysis page's display-only 10 Hz high-pass. NOKOV nonfinite points, missing observations, and coordinates with magnitude at least 1,000,000 mm are drawn as gaps; bracketed gaps get a red interval and dashed display-only linear interpolation, while leading/trailing gaps stay blank. The raw session is never modified. Unidentified NOKOV points are exposed by per-frame array slot because the SDK does not provide stable IDs for them.

The standalone **查看 NOKOV 轨迹** UI still opens an aligned NOKOV CSV or a captured `raw/nokov.h5` using the same missing-point display rule. The observed 9,999,999 mm sentinel was verified in the supplied legacy CSV; whether every live SDK/version uses the same sentinel still needs confirmation.

After stop, `derived/nokov_device_endpoint_map_iframe_v2_<id>/nokov.h5` maps the first and last `(iTimeStamp, host_receive_mono_ns)` pairs onto the session QPC axis, then maps each frame's `iTimeStamp` on that line. Consecutive frames with the same `iTimeStamp` are distributed between adjacent mapped timestamp anchors in proportion to their `iFrame` values, following the EMG equal-timestamp expansion approach. The output retains source row, callback index, `iFrame`, raw `iTimeStamp`, and callback QPC for audit; `report.json` counts frame-number gaps, duplicates, regressions, and repeated timestamp groups. This assumes a single linear clock relationship across the session; the source/timebase of `iTimeStamp` is still unverified. A frame-number gap shows a discontinuity in frames delivered to this host, not the cause or exact loss point. For an SDK-only connection check, run `python tools/nokov_preflight.py --server-ip 10.1.1.198 --seconds 3`.

## EMG aligned output

After the raw writer closes, each reconstruction creates a new `derived/emg_session_endpoint_map_v5_<unique_id>/` directory with `emg.h5` and `report.json`. Raw files are unchanged. `manifest.json` records the output path, status, and session endpoint mapping under `emg_timeline`; events record success or failure. A processing failure preserves the raw acquisition and is reported separately. Previous v1/v2/v3/v4 outputs remain historical artifacts and are not the current reconstruction rule.

For the whole session, the first and last device timestamps map to their respective QPC callback times. Every distinct device timestamp is substituted into that line, independently of intermediate callback jitter. Only samples sharing the same device timestamp are expanded from that mapped time at the configured rate (1 ms at 1000 Hz), in packet/row order. Only consecutive packets with equal device timestamps share a sample offset counter. Any change in device timestamp starts a new group at offset zero; a later return to an earlier timestamp does not reuse earlier offsets. Different timestamp groups retain their mapped positions and any gaps. For example, with a line `y=100x+10` in milliseconds, three samples at x=5 are assigned 510, 511, 512 ms, while x=6 starts at 610 ms. Times are relative to `planned_start_mono_ns`, with integer arithmetic for endpoint interpolation. Samples sharing the final device timestamp may extend beyond its mapped endpoint. Callback delays and the unverified group-first-sample convention remain limitations.

The derived HDF5 datasets are:

| Dataset | Content |
| --- | --- |
| `data` | Received sample rows and channels, copied without filtering |
| `time_ns` | Estimated per-sample time in integer nanoseconds relative to the session origin |
| `raw_record_index` | Zero-based row in raw `record_json`; a source reference, not a device sequence |
| `packet_offset` | Zero-based sample row within its packet |
| `timestamp_group_offset` | Zero-based sample position within a consecutive group of packets sharing the same device timestamp |
| `packet_json` | Per-packet source reference, derived sample start/count, anchor and inter-packet gap in ns |

HDF5 attributes (`schema_version=2.0`) and `report.json` store the algorithm version, session origin, configured sample rate (`sample_rate_source=session_configuration`), group-first-sample reconstruction convention and one session `mapping`. The mapping includes first/last raw record references, device timestamps, QPC times, packet/sample counts and observational quality counts. There is no segmentation or placeholder time dataset. A successful output has `status=complete`, which means calculation completed, not that sampling accuracy has been verified.

`packet_json` also records `timestamp_group_start_offset`; its `anchor_time_ns` is the mapped device timestamp before adding sample offsets. The gap is current first sample minus previous last sample minus one nominal sample interval; positive values describe a gap and negative values describe overlap, without asserting packet loss. Local device/host timestamp regressions are counted and flagged but never change the line or discard/reorder data. `mapping` exposes `device_timestamp_regression_count`, `host_timestamp_regression_count` and `repeated_device_timestamp_count` (packets with the same timestamp as the immediately preceding packet).

## TouchGlove force capture and alignment

The Windows SDK files are vendored under `vendor/touchglove/`: `python/touch_glove_ptp` contains the Python binding and `touch_glove.dll`, and `model/dense_3ch.onnx` is the Dense model. The GUI requires a selected TouchGlove COM port and verifies that DirectML is the active first ONNX Runtime provider when connecting the force sensor.

The TouchGlove sensor is now part of the same UI session and uses the same planned QPC start as the other devices. The separate capture command remains available for diagnostics:

```powershell
python tools/force_offline_capture.py --port COM14 --seconds 30 --baseline-seconds 1
```

The integrated UI stores every 192×192 uint8 frame in `raw/force.h5`, including `channel`, `seq_id`, `device_timestamp_us`, decoded RTC fields, `host_receive_mono_ns`, and `sensor_id`, plus the opening baseline images/sample counts. The DLL's `timestamp_us` is generated from Windows host wall time and does not expose the frame's 8-byte BCD RTC at offsets `[7:15]`; the UI therefore defaults to the Python serial parser. It stores `device_rtc_raw`, decoded UTC epoch-microseconds in `device_rtc_timestamp_us`, and `device_rtc_valid`. Set `TOUCHGLOVE_BACKEND=dll` only for legacy diagnostics; DLL captures do not contain the RTC required by the current alignment.

After capture, `derived/force_channel_rtc_endpoint_map_v2_<id>/force.h5` stores `time_ns`, raw-frame references, RTC and host timestamp fields, `disp` and `force_field`; `report.json` and `manifest.json` record the provider and per-channel mapping. Each channel uses its first and last decoded RTC values and their corresponding QPC observations to define one linear map; every frame from that channel is mapped through it. Mapping is per channel because the capture interleaves channels and their RTC values may coincide. Times are relative to the shared session QPC start. Intermediate callback jitter does not change the line; regressions are counted but do not split, reorder or discard frames. Channels absent from a session are omitted. Mapped frame time is an estimate and QPC is observed after `poll()` returns, not at exposure.

The capture is intentionally non-realtime with respect to inference: all raw frames are flushed to disk before DirectML model initialization. Thus a model/runtime failure keeps the raw capture and marks the session `INFERENCE_FAILED`. `--baseline-seconds` defaults to 1 second and estimates an unloaded baseline for each channel that actually produced frames; absent channels are left empty and omitted from inference.

Operating assumptions: the sampling rate is fixed throughout a session, and the device counter is not reset or rebooted. Expansion uses the session configuration; the SDK-reported `sample_rate_hz` is preserved directly in raw packets without a fallback and must match that configuration. A mismatch is an explicit processing error, not a reason to split the recording. Hardware-supported rate choices are validated by the SDK at configuration time; the reconstruction algorithm only requires a positive configured rate. Matrix shape, channel count and declared sample count are checked for data consistency. Empty data or nonpositive first/last timestamp spans fail explicitly, retain raw acquisition and produce no zero-placeholder timeline. Other output can be nonmonotonic; consumers must respect the quality report instead of silently sorting or closing gaps.

## Current boundary

The five acquisition streams are integrated in the graphical session with bundled SDK files and project-local Python dependencies. EMG, IMU, force, and NOKOV use session endpoint mappings; RealSense indexes host-observed frames directly on the shared QPC axis. NOKOV `iTimeStamp` is used as the relative axis for its endpoint mapping, while its absolute clock source remains unverified. Force inference runs after capture and requires DirectML. NOKOV gap interpolation is available only in its visualization and is not an analysis-grade reconstruction. Host callback/frame observation time does not claim to be the exact device sample or exposure time.

See [`docs/legacy_sample_review.md`](docs/legacy_sample_review.md) for the `example_sample` field inventory and known timing limitations.

See [`docs/emg_loss_detection_review.md`](docs/emg_loss_detection_review.md) for the direct SDK runtime inspection of device-side counters and the limits of packet-loss detection.
