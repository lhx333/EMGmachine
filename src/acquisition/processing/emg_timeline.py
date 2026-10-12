from __future__ import annotations

import json
import uuid
from pathlib import Path

import h5py
import numpy as np


ALGORITHM = "emg_session_endpoint_map_v5"
IMU_ALGORITHM = "imu_session_endpoint_map_v1"


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round integer ratios to nearest nanosecond, ties away from zero."""
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def build_emg_timeline(session_path: Path, origin_ns: int, sample_rate_hz: int) -> dict:
    return _build_sensor_timeline(session_path, origin_ns, sample_rate_hz, "emg", 9, ALGORITHM)


def build_imu_timeline(session_path: Path, origin_ns: int, sample_rate_hz: int) -> dict:
    return _build_sensor_timeline(session_path, origin_ns, sample_rate_hz, "imu", 6, IMU_ALGORITHM)


def _build_sensor_timeline(session_path: Path, origin_ns: int, sample_rate_hz: int,
                           stream: str, expected_channels: int, algorithm: str) -> dict:
    """Map each device timestamp to QPC; expand equal-timestamp sample groups.

    Two passes avoid loading an entire recording's sample matrices into memory.
    One session uses one line and one configured rate. Only consecutive equal
    timestamps share offsets. No inferred reset or segmentation.
    """
    if sample_rate_hz <= 0:
        raise ValueError("Session EMG sample rate must be positive")
    raw_path = session_path / "raw" / f"{stream}.h5"
    mapping = None
    with h5py.File(raw_path, "r") as raw:
        records = raw["record_json"].asstr()
        previous = None
        channels = None
        for index in range(len(records)):
            row = json.loads(records[index])
            device = int(row["device_timestamp_raw"])
            host = int(row["host_receive_mono_ns"])
            rate = int(row["sample_rate_hz"])
            matrix = np.asarray(row["data"])
            if rate != sample_rate_hz:
                raise ValueError(f"SDK sample rate differs from fixed session configuration at raw record {index}")
            if matrix.ndim != 2 or min(matrix.shape) == 0:
                raise ValueError(f"Invalid {stream.upper()} rate or matrix at raw record {index}")
            if matrix.shape[1] != expected_channels:
                raise ValueError(f"Expected {expected_channels} {stream.upper()} channels at raw record {index}, got {matrix.shape[1]}")
            if matrix.shape[0] != int(row["sample_count"]):
                raise ValueError(f"{stream.upper()} sample count mismatch at raw record {index}")
            if channels is not None and matrix.shape[1] != channels:
                raise ValueError(f"{stream.upper()} channel count changed at raw record {index}")
            channels = matrix.shape[1]
            if mapping is None:
                mapping = {
                    "start_raw_record_index": index,
                    "first_device_timestamp_raw": device, "first_host_receive_mono_ns": host,
                    "sample_rate_hz": sample_rate_hz, "sample_rate_source": "session_configuration",
                    "packet_count": 0, "sample_count": 0,
                    "device_timestamp_regression_count": 0,
                    "host_timestamp_regression_count": 0,
                }
            if previous is not None:
                mapping["device_timestamp_regression_count"] += int(device < previous[0])
                mapping["host_timestamp_regression_count"] += int(host < previous[1])
            mapping.update(last_device_timestamp_raw=device, last_host_receive_mono_ns=host,
                           end_raw_record_index=index)
            mapping["packet_count"] += 1
            mapping["sample_count"] += matrix.shape[0]
            previous = (device, host, rate)

        if mapping is None:
            raise ValueError(f"No {stream.upper()} packets to align")
        if (mapping["last_device_timestamp_raw"] <= mapping["first_device_timestamp_raw"]
                or mapping["last_host_receive_mono_ns"] <= mapping["first_host_receive_mono_ns"]):
            raise ValueError(f"Cannot build {stream.upper()} session endpoint line: first/last timestamps need positive spans")
        mapping["positive_gap_count"] = 0
        mapping["overlap_count"] = 0
        mapping["nonincreasing_sample_boundary_count"] = 0
        mapping["repeated_device_timestamp_count"] = 0

        output_dir = session_path / "derived" / (algorithm + "_" + uuid.uuid4().hex)
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / f"{stream}.h5"
        try:
            with h5py.File(output_path, "x") as result:
                result.attrs.update(algorithm=algorithm, schema_version="2.0",
                                    origin_mono_ns=int(origin_ns), timestamp_group_anchor="first_sample_reconstruction_convention",
                                    timestamp_semantics="estimated sampling time relative to planned QPC start",
                                    source=f"raw/{stream}.h5", sample_rate_hz=sample_rate_hz, sample_rate_source="session_configuration")
                total = mapping["sample_count"]
                result.create_dataset("data", shape=(total, channels), dtype=np.float64, chunks=True)
                for name, dtype in (("time_ns", np.int64),
                                    ("raw_record_index", np.int64), ("packet_offset", np.int32),
                                    ("timestamp_group_offset", np.int64)):
                    result.create_dataset(name, shape=(total,), dtype=dtype, chunks=True)
                packets = result.create_dataset("packet_json", shape=(len(records),), dtype=h5py.string_dtype("utf-8"))
                offset = 0
                prior_last = None
                prior_device = None
                prior_host = None
                group_offset = 0
                for index in range(mapping["start_raw_record_index"], mapping["end_raw_record_index"] + 1):
                    row = json.loads(records[index])
                    count = int(row["sample_count"])
                    device = int(row["device_timestamp_raw"])
                    host = int(row["host_receive_mono_ns"])
                    device_regression = prior_device is not None and device < prior_device
                    host_regression = prior_host is not None and host < prior_host
                    rate = mapping["sample_rate_hz"]
                    gap = None
                    if device != prior_device:
                        group_offset = 0
                    anchor = mapping["first_host_receive_mono_ns"] - int(origin_ns) + _round_ratio(
                        (mapping["last_host_receive_mono_ns"] - mapping["first_host_receive_mono_ns"])
                        * (device - mapping["first_device_timestamp_raw"]),
                        mapping["last_device_timestamp_raw"] - mapping["first_device_timestamp_raw"])
                    times = [anchor + _round_ratio((group_offset + i) * 1_000_000_000, rate)
                             for i in range(count)]
                    if prior_last is not None:
                        gap = times[0] - prior_last - _round_ratio(1_000_000_000, rate)
                        mapping["positive_gap_count"] += int(gap > 0)
                        mapping["overlap_count"] += int(gap < 0)
                        mapping["nonincreasing_sample_boundary_count"] += int(times[0] <= prior_last)
                    prior_last = times[-1]
                    mapping["repeated_device_timestamp_count"] += int(group_offset > 0)
                    prior_device = device
                    prior_host = host
                    selection = slice(offset, offset + count)
                    result["data"][selection] = row["data"]
                    result["time_ns"][selection] = times
                    result["raw_record_index"][selection] = index
                    result["packet_offset"][selection] = np.arange(count)
                    result["timestamp_group_offset"][selection] = np.arange(group_offset, group_offset + count)
                    packets[index] = json.dumps({"raw_record_index": index,
                        "sample_start_row": offset,
                        "sample_count": count, "anchor_time_ns": anchor,
                        "timestamp_group_start_offset": group_offset,
                        "device_timestamp_regressed": device_regression,
                        "host_timestamp_regressed": host_regression,
                        "gap_from_previous_packet_ns": gap})
                    offset += count
                    group_offset += count
                result.attrs["mapping_json"] = json.dumps(mapping)
            report = {"algorithm": algorithm, "path": output_path.relative_to(session_path).as_posix(),
                      "origin_mono_ns": int(origin_ns), "timestamp_group_anchor": "first_sample_reconstruction_convention",
                      "status": "complete", "sample_rate_hz": sample_rate_hz, "sample_rate_source": "session_configuration",
                      "sample_count": total, "mapping": mapping,
                      "limits": "Callback endpoint delays and unknown packet anchor; gaps are not confirmed lost packets. Completion describes calculation success, not validated sampling accuracy."}
            (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        except Exception:
            # Preserve partial artifacts for inspection, but never publish them as successful.
            (output_dir / "FAILED").write_text("Timeline generation failed; outputs are incomplete.", encoding="utf-8")
            raise
