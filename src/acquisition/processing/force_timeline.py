from __future__ import annotations

import json
import uuid
from pathlib import Path

import h5py
import numpy as np


ALGORITHM = "force_channel_rtc_endpoint_map_v2"


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest nanosecond, ties away from zero."""
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def build_force_timeline(session_path: Path, origin_ns: int) -> dict:
    """Map each channel's RTC timestamps to QPC using its own endpoint line.

    Raw row order is retained. Each channel is mapped independently because
    channel frames can be interleaved and may share the RTC millisecond value.
    Intermediate RTC/QPC jitter does not alter the endpoint line.
    """
    raw_path = session_path / "raw" / "force.h5"
    with h5py.File(raw_path, "r") as raw:
        frames = raw["frames"]
        required = {"channel", "seq_id", "device_timestamp_us", "device_rtc_timestamp_us",
                    "device_rtc_raw", "device_rtc_valid", "host_receive_mono_ns", "sensor_id"}
        missing = sorted(required.difference(frames.keys()))
        if missing:
            raise ValueError(
                "TouchGlove RTC alignment requires raw RTC fields; missing " + ", ".join(missing)
            )

        channels = np.asarray(frames["channel"][:], dtype=np.uint8)
        rtc_times = np.asarray(frames["device_rtc_timestamp_us"][:], dtype=np.uint64)
        rtc_valid = np.asarray(frames["device_rtc_valid"][:], dtype=np.uint8)
        host_times = np.asarray(frames["host_receive_mono_ns"][:], dtype=np.uint64)
        count = len(channels)
        if count == 0:
            raise ValueError("No TouchGlove frames to align")
        invalid_indices = np.flatnonzero(rtc_valid != 1)
        if invalid_indices.size:
            raise ValueError(
                f"TouchGlove RTC decode is invalid for {invalid_indices.size}/{count} frames; "
                f"first raw frame index is {int(invalid_indices[0])}"
            )

        time_ns = np.empty(count, dtype=np.int64)
        channel_mappings: dict[str, dict] = {}
        total_rtc_regressions = 0
        total_qpc_regressions = 0
        total_adjacent_rtc_repeats = 0
        for channel in np.unique(channels):
            indices = np.flatnonzero(channels == channel)
            first_index = int(indices[0])
            last_index = int(indices[-1])
            first_rtc = int(rtc_times[first_index])
            last_rtc = int(rtc_times[last_index])
            first_host = int(host_times[first_index])
            last_host = int(host_times[last_index])
            if last_rtc <= first_rtc or last_host <= first_host:
                raise ValueError(
                    f"Cannot map TouchGlove channel {int(channel)}: first/last RTC and QPC "
                    "observations need positive spans"
                )

            channel_rtc = [int(rtc_times[index]) for index in indices]
            channel_host = [int(host_times[index]) for index in indices]
            rtc_regressions = sum(b < a for a, b in zip(channel_rtc, channel_rtc[1:]))
            qpc_regressions = sum(b < a for a, b in zip(channel_host, channel_host[1:]))
            adjacent_repeats = sum(b == a for a, b in zip(channel_rtc, channel_rtc[1:]))
            total_rtc_regressions += rtc_regressions
            total_qpc_regressions += qpc_regressions
            total_adjacent_rtc_repeats += adjacent_repeats

            for index in indices:
                rtc = int(rtc_times[index])
                delta_ns = _round_ratio(
                    (last_host - first_host) * (rtc - first_rtc),
                    last_rtc - first_rtc,
                )
                time_ns[index] = first_host - int(origin_ns) + delta_ns

            channel_mappings[str(int(channel))] = {
                "frame_count": int(indices.size),
                "first_raw_frame_index": first_index,
                "last_raw_frame_index": last_index,
                "first_device_rtc_timestamp_us": first_rtc,
                "last_device_rtc_timestamp_us": last_rtc,
                "first_host_receive_mono_ns": first_host,
                "last_host_receive_mono_ns": last_host,
                "device_rtc_regression_count": int(rtc_regressions),
                "host_timestamp_regression_count": int(qpc_regressions),
                "repeated_adjacent_device_rtc_count": int(adjacent_repeats),
            }

        mapping = {
            "frame_count": count,
            "channel_count": len(channel_mappings),
            "channels": channel_mappings,
            "device_rtc_regression_count": total_rtc_regressions,
            "host_timestamp_regression_count": total_qpc_regressions,
            "repeated_adjacent_device_rtc_count": total_adjacent_rtc_repeats,
        }
        output_dir = session_path / "derived" / f"{ALGORITHM}_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / "force.h5"
        try:
            with h5py.File(output_path, "x") as result:
                result.attrs.update(
                    algorithm=ALGORITHM,
                    schema_version="2.0",
                    origin_mono_ns=int(origin_ns),
                    timestamp_semantics=("estimated frame time relative to planned QPC start; "
                                         "per-channel linear mapping from decoded device RTC to QPC endpoints"),
                    source="raw/force.h5",
                    device_rtc_source="BCD bytes [7:15] decoded as UTC epoch microseconds",
                )
                result.create_dataset("time_ns", data=time_ns, chunks=True)
                result.create_dataset("raw_frame_index", data=np.arange(count, dtype=np.uint64), chunks=True)
                for name, dtype in (
                    ("channel", np.uint8), ("seq_id", np.uint32),
                    ("device_timestamp_us", np.uint64), ("device_rtc_timestamp_us", np.uint64),
                    ("device_rtc_raw", np.uint8), ("device_rtc_valid", np.uint8),
                    ("host_receive_mono_ns", np.uint64),
                    ("sensor_id", h5py.string_dtype("utf-8")),
                ):
                    result.create_dataset(name, data=frames[name][:], dtype=dtype, chunks=True)
                result.attrs["mapping_json"] = json.dumps(mapping)
            report = {
                "algorithm": ALGORITHM,
                "path": output_path.relative_to(session_path).as_posix(),
                "origin_mono_ns": int(origin_ns),
                "timestamp_semantics": "per-channel RTC-to-QPC endpoint line; estimated time relative to planned session QPC start",
                "status": "complete",
                "frame_count": count,
                "mapping": mapping,
                "limits": ("Each channel uses its first and last decoded device RTC values and corresponding "
                           "host QPC observations to define one linear mapping. Intermediate callback jitter "
                           "does not alter the line. Capture order is retained; RTC/QPC regressions and repeated "
                           "adjacent RTC values are reported, not segmented or reordered. QPC is observed after "
                           "poll() returns, not at exposure."),
            }
            (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        except Exception:
            (output_dir / "FAILED").write_text("Force timeline generation failed; outputs are incomplete.", encoding="utf-8")
            raise
