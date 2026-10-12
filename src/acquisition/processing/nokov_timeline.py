from __future__ import annotations

import json
import uuid
from pathlib import Path

import h5py
import numpy as np


ALGORITHM = "nokov_device_endpoint_map_iframe_v2"


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round integer ratios to nearest nanosecond, ties away from zero."""
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def build_nokov_timeline(session_path: Path, origin_ns: int) -> dict:
    """Map iTimeStamp endpoints to QPC and spread equal-time frames by iFrame."""
    raw_path = session_path / "raw" / "nokov.h5"
    with h5py.File(raw_path, "r") as raw:
        records = raw["record_json"].asstr()
        host_times = raw["host_receive_mono_ns"]
        count = len(records)
        if count == 0:
            raise ValueError("No NOKOV frames to align")

        device_times = np.empty(count, dtype=np.int64)
        frame_indices = np.empty(count, dtype=np.int64)
        callback_indices = np.empty(count, dtype=np.uint64)
        host_values = np.asarray(host_times[:], dtype=np.int64)
        for index in range(count):
            row = json.loads(records[index])
            device_times[index] = int(row["device_timestamp_raw"])
            frame_indices[index] = int(row["device_frame_index"])
            callback_indices[index] = int(row["callback_index"])

        first_device = int(device_times[0])
        last_device = int(device_times[-1])
        first_host = int(host_values[0])
        last_host = int(host_values[-1])
        if last_device == first_device:
            raise ValueError("Cannot map NOKOV session endpoints: first and last iTimeStamp are equal")
        if last_host == first_host:
            raise ValueError("Cannot map NOKOV session endpoints: first and last QPC times are equal")

        device_span = last_device - first_device
        host_span = last_host - first_host
        relative_times = np.fromiter(
            (first_host - int(origin_ns) + _round_ratio(host_span * (int(value) - first_device), device_span)
             for value in device_times),
            dtype=np.int64,
            count=count,
        )

        quality = {
            "frame_gap_count": 0,
            "duplicate_frame_count": 0,
            "frame_regression_count": 0,
            "host_timestamp_regression_count": 0,
            "device_timestamp_regression_count": 0,
            "repeated_timestamp_run_count": 0,
            "repeated_timestamp_frame_count": 0,
            "repeated_timestamp_unresolved_run_count": 0,
        }
        for index in range(1, count):
            frame_delta = int(frame_indices[index]) - int(frame_indices[index - 1])
            if frame_delta > 1:
                quality["frame_gap_count"] += frame_delta - 1
            elif frame_delta == 0:
                quality["duplicate_frame_count"] += 1
            elif frame_delta < 0:
                quality["frame_regression_count"] += 1
            quality["host_timestamp_regression_count"] += int(host_values[index] < host_values[index - 1])
            quality["device_timestamp_regression_count"] += int(device_times[index] < device_times[index - 1])

        # A millisecond timestamp can label multiple motion-capture frames.
        # Keep the first frame at that timestamp's endpoint-mapped anchor and
        # distribute the following equal-timestamp frames by iFrame up to the
        # next distinct timestamp. For a repeated run at the very end, anchor
        # the last frame and interpolate the run from the preceding timestamp.
        index = 0
        while index < count:
            end = index + 1
            while end < count and device_times[end] == device_times[index]:
                end += 1
            if end - index > 1:
                quality["repeated_timestamp_run_count"] += 1
                quality["repeated_timestamp_frame_count"] += end - index - 1
                if end < count:
                    left, right = index, end
                elif index > 0:
                    left, right = index - 1, end - 1
                else:
                    left = right = index
                frame_span = int(frame_indices[right]) - int(frame_indices[left])
                if frame_span > 0:
                    start_time = int(relative_times[left])
                    end_time = int(relative_times[right])
                    for frame_row in range(index, end):
                        relative_times[frame_row] = start_time + _round_ratio(
                            (end_time - start_time)
                            * (int(frame_indices[frame_row]) - int(frame_indices[left])),
                            frame_span,
                        )
                else:
                    quality["repeated_timestamp_unresolved_run_count"] += 1
            index = end

        output_dir = session_path / "derived" / f"{ALGORITHM}_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / "nokov.h5"
        try:
            with h5py.File(output_path, "x") as result:
                result.attrs.update(
                    algorithm=ALGORITHM,
                    schema_version="2.0",
                    origin_mono_ns=int(origin_ns),
                    time_source="iTimeStamp endpoint line mapped to Windows QPC; equal timestamps interpolated by iFrame",
                    device_timestamp_field="iTimeStamp (milliseconds per SDK documentation); endpoint mapping uses recorded values without assuming their epoch",
                    device_frame_index_field="iFrame (SDK frame number); used to interpolate frames with equal iTimeStamp",
                    source="raw/nokov.h5",
                )
                for name, dtype in (("time_ns", np.int64), ("raw_record_index", np.uint64),
                                    ("callback_index", np.uint64), ("device_frame_index", np.int64),
                                    ("device_timestamp_raw", np.int64),
                                    ("host_receive_mono_ns", np.int64)):
                    result.create_dataset(name, shape=(count,), dtype=dtype, chunks=True)
                result["time_ns"][:] = relative_times
                result["raw_record_index"][:] = np.arange(count, dtype=np.uint64)
                result["callback_index"][:] = callback_indices
                result["device_frame_index"][:] = frame_indices
                result["device_timestamp_raw"][:] = device_times
                result["host_receive_mono_ns"][:] = host_values
                mapping = {
                    "frame_count": count,
                    "first_raw_record_index": 0,
                    "last_raw_record_index": count - 1,
                    "first_device_timestamp_raw": first_device,
                    "last_device_timestamp_raw": last_device,
                    "first_device_frame_index": int(frame_indices[0]),
                    "last_device_frame_index": int(frame_indices[-1]),
                    "first_host_receive_mono_ns": first_host,
                    "last_host_receive_mono_ns": last_host,
                    **quality,
                }
                result.attrs["mapping_json"] = json.dumps(mapping)
            report = {
                "algorithm": ALGORITHM,
                "path": output_path.relative_to(session_path).as_posix(),
                "origin_mono_ns": int(origin_ns),
                "status": "complete",
                "frame_count": count,
                "mapping": mapping,
                "limits": "The first and last observed (iTimeStamp, QPC) pairs define one linear map. Frames in each consecutive equal-iTimeStamp run are distributed between mapped timestamp anchors using iFrame. iTimeStamp source/timebase is not asserted; QPC callback times are retained for audit. Frame gaps report only observed iFrame differences.",
            }
            (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        except Exception:
            (output_dir / "FAILED").write_text("NOKOV timeline generation failed; outputs are incomplete.", encoding="utf-8")
            raise
