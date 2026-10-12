from __future__ import annotations

import json
import uuid
from pathlib import Path

import h5py
import numpy as np


ALGORITHM = "realsense_host_qpc_v1"


def build_realsense_timeline(session_path: Path, origin_ns: int) -> dict:
    """Index RGB frames by host QPC observation time relative to the shared session origin."""
    raw_path = session_path / "raw" / "realsense_rgb.h5"
    with h5py.File(raw_path, "r") as raw:
        records = raw["record_json"].asstr()
        host_times = raw["host_receive_mono_ns"]
        count = len(records)
        if count == 0:
            raise ValueError("No RealSense RGB frames to index")
        output_dir = session_path / "derived" / f"{ALGORITHM}_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / "realsense_rgb.h5"
        camera_domains: set[str] = set()
        previous_host = None
        host_regressions = 0
        with h5py.File(output_path, "x") as output:
            output.attrs.update(
                algorithm=ALGORITHM,
                schema_version="1.0",
                origin_mono_ns=int(origin_ns),
                time_source="host_receive_mono_ns (Windows QPC after wait_for_frames)",
                camera_timestamp_semantics="raw SDK timestamp and domain retained; not mapped",
                source="raw/realsense_rgb.h5",
            )
            for name, dtype in (("time_ns", np.int64), ("raw_record_index", np.uint64),
                                ("frame_index", np.uint64), ("camera_frame_number", np.uint64),
                                ("camera_timestamp_ms", np.float64), ("host_receive_mono_ns", np.uint64)):
                output.create_dataset(name, shape=(count,), dtype=dtype, chunks=True)
            output.create_dataset("camera_timestamp_domain", shape=(count,), dtype=h5py.string_dtype("utf-8"), chunks=True)
            for index in range(count):
                row = json.loads(records[index])
                host_ns = int(host_times[index])
                if previous_host is not None:
                    host_regressions += int(host_ns < previous_host)
                previous_host = host_ns
                domain = str(row["camera_timestamp_domain"])
                camera_domains.add(domain)
                output["time_ns"][index] = host_ns - int(origin_ns)
                output["raw_record_index"][index] = index
                output["frame_index"][index] = int(row["frame_index"])
                output["camera_frame_number"][index] = int(row["camera_frame_number"])
                output["camera_timestamp_ms"][index] = float(row["camera_timestamp_ms"])
                output["camera_timestamp_domain"][index] = domain
                output["host_receive_mono_ns"][index] = host_ns
            mapping = {"frame_count": count, "first_raw_record_index": 0,
                       "last_raw_record_index": count - 1,
                       "first_host_receive_mono_ns": int(host_times[0]),
                       "last_host_receive_mono_ns": int(host_times[-1]),
                       "camera_timestamp_domains": sorted(camera_domains),
                       "host_timestamp_regression_count": host_regressions}
            output.attrs["mapping_json"] = json.dumps(mapping)
        report = {
            "algorithm": ALGORITHM,
            "path": output_path.relative_to(session_path).as_posix(),
            "origin_mono_ns": int(origin_ns),
            "status": "complete",
            "frame_count": count,
            "mapping": mapping,
            "limits": "time_ns is the QPC observation after wait_for_frames, relative to the shared planned QPC start. Camera SDK timestamps and domains are preserved without assuming they share the host clock; QPC time is not exposure time.",
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
