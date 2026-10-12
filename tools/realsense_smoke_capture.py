from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import h5py

from acquisition.clock import WindowsQpcClock
from acquisition.devices.realsense import RealSenseAdapter
from acquisition.recording.hdf5_writer import Hdf5Writer


def main() -> int:
    parser = argparse.ArgumentParser(description="Short RealSense RGB capture and HDF5 readback")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--serial", default="")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="realsense-smoke-") as temporary:
        raw = Path(temporary) / "raw"
        raw.mkdir()
        writer = Hdf5Writer(raw)
        adapter = RealSenseAdapter(lambda record: writer.append("realsense_rgb", record), serial=args.serial)
        try:
            info = adapter.prepare(raw)
            deadline = WindowsQpcClock().now_ns() + 500_000_000
            adapter.arm(deadline)
            time.sleep(args.seconds)
            stopped = adapter.stop()
            writer.close()
            video = raw / "realsense_rgb.mp4"
            if not video.is_file() or video.stat().st_size <= 0:
                raise RuntimeError("RealSense MP4 was not created")
            with h5py.File(raw / "realsense_rgb.h5", "r") as data:
                rows = data["record_json"].asstr()[:]
            decoded = [json.loads(row) for row in rows]
            if len(decoded) != stopped["frame_count"] or not decoded:
                raise RuntimeError(f"Frame metadata count {len(decoded)} != video frame count {stopped['frame_count']}")
            print(json.dumps({"device": info, "frame_count": len(decoded),
                              "first_qpc_ns": decoded[0]["host_receive_mono_ns"],
                              "last_qpc_ns": decoded[-1]["host_receive_mono_ns"],
                              "frame_number_gap_count": stopped["frame_number_gap_count"],
                              "video_bytes": video.stat().st_size,
                              "result": "PASS"}, ensure_ascii=False, indent=2))
        finally:
            try:
                adapter.stop()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
