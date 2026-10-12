from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py

from acquisition.devices.emg import detect_emg_ports
from acquisition.session.session import Session


def summarize_stream(path: Path, name: str) -> dict:
    with h5py.File(path / "raw" / f"{name}.h5", "r") as raw:
        records = [json.loads(value) for value in raw["record_json"].asstr()[:]]
    if not records:
        raise RuntimeError(f"No {name.upper()} packets were recorded")

    sorted_device_times = sorted(int(row["device_timestamp_raw"]) for row in records)
    intervals = [right - left for left, right in zip(sorted_device_times, sorted_device_times[1:])]
    counts = Counter(intervals)
    return {
        "packet_count": len(records),
        "sample_count": sum(int(row["sample_count"]) for row in records),
        "device_time_min_raw": sorted_device_times[0],
        "device_time_max_raw": sorted_device_times[-1],
        "sorted_adjacent_interval_count": len(intervals),
        "interval_counts_descending": [
            {"interval_raw_units": interval, "count": counts[interval]}
            for interval in sorted(counts, reverse=True)
        ],
        "zero_intervals": counts.get(0, 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture EMG/IMU and report sorted device-time steps")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--port", help="Override the auto-detected ResearchKit USB COM port")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--session-id", default="device_time_stability")
    args = parser.parse_args()

    ports = detect_emg_ports()
    port = args.port or (ports[0][0] if len(ports) == 1 else None)
    if not port:
        parser.error(f"Expected exactly one detected EMG adapter or --port. Found: {ports}")

    session = Session(args.root, session_id=args.session_id)
    print(json.dumps({"phase": "prepare", "port": port, "detected": ports}, ensure_ascii=False), flush=True)
    info = session.prepare_emg(port, sample_rate_hz=1000, imu_sample_rate_hz=50)
    print(json.dumps({
        "phase": "prepared",
        "device": info.get("emg", {}).get("device_name"),
        "path": str(session.path.resolve()),
    }, ensure_ascii=False), flush=True)

    session.start_emg(delay_s=1.0)
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds and session.state not in {"ERROR", "COMPLETED"}:
            time.sleep(1.0)
            elapsed = int(time.monotonic() - started)
            if elapsed and elapsed % 15 == 0:
                print(json.dumps({
                    "phase": "capturing", "elapsed_s": elapsed,
                    "emg_samples": session.sample_count,
                    "imu_samples": session.imu_sample_count,
                    "state": session.state,
                }), flush=True)
    finally:
        session.stop()

    summary = {
        "session_path": str(session.path.resolve()),
        "requested_duration_s": args.seconds,
        "session_state": session.state,
        "streams": {
            name: summarize_stream(session.path, name)
            for name in ("emg", "imu")
        },
        "evaluation": "Packet device_time values sorted ascending; adjacent differences summarized descending. Host timestamps and acquisition-order regressions are excluded.",
    }
    report_path = session.path / "device_time_stability.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"report_path={report_path.resolve()}", flush=True)
    return 0 if session.state == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
