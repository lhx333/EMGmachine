from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import h5py

from acquisition.devices.emg import detect_emg_ports
from acquisition.session.session import Session


def main() -> int:
    parser = argparse.ArgumentParser(description="Short live EMG-to-HDF5 smoke capture; temporary data is deleted")
    parser.add_argument("--port", help="Override the auto-detected ResearchKit USB COM port")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    ports = detect_emg_ports()
    port = args.port or (ports[0][0] if len(ports) == 1 else None)
    if not port:
        parser.error(f"Expected exactly one detected EMG adapter or --port. Found: {ports}")

    with tempfile.TemporaryDirectory(prefix="full-system-emg-smoke-") as temporary:
        session = Session(Path(temporary) / "data")
        try:
            info = session.prepare_emg(port, sample_rate_hz=1000)
            session.start_emg(delay_s=1.0)
            deadline = time.monotonic() + max(0.1, args.seconds)
            while time.monotonic() < deadline and session.state not in {"ERROR", "COMPLETED"}:
                time.sleep(0.05)
        finally:
            session.stop()

        raw_path = session.path / "raw" / "emg.h5"
        with h5py.File(raw_path, "r") as raw:
            records = [json.loads(item) for item in raw["record_json"].asstr()[:]]
        if not records:
            raise RuntimeError("Capture ran, but the SDK delivered no EMG packets")
        if any("sdk_timestamp_raw" in record for record in records):
            raise RuntimeError("SDK Unix timestamp unexpectedly persisted in the EMG record")
        if any(
            int(record["sample_count"]) != len(record["data"])
            or any(len(sample) != 9 for sample in record["data"])
            for record in records
        ):
            raise RuntimeError("An EMG packet does not match its declared sample count / nine-channel shape")
        observed_ns = [int(record["host_receive_mono_ns"]) for record in records]
        if observed_ns != sorted(observed_ns):
            raise RuntimeError("Callback QPC observation times are not monotonic")
        manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
        emg_manifest = manifest["streams"]["emg"]
        imu_manifest = manifest["streams"]["imu"]
        if emg_manifest.get("record_count") != len(records):
            raise RuntimeError("Manifest packet count does not match the HDF5 packet count")
        if emg_manifest.get("sdk_unix_timestamp_recorded") is not False:
            raise RuntimeError("Manifest does not document SDK Unix timestamp exclusion")
        timeline = manifest.get("emg_timeline", {})
        if timeline.get("status") != "complete":
            raise RuntimeError(f"EMG time reconstruction did not complete: {timeline}")
        with h5py.File(session.path / timeline["path"], "r") as aligned:
            if len(aligned["time_ns"]) != sum(int(row["sample_count"]) for row in records):
                raise RuntimeError("Aligned sample count differs from raw packets")
            mapping = timeline["mapping"]
            first = mapping["start_raw_record_index"]
            last = mapping["end_raw_record_index"]
            packets = aligned["packet_json"].asstr()
            first_anchor = json.loads(packets[first])["anchor_time_ns"]
            last_anchor = json.loads(packets[last])["anchor_time_ns"]
            origin = manifest["planned_start_mono_ns"]
            if (first_anchor != records[first]["host_receive_mono_ns"] - origin
                    or last_anchor != records[last]["host_receive_mono_ns"] - origin):
                raise RuntimeError("Aligned endpoints do not match callback QPC anchors")
        imu_raw_path = session.path / "raw" / "imu.h5"
        with h5py.File(imu_raw_path, "r") as raw:
            imu_records = [json.loads(item) for item in raw["record_json"].asstr()[:]]
        if not imu_records:
            raise RuntimeError("Capture ran, but the SDK delivered no IMU packets")
        if any(int(record["sample_count"]) != len(record["data"])
               or any(len(sample) != 6 for sample in record["data"]) for record in imu_records):
            raise RuntimeError("An IMU packet does not match its declared sample count / six-channel shape")
        imu_timeline = manifest.get("imu_timeline", {})
        if imu_timeline.get("status") != "complete":
            raise RuntimeError(f"IMU time reconstruction did not complete: {imu_timeline}")
        with h5py.File(session.path / imu_timeline["path"], "r") as aligned:
            if len(aligned["time_ns"]) != sum(int(row["sample_count"]) for row in imu_records):
                raise RuntimeError("Aligned IMU sample count differs from raw packets")
            mapping = imu_timeline["mapping"]
            first = mapping["start_raw_record_index"]
            last = mapping["end_raw_record_index"]
            packets = aligned["packet_json"].asstr()
            first_anchor = json.loads(packets[first])["anchor_time_ns"]
            last_anchor = json.loads(packets[last])["anchor_time_ns"]
            origin = manifest["planned_start_mono_ns"]
            if (first_anchor != imu_records[first]["host_receive_mono_ns"] - origin
                    or last_anchor != imu_records[last]["host_receive_mono_ns"] - origin):
                raise RuntimeError("Aligned IMU endpoints do not match callback QPC anchors")
        print({
            "device": info["emg"].get("device_name"),
            "device_mac": info["emg"].get("device_mac"),
            "researchkit_sdk_version": info["emg"].get("researchkit_sdk_version"),
            "port": port,
            "packets_written": len(records),
            "samples_written": sum(int(record["sample_count"]) for record in records),
            "manifest_record_count": emg_manifest.get("record_count"),
            "first_packet_shape": [len(records[0]["data"]), len(records[0]["data"][0])],
            "timestamp_stage": records[0]["timestamp_stage"],
            "timeline_status": timeline["status"],
            "timeline_samples": timeline["sample_count"],
            "imu_packets_written": len(imu_records),
            "imu_samples_written": sum(int(record["sample_count"]) for record in imu_records),
            "imu_sample_rate_hz": imu_manifest.get("sample_rate_hz_configured"),
            "imu_timeline_status": imu_timeline["status"],
            "temporary_session_removed_on_exit": True,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
