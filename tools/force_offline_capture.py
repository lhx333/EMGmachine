from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "touchglove" / "python"))
from touch_glove_ptp import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from touch_glove_ptp._fallback import TouchGlove  # noqa: E402

from acquisition.clock import WindowsQpcClock  # noqa: E402
from acquisition.processing.force_timeline import build_force_timeline  # noqa: E402


MODEL_PATH = ROOT / "vendor" / "touchglove" / "model" / "dense_3ch.onnx"
TIMESTAMP_SEMANTICS = "device_rtc_timestamp_us decoded from per-frame BCD RTC bytes [7:15]; UTC epoch microseconds"


def _append_batch(raw: h5py.Group, images: list[np.ndarray], rows: list[dict]) -> None:
    if not rows:
        return
    start = raw["image"].shape[0]
    end = start + len(rows)
    for name in ("image", "channel", "seq_id", "device_timestamp_us", "device_rtc_timestamp_us",
                 "device_rtc_raw", "device_rtc_valid", "host_receive_mono_ns", "sensor_id"):
        raw[name].resize((end,) + raw[name].shape[1:])
    raw["image"][start:end] = np.stack(images).astype(np.uint8, copy=False)
    for name in ("channel", "seq_id", "device_timestamp_us", "device_rtc_timestamp_us",
                 "device_rtc_raw", "device_rtc_valid", "host_receive_mono_ns", "sensor_id"):
        raw[name][start:end] = [row[name] for row in rows]


def _summarize_timestamps(times: np.ndarray, channels: np.ndarray) -> dict:
    result = {}
    for channel in range(IMAGE_CHANNELS):
        values = times[channels == channel]
        diffs = np.diff(values.astype(np.int64))
        result[str(channel)] = {
            "frame_count": int(values.size),
            "first_device_rtc_timestamp_us": int(values[0]) if values.size else None,
            "last_device_rtc_timestamp_us": int(values[-1]) if values.size else None,
            "nonincreasing_adjacent_count_in_capture_order": int(np.count_nonzero(diffs <= 0)),
            "zero_timestamp_count": int(np.count_nonzero(values == 0)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture TouchGlove raw frames, then align and infer offline")
    parser.add_argument("--port", required=True, help="TouchGlove USB COM port, e.g. COM14")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0,
                        help="Opening interval used to estimate the unloaded baseline")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--flush-frames", type=int, default=32)
    args = parser.parse_args()
    if args.seconds <= args.baseline_seconds or args.baseline_seconds <= 0:
        parser.error("--seconds must exceed a positive --baseline-seconds")
    if args.flush_frames < 1:
        parser.error("--flush-frames must be positive")

    clock = WindowsQpcClock()
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    session_path = args.output_root.resolve() / session_id
    (session_path / "raw").mkdir(parents=True, exist_ok=False)
    raw_path = session_path / "raw" / "force.h5"
    pending_images: list[np.ndarray] = []
    pending_rows: list[dict] = []
    reader = TouchGlove(auto_open=False, use_c_lib=False)

    capture_start_ns: int | None = None
    try:
        with h5py.File(raw_path, "w") as capture:
            capture.attrs.update(
                schema_version="0.4",
                stream="force",
                device="TouchGlove Windows SDK",
                backend="Python serial frame parser",
                timestamp_field="device_rtc_timestamp_us",
                timestamp_semantics=TIMESTAMP_SEMANTICS,
                host_receive_mono_ns_semantics="Windows QPC sampled immediately after poll() returns; observation time",
                raw_layout="one 192x192 uint8 image per channel frame; channel, SDK sequence id, device time and QPC observation retained",
                inference_timing="all raw frames are persisted before offline inference starts",
            )
            chunk = int(args.flush_frames)
            raw = capture.create_group("frames")
            raw.create_dataset("image", shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH),
                               maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
                               chunks=(min(chunk, 32), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            for name, dtype in (("channel", np.uint8), ("seq_id", np.uint32),
                                ("device_timestamp_us", np.uint64), ("device_rtc_timestamp_us", np.uint64),
                                ("device_rtc_valid", np.uint8), ("host_receive_mono_ns", np.uint64)):
                raw.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(chunk,), dtype=dtype)
            raw.create_dataset("device_rtc_raw", shape=(0, 8), maxshape=(None, 8),
                               chunks=(chunk, 8), dtype=np.uint8)
            raw.create_dataset("sensor_id", shape=(0,), maxshape=(None,), chunks=(chunk,),
                               dtype=h5py.string_dtype("utf-8"))

            if not reader.open(args.port):
                raise RuntimeError(f"Could not open TouchGlove on {args.port}")
            reader.start()
            capture_start_ns = clock.now_ns()
            capture.attrs["state"] = "CAPTURING_RAW_FRAMES"
            capture.attrs["planned_start_mono_ns"] = capture_start_ns
            deadline = time.monotonic() + args.seconds
            started = time.monotonic()
            print(json.dumps({"phase": "capturing", "session": str(session_path),
                              "duration_s": args.seconds, "baseline_s": args.baseline_seconds,
                              "port": args.port}, ensure_ascii=False), flush=True)
            next_report = 5
            while time.monotonic() < deadline:
                frames = reader.poll()
                observed_ns = clock.now_ns()
                for frame in frames:
                    pending_images.append(np.asarray(frame.image, dtype=np.uint8).copy())
                    pending_rows.append({
                        "channel": int(frame.channel),
                        "seq_id": int(frame.seq_id) & 0xFFFFFFFF,
                        "device_timestamp_us": int(frame.timestamp_us),
                        "device_rtc_timestamp_us": int(frame.device_rtc_timestamp_us or 0),
                        "device_rtc_raw": list(frame.device_rtc_raw or bytes(8)),
                        "device_rtc_valid": int(frame.device_rtc_timestamp_us is not None),
                        "host_receive_mono_ns": observed_ns,
                        "sensor_id": str(frame.sensor_id),
                    })
                if len(pending_rows) >= chunk:
                    _append_batch(raw, pending_images, pending_rows)
                    pending_images.clear()
                    pending_rows.clear()
                    capture.flush()
                elapsed = int(time.monotonic() - started)
                if elapsed >= next_report:
                    count = raw["image"].shape[0] + len(pending_rows)
                    print(json.dumps({"phase": "capturing", "elapsed_s": elapsed, "frames_written": count}), flush=True)
                    next_report += 5
                if not frames:
                    time.sleep(0.001)
            if pending_rows:
                _append_batch(raw, pending_images, pending_rows)
                capture.flush()
            if raw["image"].shape[0] == 0:
                raise RuntimeError("Capture ended without any image frames")
            capture.attrs["state"] = "RAW_CAPTURE_COMPLETE"
            capture.flush()
    finally:
        try:
            reader.stop()
        finally:
            reader.close()

    assert capture_start_ns is not None
    with h5py.File(raw_path, "r") as capture:
        raw = capture["frames"]
        qpc = raw["host_receive_mono_ns"][:].astype(np.uint64)
        channels = raw["channel"][:].astype(np.uint8)
        rtc_times = raw["device_rtc_timestamp_us"][:].astype(np.uint64)
        baseline_end = capture_start_ns + int(args.baseline_seconds * 1_000_000_000)
        baseline_mask = qpc <= baseline_end
        sums = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)
        sample_counts = np.zeros(IMAGE_CHANNELS, dtype=np.uint64)
        for index in np.flatnonzero(baseline_mask):
            channel = int(channels[index])
            sums[channel] += raw["image"][index]
            sample_counts[channel] += 1
        active_channels = np.unique(channels)
        if any(int(sample_counts[channel]) == 0 for channel in active_channels):
            raise RuntimeError(
                "Baseline interval did not receive samples for every channel present in the capture: "
                f"active_channels={active_channels.tolist()}, baseline_samples={sample_counts.tolist()}"
            )
        baseline = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        for channel in active_channels:
            baseline[channel] = np.rint(sums[channel] / sample_counts[channel]).clip(0, 255).astype(np.uint8)

    with h5py.File(raw_path, "r+") as capture:
        baseline_group = capture.create_group("baseline")
        baseline_group.create_dataset("image", data=baseline, compression="lzf")
        baseline_group.create_dataset("sample_count_by_channel", data=sample_counts)
        baseline_group.attrs.update(duration_s=float(args.baseline_seconds), source="opening raw capture interval")
        capture.attrs["state"] = "RAW_CAPTURE_COMPLETE_INFERENCE_RUNNING"
        capture.flush()

    descriptor = {
        "device": "TouchGlove",
        "backend": "Python serial frame parser",
        "timestamp_field": "device_rtc_timestamp_us",
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "host_receive_mono_ns_semantics": "Windows QPC sampled after poll() returns; host observation time",
        "frame_layout": "192x192 uint8 image; channel, seq_id and sensor_id identify each SDK frame",
        "sample_rate_hz_configured": None,
        "inference": "DirectML after raw capture completes",
    }
    manifest = {
        "schema_version": "0.2",
        "session_id": session_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clock_id": clock.clock_id,
        "clock_frequency_hz": clock.frequency_hz,
        "planned_start_mono_ns": capture_start_ns,
        "state": "RAW_CAPTURE_COMPLETE_INFERENCE_RUNNING",
        "streams": {"force": descriptor},
    }
    (session_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        report = build_force_timeline(session_path, capture_start_ns)
    except Exception as exc:
        manifest["state"] = "TIMELINE_FAILED"
        manifest["force_timeline"] = {"status": "failed", "error": str(exc)}
        with h5py.File(raw_path, "r+") as capture:
            capture.attrs["state"] = "TIMELINE_FAILED"
            capture.attrs["timeline_error"] = str(exc)
            capture.flush()
        (session_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (session_path / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps({"event": "FORCE_TIMELINE_FAILED", "event_mono_ns": clock.now_ns(),
                                     "utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)},
                                    ensure_ascii=False) + "\n")
        raise
    output_path = session_path / report["path"]
    inferencer = None
    try:
        inferencer = TouchGlove(model=str(MODEL_PATH), device="directml", auto_open=False)
        if inferencer._ort_session is None:
            raise RuntimeError("Could not initialize the TouchGlove Dense ONNX model")
        providers = inferencer._ort_session.get_providers()
        if not providers or providers[0] != "DmlExecutionProvider":
            raise RuntimeError(f"DirectML is not the active first provider: {providers}")
        if not hasattr(inferencer, "set_baseline_images") or not hasattr(inferencer, "infer_frame"):
            raise RuntimeError("Vendored SDK does not expose offline inference helpers")
        inferencer.set_baseline_images(baseline)
        with h5py.File(raw_path, "r") as source, h5py.File(output_path, "r+") as output:
            frames = source["frames"]
            count = len(frames["channel"])
            output.create_dataset("disp", shape=(count, 32, 32, 3), dtype=np.float32,
                                  chunks=(min(8, count), 32, 32, 3), compression="lzf")
            output.create_dataset("force_field", shape=(count, 32, 32, 3), dtype=np.float32,
                                  chunks=(min(8, count), 32, 32, 3), compression="lzf")
            for index in range(count):
                field_disp, field_force = inferencer.infer_frame(int(frames["channel"][index]), frames["image"][index])
                if field_disp is None or field_force is None:
                    raise RuntimeError(f"Inference failed for raw frame {index}")
                output["disp"][index] = field_disp
                output["force_field"][index] = field_force
                if index % args.flush_frames == args.flush_frames - 1:
                    output.flush()
                    print(json.dumps({"phase": "offline_inference", "completed": index + 1,
                                      "total": count, "providers": providers}), flush=True)
            output.attrs["inference_provider_order"] = json.dumps(providers)
            output.attrs["model_path"] = str(MODEL_PATH.relative_to(ROOT))
            output.attrs["inference_status"] = "complete"
            output.flush()
        report["inference"] = {"status": "complete", "provider_order": providers,
                               "model": MODEL_PATH.relative_to(ROOT).as_posix()}
        report["raw_frame_count"] = int(len(rtc_times))
        report["baseline_samples_by_channel"] = sample_counts.tolist()
        report["timestamp_summary_by_channel"] = _summarize_timestamps(rtc_times, channels)
        report["capture_duration_s"] = args.seconds
        report["baseline_duration_s"] = args.baseline_seconds
        report["limits"] += " Dense outputs are post-capture inference; time_ns is copied from the endpoint-mapped frame timeline."
        with h5py.File(raw_path, "r+") as capture:
            capture.attrs["state"] = "COMPLETED"
            capture.attrs["inference_provider_order"] = json.dumps(providers)
            capture.flush()
        manifest["state"] = "COMPLETED"
        manifest["force_timeline"] = report
    except Exception as exc:
        report["inference"] = {"status": "failed", "error": str(exc)}
        report["status"] = "inference_failed"
        if output_path.exists():
            with h5py.File(output_path, "r+") as output:
                output.attrs["inference_status"] = "failed"
                output.attrs["inference_error"] = str(exc)
        with h5py.File(raw_path, "r+") as capture:
            capture.attrs["state"] = "INFERENCE_FAILED"
            capture.attrs["inference_error"] = str(exc)
            capture.flush()
        manifest["state"] = "INFERENCE_FAILED"
        manifest["force_timeline"] = report
        (output_path.parent / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    finally:
        if inferencer is not None:
            inferencer.close()
        (session_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (session_path / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps({"event": "FORCE_CAPTURE_AND_OFFLINE_INFERENCE_FINISHED",
                                     "event_mono_ns": clock.now_ns(), "utc": datetime.now(timezone.utc).isoformat(),
                                     "state": manifest["state"], "frame_count": int(len(rtc_times))},
                                    ensure_ascii=False) + "\n")
    (output_path.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
