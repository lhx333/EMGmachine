from __future__ import annotations

import copy
import json
import math
import os
import platform
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

from acquisition.clock import WindowsQpcClock
from acquisition.recording.hdf5_writer import Hdf5Writer
from acquisition.devices.emg import DEFAULT_EMG_MAC, EmgAdapter, normalize_emg_mac
from acquisition.devices.realsense import RealSenseAdapter
from acquisition.devices.nokov import NokovAdapter
from acquisition.devices.touchglove import TouchGloveAdapter
from acquisition.processing.emg_timeline import build_emg_timeline, build_imu_timeline
from acquisition.processing.force_inference import run_force_inference, validate_touchglove_runtime
from acquisition.processing.force_timeline import build_force_timeline
from acquisition.processing.nokov_timeline import build_nokov_timeline
from acquisition.processing.realsense_timeline import build_realsense_timeline


_ROOT = Path(__file__).resolve().parents[3]
_TOUCHGLOVE_MODEL = _ROOT / "vendor" / "touchglove" / "model" / "dense_3ch.onnx"


class Session:
    def __init__(self, root: Path, session_id: str | None = None, sample_hz: float = 50.0, experiment: dict[str, str] | None = None) -> None:
        self.root = root.resolve()
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        if not all(c.isalnum() or c in "_-" for c in self.session_id):
            raise ValueError("session_id may contain only letters, numbers, underscores, and hyphens")
        self.path = self.root / self.session_id
        self.clock = WindowsQpcClock()
        self.sample_hz = sample_hz
        self.experiment = {key: value.strip() for key, value in (experiment or {}).items() if value.strip()}
        self.state = "IDLE"
        self._writer: Hdf5Writer | None = None
        self._emg: EmgAdapter | None = None
        self._realsense: RealSenseAdapter | None = None
        self._nokov: NokovAdapter | None = None
        self._touchglove: TouchGloveAdapter | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._sample_count = 0
        self._imu_sample_count = 0
        self._realsense_frame_count = 0
        self._nokov_frame_count = 0
        self._touchglove_frame_count = 0
        self._manifest: dict[str, Any] = {}
        self._record_after_ns: int | None = None
        self._preview: dict[str, dict[str, Any]] = {}
        self._emg_preview_rows: deque[list[float]] = deque(maxlen=3000)
        self._emg_preview_sequence = 0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def imu_sample_count(self) -> int:
        return self._imu_sample_count

    @property
    def realsense_frame_count(self) -> int:
        return self._realsense_frame_count

    @property
    def nokov_frame_count(self) -> int:
        return self._nokov_frame_count

    @property
    def touchglove_frame_count(self) -> int:
        return self._touchglove_frame_count

    @property
    def preview(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result = {name: dict(value) for name, value in self._preview.items()}
            emg = result.get("emg")
            if emg is not None:
                emg["samples"] = [list(row) for row in self._emg_preview_rows]
                emg["sequence_start"] = self._emg_preview_sequence - len(self._emg_preview_rows)
            return result

    @property
    def all_devices_connected(self) -> bool:
        return all(value is not None for value in
                   (self._emg, self._realsense, self._nokov, self._touchglove))

    @property
    def emg_timeline_status(self) -> str | None:
        return self._manifest.get("emg_timeline", {}).get("status")

    def prepare(self) -> Path:
        with self._lock:
            if self.state != "IDLE":
                raise RuntimeError(f"cannot prepare in state {self.state}")
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / "raw").mkdir()
            self._writer = Hdf5Writer(self.path / "raw")
            self._manifest = {
                "schema_version": "0.2",
                "session_id": self.session_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "host_id": platform.node(),
                "boot_id": "unknown",
                "clock_id": self.clock.clock_id,
                "clock_frequency_hz": self.clock.frequency_hz,
                "clock_conversion": "floor(counter * 1e9 / frequency)",
                "state": "PREPARED",
                "experiment": self.experiment,
                "streams": {},
            }
            self._write_manifest()
            self._event("PREPARE", {"path": str(self.path)})
            self.state = "PREPARED"
            return self.path

    def start_demo(self, delay_s: float = 1.0) -> int:
        with self._lock:
            if self.state != "PREPARED":
                raise RuntimeError(f"cannot start in state {self.state}")
            deadline = self.clock.now_ns() + int(delay_s * 1_000_000_000)
            self._manifest["planned_start_mono_ns"] = deadline
            self._manifest["state"] = "ARMED"
            self.state = "ARMED"
            self._write_manifest()
            self._event("ARM", {"deadline_windows_ns": deadline, "clock_id": self.clock.clock_id})
            self._thread = threading.Thread(target=self._demo_loop, args=(deadline,), name="demo-stream", daemon=True)
            self._thread.start()
            return deadline

    def _ensure_device_setup(self) -> None:
        if self.state == "IDLE":
            self.prepare()
        if self.state not in {"PREPARED", "CONNECTING_DEVICES"}:
            raise RuntimeError(f"cannot connect a device in state {self.state}")
        self.state = "CONNECTING_DEVICES"
        self._manifest["state"] = self.state
        self._write_manifest()

    def connect_emg(self, port: str, target_mac: str,
                    sample_rate_hz: int = 1000, imu_sample_rate_hz: int = 50) -> dict[str, Any]:
        self._ensure_device_setup()
        if self._emg is not None:
            raise RuntimeError("EMG is already connected")
        adapter = EmgAdapter(self._append_emg_record, self._on_emg_event)
        self._emg = adapter
        try:
            info = adapter.prepare(port, sample_rate_hz, imu_sample_rate_hz, target_mac=target_mac)
        except Exception:
            self._emg = None
            raise
        self._manifest["streams"]["emg"] = {
            "device": info, "sample_rate_hz_configured": int(sample_rate_hz),
            "timestamp_stage": "researchkit_on_data_received_callback",
            "host_receive_mono_ns_semantics": "Windows QPC sampled in SDK callback; packet observation time",
            "device_timestamp_raw_semantics": "raw UPLOAD_DATA TIMESTAMP TLV; device-relative milliseconds",
            "device_sequence_field": "none identified in the observed UPLOAD_DATA frame",
        }
        self._manifest["streams"]["imu"] = {
            "device": info, "sample_rate_hz_configured": int(imu_sample_rate_hz),
            "channel_count": 6, "data_encoding": "SDK int16 preserved without unit conversion",
            "timestamp_stage": "researchkit_on_data_received_callback",
            "host_receive_mono_ns_semantics": "Windows QPC sampled in SDK callback; packet observation time",
        }
        self._write_manifest()
        self._event("EMG_DEVICE_CONNECTED", {"port": port, "mac": info.get("device_mac")})
        return info

    def connect_realsense(self) -> dict[str, Any]:
        self._ensure_device_setup()
        if self._realsense is not None:
            raise RuntimeError("RealSense is already connected")
        adapter = RealSenseAdapter(self._append_realsense_record, self._on_realsense_event)
        self._realsense = adapter
        try:
            info = adapter.prepare(self.path / "raw")
        except Exception:
            self._realsense = None
            raise
        self._manifest["streams"]["realsense_rgb"] = {
            "device": info, "timestamp_stage": "qpc_after_wait_for_frames",
            "host_receive_mono_ns_semantics": "QPC after wait_for_frames; host observation time",
            "camera_timestamp_semantics": "raw color-frame SDK timestamp and domain",
            "video_path": "raw/realsense_rgb.mp4", "video_codec": info["video_codec"],
            "raw_frame_pixels_preserved": False,
        }
        self._write_manifest()
        self._event("REALSENSE_DEVICE_CONNECTED", {"serial": info.get("serial_number")})
        return info

    def connect_nokov(self, server_ip: str) -> dict[str, Any]:
        self._ensure_device_setup()
        if self._nokov is not None:
            raise RuntimeError("NOKOV is already connected")
        adapter = NokovAdapter(self._append_nokov_record, self._on_nokov_event)
        self._nokov = adapter
        try:
            info = adapter.prepare(server_ip)
        except Exception:
            self._nokov = None
            raise
        self._manifest["streams"]["nokov"] = {
            "device": info, "timestamp_stage": "qpc_at_nokov_data_callback_entry",
            "host_receive_mono_ns_semantics": "QPC at SDK callback entry; host observation time",
            "device_frame_index_field": "iFrame", "device_timestamp_field": "iTimeStamp",
            "coordinate_unit": "mm", "orientation_order": "qx,qy,qz,qw",
            "validity_policy": "Raw values retained; no unverified sentinel normalization",
        }
        self._write_manifest()
        self._event("NOKOV_DEVICE_CONNECTED", {"server_ip": server_ip})
        return info

    def connect_touchglove(self, port: str, baseline_seconds: float = 1.0) -> dict[str, Any]:
        self._ensure_device_setup()
        if self._touchglove is not None:
            raise RuntimeError("TouchGlove is already connected")
        adapter = TouchGloveAdapter(self._on_touchglove_event, baseline_seconds=baseline_seconds)
        self._touchglove = adapter
        try:
            info = adapter.prepare(self.path / "raw", port)
            info["inference_runtime"] = validate_touchglove_runtime(_TOUCHGLOVE_MODEL)
        except Exception:
            try:
                adapter.stop()
            except Exception:
                pass
            self._touchglove = None
            raise
        self._manifest["streams"]["force"] = {
            "device": info, "timestamp_field": "device_rtc_timestamp_us",
            "timestamp_semantics": "decoded device RTC from BCD bytes [7:15]; UTC epoch microseconds",
            "host_receive_mono_ns_semantics": "QPC after poll() returns; observation time",
            "frame_layout": "192x192 uint8 image; channel, seq_id and sensor_id identify each SDK frame",
            "baseline_duration_s": float(baseline_seconds),
            "inference": "offline after raw capture; DirectML required; raw images retained",
        }
        self._write_manifest()
        self._event("TOUCHGLOVE_DEVICE_CONNECTED", {"port": port})
        return info

    def prepare_emg(self, port: str, sample_rate_hz: int = 1000, imu_sample_rate_hz: int = 50,
                    nokov_server_ip: str = "10.1.1.198", touchglove_port: str | None = None,
                    touchglove_baseline_seconds: float = 1.0,
                    emg_target_mac: str = DEFAULT_EMG_MAC) -> dict[str, Any]:
        """Compatibility helper; the UI uses the individual connect methods."""
        if not touchglove_port:
            raise ValueError("TouchGlove serial port must be selected")
        emg = self.connect_emg(port, emg_target_mac, sample_rate_hz, imu_sample_rate_hz)
        camera = self.connect_realsense()
        nokov = self.connect_nokov(nokov_server_ip)
        force = self.connect_touchglove(touchglove_port, touchglove_baseline_seconds)
        return {"emg": emg, "imu": {"sample_rate_hz": imu_sample_rate_hz, "channel_count": 6},
                "realsense": camera, "nokov": nokov, "force": force}

    def start_emg(self, delay_s: float = 1.0) -> int:
        with self._lock:
            if self.state not in {"PREPARED", "CONNECTING_DEVICES"}:
                raise RuntimeError(f"All devices must be connected before start; session is {self.state}")
            missing = [name for name, value in (("EMG/IMU", self._emg), ("RealSense", self._realsense),
                                                ("NOKOV", self._nokov), ("TouchGlove", self._touchglove))
                       if value is None]
            if missing:
                raise RuntimeError("Devices must be prepared before start: " + ", ".join(missing))
            deadline = self.clock.now_ns() + int(delay_s * 1_000_000_000)
            self._record_after_ns = deadline
            self._manifest["planned_start_mono_ns"] = deadline
            self._manifest["state"] = "ARMED"
            self.state = "ARMED"
            self._write_manifest()
            self._event("ARM", {"deadline_windows_ns": deadline, "clock_id": self.clock.clock_id,
                                 "streams": ["emg", "imu", "realsense_rgb", "nokov", "force"]})
            self._emg.arm(deadline)
            assert self._realsense is not None
            self._realsense.arm(deadline)
            assert self._nokov is not None
            self._nokov.arm(deadline)
            assert self._touchglove is not None
            self._touchglove.arm(deadline)
            return deadline

    def _append_emg_record(self, record: dict[str, Any]) -> None:
        sensor = str(record.get("sensor_type", "")).lower()
        if sensor not in {"emg", "imu"}:
            raise ValueError(f"Unexpected ResearchKit sensor type: {sensor}")
        values = record.get("data", [])
        with self._lock:
            preview = {
                "count": int(record.get("sample_count", 0)),
                "host_receive_mono_ns": int(record.get("host_receive_mono_ns", 0)),
                "sample_rate_hz": int(record.get("sample_rate_hz", 0)),
                "latest": values[-1] if isinstance(values, list) and values else values,
            }
            if sensor == "emg":
                rows = values if isinstance(values, list) else []
                if rows and not isinstance(rows[0], (list, tuple)):
                    rows = [rows]
                for row in rows:
                    try:
                        self._emg_preview_rows.append([float(value) for value in row])
                    except (TypeError, ValueError):
                        continue
                self._emg_preview_sequence += len(rows)
                preview["samples"] = [list(row) for row in self._emg_preview_rows]
                preview["sequence_start"] = self._emg_preview_sequence - len(self._emg_preview_rows)
            self._preview[sensor] = preview
            writer = self._writer
            record_after_ns = self._record_threshold_locked()
            should_record = (writer is not None and record_after_ns is not None
                             and int(record.get("host_receive_mono_ns", 0)) >= record_after_ns)
        if not should_record:
            return
        writer.append(sensor, record)
        with self._lock:
            if sensor == "emg":
                self._sample_count += int(record.get("sample_count", 0))
            else:
                self._imu_sample_count += int(record.get("sample_count", 0))

    def _append_realsense_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._preview["realsense_rgb"] = {
                "frame_index": int(record.get("frame_index", 0)),
                "camera_frame_number": int(record.get("camera_frame_number", 0)),
                "host_receive_mono_ns": int(record.get("host_receive_mono_ns", 0)),
                "preview_jpeg": record.get("preview_jpeg", b""),
            }
            writer = self._writer
            record_after_ns = self._record_threshold_locked()
            should_record = (writer is not None and record_after_ns is not None
                             and int(record.get("host_receive_mono_ns", 0)) >= record_after_ns)
        if not should_record:
            return
        # JPEG bytes are an ephemeral UI preview. The full RGB stream is
        # persisted separately as MP4, while this HDF5 stream stores only
        # JSON-serializable frame metadata.
        persisted_record = {key: value for key, value in record.items() if key != "preview_jpeg"}
        writer.append("realsense_rgb", persisted_record)
        with self._lock:
            self._realsense_frame_count += 1

    def _append_nokov_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            bodies = record.get("rigid_bodies", [])
            markers = record.get("labeled_markers", [])
            self._preview["nokov"] = {
                "device_frame_index": int(record.get("device_frame_index", 0)),
                "host_receive_mono_ns": int(record.get("host_receive_mono_ns", 0)),
                "rigid_body_count": len(bodies), "labeled_marker_count": len(markers),
                "first_rigid_body_mm": bodies[0].get("position_mm") if bodies else None,
            }
            writer = self._writer
            record_after_ns = self._record_threshold_locked()
            should_record = (writer is not None and record_after_ns is not None
                             and int(record.get("host_receive_mono_ns", 0)) >= record_after_ns)
        if not should_record:
            return
        writer.append("nokov", record)
        with self._lock:
            self._nokov_frame_count += 1

    def _on_touchglove_event(self, name: str, details: dict[str, Any]) -> None:
        stop_session = False
        with self._lock:
            if name != "TOUCHGLOVE_PREVIEW":
                self._event(name, details)
            if name == "TOUCHGLOVE_STATUS":
                self._touchglove_frame_count = int(details.get("frame_count", self._touchglove_frame_count))
            elif name == "TOUCHGLOVE_PREVIEW":
                self._preview["force"] = {
                    "channel": int(details.get("channel", -1)),
                    "seq_id": int(details.get("seq_id", 0)),
                    "device_timestamp_us": int(details.get("device_timestamp_us", 0)),
                    "host_receive_mono_ns": int(details.get("host_receive_mono_ns", 0)),
                    "image_96x96": details.get("image_96x96", b""),
                }
            elif name == "TOUCHGLOVE_RECORDING" and self.state not in {"ERROR", "STOPPING"}:
                self._manifest["streams"].setdefault("force", {})["actual_start_mono_ns"] = int(details["actual_start_mono_ns"])
            elif name == "TOUCHGLOVE_ERROR":
                self._manifest.setdefault("streams", {}).setdefault("force", {})["error"] = details.get("error")
                if self.state in {"ARMED", "RECORDING"}:
                    self.state = "ERROR"
                    self._manifest["state"] = "ERROR"
                    self._manifest["error"] = details.get("error")
                    stop_session = True
                self._write_manifest()
        if stop_session:
            threading.Thread(target=self.stop, name="touchglove-failure-shutdown", daemon=True).start()

    def _on_force_inference_progress(self, completed: int, total: int) -> None:
        with self._lock:
            self._manifest["force_inference"] = {
                "status": "running", "completed_frames": int(completed), "total_frames": int(total)
            }
            self._write_manifest()

    def _on_realsense_event(self, name: str, details: dict[str, Any]) -> None:
        stop_session = False
        with self._lock:
            if name != "NOKOV_PREVIEW":
                self._event(name, details)
            if name == "REALSENSE_ERROR":
                self._manifest["streams"].setdefault("realsense_rgb", {})["error"] = details.get("error")
                if self.state in {"ARMED", "RECORDING"}:
                    self.state = "ERROR"
                    self._manifest["state"] = "ERROR"
                    self._manifest["error"] = details.get("error")
                    stop_session = True
                self._write_manifest()
        if stop_session:
            threading.Thread(target=self.stop, name="device-failure-shutdown", daemon=True).start()

    def _on_emg_event(self, name: str, details: dict[str, Any]) -> None:
        with self._lock:
            self._event(name, details)
            if name == "EMG_RECORDING" and self.state != "ERROR" and not self._stop.is_set():
                self.state = "RECORDING"
                self._manifest["state"] = "RECORDING"
                self._manifest["actual_open_mono_ns"] = int(details["actual_start_mono_ns"])
                self._write_manifest()
            elif name == "EMG_ERROR":
                self._manifest["state"] = "ERROR"
                self._manifest["error"] = details.get("error")
                self.state = "ERROR"
                self._write_manifest()

    def _on_nokov_event(self, name: str, details: dict[str, Any]) -> None:
        stop_session = False
        with self._lock:
            self._event(name, details)
            if name == "NOKOV_PREVIEW":
                self._preview["nokov"] = dict(details)
            if name == "NOKOV_ERROR":
                self._manifest.setdefault("streams", {}).setdefault("nokov", {})["error"] = details.get("error")
                if self.state in {"ARMED", "RECORDING"}:
                    self.state = "ERROR"
                    self._manifest["state"] = "ERROR"
                    self._manifest["error"] = details.get("error")
                    stop_session = True
                self._write_manifest()
        if stop_session:
            threading.Thread(target=self.stop, name="nokov-failure-shutdown", daemon=True).start()

    def _record_threshold_locked(self) -> int | None:
        if self._record_after_ns is not None:
            return self._record_after_ns
        if self.state in {"ARMED", "RECORDING"}:
            value = self._manifest.get("planned_start_mono_ns")
            return int(value) if value is not None else None
        return None

    def _demo_loop(self, deadline_ns: int) -> None:
        while not self._stop.is_set():
            remaining = deadline_ns - self.clock.now_ns()
            if remaining <= 0:
                break
            self._stop.wait(min(remaining / 1_000_000_000, 0.05))
        if self._stop.is_set():
            return
        with self._lock:
            self.state = "RECORDING"
            self._manifest["state"] = "RECORDING"
            self._manifest["actual_open_mono_ns"] = self.clock.now_ns()
            self._write_manifest()
            self._event("RECORDING_OPEN", {"actual_open_mono_ns": self._manifest["actual_open_mono_ns"]})
        start_ns = self.clock.now_ns()
        period_ns = int(1_000_000_000 / self.sample_hz)
        index = 0
        while not self._stop.is_set():
            target_ns = start_ns + index * period_ns
            now_ns = self.clock.now_ns()
            if now_ns < target_ns:
                self._stop.wait((target_ns - now_ns) / 1_000_000_000)
                continue
            elapsed = (now_ns - start_ns) / 1_000_000_000
            value = math.sin(2 * math.pi * 0.5 * elapsed)
            record = {
                "record_id": index,
                "logical_index": index,
                "host_receive_mono_ns": now_ns,
                "timestamp_stage": "synthetic_sample_observed",
                "value": value,
                "observed_valid": True,
            }
            try:
                assert self._writer is not None
                self._writer.append("demo", record)
            except Exception as exc:
                with self._lock:
                    self._event("RECORDING_ERROR", {"error": str(exc)})
                    self._stop.set()
                break
            index += 1
            self._sample_count = index

    def stop(self) -> None:
        with self._lock:
            if self.state not in {"ARMED", "RECORDING", "ERROR"}:
                if self.state in {"PREPARED", "CONNECTING_DEVICES", "PREPARE_FAILED"}:
                    self._stop.set()
                else:
                    return
                emg = self._emg
                realsense = self._realsense
                nokov = self._nokov
                touchglove = self._touchglove
                thread = None
            else:
                self._stop.set()
                thread = self._thread
                emg = self._emg
                realsense = self._realsense
                nokov = self._nokov
                touchglove = self._touchglove
                self.state = "STOPPING"
                self._manifest["state"] = "STOPPING"
                self._write_manifest()
        if realsense:
            try:
                final_camera = realsense.stop()
                with self._lock:
                    descriptor = self._manifest.get("streams", {}).get("realsense_rgb", {})
                    descriptor["frame_count"] = int(final_camera["frame_count"])
                    descriptor["frame_number_gap_count"] = int(final_camera["frame_number_gap_count"])
                    descriptor["recording_started_mono_ns"] = final_camera.get("recording_started_mono_ns")
                    descriptor["video_path"] = final_camera.get("video_path")
                    self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._event("REALSENSE_STOP_FAILED", {"error": str(exc)})
                    self._manifest.setdefault("streams", {}).setdefault("realsense_rgb", {})["stop_error"] = str(exc)
        if emg:
            emg.stop()
        if nokov:
            try:
                nokov.stop()
            except Exception as exc:
                with self._lock:
                    self._event("NOKOV_STOP_FAILED", {"error": str(exc)})
                    self._manifest.setdefault("streams", {}).setdefault("nokov", {})["stop_error"] = str(exc)
        if touchglove:
            try:
                final_force = touchglove.stop()
                if final_force:
                    with self._lock:
                        descriptor = self._manifest.get("streams", {}).get("force", {})
                        descriptor["frame_count"] = int(final_force.get("frame_count", 0))
                        descriptor["baseline_samples_by_channel"] = final_force.get("baseline_samples_by_channel")
                        descriptor["raw_path"] = final_force.get("raw_path")
                        self._touchglove_frame_count = int(final_force.get("frame_count", 0))
                        self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._event("TOUCHGLOVE_STOP_FAILED", {"error": str(exc)})
                    self._manifest.setdefault("streams", {}).setdefault("force", {})["stop_error"] = str(exc)
        if thread:
            thread.join()
        # Closing the writer drains its queue and may perform substantial disk
        # I/O. Detach it under the lock, then wait without blocking UI readers.
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer:
            writer.close()
            counts = writer.counts
        else:
            counts = {}

        with self._lock:
            stream_descriptors = self._manifest.setdefault("streams", {})
            for stream_name, count in counts.items():
                stream_descriptors.setdefault(stream_name, {})["record_count"] = count
            if "emg" in stream_descriptors:
                stream_descriptors["emg"]["sample_count"] = self._sample_count
            if "imu" in stream_descriptors:
                stream_descriptors["imu"]["sample_count"] = self._imu_sample_count
            if "nokov" in stream_descriptors and self._nokov is not None:
                stream_descriptors["nokov"].update(frame_count=self._nokov_frame_count,
                                                     **self._nokov.quality_counts)
            self.state = "PROCESSING"
            self._manifest["state"] = "PROCESSING"
            self._event("POST_CAPTURE_PROCESSING", {})
            self._write_manifest()
            stream_descriptors = copy.deepcopy(stream_descriptors)
            origin_ns = self._manifest.get("planned_start_mono_ns")

        # Timeline generation and model inference can take seconds or minutes.
        # Never hold the Session lock while doing them: the GUI polls preview
        # and counters on its main thread.
        if "emg" in stream_descriptors and origin_ns is not None:
            try:
                timeline = build_emg_timeline(self.path, origin_ns,
                                              stream_descriptors["emg"]["sample_rate_hz_configured"])
                with self._lock:
                    self._manifest["emg_timeline"] = timeline
                    self._event("EMG_TIMELINE_BUILT", {"path": timeline["path"], "status": timeline["status"]})
                    self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._manifest["emg_timeline"] = {"status": "failed", "error": str(exc)}
                    self._event("EMG_TIMELINE_FAILED", {"error": str(exc)})
                    self._write_manifest()
        if "imu" in stream_descriptors and origin_ns is not None:
            try:
                timeline = build_imu_timeline(self.path, origin_ns,
                                              stream_descriptors["imu"]["sample_rate_hz_configured"])
                with self._lock:
                    self._manifest["imu_timeline"] = timeline
                    self._event("IMU_TIMELINE_BUILT", {"path": timeline["path"], "status": timeline["status"]})
                    self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._manifest["imu_timeline"] = {"status": "failed", "error": str(exc)}
                    self._event("IMU_TIMELINE_FAILED", {"error": str(exc)})
                    self._write_manifest()
        if "nokov" in stream_descriptors and origin_ns is not None:
            try:
                timeline = build_nokov_timeline(self.path, origin_ns)
                with self._lock:
                    self._manifest["nokov_timeline"] = timeline
                    self._event("NOKOV_TIMELINE_BUILT", {"path": timeline["path"], "status": timeline["status"]})
                    self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._manifest["nokov_timeline"] = {"status": "failed", "error": str(exc)}
                    self._event("NOKOV_TIMELINE_FAILED", {"error": str(exc)})
                    self._write_manifest()
        if "realsense_rgb" in stream_descriptors and origin_ns is not None:
            try:
                timeline = build_realsense_timeline(self.path, origin_ns)
                with self._lock:
                    self._manifest["realsense_timeline"] = timeline
                    self._event("REALSENSE_TIMELINE_BUILT", {"path": timeline["path"], "status": timeline["status"]})
                    self._write_manifest()
            except Exception as exc:
                with self._lock:
                    self._manifest["realsense_timeline"] = {"status": "failed", "error": str(exc)}
                    self._event("REALSENSE_TIMELINE_FAILED", {"error": str(exc)})
                    self._write_manifest()
        if "force" in stream_descriptors and origin_ns is not None:
            raw_force_path = self.path / "raw" / "force.h5"
            if raw_force_path.exists():
                force_output_path = None
                try:
                    timeline = build_force_timeline(self.path, origin_ns)
                    force_output_path = self.path / timeline["path"]
                    inference = run_force_inference(self.path, force_output_path, _TOUCHGLOVE_MODEL,
                                                    on_progress=self._on_force_inference_progress)
                    timeline["inference"] = inference
                    timeline["frame_count"] = inference["raw_frame_count"]
                    timeline["baseline_samples_by_channel"] = inference["baseline_samples_by_channel"]
                    with h5py.File(raw_force_path, "r+") as raw:
                        raw.attrs["state"] = "COMPLETED"
                        raw.attrs["inference_provider_order"] = json.dumps(inference["provider_order"])
                        raw.flush()
                    with self._lock:
                        self._manifest["force_timeline"] = timeline
                        self._manifest["force_inference"] = {"status": "complete", **inference}
                        self._event("FORCE_TIMELINE_BUILT", {"path": timeline["path"], "status": "complete",
                                                               "frame_count": timeline["frame_count"]})
                        self._write_manifest()
                except Exception as exc:
                    if force_output_path is not None and force_output_path.exists():
                        try:
                            with h5py.File(force_output_path, "r+") as output:
                                output.attrs["inference_status"] = "failed"
                                output.attrs["inference_error"] = str(exc)
                                output.flush()
                        except Exception:
                            pass
                    try:
                        with h5py.File(raw_force_path, "r+") as raw:
                            raw.attrs["state"] = "POSTPROCESSING_FAILED"
                            raw.attrs["postprocessing_error"] = str(exc)
                            raw.flush()
                    except Exception:
                        pass
                    with self._lock:
                        self._manifest["force_timeline"] = {"status": "failed", "error": str(exc)}
                        self._manifest["force_inference"] = {"status": "failed", "error": str(exc)}
                        self._event("FORCE_TIMELINE_FAILED", {"error": str(exc), "raw_preserved": True})
                        self._write_manifest()
        with self._lock:
            timeline_names = ("emg_timeline", "imu_timeline", "nokov_timeline",
                              "realsense_timeline", "force_timeline")
            has_processing_failure = any(
                self._manifest.get(name, {}).get("status") == "failed" for name in timeline_names
            )
            final_state = "COMPLETED_WITH_ERRORS" if has_processing_failure else "COMPLETED"
            self._manifest["state"] = final_state
            self._manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            self.state = final_state
            self._emg = None
            self._realsense = None
            self._nokov = None
            self._touchglove = None
            self._event("STOP", {"record_counts": counts})
            self._write_manifest()

    def _event(self, event: str, details: dict[str, Any]) -> None:
        path = self.path / "events.jsonl"
        row = {"event": event, "event_mono_ns": self.clock.now_ns(), "utc": datetime.now(timezone.utc).isoformat(), **details}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _write_manifest(self) -> None:
        target = self.path / "manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
