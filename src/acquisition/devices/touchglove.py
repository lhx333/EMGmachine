from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from acquisition.clock import WindowsQpcClock


def detect_touchglove_ports() -> list[tuple[str, str]]:
    """List serial ports other than the ResearchKit receiver."""
    from serial.tools import list_ports

    found = []
    for port in list_ports.comports():
        hwid = (port.hwid or "").upper()
        if "VID:PID=1915:C00A" in hwid:
            continue
        found.append((port.device, port.description or port.hwid or port.device))
    return found


def _append_frames(group: Any, images: list[Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import numpy as np

    start = group["image"].shape[0]
    end = start + len(rows)
    for name in ("image", "channel", "seq_id", "device_timestamp_us", "host_receive_mono_ns",
                 "device_rtc_raw", "device_rtc_timestamp_us", "device_rtc_valid", "sensor_id"):
        group[name].resize((end,) + group[name].shape[1:])
    group["image"][start:end] = np.stack(images).astype(np.uint8, copy=False)
    for name in ("channel", "seq_id", "device_timestamp_us", "host_receive_mono_ns",
                 "device_rtc_timestamp_us", "device_rtc_valid", "sensor_id"):
        group[name][start:end] = [row[name] for row in rows]
    group["device_rtc_raw"][start:end] = np.asarray(
        [row["device_rtc_raw"] for row in rows], dtype=np.uint8)


def _capture_process(connection: Any, raw_path: str, port: str, baseline_seconds: float,
                     flush_frames: int) -> None:
    reader = None
    frame_count = 0
    try:
        import h5py
        import numpy as np

        sdk_path = Path(__file__).resolve().parents[3] / "vendor" / "touchglove" / "python"
        sys.path.insert(0, str(sdk_path))
        backend = os.environ.get("TOUCHGLOVE_BACKEND", "python").strip().lower()
        if backend not in {"dll", "python"}:
            raise ValueError("TOUCHGLOVE_BACKEND must be 'dll' or 'python'")
        from touch_glove_ptp import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
        if backend == "python":
            from touch_glove_ptp._fallback import TouchGlove
            reader = TouchGlove(auto_open=False, use_c_lib=False)
        else:
            from touch_glove_ptp import TouchGlove
            reader = TouchGlove(auto_open=False)
        if backend == "dll" and not reader.use_c_lib:
            raise RuntimeError(
                "TouchGlove touch_glove.dll backend is unavailable; the RTC-aligned capture uses "
                "the Python raw serial parser by default, or install/repair the vendor DLL"
            )
        if not reader.open(port):
            raise RuntimeError(f"Could not open TouchGlove on {port}")
        clock = WindowsQpcClock()
        connection.send({"type": "READY", "info": {
            "device_name": "TouchGlove",
            "backend": "vendored touch_glove.dll" if backend == "dll" else "Python serial frame parser",
            "port": port,
            "channels": IMAGE_CHANNELS,
            "frame_width": IMAGE_WIDTH,
            "frame_height": IMAGE_HEIGHT,
            "clock_id": clock.clock_id,
            "clock_frequency_hz": clock.frequency_hz,
        }})

        raw_file = Path(raw_path)
        pending_images: list[Any] = []
        pending_rows: list[dict[str, Any]] = []
        baseline_sum = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)
        baseline_count = np.zeros(IMAGE_CHANNELS, dtype=np.uint64)
        reader.start()
        deadline_ns = None
        capture = None
        group = None
        recording_started_ns = None
        next_status_ns = 0
        next_preview_ns = 0
        try:
            while True:
                if connection.poll(0):
                    command = connection.recv()
                    if command.get("type") == "STOP":
                        break
                    if command.get("type") == "ARM":
                        deadline_ns = int(command["deadline_mono_ns"])
                        capture = h5py.File(raw_file, "w")
                        capture.attrs.update(
                            schema_version="0.4", stream="force", device="TouchGlove",
                            backend="touch_glove.dll" if backend == "dll" else "Python serial frame parser",
                            timestamp_field="device_rtc_timestamp_us (when device_rtc_valid=1)",
                            timestamp_semantics=("device_timestamp_us retains the backend Frame timestamp (Windows DLL: "
                                                 "host wall clock); device_rtc_timestamp_us is decoded from device BCD "
                                                 "bytes [7:15] when the Python serial backend is selected"),
                            device_timestamp_us_semantics="backend Frame.timestamp_us; Windows DLL generates host wall-clock epoch microseconds",
                            device_rtc_timestamp_us_semantics="UTC epoch microseconds decoded from raw BCD bytes [7:15]; valid only when device_rtc_valid=1",
                            host_receive_mono_ns_semantics=("Windows QPC sampled immediately after poll() "
                                                            "returns; observation time"),
                            raw_layout=("one 192x192 uint8 image per channel frame with SDK channel, seq_id, "
                                        "device time and QPC observation"),
                            inference_timing="all raw frames are persisted before post-capture DirectML inference",
                            planned_start_mono_ns=deadline_ns, baseline_duration_s=float(baseline_seconds),
                            state="WAITING_FOR_PLANNED_START",
                        )
                        group = capture.create_group("frames")
                        group.create_dataset("image", shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH),
                                             maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
                                             chunks=(min(flush_frames, 32), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                        for name, dtype in (("channel", np.uint8), ("seq_id", np.uint32),
                                            ("device_timestamp_us", np.uint64), ("host_receive_mono_ns", np.uint64)):
                            group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(flush_frames,), dtype=dtype)
                        group.create_dataset("device_rtc_raw", shape=(0, 8), maxshape=(None, 8),
                                             chunks=(flush_frames, 8), dtype=np.uint8)
                        for name, dtype in (("device_rtc_timestamp_us", np.uint64), ("device_rtc_valid", np.uint8)):
                            group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(flush_frames,), dtype=dtype)
                        group.create_dataset("sensor_id", shape=(0,), maxshape=(None,), chunks=(flush_frames,),
                                             dtype=h5py.string_dtype("utf-8"))
                # FrameBatch implements iteration but intentionally has no
                # indexing API in the native SDK. Convert it once so both the
                # preview and capture paths consume the same batch safely.
                frames = list(reader.poll())
                observed_ns = clock.now_ns()
                if frames and observed_ns >= next_preview_ns:
                    latest = frames[-1]
                    connection.send({"type": "PREVIEW", "frame_count": len(frames),
                                     "channel": int(latest.channel), "seq_id": int(latest.seq_id),
                                     "device_timestamp_us": int(latest.timestamp_us),
                                     "device_rtc_timestamp_us": getattr(latest, "device_rtc_timestamp_us", None),
                                     "host_receive_mono_ns": observed_ns,
                                     "image_96x96": np.asarray(latest.image, dtype=np.uint8)[::2, ::2].copy().tobytes()})
                    next_preview_ns = observed_ns + 250_000_000
                if deadline_ns is None or observed_ns < deadline_ns:
                    if not frames:
                        time.sleep(0.001)
                    continue
                if recording_started_ns is None:
                    recording_started_ns = observed_ns
                    capture.attrs["state"] = "CAPTURING_RAW_FRAMES"
                    connection.send({"type": "RECORDING", "actual_start_mono_ns": recording_started_ns})
                    next_status_ns = recording_started_ns + 5_000_000_000
                for frame in frames:
                    image = np.asarray(frame.image, dtype=np.uint8).copy()
                    channel = int(frame.channel)
                    pending_images.append(image)
                    pending_rows.append({
                        "channel": channel,
                        "seq_id": int(frame.seq_id) & 0xFFFFFFFF,
                        "device_timestamp_us": int(frame.timestamp_us),
                        "device_rtc_raw": list(getattr(frame, "device_rtc_raw", None) or bytes(8)),
                        "device_rtc_timestamp_us": int(getattr(frame, "device_rtc_timestamp_us", None) or 0),
                        "device_rtc_valid": int(getattr(frame, "device_rtc_timestamp_us", None) is not None),
                        "host_receive_mono_ns": observed_ns,
                        "sensor_id": str(frame.sensor_id),
                    })
                    if deadline_ns <= observed_ns <= deadline_ns + int(baseline_seconds * 1_000_000_000):
                        baseline_sum[channel] += image
                        baseline_count[channel] += 1
                frame_count += len(frames)
                if group is not None and len(pending_rows) >= flush_frames:
                    _append_frames(group, pending_images, pending_rows)
                    pending_images.clear()
                    pending_rows.clear()
                    capture.flush()
                if observed_ns >= next_status_ns:
                    connection.send({"type": "STATUS", "frame_count": frame_count + len(pending_rows)})
                    next_status_ns += 5_000_000_000
                if not frames:
                    time.sleep(0.001)
            if group is not None and pending_rows:
                _append_frames(group, pending_images, pending_rows)
                pending_images.clear()
                pending_rows.clear()
            if group is not None and group["image"].shape[0] == 0:
                raise RuntimeError("TouchGlove stopped without any frames")
            if capture is not None:
                baseline = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                for channel in range(IMAGE_CHANNELS):
                    if baseline_count[channel]:
                        baseline[channel] = np.rint(baseline_sum[channel] / baseline_count[channel]).clip(0, 255).astype(np.uint8)
                baseline_group = capture.create_group("baseline")
                baseline_group.create_dataset("image", data=baseline, compression="lzf")
                baseline_group.create_dataset("sample_count_by_channel", data=baseline_count)
                baseline_group.attrs.update(duration_s=float(baseline_seconds), source="initial session QPC interval")
                capture.attrs["state"] = "RAW_CAPTURE_COMPLETE"
                capture.attrs["frame_count"] = frame_count
                capture.flush()
        finally:
            reader.stop()
            if capture is not None:
                capture.close()
        connection.send({"type": "STOPPED", "frame_count": frame_count,
                         "baseline_samples_by_channel": baseline_count.tolist(),
                         "recording_started_mono_ns": recording_started_ns,
                         "raw_path": "raw/force.h5"})
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send({"type": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            connection.send({"type": "STOPPED", "frame_count": frame_count, "raw_path": "raw/force.h5"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
            try:
                reader.close()
            except Exception:
                pass
        connection.close()


class TouchGloveAdapter:
    """Raw TouchGlove acquisition worker; all inference happens after capture."""

    def __init__(self, on_event: Callable[[str, dict[str, Any]], None] | None = None,
                 baseline_seconds: float = 1.0, flush_frames: int = 32) -> None:
        if baseline_seconds <= 0 or flush_frames < 1:
            raise ValueError("baseline_seconds and flush_frames must be positive")
        self.on_event = on_event or (lambda _name, _details: None)
        self.baseline_seconds = float(baseline_seconds)
        self.flush_frames = int(flush_frames)
        self._ctx = mp.get_context("spawn")
        self.process = None
        self.connection = None
        self.reader: threading.Thread | None = None
        self.ready = threading.Event()
        self.preview_ready = threading.Event()
        self.stopped = threading.Event()
        self.info: dict[str, Any] = {}
        self.stopped_info: dict[str, Any] = {}
        self.error: str | None = None
        self.frame_count = 0

    def prepare(self, raw_dir: Path, port: str, timeout_s: float = 15.0) -> dict[str, Any]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        parent, child = self._ctx.Pipe(duplex=True)
        raw_path = raw_dir / "force.h5"
        self.connection = parent
        self.process = self._ctx.Process(
            target=_capture_process,
            args=(child, str(raw_path), port, self.baseline_seconds, self.flush_frames),
            name="touchglove-capture", daemon=True,
        )
        self.process.start()
        child.close()
        self.reader = threading.Thread(target=self._read_loop, name="touchglove-events", daemon=True)
        self.reader.start()
        if not self.ready.wait(timeout_s):
            self.stop()
            raise TimeoutError("Timed out preparing TouchGlove Windows SDK")
        if self.error:
            error = self.error
            self.stop()
            raise RuntimeError(error)
        if not self.preview_ready.wait(timeout_s):
            self.stop()
            if self.error:
                raise RuntimeError(self.error)
            raise TimeoutError("TouchGlove opened, but no live image frames arrived")
        if self.error:
            error = self.error
            self.stop()
            raise RuntimeError(error)
        return dict(self.info)

    def _read_loop(self) -> None:
        try:
            while self.connection is not None:
                message = self.connection.recv()
                kind = message.get("type")
                if kind == "READY":
                    self.info = message["info"]
                    self.ready.set()
                    self.on_event("TOUCHGLOVE_READY", self.info)
                elif kind == "RECORDING":
                    self.on_event("TOUCHGLOVE_RECORDING", message)
                elif kind == "STATUS":
                    self.frame_count = int(message.get("frame_count", self.frame_count))
                    self.on_event("TOUCHGLOVE_STATUS", message)
                elif kind == "PREVIEW":
                    self.frame_count += int(message.get("frame_count", 0))
                    self.preview_ready.set()
                    self.on_event("TOUCHGLOVE_PREVIEW", message)
                elif kind == "ERROR":
                    self.error = str(message.get("error", "TouchGlove worker failed"))
                    self.on_event("TOUCHGLOVE_ERROR", {"error": self.error})
                    self.ready.set()
                    self.preview_ready.set()
                elif kind == "STOPPED":
                    self.stopped_info = message
                    self.frame_count = int(message.get("frame_count", self.frame_count))
                    self.stopped.set()
                    self.on_event("TOUCHGLOVE_STOPPED", message)
                    return
        except (EOFError, OSError) as exc:
            if self.process is not None and self.process.exitcode not in (0, None):
                self.error = f"TouchGlove worker exited with code {self.process.exitcode}: {exc}"
            self.ready.set()
            self.stopped.set()

    def arm(self, deadline_mono_ns: int) -> None:
        if not self.ready.is_set() or self.error or self.connection is None:
            raise RuntimeError(self.error or "TouchGlove adapter is not ready")
        self.connection.send({"type": "ARM", "deadline_mono_ns": int(deadline_mono_ns)})

    def stop(self) -> dict[str, Any] | None:
        proc = self.process
        if proc is None:
            return self.stopped_info or None
        try:
            if proc.is_alive() and self.connection is not None:
                self.connection.send({"type": "STOP"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        proc.join(15)
        if proc.is_alive():
            proc.terminate()
            proc.join(3)
            self.error = self.error or "TouchGlove worker did not stop within 15 seconds"
        if self.reader and self.reader is not threading.current_thread():
            self.reader.join(2)
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.process = None
        if self.error:
            raise RuntimeError(self.error)
        return self.stopped_info or None
