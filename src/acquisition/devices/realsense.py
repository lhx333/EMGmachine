from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from acquisition.clock import WindowsQpcClock


def _capture_process(connection, raw_dir: str, serial: str, width: int, height: int,
                     fps: int, codec: str) -> None:
    """Own librealsense and the RGB encoder outside the GUI process."""
    pipeline = None
    writer = None
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipeline.start(config)
        device = profile.get_device()
        actual_sensor = profile.get_stream(rs.stream.color).as_video_stream_profile()
        video_profile = actual_sensor.get_intrinsics()
        actual_fps = actual_sensor.fps()
        actual_width, actual_height = actual_sensor.width(), actual_sensor.height()
        selected_serial = device.get_info(rs.camera_info.serial_number)
        color_sensor = next((s for s in device.query_sensors()
                             if s.get_info(rs.camera_info.name).lower().find("rgb") >= 0
                             or s.get_info(rs.camera_info.name).lower().find("color") >= 0), None)
        firmware = device.get_info(rs.camera_info.firmware_version)
        name = device.get_info(rs.camera_info.name)
        first = pipeline.wait_for_frames(5000).get_color_frame()
        if not first:
            raise RuntimeError("RealSense did not produce a color frame during PREPARE")

        raw = Path(raw_dir)
        staging = raw / "realsense_rgb.part.mp4"
        output = raw / "realsense_rgb.mp4"
        encoder = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(staging), encoder, actual_fps, (actual_width, actual_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open RealSense video writer for {staging}")
        clock = WindowsQpcClock()
        connection.send({"type": "READY", "info": {
            "device_name": name, "serial_number": selected_serial,
            "firmware_version": firmware, "stream": "RGB",
            "width": actual_width, "height": actual_height, "fps": actual_fps,
            "format": "bgr8", "video_codec": codec,
            "camera_timestamp_semantics": "SDK color-frame timestamp in milliseconds; preserve raw value and domain",
            "camera_timestamp_domain": "reported per frame",
            "intrinsics": {"fx": video_profile.fx, "fy": video_profile.fy,
                           "ppx": video_profile.ppx, "ppy": video_profile.ppy,
                           "model": str(video_profile.model),
                           "coeffs": list(video_profile.coeffs)},
            "color_sensor_name": color_sensor.get_info(rs.camera_info.name) if color_sensor else None,
            "qpc_clock_id": clock.clock_id,
        }})

        armed_at = None
        frame_index = 0
        recording_started_ns = None
        frame_drops = 0
        previous_frame_number = None
        next_preview_ns = 0
        while True:
            while connection.poll():
                command = connection.recv()
                if command["type"] == "ARM":
                    armed_at = int(command["deadline_ns"])
                elif command["type"] == "STOP":
                    raise StopIteration
            frames = pipeline.wait_for_frames(1000)
            color = frames.get_color_frame()
            if not color:
                continue
            observed_ns = clock.now_ns()
            if armed_at is None or observed_ns < armed_at:
                if observed_ns >= next_preview_ns:
                    preview = cv2.resize(np.asanyarray(color.get_data()), (320, 240))
                    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    connection.send({"type": "PREVIEW", "record": {
                        "frame_index": frame_index,
                        "camera_frame_number": int(color.get_frame_number()),
                        "host_receive_mono_ns": observed_ns,
                        "preview_jpeg": encoded.tobytes() if ok else b"",
                    }})
                    next_preview_ns = observed_ns + 200_000_000
                continue
            if recording_started_ns is None:
                recording_started_ns = observed_ns
            number = int(color.get_frame_number())
            if previous_frame_number is not None and number > previous_frame_number + 1:
                frame_drops += number - previous_frame_number - 1
            previous_frame_number = number
            writer.write(np.asanyarray(color.get_data()))
            preview_jpeg = b""
            if observed_ns >= next_preview_ns:
                preview = cv2.resize(np.asanyarray(color.get_data()), (320, 240))
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    preview_jpeg = encoded.tobytes()
                next_preview_ns = observed_ns + 200_000_000
            connection.send({"type": "FRAME", "record": {
                "frame_index": frame_index,
                "camera_frame_number": number,
                "camera_timestamp_ms": float(color.get_timestamp()),
                "camera_timestamp_domain": str(color.get_frame_timestamp_domain()),
                "host_receive_mono_ns": observed_ns,
                "timestamp_stage": "qpc_after_wait_for_frames",
                "clock_id": clock.clock_id,
                "preview_jpeg": preview_jpeg,
            }})
            frame_index += 1
    except StopIteration:
        pass
    except Exception as exc:
        try:
            connection.send({"type": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if writer is not None:
            writer.release()
        try:
            if "frame_index" in locals() and frame_index:
                os.replace(staging, output)
            connection.send({"type": "STOPPED", "frame_count": locals().get("frame_index", 0),
                             "frame_number_gap_count": locals().get("frame_drops", 0),
                             "recording_started_mono_ns": locals().get("recording_started_ns"),
                             "video_path": "raw/realsense_rgb.mp4" if locals().get("frame_index", 0) else None})
        except Exception as exc:
            try:
                connection.send({"type": "ERROR", "error": f"finalize video: {exc}"})
            except Exception:
                pass
        connection.close()


class RealSenseAdapter:
    def __init__(self, on_frame: Callable[[dict[str, Any]], None],
                 on_event: Callable[[str, dict[str, Any]], None] | None = None,
                 serial: str = "", width: int = 640, height: int = 480,
                 fps: int = 30, codec: str = "mp4v") -> None:
        self.on_frame = on_frame
        self.on_event = on_event or (lambda _name, _details: None)
        self.serial = serial
        self.width, self.height, self.fps, self.codec = width, height, fps, codec
        self.process = None
        self.connection = None
        self.reader: threading.Thread | None = None
        self.info: dict[str, Any] = {}
        self.error: str | None = None
        self.stopped: dict[str, Any] | None = None

    def prepare(self, raw_dir: Path) -> dict[str, Any]:
        parent, child = mp.get_context("spawn").Pipe()
        self.connection = parent
        self.process = mp.get_context("spawn").Process(
            target=_capture_process,
            args=(child, str(raw_dir.resolve()), self.serial, self.width, self.height, self.fps, self.codec),
            name="realsense-capture", daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(20):
            self.stop()
            raise RuntimeError("Timed out waiting for RealSense PREPARE")
        message = parent.recv()
        if message["type"] != "READY":
            self.stop()
            raise RuntimeError(message.get("error", "RealSense PREPARE failed"))
        self.info = message["info"]
        self.reader = threading.Thread(target=self._read_messages, name="realsense-events", daemon=True)
        self.reader.start()
        return self.info

    def _read_messages(self) -> None:
        try:
            while self.connection is not None:
                message = self.connection.recv()
                kind = message["type"]
                if kind == "FRAME":
                    try:
                        self.on_frame(message["record"])
                    except Exception as exc:
                        self.error = f"Failed to record RealSense frame metadata: {exc}"
                        self.on_event("REALSENSE_ERROR", {"error": self.error})
                        self.connection.send({"type": "STOP"})
                elif kind == "PREVIEW":
                    self.on_frame(message["record"])
                elif kind == "ERROR":
                    self.error = message["error"]
                    self.on_event("REALSENSE_ERROR", {"error": self.error})
                elif kind == "STOPPED":
                    self.stopped = message
                    self.on_event("REALSENSE_STOPPED", message)
                    return
        except (EOFError, OSError) as exc:
            if self.stopped is None:
                self.error = f"RealSense worker pipe closed unexpectedly: {exc}"
                self.on_event("REALSENSE_ERROR", {"error": self.error})
        except Exception as exc:
            self.error = f"RealSense frame reader failed: {type(exc).__name__}: {exc}"
            self.on_event("REALSENSE_ERROR", {"error": self.error})
            if self.connection is not None:
                try:
                    self.connection.send({"type": "STOP"})
                except (BrokenPipeError, OSError):
                    pass

    def arm(self, deadline_ns: int) -> None:
        if not self.process or not self.process.is_alive():
            raise RuntimeError("RealSense worker is not ready")
        self.connection.send({"type": "ARM", "deadline_ns": int(deadline_ns)})

    def stop(self) -> dict[str, Any] | None:
        if self.process is None:
            return self.stopped
        if self.process.is_alive() and self.connection is not None:
            try:
                self.connection.send({"type": "STOP"})
            except (BrokenPipeError, OSError):
                pass
            self.process.join(10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
                self.error = self.error or "RealSense worker did not stop within 10 seconds"
        if self.reader and self.reader is not threading.current_thread():
            self.reader.join(2)
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.process = None
        if self.error:
            raise RuntimeError(self.error)
        if self.stopped is None:
            raise RuntimeError("RealSense worker exited without final status")
        return self.stopped
