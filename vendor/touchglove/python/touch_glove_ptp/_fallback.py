"""
Fallback implementation of TouchGlove for Python when native extension module is unavailable or restricted.
Uses ctypes with touch_glove C library if available, and falls back to pure Python pyserial driver.
"""
from __future__ import annotations

import ctypes
import datetime as _datetime
import os
import queue
import sys
import threading
import time
from typing import Iterator, List, Optional
import numpy as np

IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192
IMAGE_CHANNELS = 5
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT

class CtsFrame(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint8),
        ("type", ctypes.c_uint8),
        ("cmd", ctypes.c_uint8),
        ("param_len", ctypes.c_uint16),
        ("seq_id", ctypes.c_uint32),
        ("timestamp_us", ctypes.c_uint64),
        ("sensor_id", ctypes.c_uint8 * 8),
        ("param", ctypes.c_uint8 * IMAGE_BYTES),
    ]

# Load C library fallback if available
lib = None
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dll_name = "touch_glove.dll" if sys.platform.startswith("win") else "libtouch_glove.so"
_search_paths = [
    os.path.join(_script_dir, _dll_name),
    os.path.join(_script_dir, "build", _dll_name),
    os.path.join(_script_dir, "..", "sdk", "api", "build", _dll_name),
    os.path.join(_script_dir, "..", "..", "sdk", "api", "build", _dll_name),
]

for _path in _search_paths:
    if os.path.exists(_path):
        try:
            lib = ctypes.CDLL(_path)
            lib.touch_glove_create.argtypes = []
            lib.touch_glove_create.restype = ctypes.c_void_p
            lib.touch_glove_destroy.argtypes = [ctypes.c_void_p]
            lib.touch_glove_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.touch_glove_open.restype = ctypes.c_int
            lib.touch_glove_close.argtypes = [ctypes.c_void_p]
            lib.touch_glove_send_start.argtypes = [ctypes.c_void_p]
            lib.touch_glove_send_stop.argtypes = [ctypes.c_void_p]
            lib.touch_glove_send_get_unique_id.argtypes = [ctypes.c_void_p]
            lib.touch_glove_send_get_unique_id.restype = ctypes.c_int
            lib.touch_glove_is_open.argtypes = [ctypes.c_void_p]
            lib.touch_glove_is_open.restype = ctypes.c_int
            lib.touch_glove_poll_read.argtypes = [ctypes.c_void_p]
            lib.touch_glove_poll_read.restype = ctypes.c_size_t
            lib.touch_glove_get_bytes_per_second.argtypes = [ctypes.c_void_p]
            lib.touch_glove_get_bytes_per_second.restype = ctypes.c_double
            lib.touch_glove_get_total_bytes.argtypes = [ctypes.c_void_p]
            lib.touch_glove_get_total_bytes.restype = ctypes.c_uint64
            lib.touch_glove_pop_image_frames.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_size_t]
            lib.touch_glove_pop_image_frames.restype = ctypes.c_size_t
            lib.touch_glove_peek_image_frames.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.POINTER(CtsFrame))]
            lib.touch_glove_peek_image_frames.restype = ctypes.c_size_t
            break
        except Exception as _exc:
            print(f"[Warning] Failed loading C library {_path}: {_exc}")
            lib = None

class Frame:
    def __init__(
        self,
        channel: int,
        sensor_id: str,
        seq_id: int,
        timestamp_us: int,
        image: np.ndarray,
        disp: Optional[np.ndarray] = None,
        force_field: Optional[np.ndarray] = None,
        device_rtc_raw: Optional[bytes] = None,
        device_rtc_timestamp_us: Optional[int] = None,
    ):
        self.channel = channel
        self.sensor_id = sensor_id
        self.seq_id = seq_id
        self.timestamp_us = timestamp_us
        self.image = image
        self.disp = disp
        self.force_field = force_field
        self.device_rtc_raw = device_rtc_raw
        self.device_rtc_timestamp_us = device_rtc_timestamp_us

    def __repr__(self) -> str:
        return (
            f"Frame(channel={self.channel}, seq_id={self.seq_id}, "
            f"timestamp_us={self.timestamp_us}, sensor_id='{self.sensor_id}')"
        )

class FrameBatch:
    def __init__(self, frames: List[Frame]):
        self.frames = frames

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f"FrameBatch(total_frames={len(self.frames)})"

def list_ports() -> List[str]:
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        ports = []
        if sys.platform.startswith("win"):
            for i in range(1, 257):
                ports.append(f"COM{i}")
        return ports

class TouchGlove:
    def __init__(
        self,
        port: Optional[str] = None,
        model: Optional[str] = None,
        device: str = "auto",
        auto_open: bool = True,
        decouple_inference: bool = False,
        inference_queue_size: int = 32,
        use_c_lib: Optional[bool] = None,
    ):
        self.use_c_lib = (lib is not None) if use_c_lib is None else (bool(use_c_lib) and lib is not None)
        self.handle = None
        self._is_open = False
        self._is_streaming = False
        self.port = None
        self.ser = None
        self.buffer = bytearray()
        self.queues = [[] for _ in range(IMAGE_CHANNELS)]
        self.seq_ids = [0] * IMAGE_CHANNELS
        self.total_bytes = 0
        self.start_t = time.time()
        self._sensor_ids = [""] * IMAGE_CHANNELS
        self._ort_session = None
        self._baseline_images = [None] * IMAGE_CHANNELS
        self._dense_lock = threading.Lock()
        self._dense_cache = {
            "disp": np.zeros((IMAGE_CHANNELS, 32, 32, 3), dtype=np.float32),
            "force": np.zeros((IMAGE_CHANNELS, 32, 32, 3), dtype=np.float32),
            "update_count": 0,
        }
        self.decouple_inference = bool(decouple_inference)
        if inference_queue_size < 1:
            raise ValueError("inference_queue_size must be at least one")
        self._inference_queue = queue.Queue(maxsize=int(inference_queue_size))
        self._inference_thread = None
        self._inference_stop = threading.Event()
        self._inference_enabled = threading.Event()
        self._inference_enabled.set()
        self._inference_idle = threading.Event()
        self._inference_idle.set()
        self._inference_stats_lock = threading.Lock()
        self._inference_dropped_count = 0
        self._inference_completed_count = 0

        if self.use_c_lib:
            self.handle = lib.touch_glove_create()
            if not self.handle:
                self.use_c_lib = False

        if model:
            try:
                import onnxruntime as ort
                avail = ort.get_available_providers()
                providers = []
                dev_lower = str(device).lower()
                if dev_lower in ("cuda", "gpu", "directml", "auto"):
                    if "CUDAExecutionProvider" in avail:
                        providers.append("CUDAExecutionProvider")
                    if "DmlExecutionProvider" in avail:
                        providers.append("DmlExecutionProvider")
                providers.append("CPUExecutionProvider")
                self._ort_session = ort.InferenceSession(model, providers=providers)
                active = self._ort_session.get_providers()
                print(f"[TouchGlove] Loaded ONNX model with providers: {active}")
            except Exception as e:
                print(f"[Warning] Failed to initialize ONNX model: {e}")

        if port and auto_open:
            self.open(port)

    def __enter__(self) -> "TouchGlove":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False

    def open(self, port: str) -> bool:
        if self._is_open:
            self.close()
        self.port = port
        if self.use_c_lib:
            res = lib.touch_glove_open(self.handle, port.encode("utf-8"))
            if res == 0:
                self._is_open = True
                for _ in range(5):
                    lib.touch_glove_send_stop(self.handle)
                    time.sleep(0.005)
                lib.touch_glove_send_get_unique_id(self.handle)
                for _ in range(20):
                    lib.touch_glove_poll_read(self.handle)
                    time.sleep(0.01)
                return True
            return False
        else:
            import serial
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.01, write_timeout=0.1, rtscts=False, dsrdtr=False)
                try:
                    self.ser.setDTR(True)
                    self.ser.setRTS(True)
                except Exception:
                    pass
                self._is_open = True
                self._send_cmd_pure(0x51, 0x02)
                time.sleep(0.01)
                self._send_cmd_pure(0x51, 0x04)
                time.sleep(0.05)
                return True
            except Exception as e:
                print(f"Error opening port {port}: {e}")
                return False

    def close(self) -> None:
        if self.use_c_lib and hasattr(self, "handle") and self.handle:
            if self._is_streaming:
                self.stop()
            lib.touch_glove_close(self.handle)
            self._is_open = False
        else:
            if self.ser and self.ser.is_open:
                if self._is_streaming:
                    self.stop()
                self.ser.close()
            self._is_open = False
        self._stop_inference_worker()

    def __del__(self):
        if self.use_c_lib and hasattr(self, "handle") and self.handle:
            try:
                self.close()
                lib.touch_glove_destroy(self.handle)
            except Exception:
                pass
            self.handle = None

    def _send_cmd_pure(self, type_byte: int, cmd_byte: int, param: bytes = b"") -> bool:
        if not self.ser or not self.ser.is_open:
            return False
        length = 2 + len(param)
        len_l = length & 0xFF
        len_h = (length >> 8) & 0xFF
        pkt = bytearray([0x55, 0xAA, 0x01, len_l, len_h, type_byte, cmd_byte])
        pkt.extend(param)
        chk = 0
        chk ^= len_l
        chk ^= len_h
        chk ^= type_byte
        chk ^= cmd_byte
        for b in param:
            chk ^= b
        pkt.append(chk & 0xFF)
        pkt.extend([0xCC, 0x33])
        self.ser.write(pkt)
        return True

    def start(self) -> None:
        if self._is_open:
            self._start_inference_worker()
            if self.use_c_lib:
                lib.touch_glove_send_start(self.handle)
            else:
                self._send_cmd_pure(0x51, 0x01)
            self._is_streaming = True

    def stop(self) -> None:
        if self._is_open:
            if self.use_c_lib:
                lib.touch_glove_send_stop(self.handle)
            else:
                self._send_cmd_pure(0x51, 0x02)
            self._is_streaming = False

    def calibrate_baseline(self, duration_sec: float = 1.0) -> None:
        if self.decouple_inference:
            self._inference_enabled.clear()
            self._inference_idle.wait()
            self._clear_inference_queue()
        try:
            start_t = time.time()
            sums = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64) for _ in range(IMAGE_CHANNELS)]
            counts = [0] * IMAGE_CHANNELS

            while time.time() - start_t < duration_sec:
                batch = self._collect_raw_frames()
                for frame in batch:
                    sums[frame.channel] += frame.image.astype(np.float64)
                    counts[frame.channel] += 1
                time.sleep(0.005)

            for ch in range(IMAGE_CHANNELS):
                if counts[ch] > 0:
                    self._baseline_images[ch] = (sums[ch] / counts[ch]).astype(np.uint8)
                else:
                    self._baseline_images[ch] = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        finally:
            if self.decouple_inference:
                self._inference_enabled.set()

    def set_baseline_images(self, images: np.ndarray) -> None:
        """Set per-channel uint8 baseline images for later frame inference."""
        baseline = np.asarray(images)
        expected = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if baseline.shape != expected:
            raise ValueError(f"Expected baseline shape {expected}, got {baseline.shape}")
        baseline = np.clip(baseline, 0, 255).astype(np.uint8, copy=True)
        if self.decouple_inference:
            self._inference_enabled.clear()
            self._inference_idle.wait()
            self._clear_inference_queue()
        self._baseline_images = [baseline[ch] for ch in range(IMAGE_CHANNELS)]
        if self.decouple_inference:
            self._inference_enabled.set()

    def infer_frame(self, channel: int, image: np.ndarray):
        """Run Dense inference for one saved raw frame using the configured model."""
        if not 0 <= int(channel) < IMAGE_CHANNELS:
            raise ValueError(f"Channel must be in [0, {IMAGE_CHANNELS - 1}]")
        image = np.asarray(image)
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"Expected image shape {(IMAGE_HEIGHT, IMAGE_WIDTH)}, got {image.shape}")
        return self._run_inference(int(channel), image.astype(np.uint8, copy=False))

    def _run_inference(self, ch: int, img: np.ndarray):
        if not self._ort_session:
            return None, None
        base_img = self._baseline_images[ch] if self._baseline_images[ch] is not None else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        diff = (img.astype(np.float32) - base_img.astype(np.float32)) / 255.0
        inp = np.stack([diff, diff, diff], axis=0)[np.newaxis, :, :, :]
        try:
            inputs = {self._ort_session.get_inputs()[0].name: inp}
            outs = self._ort_session.run(None, inputs)
            if len(outs) >= 2:
                disp_out = outs[0][0].transpose(1, 2, 0)
                force_out = outs[1][0].transpose(1, 2, 0)
                return disp_out, force_out
        except Exception:
            pass
        return None, None

    def _poll_pure(self) -> List[Frame]:
        if not self.ser or not self.ser.is_open:
            return []
        try:
            data = self.ser.read(65536)
            if data:
                self.total_bytes += len(data)
                self.buffer.extend(data)
        except Exception:
            pass

        buf = self.buffer
        while len(buf) >= 9:
            idx = buf.find(b"\x55\xAA")
            if idx < 0:
                buf.clear()
                break
            if idx > 0:
                del buf[:idx]
            if len(buf) < 7:
                break

            length = buf[3] | (buf[4] << 8)
            command = buf[6]
            # Image frames include the 8-byte RTC between the 7-byte header
            # and image payload. Their length field is payload size + 2.
            needed = length + (16 if 0x11 <= command <= 0x15 else 8)
            if len(buf) < needed:
                break

            pkt = buf[:needed]
            del buf[:needed]

            if pkt[-2] != 0xCC or pkt[-1] != 0x33:
                continue

            cmd = pkt[6]
            if not getattr(self, "_printed_debug", False):
                print(f"[DEBUG] pkt_len={len(pkt)}, length={length}, type=0x{pkt[5]:02X}, cmd=0x{cmd:02X}, raw={list(pkt[:15])}")
                self._printed_debug = True
            if 0x11 <= cmd <= 0x15:
                channel = cmd - 0x11
                img_bytes = bytes(pkt[15:15+IMAGE_BYTES])
                if len(img_bytes) == IMAGE_BYTES:
                    rtc_raw = bytes(pkt[7:15])
                    try:
                        rtc_timestamp_us = decode_device_rtc_bcd(rtc_raw)
                    except ValueError:
                        rtc_timestamp_us = None
                    self.seq_ids[channel] += 1
                    seq_id = self.seq_ids[channel]
                    ts_us = int(time.time() * 1e6)
                    img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).copy()
                    sid = f"CH{channel+1:02d}"

                    self.queues[channel].append({
                        "channel": channel,
                        "sensor_id": sid,
                        "seq_id": seq_id,
                        "timestamp_us": ts_us,
                        "device_rtc_raw": rtc_raw,
                        "device_rtc_timestamp_us": rtc_timestamp_us,
                        "image": img_np,
                    })

        frames = []
        for ch in range(IMAGE_CHANNELS):
            while self.queues[ch]:
                item = self.queues[ch].pop(0)
                img = item["image"]
                frames.append(
                    Frame(
                        channel=ch,
                        sensor_id=item["sensor_id"],
                        seq_id=item["seq_id"],
                        timestamp_us=item["timestamp_us"],
                        image=img,
                        device_rtc_raw=item["device_rtc_raw"],
                        device_rtc_timestamp_us=item["device_rtc_timestamp_us"],
                    )
                )
        return frames

    def _collect_raw_frames(self) -> List[Frame]:
        if not self._is_open:
            return []

        if not self.use_c_lib:
            return self._poll_pure()

        lib.touch_glove_poll_read(self.handle)
        collected_frames = []

        ptr = ctypes.POINTER(CtsFrame)()
        for ch in range(IMAGE_CHANNELS):
            count = lib.touch_glove_peek_image_frames(self.handle, ch, ctypes.byref(ptr))
            if count > 0:
                for i in range(count):
                    f = ptr[i]
                    sid = "".join(f"{b:02X}" for b in f.sensor_id)
                    img = np.ctypeslib.as_array(f.param).reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).copy()

                    collected_frames.append(
                        Frame(
                            channel=ch,
                            sensor_id=sid,
                            seq_id=f.seq_id,
                            timestamp_us=f.timestamp_us,
                            image=img,
                        )
                    )
                lib.touch_glove_pop_image_frames(self.handle, ch, count)

        return collected_frames

    def poll(self) -> FrameBatch:
        frames = self._collect_raw_frames()
        if self.decouple_inference and self._inference_thread and self._inference_thread.is_alive():
            for frame in frames:
                self._enqueue_inference(frame)
            return FrameBatch(frames)
        for frame in frames:
            frame.disp, frame.force_field = self._run_inference(frame.channel, frame.image)
            if frame.disp is not None:
                self._publish_dense(frame.channel, frame.disp, frame.force_field)
        return FrameBatch(frames)

    def _start_inference_worker(self) -> None:
        if not self.decouple_inference or self._ort_session is None:
            return
        if self._inference_thread and self._inference_thread.is_alive():
            return
        self._inference_stop.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="touchglove-dense-inference",
            daemon=True,
        )
        self._inference_thread.start()

    def _stop_inference_worker(self) -> None:
        thread = self._inference_thread
        if thread is None:
            return
        self._inference_stop.set()
        thread.join(timeout=5.0)
        if not thread.is_alive():
            self._inference_thread = None
            self._inference_stop.clear()

    def _clear_inference_queue(self) -> None:
        while True:
            try:
                self._inference_queue.get_nowait()
                self._inference_queue.task_done()
            except queue.Empty:
                break

    def _enqueue_inference(self, frame: Frame) -> None:
        try:
            self._inference_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        # Keep latency bounded by discarding the oldest not-yet-inferred frame.
        try:
            self._inference_queue.get_nowait()
            self._inference_queue.task_done()
        except queue.Empty:
            pass
        with self._inference_stats_lock:
            self._inference_dropped_count += 1
        try:
            self._inference_queue.put_nowait(frame)
        except queue.Full:
            with self._inference_stats_lock:
                self._inference_dropped_count += 1

    def _inference_loop(self) -> None:
        while not self._inference_stop.is_set() or not self._inference_queue.empty():
            try:
                frame = self._inference_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._inference_idle.clear()
            try:
                if not self._inference_enabled.is_set():
                    continue
                disp, force = self._run_inference(frame.channel, frame.image)
                if disp is not None and force is not None:
                    self._publish_dense(frame.channel, disp, force)
                    with self._inference_stats_lock:
                        self._inference_completed_count += 1
            finally:
                self._inference_idle.set()
                self._inference_queue.task_done()

    def _publish_dense(self, channel: int, disp: np.ndarray, force: np.ndarray) -> None:
        with self._dense_lock:
            self._dense_cache["disp"][channel] = disp
            self._dense_cache["force"][channel] = force
            if channel == 0:
                self._dense_cache["update_count"] += 1

    def get_dense_fields(self):
        with self._dense_lock:
            return (
                self._dense_cache["disp"].copy(),
                self._dense_cache["force"].copy(),
                int(self._dense_cache["update_count"]),
            )

    def get_inference_stats(self) -> dict:
        with self._inference_stats_lock:
            dropped = self._inference_dropped_count
            completed = self._inference_completed_count
        return {
            "queue_depth": self._inference_queue.qsize(),
            "queue_capacity": self._inference_queue.maxsize,
            "dropped_oldest_count": dropped,
            "completed_count": completed,
            "worker_alive": bool(self._inference_thread and self._inference_thread.is_alive()),
        }

    def get_bytes_per_second(self) -> float:
        if self.use_c_lib and self.handle:
            return float(lib.touch_glove_get_bytes_per_second(self.handle))
        elapsed = time.time() - self.start_t
        return self.total_bytes / elapsed if elapsed > 0 else 0.0

    def get_total_bytes(self) -> int:
        if self.use_c_lib and self.handle:
            return int(lib.touch_glove_get_total_bytes(self.handle))
        return self.total_bytes

    def get_dense_update_count(self) -> int:
        with self._dense_lock:
            return int(self._dense_cache["update_count"])

    @property
    def is_open(self) -> bool:
        if self.use_c_lib and self.handle:
            return bool(lib.touch_glove_is_open(self.handle))
        return self._is_open

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def sensor_ids(self) -> List[str]:
        return self._sensor_ids

    @staticmethod
    def list_ports() -> List[str]:
        return list_ports()


TouchGlovePTP = TouchGlove


def _decode_bcd_byte(value: int) -> int:
    high, low = value >> 4, value & 0x0F
    if high > 9 or low > 9:
        raise ValueError("invalid BCD digit")
    return high * 10 + low


def decode_device_rtc_bcd(raw: bytes) -> int:
    """Decode the device's 8-byte RTC field at image-frame bytes [7:15] to UTC µs."""
    if len(raw) != 8:
        raise ValueError("TouchGlove RTC field must contain exactly 8 bytes")
    year = 2000 + _decode_bcd_byte(raw[0])
    month = _decode_bcd_byte(raw[1])
    day = _decode_bcd_byte(raw[2])
    hour = _decode_bcd_byte(raw[3])
    minute = _decode_bcd_byte(raw[4])
    second = _decode_bcd_byte(raw[5])
    if raw[6] & 0xF0:
        raise ValueError("invalid BCD milliseconds hundreds digit")
    millisecond = (raw[6] & 0x0F) * 100 + _decode_bcd_byte(raw[7])
    if millisecond > 999:
        raise ValueError("invalid TouchGlove RTC millisecond value")
    try:
        instant = _datetime.datetime(year, month, day, hour, minute, second,
                                     millisecond * 1000, tzinfo=_datetime.timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid TouchGlove RTC calendar value: {exc}") from exc
    delta = instant - _datetime.datetime(1970, 1, 1, tzinfo=_datetime.timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
