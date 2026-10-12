from __future__ import annotations

import multiprocessing
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from acquisition.clock import WindowsQpcClock


DEFAULT_EMG_MAC = "ED:9D:0F:48:F3:BA"


def normalize_emg_mac(value: str) -> str:
    normalized = value.strip().replace("-", ":").upper()
    if not re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", normalized):
        raise ValueError(f"Invalid EMG band MAC address: {value!r}")
    return normalized


def detect_emg_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports

    ports = [
        (port.device, port.description or port.hwid)
        for port in list_ports.comports()
        if "VID:PID=1915:C00A" in (port.hwid or "").upper()
    ]
    configured = os.environ.get("EMG_PORT", "").strip()
    if configured and all(device.upper() != configured.upper() for device, _ in ports):
        return [(configured, "EMG_PORT override")]
    return ports


def scan_emg_devices(port: str, duration_s: float = 5.0) -> list[dict[str, str]]:
    """Scan BLE bands through the selected ResearchKit USB receiver."""
    sys.path.insert(0, str(_find_sdk_path()))
    from ResearchKit_SDK import ResearchKitClient

    client = ResearchKitClient(port=port)
    try:
        client.open()
        devices = client.scan_devices(duration=float(duration_s))
        return [{"name": str(device.name), "mac_address": normalize_emg_mac(device.mac_address)}
                for device in devices]
    finally:
        for method_name in ("stop_capture", "close"):
            method = getattr(client, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


def _find_sdk_path() -> Path:
    bundled = Path(__file__).resolve().parents[3] / "vendor" / "researchkit"
    if not (bundled / "ResearchKit_SDK" / "__init__.py").is_file():
        raise FileNotFoundError(f"Bundled ResearchKit SDK is missing: {bundled}")
    return bundled


def _emg_process(connection, port: str, sample_rate_hz: int, imu_sample_rate_hz: int,
                 target_mac: str, start_deadline_ns: int) -> None:
    """Own SDK/BLE in a child process; timestamp at the SDK callback boundary."""
    client = None
    try:
        sys.path.insert(0, str(_find_sdk_path()))
        from ResearchKit_SDK import ResearchKitClient, SensorConfig, __version__ as sdk_version
        connection.send({"type": "STATUS", "status": "researchkit_loaded"})

        clock = WindowsQpcClock()
        target_mac = normalize_emg_mac(target_mac)
        target = None
        last_error = None
        for attempt in range(1, 4):
            client = ResearchKitClient(port=port)
            try:
                connection.send({"type": "STATUS", "status": "opening_ble_adapter", "attempt": attempt})
                client.open()
                connection.send({"type": "STATUS", "status": "scanning_ble", "attempt": attempt})
                devices = client.scan_devices(duration=5)
                target = next((device for device in devices if device.mac_address.upper() == target_mac), None)
                if target is None:
                    found = [f"{device.name} ({device.mac_address})" for device in devices]
                    raise RuntimeError(f"Target EMG band {target_mac} not found. BLE devices found: {found}")
                connection.send({"type": "STATUS", "status": "connecting_band", "device": target.name, "attempt": attempt})
                client.connect(target.mac_address)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                connection.send({"type": "STATUS", "status": "connect_retry", "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                for method_name in ("stop_capture", "close"):
                    method = getattr(client, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass
                client = None
                if attempt < 3:
                    time.sleep(1.0)
        if last_error is not None or target is None or client is None:
            raise RuntimeError(f"Could not connect to EMG band after 3 attempts: {last_error}")
        config = SensorConfig(
            emg_enabled=True,
            imu_enabled=True,
            emg_sample_rate=int(sample_rate_hz),
            imu_sample_rate=int(imu_sample_rate_hz),
            capture_name="FullSystem",
            capture_hand="R",
            capture_action=1,
        )
        connection.send({
            "type": "READY",
            "port": port,
            "device_name": target.name,
            "device_mac": target.mac_address,
            "sample_rate_hz": int(sample_rate_hz),
            "imu_sample_rate_hz": int(imu_sample_rate_hz),
            "researchkit_sdk_version": sdk_version,
            "clock_id": clock.clock_id,
            "clock_frequency_hz": clock.frequency_hz,
        })

        def on_data(data: Any) -> None:
            observed_ns = clock.now_ns()
            sensor_name = getattr(data.sensor_type, "name", "")
            if sensor_name not in {"EMG", "IMU"}:
                return
            payload = data.data
            try:
                values = payload.tolist()
            except AttributeError:
                values = payload
            connection.send({
                "type": "DATA",
                "payload": {
                    "sensor_type": str(sensor_name),
                    "device_timestamp_raw": int(data.device_timestamp),
                    "sample_count": int(data.sample_count),
                    "sample_rate_hz": int(data.sample_rate),
                    "data": values,
                    "host_receive_mono_ns": observed_ns,
                    "timestamp_stage": "researchkit_on_data_received_callback",
                    "clock_id": clock.clock_id,
                },
            })

        client.on_data_received = on_data
        client.start_capture(config)
        connection.send({"type": "STATUS", "status": "preview_streaming"})
        deadline = None
        while True:
            if connection.poll(0.05):
                command = connection.recv()
                if command.get("type") == "STOP":
                    break
                if command.get("type") == "ARM":
                    deadline = int(command["deadline_mono_ns"])
                    connection.send({"type": "ARMED", "deadline_mono_ns": deadline})
            if not getattr(client, "is_capturing", True):
                break
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send({"type": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if client is not None:
            for method_name in ("stop_capture", "close"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        try:
            connection.send({"type": "STOPPED"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


class EmgAdapter:
    """ResearchKit EMG worker with callback-time stamps in the Windows QPC domain."""

    def __init__(self, on_record: Callable[[dict[str, Any]], None], on_event: Callable[[str, dict], None]):
        self.on_record = on_record
        self.on_event = on_event
        self._ctx = multiprocessing.get_context("spawn")
        self._process = None
        self._connection = None
        self._reader: threading.Thread | None = None
        self._error: str | None = None
        self.ready = threading.Event()
        self.recording = threading.Event()
        self.stopped = threading.Event()

    def prepare(self, port: str, sample_rate_hz: int = 1000, imu_sample_rate_hz: int = 50,
                timeout_s: float = 30.0, target_mac: str = DEFAULT_EMG_MAC) -> dict[str, Any]:
        if self._process is not None:
            raise RuntimeError("EMG adapter already prepared")
        target_mac = normalize_emg_mac(target_mac)
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_emg_process,
                                 args=(child, port, sample_rate_hz, imu_sample_rate_hz, target_mac, 0),
                                 name="researchkit-sensors")
        proc.start()
        child.close()
        self._connection, self._process = parent, proc
        self._reader = threading.Thread(target=self._read_loop, name="emg-ipc-reader", daemon=True)
        self._reader.start()
        deadline = time.monotonic() + timeout_s
        while not self.ready.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
            if not proc.is_alive():
                exit_code = proc.exitcode
                self.stop()
                raise RuntimeError(f"EMG worker exited during PREPARE (exit code {exit_code})")
            if time.monotonic() >= deadline:
                self.stop()
                raise TimeoutError("Timed out connecting to the EMG adapter/device")
        if self._error:
            error = self._error
            self.stop()
            raise RuntimeError(error)
        return self._ready_info

    def _read_loop(self) -> None:
        try:
            while True:
                message = self._connection.recv()
                kind = message.get("type")
                if kind == "READY":
                    self._ready_info = message
                    self.ready.set()
                    self.on_event("EMG_READY", message)
                elif kind == "RECORDING":
                    self.recording.set()
                    self.on_event("EMG_RECORDING", message)
                elif kind == "DATA":
                    self.on_record(message["payload"])
                elif kind == "ERROR":
                    self._error = str(message.get("error", "EMG worker failed"))
                    self.on_event("EMG_ERROR", {"error": self._error})
                    self.ready.set()
                elif kind == "STATUS":
                    self.on_event("EMG_STATUS", {"status": message.get("status")})
                elif kind == "STOPPED":
                    self.stopped.set()
                    return
        except (EOFError, OSError) as exc:
            if self._process is not None and self._process.exitcode not in (0, None):
                self._error = f"EMG worker exited with code {self._process.exitcode}: {exc}"
            self.stopped.set()

    def arm(self, deadline_mono_ns: int) -> None:
        if not self.ready.is_set() or self._error:
            raise RuntimeError(self._error or "EMG adapter is not ready")
        self._connection.send({"type": "ARM", "deadline_mono_ns": int(deadline_mono_ns)})

    def stop(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.is_alive():
                self._connection.send({"type": "STOP"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        proc.join(timeout=8)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=2)
        try:
            self._connection.close()
        except Exception:
            pass
        try:
            proc.close()
        except Exception:
            pass
        self._process = self._connection = self._reader = None
