from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from acquisition.visualization.nokov_trajectory import INVALID_COORDINATE_MAGNITUDE_MM


@dataclass
class SessionResult:
    path: Path
    duration_s: float = 0.0
    emg_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    emg_data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    imu_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    imu_data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    force: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    nokov_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    nokov_tracks: dict[str, np.ndarray] = field(default_factory=dict)
    video_path: Path | None = None
    video_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    stream_notes: list[str] = field(default_factory=list)


def _read_timeline(session_path: Path, descriptor: dict[str, Any], fallback_pattern: str,
                   filename: str) -> Path | None:
    relative = descriptor.get("path")
    if relative:
        candidate = session_path / relative
        return candidate if candidate.is_file() else None
    candidates = sorted((session_path / "derived").glob(f"{fallback_pattern}*/{filename}"))
    return candidates[-1] if candidates else None


def _seconds(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / 1_000_000_000.0


def _emg_highpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 10.0) -> np.ndarray:
    """Match the old analysis page's display-only second-order 10 Hz high-pass."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 10:
        return data.copy()
    k = np.tan(np.pi * float(cutoff_hz) / float(sample_rate_hz))
    norm = 1.0 / (1.0 + np.sqrt(2.0) * k + k * k)
    b0, b1, b2 = norm, -2.0 * norm, norm
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - np.sqrt(2.0) * k + k * k) * norm
    output = np.empty_like(data)
    x1 = x2 = y1 = y2 = 0.0
    for index, value in enumerate(data):
        current = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = current
        x2, x1 = x1, value
        y2, y1 = y1, current
    return output


def _load_nokov(session_path: Path, timeline_path: Path | None,
                 planned_start_ns: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_path = session_path / "raw" / "nokov.h5"
    if not raw_path.is_file():
        return np.empty(0), {}
    with h5py.File(raw_path, "r") as raw:
        records = raw["record_json"].asstr()
        host_times = raw["host_receive_mono_ns"]
        if timeline_path and timeline_path.is_file():
            with h5py.File(timeline_path, "r") as aligned:
                time_ns = aligned["time_ns"][:]
                raw_indices = aligned["raw_record_index"][:]
        else:
            host_values = np.asarray(host_times[:], dtype=np.int64)
            origin_ns = int(planned_start_ns) if planned_start_ns else (int(host_values[0]) if len(host_values) else 0)
            time_ns = host_values - origin_ns
            raw_indices = np.arange(len(records), dtype=np.int64)
        frames: list[dict[str, list[float]]] = []
        for raw_index in raw_indices:
            row = json.loads(records[int(raw_index)])
            points: dict[str, list[float]] = {}
            for marker_set in row.get("marker_sets", []):
                set_name = marker_set.get("name") or f"set {marker_set.get('set_index', 0)}"
                for marker in marker_set.get("markers", []):
                    key = f"{set_name} · 点 {marker.get('marker_index', 0)}"
                    points[key] = [float(value) for value in marker["position_mm"]]
            frames.append(points)
    if not frames:
        return np.empty(0), {}
    names = sorted({name for frame in frames for name in frame})
    tracks = {
        name: np.asarray([frame.get(name, [np.nan, np.nan, np.nan]) for frame in frames], dtype=np.float64)
        for name in names
    }
    return _seconds(time_ns), tracks


def _load_force(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not path.is_file():
        return {}
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with h5py.File(path, "r") as data:
        required = {"time_ns", "channel", "disp", "force_field"}
        if not required.issubset(data.keys()) or data.attrs.get("inference_status") != "complete":
            return {}
        count = len(data["channel"])
        channels = data["channel"][:]
        times = data["time_ns"][:]
        vectors = np.empty((count, 3), dtype=np.float64)
        valid = np.zeros(count, dtype=bool)
        for start in range(0, count, 64):
            stop = min(start + 64, count)
            force = data["force_field"][start:stop]
            disp = data["disp"][start:stop]
            vectors[start:stop] = np.sum(force, axis=(1, 2), dtype=np.float64)
            valid[start:stop] = np.max(np.abs(disp[..., 2]), axis=(1, 2)) >= 0.3
        vectors[~valid] = 0.0
        seconds = _seconds(times)
        for channel in np.unique(channels):
            selected = channels == channel
            channel_vectors = vectors[selected]
            magnitude = np.linalg.norm(channel_vectors, axis=1)
            result[int(channel)] = (seconds[selected], channel_vectors, magnitude)
    return result


def load_session_result(session_path: Path) -> SessionResult:
    """Read available derived streams from a Full_system session without mutation."""
    session_path = Path(session_path).expanduser().resolve()
    manifest_path = session_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("目录中没有 manifest.json；请选择 Full_system 的 Session 目录")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned_start_ns = int(manifest.get("planned_start_mono_ns", 0))
    streams = manifest.get("streams", {})
    result = SessionResult(path=session_path)
    time_sources: list[np.ndarray] = []

    for name, key in (("EMG", "emg"), ("IMU", "imu")):
        descriptor = manifest.get(f"{key}_timeline", {})
        timeline_path = _read_timeline(session_path, descriptor, f"{key}_session_endpoint_map_", f"{key}.h5")
        if timeline_path is None:
            if key in streams:
                result.stream_notes.append(f"{name}：没有可用的对齐结果")
            continue
        with h5py.File(timeline_path, "r") as aligned:
            values = np.asarray(aligned["data"][:], dtype=np.float64)
            times = _seconds(aligned["time_ns"][:])
        if key == "emg":
            sample_rate = float(descriptor.get("sample_rate_hz")
                                or streams.get("emg", {}).get("sample_rate_hz_configured") or 1000)
            if values.ndim == 2:
                values = np.column_stack([
                    _emg_highpass(values[:, channel], sample_rate)
                    for channel in range(values.shape[1])
                ])
            result.emg_time_s, result.emg_data = times, values
        else:
            result.imu_time_s, result.imu_data = times, values
        if len(times):
            time_sources.append(times)

    nokov_descriptor = manifest.get("nokov_timeline", {})
    nokov_path = _read_timeline(session_path, nokov_descriptor, "nokov_", "nokov.h5")
    result.nokov_time_s, result.nokov_tracks = _load_nokov(session_path, nokov_path, planned_start_ns)
    if len(result.nokov_time_s):
        time_sources.append(result.nokov_time_s)
    elif "nokov" in streams:
        result.stream_notes.append("NOKOV：没有可用帧")

    force_descriptor = manifest.get("force_timeline", {})
    force_path = _read_timeline(session_path, force_descriptor, "force_channel_rtc_endpoint_map_v2_", "force.h5")
    if force_path is None and not force_descriptor.get("path"):
        force_path = _read_timeline(session_path, {}, "force_session_endpoint_map_v1_", "force.h5")
    result.force = _load_force(force_path) if force_path else {}
    for force_time, _, _ in result.force.values():
        if len(force_time):
            time_sources.append(force_time)
    if "force" in streams and not result.force:
        result.stream_notes.append("Force：没有可用的推理结果")

    camera_descriptor = streams.get("realsense_rgb", {})
    video_relative = camera_descriptor.get("video_path") or "raw/realsense_rgb.mp4"
    candidate_video = session_path / video_relative
    if candidate_video.is_file():
        result.video_path = candidate_video
        camera_timeline = manifest.get("realsense_timeline", {})
        camera_path = _read_timeline(session_path, camera_timeline, "realsense_host_qpc_v1_", "realsense_rgb.h5")
        if camera_path:
            with h5py.File(camera_path, "r") as aligned:
                result.video_time_s = _seconds(aligned["time_ns"][:])
            if len(result.video_time_s):
                time_sources.append(result.video_time_s)
        else:
            result.stream_notes.append("RealSense：视频存在，但没有对齐帧时间")
    elif "realsense_rgb" in streams:
        result.stream_notes.append("RealSense：没有可用视频")

    if time_sources:
        result.duration_s = max(0.0, max(float(np.nanmax(times)) for times in time_sources if len(times)))
    if not any((len(result.emg_time_s), len(result.imu_time_s), len(result.nokov_time_s),
                bool(result.force), result.video_path is not None)):
        raise ValueError("这个 Session 中没有可预览的派生数据")
    return result


def latest_completed_session(data_root: Path) -> Path | None:
    """Return the most recently updated successfully sealed Full_system session."""
    candidates: list[tuple[float, Path]] = []
    for manifest_path in Path(data_root).glob("session_*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = manifest.get("state")
        except (OSError, json.JSONDecodeError):
            continue
        if state in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
            session_path = manifest_path.parent
            has_derived = any(
                isinstance(manifest.get(f"{key}_timeline"), dict)
                and (relative := manifest[f"{key}_timeline"].get("path"))
                and (session_path / relative).is_file()
                for key in ("emg", "imu", "nokov", "realsense_rgb", "force")
            )
            video_path = manifest.get("streams", {}).get("realsense_rgb", {}).get("video_path")
            has_video = bool(video_path and (session_path / video_path).is_file())
            if has_derived or has_video:
                candidates.append((manifest_path.stat().st_mtime, session_path))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]
