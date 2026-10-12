from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# The legacy NOKOV CSV uses this exact missing-point sentinel. Values at this
# scale cannot be marker coordinates in the documented millimetre workspace.
INVALID_COORDINATE_MAGNITUDE_MM = 1_000_000.0
_CSV_COORDINATE = re.compile(r"^(?P<name>.+)_(?P<axis>[XYZ])_mm$")


def load_nokov_points(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load marker tracks from the legacy aligned CSV or raw session HDF5."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_hdf5(path)
    raise ValueError("Choose a NOKOV aligned CSV or raw NOKOV HDF5 file")


def _load_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'time' column")
        axes: dict[str, dict[str, str]] = {}
        for column in reader.fieldnames:
            match = _CSV_COORDINATE.match(column)
            if match:
                axes.setdefault(match.group("name"), {})[match.group("axis")] = column
        axes = {name: columns for name, columns in axes.items() if set(columns) == {"X", "Y", "Z"}}
        if not axes:
            raise ValueError("CSV has no complete marker XYZ columns ending in _mm")
        times: list[float] = []
        data: dict[str, list[list[float]]] = {name: [] for name in axes}
        for row in reader:
            times.append(float(row["time"]))
            for name, columns in axes.items():
                data[name].append([_float_or_nan(row[columns[axis]]) for axis in "XYZ"])
    return np.asarray(times, dtype=np.float64), {
        name: np.asarray(values, dtype=np.float64) for name, values in data.items()
    }


def _load_hdf5(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_points: list[dict[str, list[float]]] = []
    times_ns: list[int] = []
    with h5py.File(path, "r") as raw:
        if "record_json" not in raw:
            raise ValueError("HDF5 file does not contain NOKOV record_json")
        for encoded in raw["record_json"]:
            if isinstance(encoded, bytes):
                encoded = encoded.decode("utf-8")
            record: dict[str, Any] = json.loads(str(encoded))
            times_ns.append(int(record["host_receive_mono_ns"]))
            points: dict[str, list[float]] = {}
            for marker_set in record.get("marker_sets", []):
                set_name = marker_set.get("name") or f"set {marker_set.get('set_index', 0)}"
                for marker in marker_set.get("markers", []):
                    key = f"{set_name} · 点 {marker.get('marker_index', 0)}"
                    points[key] = [float(value) for value in marker["position_mm"]]
            frame_points.append(points)
    if not frame_points:
        raise ValueError("NOKOV HDF5 contains no frames")
    origin_ns = times_ns[0]
    times = (np.asarray(times_ns, dtype=np.int64) - origin_ns).astype(np.float64) / 1e9
    keys = sorted({key for points in frame_points for key in points})
    tracks = {
        key: np.asarray([points.get(key, [np.nan, np.nan, np.nan]) for points in frame_points],
                        dtype=np.float64)
        for key in keys
    }
    return times, tracks


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def prepare_marker_track(times_s: np.ndarray, positions_mm: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw-valid coordinates, display interpolation, and missing-frame mask.

    Missing samples are linearly interpolated in time between valid observations.
    Leading and trailing gaps use the nearest valid endpoint value. The raw
    input is left untouched; a track with no valid observation remains missing.
    """
    times = np.asarray(times_s, dtype=np.float64)
    raw = np.asarray(positions_mm, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) != len(times):
        raise ValueError("marker positions must have shape (frame_count, 3)")
    valid = np.isfinite(raw).all(axis=1) & (np.abs(raw) < INVALID_COORDINATE_MAGNITUDE_MM).all(axis=1)
    clean = raw.copy()
    clean[~valid] = np.nan
    interpolated = clean.copy()
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        for axis in range(3):
            interpolated[:, axis] = np.interp(times, times[valid], clean[valid, axis])
    return clean, interpolated, ~valid


def gap_runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end frame indices of contiguous missing runs."""
    mask = np.asarray(missing, dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def create_nokov_trajectory_dialog(parent, path: Path):
    """Build a Qt review window without changing the stored NOKOV records."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtWidgets

    path = Path(path)
    times, tracks = load_nokov_points(path)
    if not tracks:
        raise ValueError("No marker tracks were found")

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(f"NOKOV 轨迹查看 · {path.name}")
    dialog.setWindowFlags(
        dialog.windowFlags()
        | QtCore.Qt.WindowType.WindowMinimizeButtonHint
        | QtCore.Qt.WindowType.WindowMaximizeButtonHint
    )
    screen = parent.screen() if parent is not None else QtWidgets.QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = min(1100, max(480, int(available.width() * 0.92)))
        height = min(780, max(360, int(available.height() * 0.90)))
    else:
        width, height = 1100, 780
    dialog.setMinimumSize(min(600, width), min(400, height))
    dialog.resize(width, height)
    layout = QtWidgets.QVBoxLayout(dialog)
    controls = QtWidgets.QHBoxLayout()
    controls.addWidget(QtWidgets.QLabel("点位："))
    select_all = QtWidgets.QPushButton("全选")
    clear_all = QtWidgets.QPushButton("清空")
    controls.addWidget(select_all)
    controls.addWidget(clear_all)
    selector = QtWidgets.QScrollArea()
    selector.setWidgetResizable(True)
    selector.setMaximumHeight(100)
    selector_body = QtWidgets.QWidget()
    selector_layout = QtWidgets.QGridLayout(selector_body)
    selector_layout.setContentsMargins(0, 0, 0, 0)
    selector.setWidget(selector_body)
    controls.addWidget(selector, 1)
    checks: dict[str, QtWidgets.QCheckBox] = {}
    layout.addLayout(controls)
    figure = Figure(figsize=(10, 5), tight_layout=True)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setMinimumSize(0, 0)
    canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                         QtWidgets.QSizePolicy.Policy.Expanding)
    layout.addWidget(canvas, 1)
    status = QtWidgets.QLabel()
    status.setWordWrap(True)
    layout.addWidget(status)

    def draw_tracks(*_args) -> None:
        figure.clear()
        axes = figure.subplots(3, 1, sharex=True)
        selected = [name for name, box in checks.items() if box.isChecked()]
        missing_total = 0
        for name in selected:
            clean, interpolated, missing = prepare_marker_track(times, tracks[name])
            missing_total += int(missing.sum())
            for axis_index, axis in enumerate(axes):
                axis.plot(times, clean[:, axis_index], linewidth=1.0, label=name)
                if missing.any() and np.isfinite(interpolated[:, axis_index]).any():
                    axis.plot(times, interpolated[:, axis_index], linestyle="--", linewidth=1.0,
                              alpha=0.75, label=f"{name} 插补")
        for axis_index, axis in enumerate(axes):
            axis.set_ylabel(f"{('X', 'Y', 'Z')[axis_index]} (mm)")
            axis.grid(True, alpha=0.25)
            if axis_index == 0 and axis.lines:
                axis.legend(loc="upper right", fontsize=8, ncols=2)
        axes[-1].set_xlabel("时间 (s)")
        figure.suptitle("NOKOV 点位轨迹（丢点插补显示）")
        canvas.draw_idle()
        status.setText(
            f"共 {len(times)} 帧；已勾选 {len(selected)}/{len(checks)} 个点位，所选点位有 {missing_total} 个无效/缺失观测。"
            "虚线为按时间线性插补的显示轨迹；首尾缺失沿用最近有效值，全缺失轨迹保持空白。"
            "原始文件不会被修改。"
        )

    for index, name in enumerate(tracks):
        box = QtWidgets.QCheckBox(name)
        box.setChecked(index == 0)
        box.toggled.connect(draw_tracks)
        checks[name] = box
        selector_layout.addWidget(box, index // 3, index % 3)
    select_all.clicked.connect(lambda: [box.setChecked(True) for box in checks.values()])
    clear_all.clicked.connect(lambda: [box.setChecked(False) for box in checks.values()])
    draw_tracks()
    return dialog
