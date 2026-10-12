from __future__ import annotations

from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
from matplotlib import rcParams
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from acquisition.visualization.nokov_trajectory import prepare_marker_track
from acquisition.visualization.result_data import SessionResult, latest_completed_session, load_session_result

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, axes_count: int = 1, share_x: bool = False):
        self.figure = Figure(figsize=(10, 4.5), dpi=100, tight_layout=True)
        if axes_count == 1:
            self.axes = [self.figure.add_subplot(111)]
        else:
            self.axes = list(self.figure.subplots(axes_count, 1, sharex=share_x))
        super().__init__(self.figure)
        self.setMinimumHeight(180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)


class ResultPreviewDialog(QtWidgets.QDialog):
    """Read-only, synchronized result review for a sealed Full_system session."""

    def __init__(self, data_root: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.data_root = Path(data_root)
        self.result: SessionResult | None = None
        self._video: cv2.VideoCapture | None = None
        self._video_frame = -1
        self._timed = False
        self._play_anchor_s = 0.0
        self._play_clock = QtCore.QElapsedTimer()
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(100)
        self._play_timer.timeout.connect(self._advance)
        self.emg_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.imu_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.force_checks: dict[int, QtWidgets.QCheckBox] = {}
        self.nokov_checks: dict[str, QtWidgets.QCheckBox] = {}

        self.setWindowTitle("采集结果可视化预览")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )
        screen = parent.screen() if parent is not None else QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(1360, max(640, int(available.width() * 0.92)))
            height = min(950, max(480, int(available.height() * 0.90)))
            self.setMaximumSize(available.size())
        else:
            width, height = 1360, 950
        self.setMinimumSize(min(800, width), min(600, height))
        self.resize(width, height)
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLineEdit()
        self.path_label.setReadOnly(True)
        self.path_label.setPlaceholderText("选择已封存的 Full_system Session")
        self.latest_button = QtWidgets.QPushButton("打开最新结果")
        self.choose_button = QtWidgets.QPushButton("选择历史 Session")
        header.addWidget(self.path_label, 1)
        header.addWidget(self.latest_button)
        header.addWidget(self.choose_button)
        layout.addLayout(header)

        self.summary = QtWidgets.QLabel("读取 EMG、IMU、Force、NOKOV 与 RealSense 的对齐结果")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        transport = QtWidgets.QHBoxLayout()
        self.restart_button = QtWidgets.QPushButton("从头播放")
        self.play_button = QtWidgets.QPushButton("播放")
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 10000)
        self.time_label = QtWidgets.QLabel("0.00 / 0.00 s")
        self.time_label.setMinimumWidth(105)
        transport.addWidget(self.restart_button)
        transport.addWidget(self.play_button)
        transport.addWidget(self.time_slider, 1)
        transport.addWidget(self.time_label)
        layout.addLayout(transport)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_emg_tab()
        self._build_force_tab()
        self._build_nokov_tab()
        self._build_video_tab()

        self.latest_button.clicked.connect(self.open_latest)
        self.choose_button.clicked.connect(self.choose_session)
        self.restart_button.clicked.connect(self.restart)
        self.play_button.clicked.connect(self.toggle_play)
        self.time_slider.valueChanged.connect(self.seek)
        self._set_tabs_enabled(False)
        self.open_latest()

    def _build_emg_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.emg_controls = QtWidgets.QWidget()
        self.emg_control_layout = QtWidgets.QHBoxLayout(self.emg_controls)
        self.emg_control_layout.setContentsMargins(0, 0, 0, 0)
        self.emg_control_layout.addWidget(QtWidgets.QLabel("通道："))
        self.emg_all_button = QtWidgets.QPushButton("EMG 全选")
        self.emg_clear_button = QtWidgets.QPushButton("清空 EMG")
        self.emg_control_layout.addWidget(self.emg_all_button)
        self.emg_control_layout.addWidget(self.emg_clear_button)
        self.emg_channel_widget = QtWidgets.QWidget()
        self.emg_channel_layout = QtWidgets.QHBoxLayout(self.emg_channel_widget)
        self.emg_channel_layout.setContentsMargins(0, 0, 0, 0)
        self.emg_control_layout.addWidget(self.emg_channel_widget)
        self.emg_control_layout.addSpacing(16)
        self.emg_control_layout.addWidget(QtWidgets.QLabel("IMU"))
        self.imu_channel_widget = QtWidgets.QWidget()
        self.imu_channel_layout = QtWidgets.QHBoxLayout(self.imu_channel_widget)
        self.imu_channel_layout.setContentsMargins(0, 0, 0, 0)
        self.emg_control_layout.addWidget(self.imu_channel_widget)
        self.emg_control_layout.addStretch(1)
        layout.addWidget(self.emg_controls)
        self.emg_canvas = PlotCanvas()
        self.emg_imu_axis = self.emg_canvas.axes[0].twinx()
        self.emg_canvas.axes[0].set_ylabel("EMG (SDK units)")
        layout.addWidget(self.emg_canvas, 1)
        self.tabs.addTab(tab, "EMG + IMU")

    def _build_force_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("通道"))
        self.force_all_button = QtWidgets.QPushButton("全选")
        self.force_clear_button = QtWidgets.QPushButton("清空")
        controls.addWidget(self.force_all_button)
        controls.addWidget(self.force_clear_button)
        self.force_control_widget = QtWidgets.QWidget()
        self.force_control_layout = QtWidgets.QHBoxLayout(self.force_control_widget)
        self.force_control_layout.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.force_control_widget)
        controls.addSpacing(24)
        controls.addWidget(QtWidgets.QLabel("显示"))
        self.force_metric = QtWidgets.QComboBox()
        self.force_metric.addItems(["合力", "X", "Y", "Z"])
        controls.addWidget(self.force_metric)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.force_canvas = PlotCanvas()
        self.force_canvas.axes[0].set_ylabel("力场合计 (N)")
        self.force_canvas.axes[0].set_xlabel("相对时间 (s)")
        layout.addWidget(self.force_canvas, 1)
        self.force_metric.currentIndexChanged.connect(self.redraw_force)
        self.emg_all_button.clicked.connect(lambda: self._set_checked(self.emg_checks, True))
        self.emg_clear_button.clicked.connect(lambda: self._set_checked(self.emg_checks, False))
        self.force_all_button.clicked.connect(lambda: self._set_checked(self.force_checks, True))
        self.force_clear_button.clicked.connect(lambda: self._set_checked(self.force_checks, False))
        self.tabs.addTab(tab, "TouchGlove 力")

    def _build_nokov_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("点位："))
        self.nokov_all_button = QtWidgets.QPushButton("全选")
        self.nokov_clear_button = QtWidgets.QPushButton("清空")
        controls.addWidget(self.nokov_all_button)
        controls.addWidget(self.nokov_clear_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.nokov_check_scroll = QtWidgets.QScrollArea()
        self.nokov_check_scroll.setWidgetResizable(True)
        self.nokov_check_scroll.setMaximumHeight(112)
        self.nokov_check_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nokov_check_widget = QtWidgets.QWidget()
        self.nokov_check_layout = QtWidgets.QGridLayout(self.nokov_check_widget)
        self.nokov_check_layout.setContentsMargins(4, 2, 4, 2)
        self.nokov_check_scroll.setWidget(self.nokov_check_widget)
        layout.addWidget(self.nokov_check_scroll)
        self.nokov_canvas = PlotCanvas(3, share_x=True)
        layout.addWidget(self.nokov_canvas, 1)
        self.nokov_note = QtWidgets.QLabel("超大坐标、NaN 和缺帧按时间插补显示；插补仅用于查看。")
        self.nokov_note.setWordWrap(True)
        layout.addWidget(self.nokov_note)
        self.nokov_all_button.clicked.connect(lambda: self._set_checked(self.nokov_checks, True))
        self.nokov_clear_button.clicked.connect(lambda: self._set_checked(self.nokov_checks, False))
        self.tabs.addTab(tab, "NOKOV 点位")

    def _build_video_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.video_label = QtWidgets.QLabel("选择包含 RealSense 视频的 Session")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setStyleSheet("background:#111820;color:#DCE6EE;border:1px solid #526779;")
        layout.addWidget(self.video_label, 1)
        self.video_status = QtWidgets.QLabel("等待视频")
        self.video_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_status)
        self.tabs.addTab(tab, "RealSense 视频")

    def _set_tabs_enabled(self, enabled: bool) -> None:
        for index in range(self.tabs.count()):
            self.tabs.setTabEnabled(index, enabled)
        self.time_slider.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.restart_button.setEnabled(enabled)

    @staticmethod
    def _clear_layout(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _set_checked(mapping: dict, checked: bool) -> None:
        for box in mapping.values():
            box.setChecked(checked)

    def open_latest(self) -> None:
        session_path = latest_completed_session(self.data_root)
        if session_path is None:
            self.summary.setText("还没有已封存的 Session。完成一次采集后可在这里预览结果，或选择历史 Session。")
            return
        self.load_path(session_path)

    def choose_session(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择已封存的 Session", str(self.data_root)
        )
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path) -> None:
        try:
            result = load_session_result(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "结果加载失败", str(exc))
            return
        self._release_video()
        self.result = result
        self._timed = False
        self._populate_controls()
        self.path_label.setText(str(result.path))
        maxima = [
            len(result.emg_time_s), len(result.imu_time_s), len(result.nokov_time_s),
            sum(len(item[0]) for item in result.force.values()), len(result.video_time_s),
        ]
        notes = "；".join(result.stream_notes) if result.stream_notes else "五路结果已读取"
        self.summary.setText(
            f"{result.path.name} · {notes} · EMG {maxima[0]} 点 · IMU {maxima[1]} 点 · "
            f"Force {maxima[3]} 帧 · NOKOV {maxima[2]} 帧 · RealSense {maxima[4]} 帧 · "
            f"时长 {result.duration_s:.2f} s。预览只读，不会修改原始记录。"
        )
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self._set_tabs_enabled(True)
        self._redraw_all()
        self._open_video()
        self._update_time_label(0.0)

    def _populate_controls(self) -> None:
        assert self.result is not None
        self._clear_layout(self.emg_channel_layout)
        self._clear_layout(self.imu_channel_layout)
        self.emg_checks.clear()
        self.imu_checks.clear()
        for channel in range(self.result.emg_data.shape[1] if self.result.emg_data.ndim == 2 else 0):
            name = f"EMG_CH{channel + 1}"
            box = QtWidgets.QCheckBox(name)
            box.setChecked(channel == 0)
            box.toggled.connect(self.redraw_emg)
            self.emg_checks[name] = box
            self.emg_channel_layout.addWidget(box)
        for axis in range(self.result.imu_data.shape[1] if self.result.imu_data.ndim == 2 else 0):
            name = f"IMU_{axis + 1}"
            box = QtWidgets.QCheckBox(name)
            box.setChecked(axis == 0)
            box.toggled.connect(self.redraw_emg)
            self.imu_checks[name] = box
            self.imu_channel_layout.addWidget(box)

        self._clear_layout(self.force_control_layout)
        self.force_checks.clear()
        for channel in sorted(self.result.force):
            box = QtWidgets.QCheckBox(f"通道 {channel + 1}")
            box.setChecked(channel == min(self.result.force))
            box.toggled.connect(self.redraw_force)
            self.force_checks[channel] = box
            self.force_control_layout.addWidget(box)

        self._clear_layout(self.nokov_check_layout)
        self.nokov_checks.clear()
        for index, name in enumerate(self.result.nokov_tracks):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(index == 0)
            box.toggled.connect(self.redraw_nokov)
            self.nokov_checks[name] = box
            self.nokov_check_layout.addWidget(box, index // 4, index % 4)

    def _visible_indices(self, times: np.ndarray) -> np.ndarray:
        if not len(times) or not self._timed:
            return np.arange(len(times), dtype=np.int64)
        # Device-derived streams can contain timestamp regressions. Filter in
        # capture order instead of relying on searchsorted's sorted-input rule.
        return np.flatnonzero(times <= self._playhead_s)

    @staticmethod
    def _thin(times: np.ndarray, values: np.ndarray, limit: int = 6000):
        step = max(1, int(np.ceil(len(times) / limit)))
        return times[::step], values[::step]

    def _redraw_all(self) -> None:
        self.redraw_emg()
        self.redraw_force()
        self.redraw_nokov()

    def redraw_emg(self, *_args) -> None:
        if self.result is None:
            return
        axis = self.emg_canvas.axes[0]
        axis.clear()
        right = self.emg_imu_axis
        right.clear()
        for index, (name, box) in enumerate(self.emg_checks.items()):
            if not box.isChecked():
                continue
            visible = self._visible_indices(self.result.emg_time_s)
            x, y = self._thin(self.result.emg_time_s[visible], self.result.emg_data[visible, index])
            axis.plot(x, y, linewidth=0.8, label=name)
        if self.result.imu_data.ndim == 2 and self.result.imu_data.shape[1]:
            for index, (name, box) in enumerate(self.imu_checks.items()):
                if not box.isChecked():
                    continue
                visible = self._visible_indices(self.result.imu_time_s)
                x, y = self._thin(self.result.imu_time_s[visible], self.result.imu_data[visible, index])
                right.plot(x, y, linewidth=0.75, linestyle="--", alpha=0.8, label=name)
            right.set_ylabel("IMU (SDK units)")
            handles, labels = axis.get_legend_handles_labels()
            other_handles, other_labels = right.get_legend_handles_labels()
            if other_handles:
                axis.legend(handles + other_handles, labels + other_labels, loc="upper right", ncols=3, fontsize=8)
        elif axis.lines:
            axis.legend(loc="upper right", ncols=3, fontsize=8)
        axis.set_ylabel("EMG (SDK units)")
        axis.set_xlabel("相对统一主机时间 (s)")
        axis.grid(True, alpha=0.25)
        self.emg_canvas.draw_idle()

    def redraw_force(self, *_args) -> None:
        if self.result is None:
            return
        axis = self.force_canvas.axes[0]
        axis.clear()
        metric_index = self.force_metric.currentIndex() - 1
        for channel, box in self.force_checks.items():
            if not box.isChecked() or channel not in self.result.force:
                continue
            times, vectors, magnitude = self.result.force[channel]
            visible = self._visible_indices(times)
            values = magnitude if metric_index < 0 else vectors[:, metric_index]
            x, y = self._thin(times[visible], values[visible])
            axis.plot(x, y, linewidth=1.0, label=f"通道 {channel + 1}")
        axis.set_ylabel(f"力场 {'合力' if metric_index < 0 else 'XYZ'[metric_index]} (N)")
        axis.set_xlabel("相对统一主机时间 (s)")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(loc="upper right")
        self.force_canvas.draw_idle()

    def redraw_nokov(self, *_args) -> None:
        if self.result is None:
            return
        axes = self.nokov_canvas.axes
        for axis in axes:
            axis.clear()
        visible = self._visible_indices(self.result.nokov_time_s)
        times = self.result.nokov_time_s[visible]
        selected_names = [name for name, box in self.nokov_checks.items() if box.isChecked()]
        selected_missing = 0
        for name in selected_names:
            track = self.result.nokov_tracks[name][visible]
            clean, interpolated, missing = prepare_marker_track(times, track)
            selected_missing += int(missing.sum())
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
        axes[-1].set_xlabel("相对统一主机时间 (s)")
        self.nokov_note.setText(
            f"已勾选 {len(selected_names)}/{len(self.nokov_checks)} 个点位；所选点位共 {selected_missing} 个无效/缺失观测。"
            "虚线为按时间线性插补的显示轨迹；首尾缺失沿用最近有效值，全缺失轨迹保持空白。原始记录不变。"
        )
        self.nokov_canvas.draw_idle()

    def _open_video(self) -> None:
        if self.result is None or self.result.video_path is None:
            self.video_label.setText("此 Session 没有 RealSense 视频")
            self.video_status.setText("无视频")
            return
        self._video = cv2.VideoCapture(str(self.result.video_path))
        if not self._video.isOpened():
            self._video.release()
            self._video = None
            self.video_label.setText("无法解码 RealSense 视频")
            self.video_status.setText(str(self.result.video_path))
            return
        self._video_frame = -1
        self._show_video_at(0.0)

    def _show_video_at(self, seconds: float) -> None:
        if self.result is None or self._video is None or not len(self.result.video_time_s):
            return
        eligible = np.flatnonzero(self.result.video_time_s <= seconds)
        if not len(eligible):
            if self._timed:
                self.video_label.setText("等待 RealSense 首帧")
                self.video_status.setText(f"{seconds:.2f} s · 首帧尚未到达")
                self._video_frame = -1
                return
            index = 0
        else:
            index = int(eligible[-1])
        if index == self._video_frame:
            return
        self._video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._video.read()
        if not ok:
            self.video_status.setText(f"无法读取视频帧 {index + 1}")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                             QtGui.QImage.Format.Format_RGB888).copy()
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))
        self._video_frame = index
        self.video_status.setText(
            f"帧 {index + 1} / {len(self.result.video_time_s)} · "
            f"对齐时间 {self.result.video_time_s[index]:.3f} s · 播放位置 {seconds:.3f} s"
        )

    def seek(self, value: int) -> None:
        if self.result is None:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("播放")
        self._timed = True
        self._playhead_s = self.result.duration_s * int(value) / 10000.0
        self._redraw_all()
        self._show_video_at(self._playhead_s)
        self._update_time_label(self._playhead_s)

    def _update_time_label(self, seconds: float) -> None:
        duration = self.result.duration_s if self.result else 0.0
        self.time_label.setText(f"{seconds:.2f} / {duration:.2f} s")

    def toggle_play(self) -> None:
        if self.result is None or self.result.duration_s <= 0:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("播放")
            return
        if not self._timed:
            self._timed = True
            self._playhead_s = 0.0
        elif self._playhead_s >= self.result.duration_s:
            self._playhead_s = 0.0
        self._play_anchor_s = self._playhead_s
        self._play_clock.restart()
        self._play_timer.start()
        self.play_button.setText("暂停")
        self._set_playhead(self._playhead_s)

    def _advance(self) -> None:
        seconds = self._play_anchor_s + self._play_clock.elapsed() / 1000.0
        if self.result is None or seconds >= self.result.duration_s:
            seconds = self.result.duration_s if self.result else 0.0
            self._set_playhead(seconds)
            self._play_timer.stop()
            self.play_button.setText("播放")
            return
        self._set_playhead(seconds)

    def _set_playhead(self, seconds: float) -> None:
        if self.result is None:
            return
        self._playhead_s = max(0.0, min(float(seconds), self.result.duration_s))
        value = round(10000 * self._playhead_s / self.result.duration_s) if self.result.duration_s else 0
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(value)
        self.time_slider.blockSignals(False)
        self._redraw_all()
        self._show_video_at(self._playhead_s)
        self._update_time_label(self._playhead_s)

    def restart(self) -> None:
        if self.result is None:
            return
        self._timed = True
        self._play_timer.stop()
        self.play_button.setText("播放")
        self._set_playhead(0.0)

    def _release_video(self) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None
        self._video_frame = -1

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._play_timer.stop()
        self._release_video()
        super().closeEvent(event)
