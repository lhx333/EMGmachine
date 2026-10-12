from collections import deque
import math

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


CHANNEL_NAMES = [f"CH{i}" for i in range(1, 10)]
CHANNEL_COLORS = [
    "#D62728",
    "#1F77B4",
    "#2CA02C",
    "#FF7F0E",
    "#9467BD",
    "#17BECF",
    "#E377C2",
    "#8C564B",
    "#7F7F7F",
]
CHANNEL_Y_MIN = -500.0
CHANNEL_Y_MAX = 500.0


class EmgPlotWidget(QtWidgets.QWidget):
    """Fast QPainter-based rolling plot with per-channel clipping."""

    def __init__(self, window_seconds=3.0, parent=None):
        super().__init__(parent)
        self.window_seconds = float(window_seconds)
        self.sample_rate_hz = 1000.0
        self.series = {}
        self.spacing = 0.0
        self.setMinimumHeight(360)
        self.setAutoFillBackground(True)

    @property
    def global_y_limits(self):
        count = len(self.series)
        if not count:
            return CHANNEL_Y_MIN, CHANNEL_Y_MAX
        center = (count - 1) / 2.0
        baselines = [(center - index) * self.spacing for index in range(count)]
        return min(baselines) + CHANNEL_Y_MIN, max(baselines) + CHANNEL_Y_MAX

    def set_series(self, series, spacing, sample_rate_hz):
        self.series = series
        self.spacing = float(spacing)
        self.sample_rate_hz = float(sample_rate_hz)
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor("#FBFCFD"))
        painter.setFont(QtGui.QFont("Microsoft YaHei UI", 9))
        plot = QtCore.QRectF(self.rect()).adjusted(62, 18, -18, -42)
        if plot.width() <= 0 or plot.height() <= 0:
            return

        painter.setPen(QtGui.QPen(QtGui.QColor("#52606D"), 1))
        painter.drawRect(plot)
        self._draw_time_grid(painter, plot)

        selected = list(self.series)
        longest = max((len(values) for values in self.series.values()), default=0)
        if not selected or not longest:
            painter.setPen(QtGui.QColor("#52606D"))
            painter.drawText(
                plot,
                QtCore.Qt.AlignCenter,
                "准备完成后显示实时肌电",
            )
            return

        y_min, y_max = self.global_y_limits
        y_span = max(1.0, y_max - y_min)

        def map_y(value):
            return plot.bottom() - (float(value) - y_min) / y_span * plot.height()

        center = (len(selected) - 1) / 2.0
        baselines = {
            name: (center - index) * self.spacing
            for index, name in enumerate(selected)
        }
        if self.spacing == 0:
            label_groups = {0.0: "/".join(selected)}
        else:
            label_groups = {baseline: name for name, baseline in baselines.items()}

        for baseline, label in label_groups.items():
            painter.setPen(QtGui.QColor("#17202A"))
            y = map_y(baseline)
            painter.drawText(
                QtCore.QRectF(2, y - 10, 55, 20),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                label,
            )

        overlay = np.zeros((self.height(), self.width(), 4), dtype=np.uint8)
        for name in selected:
            color = QtGui.QColor(CHANNEL_COLORS[CHANNEL_NAMES.index(name)])
            baseline = baselines[name]
            baseline_y = map_y(baseline)
            baseline_pen = QtGui.QPen(color, 1, QtCore.Qt.DashLine)
            baseline_pen.setColor(QtGui.QColor(color.red(), color.green(), color.blue(), 70))
            painter.setPen(baseline_pen)
            painter.drawLine(
                QtCore.QPointF(plot.left(), baseline_y),
                QtCore.QPointF(plot.right(), baseline_y),
            )
            self._draw_channel(
                overlay,
                plot,
                np.asarray(self.series[name], dtype=np.float64),
                baseline,
                color,
            )

        image = QtGui.QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.strides[0],
            QtGui.QImage.Format_RGBA8888,
        )
        painter.drawImage(0, 0, image)

        painter.setPen(QtGui.QColor("#334E68"))
        painter.drawText(
            QtCore.QRectF(plot.left(), plot.bottom() + 20, plot.width(), 20),
            QtCore.Qt.AlignCenter,
            "相对当前时刻（s）",
        )
        painter.drawText(8, 14, "各通道固定 -500～+500")

    def _draw_time_grid(self, painter, plot):
        ticks = np.linspace(-self.window_seconds, 0.0, 7)
        for value in ticks:
            ratio = (value + self.window_seconds) / self.window_seconds
            x = plot.left() + ratio * plot.width()
            painter.setPen(QtGui.QPen(QtGui.QColor("#DCE3E9"), 1))
            painter.drawLine(
                QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom())
            )
            painter.setPen(QtGui.QColor("#334E68"))
            painter.drawText(
                QtCore.QRectF(x - 25, plot.bottom() + 2, 50, 18),
                QtCore.Qt.AlignCenter,
                f"{value:.1f}",
            )

    def _draw_channel(self, overlay, plot, values, baseline, color):
        if values.size == 0:
            return
        sample_indexes = np.arange(values.size, dtype=np.float64)
        relative_s = (sample_indexes - (values.size - 1)) / self.sample_rate_hz
        x_values = plot.left() + (
            (relative_s + self.window_seconds) / self.window_seconds
        ) * plot.width()
        y_min, y_max = self.global_y_limits
        y_span = max(1.0, y_max - y_min)
        y_values = plot.bottom() - (
            (values - y_min) / y_span * plot.height()
        )
        local_values = values - baseline
        valid = (
            np.isfinite(values)
            & (local_values >= CHANNEL_Y_MIN)
            & (local_values <= CHANNEL_Y_MAX)
            & (x_values >= plot.left())
            & (x_values <= plot.right())
        )
        if np.count_nonzero(valid) < 2:
            return
        changes = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
        rgba = (color.red(), color.green(), color.blue(), 255)
        for start, end in changes.reshape(-1, 2):
            if end - start < 2:
                continue
            points = np.column_stack(
                (np.rint(x_values[start:end]), np.rint(y_values[start:end]))
            ).astype(np.int32)
            cv2.polylines(
                overlay,
                [points.reshape(-1, 1, 2)],
                False,
                rgba,
                1,
                cv2.LINE_AA,
            )


class RealtimeEmgPage(QtWidgets.QWidget):
    """One large, vertically offset plot for live nine-channel EMG preview."""

    def __init__(self, parent=None, window_seconds=3.0):
        super().__init__(parent)
        self.window_seconds = float(window_seconds)
        self.sample_rate_hz = 1000.0
        self.last_sequence = -1
        self.received_samples = 0
        self.buffers = {name: deque(maxlen=6000) for name in CHANNEL_NAMES}
        self.filtered_buffers = {
            name: deque(maxlen=6000) for name in CHANNEL_NAMES
        }
        self._filter_states = {
            name: [0.0, 0.0, 0.0, 0.0] for name in CHANNEL_NAMES
        }
        self._filter_coefficients = self._highpass_coefficients()
        self._build()
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(50)
        self._render_timer.timeout.connect(self._render_if_visible)
        self._render_timer.start()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("实时肌电")
        title.setObjectName("realtimeTitle")
        self.state_label = QtWidgets.QLabel("● 等待准备")
        self.state_label.setObjectName("realtimeState")
        self.rate_label = QtWidgets.QLabel("预览采样率：—")
        self.count_label = QtWidgets.QLabel("接收样本：0")
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(self.state_label)
        header.addStretch()
        header.addWidget(self.rate_label)
        header.addSpacing(16)
        header.addWidget(self.count_label)
        layout.addLayout(header)

        content = QtWidgets.QHBoxLayout()
        controls = QtWidgets.QGroupBox("显示设置")
        controls.setFixedWidth(190)
        control_layout = QtWidgets.QVBoxLayout(controls)
        self.channel_checks = {}
        for name, color in zip(CHANNEL_NAMES, CHANNEL_COLORS):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(True)
            check.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 700; }}")
            check.toggled.connect(self.render)
            self.channel_checks[name] = check
            control_layout.addWidget(check)

        button_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("全选")
        clear_all = QtWidgets.QPushButton("清空")
        select_all.clicked.connect(lambda: self._set_all_channels(True))
        clear_all.clicked.connect(lambda: self._set_all_channels(False))
        button_row.addWidget(select_all)
        button_row.addWidget(clear_all)
        control_layout.addLayout(button_row)

        self.offset_check = QtWidgets.QCheckBox("通道纵向错位")
        self.offset_check.setChecked(True)
        self.offset_check.toggled.connect(self.render)
        control_layout.addWidget(self.offset_check)

        spacing_row = QtWidgets.QHBoxLayout()
        spacing_row.addWidget(QtWidgets.QLabel("通道间距 K"))
        self.spacing_spin = QtWidgets.QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 1_000_000.0)
        self.spacing_spin.setDecimals(2)
        self.spacing_spin.setValue(1000.0)
        self.spacing_spin.valueChanged.connect(self.render)
        spacing_row.addWidget(self.spacing_spin, 1)
        control_layout.addLayout(spacing_row)

        self.highpass_check = QtWidgets.QCheckBox("10 Hz 高通（仅显示）")
        self.highpass_check.setChecked(True)
        self.highpass_check.toggled.connect(self.render)
        control_layout.addWidget(self.highpass_check)
        control_layout.addStretch()
        content.addWidget(controls)

        plot_box = QtWidgets.QGroupBox("最近 3 秒")
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        self.plot_widget = EmgPlotWidget(self.window_seconds)
        plot_layout.addWidget(self.plot_widget, 1)
        content.addWidget(plot_box, 1)
        layout.addLayout(content, 1)

        note = QtWidgets.QLabel(
            "纵向偏移和高通滤波只影响屏幕显示，不修改正式保存的肌电数据。"
        )
        note.setObjectName("realtimeNote")
        layout.addWidget(note)
        self.render()

    def _set_all_channels(self, checked):
        for box in self.channel_checks.values():
            box.blockSignals(True)
            box.setChecked(bool(checked))
            box.blockSignals(False)
        self.render()

    def clear_preview(self):
        for buffer in self.buffers.values():
            buffer.clear()
        for buffer in self.filtered_buffers.values():
            buffer.clear()
        self._filter_states = {
            name: [0.0, 0.0, 0.0, 0.0] for name in CHANNEL_NAMES
        }
        self.last_sequence = -1
        self.received_samples = 0
        self.count_label.setText("接收样本：0")
        self.render()

    def set_state(self, state):
        labels = {
            "PREPARING": ("● 正在准备", "#B54708"),
            "READY": ("● 实时预览中", "#17682A"),
            "RECORDING": ("● 正在记录", "#B42318"),
            "COMPLETED": ("● 采集完成", "#087F5B"),
            "ABORTED": ("● 已中止", "#8F1C13"),
            "ERROR": ("● 预览错误", "#B42318"),
        }
        text, color = labels.get(state, ("● 等待准备", "#52606D"))
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color}; font-weight: 800;")

    def ingest_response(self, response):
        self.set_state(response.get("state"))
        preview = response.get("emg_preview") or {}
        rows = preview.get("samples") or []
        if not rows:
            return
        rate = float(preview.get("sample_rate_hz") or self.sample_rate_hz)
        if rate > 0 and rate != self.sample_rate_hz:
            self.sample_rate_hz = rate
            self._filter_coefficients = self._highpass_coefficients()
            self._filter_states = {
                name: [0.0, 0.0, 0.0, 0.0] for name in CHANNEL_NAMES
            }
        sequence = int(preview.get("sequence_start", 0))
        for index, row in enumerate(rows):
            current_sequence = sequence + index
            if current_sequence <= self.last_sequence:
                continue
            values = list(row)[: len(CHANNEL_NAMES)]
            values.extend([0.0] * (len(CHANNEL_NAMES) - len(values)))
            for name, value in zip(CHANNEL_NAMES, values):
                raw_value = float(value)
                self.buffers[name].append(raw_value)
                self.filtered_buffers[name].append(
                    self._highpass_sample(name, raw_value)
                )
            self.last_sequence = current_sequence
            self.received_samples += 1
        self._trim_buffers()
        self.rate_label.setText(f"预览采样率：{self.sample_rate_hz:g} Hz")
        self.count_label.setText(f"接收样本：{self.received_samples}")

    def _trim_buffers(self):
        target = max(10, int(round(self.sample_rate_hz * self.window_seconds)))
        for collection in (self.buffers, self.filtered_buffers):
            for name, old_buffer in collection.items():
                if old_buffer.maxlen == target:
                    continue
                collection[name] = deque(list(old_buffer)[-target:], maxlen=target)

    def _highpass_coefficients(self, cutoff_hz=10.0):
        k = math.tan(math.pi * float(cutoff_hz) / self.sample_rate_hz)
        norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
        return (
            norm,
            -2.0 * norm,
            norm,
            2.0 * (k * k - 1.0) * norm,
            (1.0 - math.sqrt(2.0) * k + k * k) * norm,
        )

    def _highpass_sample(self, name, value):
        b0, b1, b2, a1, a2 = self._filter_coefficients
        x1, x2, y1, y2 = self._filter_states[name]
        current = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        self._filter_states[name] = [value, x1, current, y1]
        return current

    def display_series(self):
        selected = [
            name for name, box in self.channel_checks.items() if box.isChecked()
        ]
        signals = {}
        for name in selected:
            source = (
                self.filtered_buffers[name]
                if self.highpass_check.isChecked()
                else self.buffers[name]
            )
            signals[name] = np.asarray(source, dtype=np.float64)
        spacing = 0.0
        if selected and self.offset_check.isChecked():
            spacing = float(self.spacing_spin.value())
        result = {}
        center = (len(selected) - 1) / 2.0
        for index, name in enumerate(selected):
            result[name] = signals[name] + (center - index) * spacing
        return result, spacing

    @QtCore.Slot()
    def render(self):
        series, spacing = self.display_series()
        self.plot_widget.set_series(series, spacing, self.sample_rate_hz)

    @QtCore.Slot()
    def _render_if_visible(self):
        if self.isVisible():
            self.render()
