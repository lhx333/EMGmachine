from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets

from acquisition.dataset.legacy_import import LegacyImporter
from acquisition.devices.emg import detect_emg_ports, scan_emg_devices
from acquisition.devices.touchglove import detect_touchglove_ports
from acquisition.session.session import Session
from acquisition.visualization.realtime_emg import RealtimeEmgPage
from acquisition.visualization.result_preview import ResultPreviewDialog


class OperationThread(QtCore.QThread):
    completed = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)

    def __init__(self, operation: str, function: Callable[[], Any]):
        super().__init__()
        self.operation = operation
        self.function = function

    def run(self) -> None:
        try:
            self.completed.emit(self.operation, self.function())
        except Exception as exc:
            self.failed.emit(self.operation, str(exc))


class SessionStopThread(QtCore.QThread):
    completed = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    def run(self) -> None:
        try:
            self.session.stop()
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class DevicePreviewDialog(QtWidgets.QDialog):
    """Independent, non-modal live preview window for one connected device."""

    def __init__(self, key: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        titles = {
            "emg": "EMG + IMU 实时预览",
            "nokov": "NOKOV 实时预览",
            "realsense_rgb": "RealSense 实时预览",
            "force": "TouchGlove 实时预览",
        }
        self.setWindowTitle(titles.get(key, "设备实时预览"))
        self.resize(1100 if key == "emg" else 720, 650 if key == "emg" else 560)
        layout = QtWidgets.QVBoxLayout(self)
        self.details = QtWidgets.QLabel("等待实时数据…")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.emg_page: RealtimeEmgPage | None = None
        self.image: QtWidgets.QLabel | None = None
        if key == "emg":
            self.emg_page = RealtimeEmgPage(window_seconds=3.0)
            layout.addWidget(self.emg_page, 1)
        else:
            self.image = QtWidgets.QLabel("等待图像帧")
            self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.image.setMinimumSize(640, 420)
            self.image.setStyleSheet("background: #20242a; color: #c7cbd1; font-size: 18px")
            layout.addWidget(self.image, 1)

    def update_preview(self, preview: dict[str, dict[str, Any]], state: str) -> None:
        if self.key == "emg":
            if self.emg_page:
                emg_state = (state if state in {"RECORDING", "COMPLETED", "ABORTED", "ERROR"}
                             else "READY")
                self.emg_page.set_state(emg_state)
                self.emg_page.ingest_response({"state": emg_state, "emg_preview": preview.get("emg", {})})
                imu = preview.get("imu", {})
                latest_imu = imu.get("latest")
                self.details.setText(f"IMU：{latest_imu if latest_imu is not None else '等待数据'}")
            return

        value = preview.get(self.key, {})
        self.details.setText(HostWindow._format_preview(self.key, value) if value else "等待实时数据…")
        if self.image is None:
            return
        image = QtGui.QImage()
        if self.key == "realsense_rgb" and value.get("preview_jpeg"):
            image = QtGui.QImage.fromData(value["preview_jpeg"], "JPG")
        elif self.key == "force" and value.get("image_96x96"):
            image = QtGui.QImage(value["image_96x96"], 96, 96, 96,
                                 QtGui.QImage.Format.Format_Grayscale8).copy()
        if not image.isNull():
            self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                self.image.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        elif self.key == "nokov" and value:
            self.image.setText(
                f"NOKOV 数据已到达\n帧：{value.get('device_frame_index', '—')}\n"
                f"刚体：{value.get('rigid_body_count', 0)}\n"
                f"标记点：{value.get('labeled_marker_count', 0)}\n"
                f"首个刚体坐标（mm）：{value.get('first_rigid_body_mm')}"
            )


class HostWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("多模态采集上位机")
        self.resize(980, 790)
        self.session: Session | None = None
        self.operation_thread: OperationThread | None = None
        self.stop_thread: SessionStopThread | None = None
        self._connected: set[str] = set()
        self._preview_dialogs: dict[str, DevicePreviewDialog] = {}
        self._results_dialog: ResultPreviewDialog | None = None

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        title = QtWidgets.QLabel("多模态采集 · Windows")
        title.setStyleSheet("font-size: 22px; font-weight: 600")
        layout.addWidget(title)

        self.state_label = QtWidgets.QLabel("状态：逐项连接设备；全部连接后才能采集")
        self.path_label = QtWidgets.QLabel("Session：连接设备后创建")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.count_label = QtWidgets.QLabel("正式记录数：0")
        layout.addWidget(self.state_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.count_label)

        experiment_form = QtWidgets.QFormLayout()
        self.subject_edit = QtWidgets.QLineEdit()
        self.subject_edit.setPlaceholderText("可选")
        self.action_edit = QtWidgets.QLineEdit()
        self.action_edit.setPlaceholderText("可选")
        self.trial_edit = QtWidgets.QLineEdit()
        self.trial_edit.setPlaceholderText("可选")
        experiment_form.addRow("受试者", self.subject_edit)
        experiment_form.addRow("动作/任务", self.action_edit)
        experiment_form.addRow("试次", self.trial_edit)
        layout.addLayout(experiment_form)

        self.device_status: dict[str, QtWidgets.QLabel] = {}

        emg_box = QtWidgets.QGroupBox("1 · EMG + IMU")
        emg_layout = QtWidgets.QVBoxLayout(emg_box)
        emg_controls = QtWidgets.QHBoxLayout()
        self.emg_port_combo = QtWidgets.QComboBox()
        self.refresh_emg_ports_button = QtWidgets.QPushButton("刷新接收器")
        self.scan_emg_button = QtWidgets.QPushButton("扫描手环 MAC")
        self.emg_mac_combo = QtWidgets.QComboBox()
        self.emg_connect_button = QtWidgets.QPushButton("连接所选手环")
        emg_controls.addWidget(QtWidgets.QLabel("接收器"))
        emg_controls.addWidget(self.emg_port_combo, 1)
        emg_controls.addWidget(self.refresh_emg_ports_button)
        emg_layout.addLayout(emg_controls)
        emg_connect_row = QtWidgets.QHBoxLayout()
        emg_connect_row.addWidget(self.scan_emg_button)
        emg_connect_row.addWidget(self.emg_mac_combo, 1)
        emg_connect_row.addWidget(self.emg_connect_button)
        emg_layout.addLayout(emg_connect_row)
        self._add_device_status(emg_layout, "emg")
        layout.addWidget(emg_box)

        nokov_box = QtWidgets.QGroupBox("2 · NOKOV 动捕")
        nokov_layout = QtWidgets.QVBoxLayout(nokov_box)
        nokov_controls = QtWidgets.QHBoxLayout()
        self.nokov_ip_edit = QtWidgets.QLineEdit("10.1.1.198")
        self.nokov_connect_button = QtWidgets.QPushButton("连接 NOKOV")
        nokov_controls.addWidget(QtWidgets.QLabel("XINGYING 服务端 IP"))
        nokov_controls.addWidget(self.nokov_ip_edit, 1)
        nokov_controls.addWidget(self.nokov_connect_button)
        nokov_layout.addLayout(nokov_controls)
        self._add_device_status(nokov_layout, "nokov")
        layout.addWidget(nokov_box)

        camera_box = QtWidgets.QGroupBox("3 · RealSense RGB")
        camera_layout = QtWidgets.QVBoxLayout(camera_box)
        self.realsense_connect_button = QtWidgets.QPushButton("连接并预览相机")
        camera_layout.addWidget(self.realsense_connect_button)
        self._add_device_status(camera_layout, "realsense_rgb")
        layout.addWidget(camera_box)

        force_box = QtWidgets.QGroupBox("4 · TouchGlove 力传感器")
        force_layout = QtWidgets.QVBoxLayout(force_box)
        force_controls = QtWidgets.QHBoxLayout()
        self.touchglove_port_combo = QtWidgets.QComboBox()
        self.refresh_touchglove_button = QtWidgets.QPushButton("刷新端口")
        self.touchglove_connect_button = QtWidgets.QPushButton("连接力手套")
        force_controls.addWidget(QtWidgets.QLabel("USB 串口"))
        force_controls.addWidget(self.touchglove_port_combo, 1)
        force_controls.addWidget(self.refresh_touchglove_button)
        force_controls.addWidget(self.touchglove_connect_button)
        force_layout.addLayout(force_controls)
        self._add_device_status(force_layout, "force")
        layout.addWidget(force_box)
        self.device_buttons = {
            "emg": self.emg_connect_button,
            "nokov": self.nokov_connect_button,
            "realsense_rgb": self.realsense_connect_button,
            "force": self.touchglove_connect_button,
        }

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("定时开始五路采集")
        self.stop_button = QtWidgets.QPushButton("停止并封存")
        self.import_button = QtWidgets.QPushButton("导入旧采集目录")
        self.results_view_button = QtWidgets.QPushButton("结果可视化预览")
        self.nokov_view_button = QtWidgets.QPushButton("查看 NOKOV 轨迹")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        for button in (self.start_button, self.stop_button, self.import_button,
                       self.results_view_button, self.nokov_view_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)

        self.refresh_emg_ports_button.clicked.connect(self.refresh_emg_ports)
        self.refresh_touchglove_button.clicked.connect(self.refresh_touchglove_ports)
        self.scan_emg_button.clicked.connect(self.scan_emg)
        self.emg_connect_button.clicked.connect(self.connect_emg)
        self.nokov_connect_button.clicked.connect(self.connect_nokov)
        self.realsense_connect_button.clicked.connect(self.connect_realsense)
        self.touchglove_connect_button.clicked.connect(self.connect_touchglove)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.import_button.clicked.connect(self.import_legacy)
        self.results_view_button.clicked.connect(self.view_results)
        self.nokov_view_button.clicked.connect(self.view_nokov_trajectory)
        self.refresh_emg_ports()
        self.refresh_touchglove_ports()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)

    def _add_device_status(self, layout: QtWidgets.QVBoxLayout, name: str) -> None:
        status = QtWidgets.QLabel("状态：待连接")
        self.device_status[name] = status
        layout.addWidget(status)

    def _ensure_session(self) -> Session:
        if self.session is None or self.session.state in {"COMPLETED", "COMPLETED_WITH_ERRORS", "STOPPED"}:
            self._connected.clear()
            self.session = Session(
                Path(__file__).resolve().parents[2] / "data",
                experiment={"subject": self.subject_edit.text(), "action": self.action_edit.text(),
                            "trial": self.trial_edit.text()},
            )
            self.session.prepare()
            self.path_label.setText(f"Session：{self.session.path}")
        return self.session

    def _run_operation(self, name: str, function: Callable[[], Any], button: QtWidgets.QPushButton) -> None:
        if self.operation_thread and self.operation_thread.isRunning():
            return
        self._set_connection_controls_busy(True)
        self.state_label.setText(f"状态：正在连接 {name}…")
        self.operation_thread = OperationThread(name, function)
        self.operation_thread.completed.connect(self._operation_completed)
        self.operation_thread.failed.connect(self._operation_failed)
        self.operation_thread.start()

    def scan_emg(self) -> None:
        if self.operation_thread and self.operation_thread.isRunning():
            return
        port = self.emg_port_combo.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "未发现接收器", "请先连接 ResearchKit USB 接收器并刷新端口。")
            return
        self._set_connection_controls_busy(True)
        self.state_label.setText("状态：正在扫描附近的 EMG 手环（约 5 秒）…")
        self.operation_thread = OperationThread("scan_emg", lambda: scan_emg_devices(port))
        self.operation_thread.completed.connect(self._operation_completed)
        self.operation_thread.failed.connect(self._operation_failed)
        self.operation_thread.start()

    def connect_emg(self) -> None:
        if "EMG + IMU" in self._connected:
            self._open_device_preview("emg")
            return
        port = self.emg_port_combo.currentData()
        mac = self.emg_mac_combo.currentData()
        if not port or not mac:
            QtWidgets.QMessageBox.warning(self, "尚未选择手环", "先扫描并选择一个手环 MAC，再点击连接。")
            return
        try:
            session = self._ensure_session()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建 Session 失败", str(exc))
            return
        self.device_status["emg"].setText(f"状态：正在连接 {mac}…")
        self._run_operation("EMG + IMU", lambda: session.connect_emg(port, mac), self.emg_connect_button)

    def connect_nokov(self) -> None:
        if "NOKOV" in self._connected:
            self._open_device_preview("nokov")
            return
        ip = self.nokov_ip_edit.text().strip()
        if not ip:
            QtWidgets.QMessageBox.warning(self, "地址为空", "请输入 XINGYING/NOKOV 服务端 IP。")
            return
        try:
            session = self._ensure_session()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建 Session 失败", str(exc))
            return
        self.device_status["nokov"].setText(f"状态：正在连接 {ip}…")
        self._run_operation("NOKOV", lambda: session.connect_nokov(ip), self.nokov_connect_button)

    def connect_realsense(self) -> None:
        if "RealSense" in self._connected:
            self._open_device_preview("realsense_rgb")
            return
        try:
            session = self._ensure_session()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建 Session 失败", str(exc))
            return
        self.device_status["realsense_rgb"].setText("状态：正在打开相机并等待首帧…")
        self._run_operation("RealSense", session.connect_realsense, self.realsense_connect_button)

    def connect_touchglove(self) -> None:
        if "TouchGlove" in self._connected:
            self._open_device_preview("force")
            return
        port = self.touchglove_port_combo.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "未发现力手套", "请选择 TouchGlove USB 串口。")
            return
        try:
            session = self._ensure_session()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建 Session 失败", str(exc))
            return
        self.device_status["force"].setText(f"状态：正在连接 {port} 并检查 DirectML…")
        self._run_operation("TouchGlove", lambda: session.connect_touchglove(port), self.touchglove_connect_button)

    def _operation_completed(self, operation: str, result: Any) -> None:
        if operation == "scan_emg":
            self.emg_mac_combo.clear()
            for device in result:
                self.emg_mac_combo.addItem(f"{device['name']} · {device['mac_address']}", device["mac_address"])
            self._set_connection_controls_busy(False)
            if result:
                self.state_label.setText(f"状态：扫描到 {len(result)} 个手环，请选择 MAC 后点击连接")
            else:
                self.state_label.setText("状态：没有扫描到手环；确认手环已开机且未连接其他主机")
            return

        self._connected.add(operation)
        names = {"EMG + IMU": "emg", "NOKOV": "nokov", "RealSense": "realsense_rgb", "TouchGlove": "force"}
        key = names[operation]
        self._set_connection_controls_busy(False)
        self.device_buttons[key].setText("打开预览窗口")
        self.device_status[key].setText(f"状态：已连接 · {self._connection_description(operation, result)}")
        self.state_label.setText("状态：设备持续上报预览数据；全部连接后可开始采集")
        if self.session:
            self.path_label.setText(f"Session：{self.session.path}")
        self._update_start_enabled()
        self._open_device_preview(key)

    @staticmethod
    def _connection_description(operation: str, result: dict[str, Any]) -> str:
        if operation == "EMG + IMU":
            return f"{result.get('device_name', '手环')} · {result.get('device_mac', '')}"
        if operation == "NOKOV":
            return f"{result.get('server_ip')} · SDK {result.get('sdk_runtime_version')}"
        if operation == "RealSense":
            return f"D435i {result.get('width')}×{result.get('height')}@{result.get('fps')} fps"
        return f"{result.get('port')} · DirectML 已验证"

    def _operation_failed(self, operation: str, error: str) -> None:
        if operation == "scan_emg":
            self._set_connection_controls_busy(False)
        else:
            key = {"EMG + IMU": "emg", "NOKOV": "nokov", "RealSense": "realsense_rgb",
                   "TouchGlove": "force"}.get(operation)
            if key:
                self.device_status[key].setText("状态：连接失败")
            self._set_connection_controls_busy(False)
        self.state_label.setText(f"状态：{operation} 连接失败")
        QtWidgets.QMessageBox.critical(self, f"{operation} 连接失败", error)
        self._update_start_enabled()

    def _set_connection_controls_busy(self, busy: bool) -> None:
        for key, button in self.device_buttons.items():
            button.setEnabled(not busy)
            if key not in self._connected:
                button.setText({"emg": "连接所选手环", "nokov": "连接 NOKOV",
                                "realsense_rgb": "连接并预览相机", "force": "连接力手套"}[key])
        self.scan_emg_button.setEnabled(not busy and "EMG + IMU" not in self._connected)

    def _open_device_preview(self, key: str) -> None:
        dialog = self._preview_dialogs.get(key)
        if dialog is None:
            dialog = DevicePreviewDialog(key, self)
            self._preview_dialogs[key] = dialog
        dialog.update_preview(self.session.preview if self.session else {},
                              self.session.state if self.session else "READY")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _update_start_enabled(self) -> None:
        ready = bool(self.session and self.session.all_devices_connected)
        self.start_button.setEnabled(ready and self.session.state in {"PREPARED", "CONNECTING_DEVICES"})

    @staticmethod
    def _format_preview(name: str, value: dict[str, Any]) -> str:
        if name == "emg":
            latest = value.get("latest")
            return f"EMG 数据包 {value.get('count')} 个样本 · {value.get('sample_rate_hz')} Hz · 最新值 {latest}"
        if name == "imu":
            return f"IMU 数据包 {value.get('count')} 个样本 · 最新值 {value.get('latest')}"
        if name == "nokov":
            return (f"帧 {value.get('device_frame_index')} · 刚体 {value.get('rigid_body_count')} · "
                    f"标记点 {value.get('labeled_marker_count')} · 首个刚体坐标 mm {value.get('first_rigid_body_mm')}")
        if name == "realsense_rgb":
            return f"收到相机帧 #{value.get('camera_frame_number')} · QPC {value.get('host_receive_mono_ns')}"
        return (f"收到力图像 · channel {value.get('channel')} · seq {value.get('seq_id')} · "
                f"device_time {value.get('device_timestamp_us')} µs")

    def start(self) -> None:
        if not self.session:
            return
        try:
            self.session.start_emg()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "启动失败", str(exc))
            if self.session.state in {"ARMED", "RECORDING", "ERROR"}:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
            return
        self.state_label.setText("状态：ARMED · 五路设备将按同一 QPC 时刻开始正式记录")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop(self) -> None:
        if self.session and (self.stop_thread is None or not self.stop_thread.isRunning()):
            self.stop_button.setEnabled(False)
            self.state_label.setText("状态：停止采集并执行时间轴生成与离线力推理…")
            self.stop_thread = SessionStopThread(self.session)
            self.stop_thread.completed.connect(self._on_stop_completed)
            self.stop_thread.failed.connect(self._on_stop_failed)
            # Stop includes disk draining, timeline generation and offline
            # inference. Keep it below the GUI thread's scheduler priority.
            self.stop_thread.start(QtCore.QThread.Priority.LowPriority)

    def _on_stop_completed(self) -> None:
        self.refresh()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self._connected.clear()
        defaults = {"emg": "连接所选手环", "nokov": "连接 NOKOV",
                    "realsense_rgb": "连接并预览相机", "force": "连接力手套"}
        for key, button in self.device_buttons.items():
            button.setEnabled(True)
            button.setText(defaults[key])
        self.scan_emg_button.setEnabled(True)
        for label in self.device_status.values():
            label.setText("状态：本次 Session 已封存")

    def _on_stop_failed(self, error: str) -> None:
        QtWidgets.QMessageBox.critical(self, "封存失败", error)
        self.refresh()
        self.stop_button.setEnabled(False)

    def import_legacy(self) -> None:
        source = QtWidgets.QFileDialog.getExistingDirectory(self, "选择旧采集目录", str(Path(__file__).resolve().parents[3]))
        if not source:
            return
        try:
            result = LegacyImporter().import_directory(Path(source), Path(__file__).resolve().parents[2] / "data")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入失败", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "导入完成", f"源文件已只读复制并生成字段/时间检查报告：\n{result}")

    def view_nokov_trajectory(self) -> None:
        default_sample = Path(__file__).resolve().parents[3] / "example_sample" / "windows" / "02_nokov_human_markers_aligned.csv"
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 NOKOV 数据", str(default_sample if default_sample.exists() else Path(__file__).resolve().parents[2] / "data"),
            "NOKOV 数据 (*.csv *.h5 *.hdf5)",
        )
        if not selected:
            return
        try:
            from acquisition.visualization.nokov_trajectory import create_nokov_trajectory_dialog
            create_nokov_trajectory_dialog(self, Path(selected)).exec()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "NOKOV 轨迹读取失败", str(exc))

    def view_results(self) -> None:
        data_root = Path(__file__).resolve().parents[2] / "data"
        if self._results_dialog is None:
            self._results_dialog = ResultPreviewDialog(data_root, self)
        elif not self._results_dialog.isVisible():
            self._results_dialog.open_latest()
        self._results_dialog.show()
        self._results_dialog.raise_()
        self._results_dialog.activateWindow()

    def refresh(self) -> None:
        if self.session is None:
            return
        session = self.session
        self.count_label.setText(
            f"正式记录：EMG {session.sample_count} · IMU {session.imu_sample_count} · "
            f"NOKOV {session.nokov_frame_count} 帧 · RGB {session.realsense_frame_count} 帧 · "
            f"力图像 {session.touchglove_frame_count} 帧"
        )
        previews = session.preview
        for key in ("emg", "nokov", "realsense_rgb", "force"):
            dialog = self._preview_dialogs.get(key)
            if dialog is not None and dialog.isVisible():
                dialog.update_preview(previews, session.state)
        if session.state in {"ARMED", "RECORDING", "STOPPING", "PROCESSING"}:
            self.state_label.setText(f"状态：{session.state}")
        if session.state in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
            self.state_label.setText(f"状态：{session.state} · Session 已封存：{session.path}")
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(False)
        self._update_start_enabled()

    def refresh_emg_ports(self) -> None:
        current = self.emg_port_combo.currentData() if hasattr(self, "emg_port_combo") else None
        self.emg_port_combo.clear()
        try:
            ports = detect_emg_ports()
        except Exception as exc:
            ports = []
            self.state_label.setText(f"状态：EMG 接收器枚举失败：{exc}")
        for device, description in ports:
            self.emg_port_combo.addItem(f"{device} · {description}", device)
        if current:
            index = self.emg_port_combo.findData(current)
            if index >= 0:
                self.emg_port_combo.setCurrentIndex(index)
        if not ports:
            self.emg_port_combo.addItem("未发现 ResearchKit 接收器", None)

    def refresh_touchglove_ports(self) -> None:
        current = self.touchglove_port_combo.currentData()
        self.touchglove_port_combo.clear()
        try:
            ports = detect_touchglove_ports()
        except Exception as exc:
            ports = []
            self.state_label.setText(f"状态：力手套串口枚举失败：{exc}")
        for device, description in ports:
            self.touchglove_port_combo.addItem(f"{device} · {description}", device)
        if current:
            index = self.touchglove_port_combo.findData(current)
            if index >= 0:
                self.touchglove_port_combo.setCurrentIndex(index)
        if not ports:
            self.touchglove_port_combo.addItem("未发现可用串口", None)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.operation_thread and self.operation_thread.isRunning():
            QtWidgets.QMessageBox.information(self, "设备操作进行中", "请等待当前设备连接操作完成后再关闭窗口。")
            event.ignore()
            return
        if self.stop_thread and self.stop_thread.isRunning():
            QtWidgets.QMessageBox.information(self, "正在封存", "请等待时间轴生成和力场推理完成后再关闭窗口。")
            event.ignore()
            return
        if self.session and self.session.state in {
            "ARMED", "RECORDING", "PREPARED", "CONNECTING_DEVICES", "ERROR"
        }:
            self.session.stop()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = HostWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
