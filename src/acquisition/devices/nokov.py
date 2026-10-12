from __future__ import annotations

import sys
import threading
import ctypes
from pathlib import Path
from typing import Any, Callable

from acquisition.clock import WindowsQpcClock


_SDK_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "nokov"
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def frame_to_record(frame: Any, callback_index: int, host_receive_mono_ns: int,
                    clock_id: str) -> dict[str, Any]:
    """Copy SDK-owned frame memory into JSON-safe raw Python values."""
    marker_sets = []
    for set_index in range(int(frame.nMarkerSets)):
        marker_set = frame.MocapData[set_index]
        marker_sets.append({
            "set_index": set_index,
            "name": _text(marker_set.szName),
            "markers": [
                {"marker_index": marker_index,
                 "position_mm": [float(v) for v in marker_set.Markers[marker_index]]}
                for marker_index in range(int(marker_set.nMarkers))
            ],
        })

    rigid_bodies = []
    for body_index in range(int(frame.nRigidBodies)):
        body = frame.RigidBodies[body_index]
        rigid_bodies.append({
            "body_index": body_index,
            "id": int(body.ID),
            "position_mm": [float(body.x), float(body.y), float(body.z)],
            "orientation_xyzw": [float(body.qx), float(body.qy), float(body.qz), float(body.qw)],
            "mean_error": float(body.MeanError),
            "params": int(body.params),
            "markers": [
                {"id": int(body.MarkerIDs[i]),
                 "position_mm": [float(v) for v in body.Markers[i]],
                 "size": float(body.MarkerSizes[i])}
                for i in range(int(body.nMarkers))
            ],
        })

    skeletons = []
    for skeleton_index in range(int(frame.nSkeletons)):
        skeleton = frame.Skeletons[skeleton_index]
        segments = []
        for segment_index in range(int(skeleton.nRigidBodies)):
            body = skeleton.RigidBodyData[segment_index]
            segments.append({
                "segment_index": segment_index,
                "id": int(body.ID),
                "position_mm": [float(body.x), float(body.y), float(body.z)],
                "orientation_xyzw": [float(body.qx), float(body.qy), float(body.qz), float(body.qw)],
                "mean_error": float(body.MeanError),
                "params": int(body.params),
            })
        skeletons.append({"skeleton_index": skeleton_index,
                          "id": int(skeleton.skeletonID), "segments": segments})

    labeled_markers = []
    for index in range(int(frame.nLabeledMarkers)):
        marker = frame.LabeledMarkers[index]
        labeled_markers.append({
            "id": int(marker.ID),
            "position_mm": [float(marker.x), float(marker.y), float(marker.z)],
            "size": float(marker.size),
            "params": int(marker.params),
        })

    other_markers = [
        [float(v) for v in frame.OtherMarkers[index]]
        for index in range(int(frame.nOtherMarkers))
    ]
    analog = [float(frame.Analogdata[index]) for index in range(int(frame.nAnalogdatas))]
    return {
        "callback_index": int(callback_index),
        "device_frame_index": int(frame.iFrame),
        "device_timestamp_raw": int(frame.iTimeStamp),
        "device_timestamp_semantics": (
            "NOKOV SDK documentation defines iTimeStamp in milliseconds; live values track "
            "Unix milliseconds and were about 39 ms behind sampled PC time in this test, "
            "but whether the clock is camera/device-side or XINGYING-host-side is unconfirmed"
        ),
        "timecode": int(frame.Timecode),
        "timecode_subframe": int(frame.TimecodeSubframe),
        "frame_params": int(frame.params),
        "latency_sdk_raw": float(frame.fLatency),
        "host_receive_mono_ns": int(host_receive_mono_ns),
        "timestamp_stage": "qpc_at_nokov_data_callback_entry",
        "clock_id": clock_id,
        "position_unit": "mm (NOKOV SDK sample convention)",
        "orientation_order": "qx,qy,qz,qw",
        "marker_sets": marker_sets,
        "rigid_bodies": rigid_bodies,
        "skeletons": skeletons,
        "labeled_markers": labeled_markers,
        "unidentified_markers_mm": other_markers,
        "analog_values_raw": analog,
    }


class NokovAdapter:
    """NOKOV callback adapter; SDK pointers are copied before callback return."""

    def __init__(self, on_record: Callable[[dict[str, Any]], None],
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.on_record = on_record
        self.on_event = on_event or (lambda _name, _details: None)
        self.clock = WindowsQpcClock()
        self.client = None
        self.info: dict[str, Any] = {}
        self.error: str | None = None
        self.sdk_messages: list[str] = []
        self._lock = threading.Lock()
        self._armed_at_ns: int | None = None
        self._callback_index = 0
        self._previous_frame: int | None = None
        self._frame_gap_count = 0
        self._duplicate_frame_count = 0
        self._frame_regression_count = 0
        self._callback_exception_count = 0

    def prepare(self, server_ip: str) -> dict[str, Any]:
        if self.client is not None:
            raise RuntimeError("NOKOV is already prepared")
        try:
            from nokov.nokovsdk import PySDKClient
        except Exception as exc:
            raise RuntimeError(f"Could not load the vendored NOKOV Windows SDK: {exc}") from exc

        client = PySDKClient()
        self.client = client
        client.PySetVerbosityLevel(4)
        client.PySetMessageCallback(self._on_message)
        client.PySetNotifyMsgCallback(self._on_notify, None)
        client.PySetDataCallback(self._on_frame, None)
        result = int(client.Initialize(server_ip.encode("utf-8")))
        if result != 0:
            diagnostic = "; ".join(self.sdk_messages[-4:])
            self._uninitialize()
            raise RuntimeError(
                f"NOKOV SDK Initialize({server_ip}) failed with code {result}; "
                f"SDK log: {diagnostic or 'no diagnostic message'}; "
                "check that XINGYING is streaming on this address and that the capture network is reachable"
            )
        client.PySetVerbosityLevel(0)
        version = [int(value) for value in client.PyNokovVersion()]
        self.info = {
            "server_ip": server_ip,
            "sdk_runtime_version": version,
            "python_binding_version": "nokovpy 3.0.1 (XING Python SDK package 2.4.0.5428)",
            "timestamp_field": "iTimeStamp",
            "timestamp_semantics": (
                "SDK documentation confirms milliseconds. In a 6 s live test it advanced "
                "6011 ms over 541 frame intervals and stayed about 39 ms behind sampled PC "
                "Unix milliseconds; clock origin/source is unconfirmed, so retain raw and "
                "do not use for cross-device alignment yet."
            ),
            "frame_order_field": "iFrame",
            "coordinate_unit": "mm",
            "host_clock_id": self.clock.clock_id,
            "host_receive_semantics": "QPC sampled on entry to the SDK data callback",
        }
        self.on_event("NOKOV_READY", self.info)
        return dict(self.info)

    def arm(self, deadline_ns: int) -> None:
        if self.client is None:
            raise RuntimeError("NOKOV must be prepared before start")
        with self._lock:
            self._armed_at_ns = int(deadline_ns)

    def _on_message(self, level: int, message: Any) -> None:
        try:
            raw = ctypes.cast(message, ctypes.c_char_p).value if message else b""
            decoded = raw.decode("utf-8", errors="replace") if raw else ""
            if decoded:
                self.sdk_messages.append(f"level={int(level)} {decoded}")
                del self.sdk_messages[:-16]
        except Exception:
            pass

    def _on_notify(self, pointer: Any, _user_data: Any) -> None:
        if not pointer:
            return
        notify = pointer.contents
        try:
            self.on_event("NOKOV_NOTIFY", {
                "notify_type": int(notify.nType), "value": int(notify.nValue),
                "device_timestamp_raw": int(notify.nTimeStamp), "message": _text(notify.sMsg),
                "param1": int(notify.nParam1), "param2": int(notify.nParam2),
                "param3": int(notify.nParam3), "param4": int(notify.nParam4),
            })
        except Exception:
            pass

    def _on_frame(self, pointer: Any, _user_data: Any) -> None:
        if not pointer:
            return
        observed_ns = self.clock.now_ns()
        with self._lock:
            deadline = self._armed_at_ns
            should_record = deadline is not None and observed_ns >= deadline
            callback_index = self._callback_index if should_record else -1
            if should_record:
                self._callback_index += 1
        try:
            if not should_record:
                frame = pointer.contents
                bodies = frame.RigidBodies
                first_body = None
                if int(frame.nRigidBodies):
                    body = bodies[0]
                    first_body = [float(body.x), float(body.y), float(body.z)]
                self.on_event("NOKOV_PREVIEW", {
                    "device_frame_index": int(frame.iFrame),
                    "device_timestamp_raw": int(frame.iTimeStamp),
                    "host_receive_mono_ns": observed_ns,
                    "rigid_body_count": int(frame.nRigidBodies),
                    "labeled_marker_count": int(frame.nLabeledMarkers),
                    "first_rigid_body_mm": first_body,
                })
                return
            record = frame_to_record(pointer.contents, callback_index, observed_ns, self.clock.clock_id)
            frame_index = record["device_frame_index"]
            with self._lock:
                previous = self._previous_frame
                if previous is not None:
                    if frame_index > previous + 1:
                        self._frame_gap_count += frame_index - previous - 1
                    elif frame_index == previous:
                        self._duplicate_frame_count += 1
                    elif frame_index < previous:
                        self._frame_regression_count += 1
                self._previous_frame = frame_index
            self.on_record(record)
        except Exception as exc:
            with self._lock:
                self._callback_exception_count += 1
                first_error = self.error is None
                self.error = f"NOKOV frame callback failed: {type(exc).__name__}: {exc}"
            if first_error:
                self.on_event("NOKOV_ERROR", {"error": self.error})

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._callback_index

    @property
    def quality_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "frame_gap_count": self._frame_gap_count,
                "duplicate_frame_count": self._duplicate_frame_count,
                "frame_regression_count": self._frame_regression_count,
                "callback_exception_count": self._callback_exception_count,
            }

    def _uninitialize(self) -> None:
        client = self.client
        self.client = None
        if client is not None:
            try:
                uninitialize = getattr(client, "Uninitialize", None) or getattr(
                    client, "_PySDKClient__Uninitialize", None
                )
                if uninitialize:
                    uninitialize()
            finally:
                # Keep callback objects alive until the native SDK has stopped.
                pass

    def stop(self) -> None:
        self._uninitialize()

