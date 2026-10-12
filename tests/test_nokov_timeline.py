import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from acquisition.processing.nokov_timeline import build_nokov_timeline
from acquisition.devices.nokov import frame_to_record
from acquisition.recording.hdf5_writer import Hdf5Writer
from acquisition.session.session import Session


class NokovTimelineTests(unittest.TestCase):
    def test_frame_serializer_copies_identity_and_xyz_values(self):
        marker_set = SimpleNamespace(szName=b"Hand", nMarkers=1, Markers=[[1.0, 2.0, 3.0]])
        body = SimpleNamespace(ID=7, x=10.0, y=20.0, z=30.0, qx=0.0, qy=0.0,
                               qz=0.0, qw=1.0, MeanError=0.1, params=1, nMarkers=0,
                               MarkerIDs=[], Markers=[], MarkerSizes=[])
        segment = SimpleNamespace(ID=8, x=4.0, y=5.0, z=6.0, qx=0.0, qy=0.0,
                                  qz=0.0, qw=1.0, MeanError=0.2, params=1)
        skeleton = SimpleNamespace(skeletonID=2, nRigidBodies=1, RigidBodyData=[segment])
        frame = SimpleNamespace(iFrame=123, iTimeStamp=456, Timecode=10, TimecodeSubframe=2,
                                params=3, fLatency=0.004, nMarkerSets=1, MocapData=[marker_set],
                                nRigidBodies=1, RigidBodies=[body], nSkeletons=1, Skeletons=[skeleton],
                                nLabeledMarkers=0, LabeledMarkers=[], nOtherMarkers=0, OtherMarkers=[],
                                nAnalogdatas=0, Analogdata=[])
        row = frame_to_record(frame, 4, 1_000_000, "windows_qpc")
        self.assertEqual(row["device_frame_index"], 123)
        self.assertEqual(row["device_timestamp_raw"], 456)
        self.assertEqual(row["marker_sets"][0]["name"], "Hand")
        self.assertEqual(row["marker_sets"][0]["markers"][0]["position_mm"], [1.0, 2.0, 3.0])
        self.assertEqual(row["rigid_bodies"][0]["id"], 7)
        self.assertEqual(row["skeletons"][0]["segments"][0]["id"], 8)

    def test_endpoint_mapping_and_frame_quality_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            origin = 10**16
            writer = Hdf5Writer(session / "raw")
            frames = [10, 11, 13, 13, 12]
            for index, frame in enumerate(frames):
                writer.append("nokov", {
                    "callback_index": index,
                    "device_frame_index": frame,
                    "device_timestamp_raw": 123456 + index,
                    "host_receive_mono_ns": origin + index * 10_000_000,
                    "rigid_bodies": [{"id": 7, "position_mm": [1, 2, 3]}],
                })
            writer.close()
            raw_before = (session / "raw" / "nokov.h5").read_bytes()
            report = build_nokov_timeline(session, origin)
            self.assertEqual(raw_before, (session / "raw" / "nokov.h5").read_bytes())
            self.assertEqual(report["mapping"]["frame_gap_count"], 1)
            self.assertEqual(report["mapping"]["duplicate_frame_count"], 1)
            self.assertEqual(report["mapping"]["frame_regression_count"], 1)
            with h5py.File(session / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:], [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000])
                np.testing.assert_array_equal(result["device_frame_index"][:], frames)
                self.assertIn("iTimeStamp endpoint line", result.attrs["time_source"])
                np.testing.assert_array_equal(result["host_receive_mono_ns"][:],
                                              [origin + index * 10_000_000 for index in range(5)])
            self.assertEqual(json.loads((session / report["path"]).parent.joinpath("report.json").read_text())["status"], "complete")

    def test_equal_device_timestamps_are_spread_by_iframe_between_mapped_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            origin = 10**16
            writer = Hdf5Writer(session / "raw")
            rows = [(100, 10), (101, 11), (101, 12), (101, 14), (103, 15)]
            for index, (device, frame) in enumerate(rows):
                writer.append("nokov", {
                    "callback_index": index,
                    "device_frame_index": frame,
                    "device_timestamp_raw": device,
                    "host_receive_mono_ns": origin + index * 10_000_000,
                    "rigid_bodies": [],
                })
            writer.close()

            report = build_nokov_timeline(session, origin)

            self.assertEqual(report["mapping"]["repeated_timestamp_run_count"], 1)
            self.assertEqual(report["mapping"]["repeated_timestamp_frame_count"], 2)
            with h5py.File(session / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:],
                                              [0, 13_333_333, 20_000_000, 33_333_333, 40_000_000])
                np.testing.assert_array_equal(result["device_timestamp_raw"][:], [100, 101, 101, 101, 103])
                np.testing.assert_array_equal(result["device_frame_index"][:], [10, 11, 12, 14, 15])

    def test_empty_nokov_stream_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            raw_dir = session / "raw"
            raw_dir.mkdir()
            with h5py.File(raw_dir / "nokov.h5", "w") as raw:
                raw.create_dataset("record_json", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
                raw.create_dataset("host_receive_mono_ns", shape=(0,), maxshape=(None,), dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "No NOKOV frames"):
                build_nokov_timeline(session, 10**16)

    def test_session_stop_builds_nokov_endpoint_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session.prepare()
            origin = 10**16
            session._manifest["planned_start_mono_ns"] = origin
            session._manifest["streams"]["nokov"] = {"device_frame_index_field": "iFrame"}
            session.state = "ARMED"
            for index, (frame, device) in enumerate([(20, 1_000), (21, 1_001), (23, 1_003)]):
                session._append_nokov_record({
                    "callback_index": index,
                    "device_frame_index": frame,
                    "device_timestamp_raw": device,
                    "host_receive_mono_ns": origin + (index + 1) * 5_000_000,
                    "timestamp_stage": "qpc_at_nokov_data_callback_entry",
                    "clock_id": "windows_qpc",
                })
            session.stop()
            manifest = json.loads((session.path / "manifest.json").read_text())
            self.assertEqual(manifest["nokov_timeline"]["status"], "complete")
            self.assertEqual(manifest["streams"]["nokov"]["record_count"], 3)
            self.assertTrue((session.path / manifest["nokov_timeline"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
