import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from acquisition.processing.realsense_timeline import build_realsense_timeline
from acquisition.recording.hdf5_writer import Hdf5Writer
from acquisition.session.session import Session


class RealSenseTimelineTests(unittest.TestCase):
    def test_writer_summarizes_repeated_serialization_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = Hdf5Writer(Path(tmp))
            record = {"host_receive_mono_ns": 1, "payload": b"not-json"}
            writer.append("broken", record)
            writer.append("broken", record)

            with self.assertRaisesRegex(RuntimeError, r"bytes is not JSON serializable.*×2"):
                writer.close()

    def test_preview_jpeg_stays_in_preview_but_is_not_sent_to_json_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session._writer = Hdf5Writer(Path(tmp) / "raw")
            session._record_after_ns = 1
            record = {
                "frame_index": 0,
                "camera_frame_number": 17,
                "camera_timestamp_ms": 3.5,
                "camera_timestamp_domain": "hardware_clock",
                "host_receive_mono_ns": 2,
                "preview_jpeg": b"jpeg-preview",
            }

            session._append_realsense_record(record)
            session._writer.close()

            self.assertEqual(session._preview["realsense_rgb"]["preview_jpeg"], b"jpeg-preview")
            with h5py.File(Path(tmp) / "raw" / "realsense_rgb.h5", "r") as raw:
                stored = json.loads(raw["record_json"].asstr()[0])
            self.assertNotIn("preview_jpeg", stored)
            self.assertEqual(stored["camera_frame_number"], 17)

    def test_frames_are_indexed_on_session_qpc_and_camera_domain_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            raw = Hdf5Writer(session / "raw")
            origin = 10**16
            records = [
                {"frame_index": 0, "camera_frame_number": 100, "camera_timestamp_ms": 12.5,
                 "camera_timestamp_domain": "hardware_clock", "host_receive_mono_ns": origin + 10_000_000},
                {"frame_index": 1, "camera_frame_number": 101, "camera_timestamp_ms": 45.8,
                 "camera_timestamp_domain": "hardware_clock", "host_receive_mono_ns": origin + 43_000_000},
            ]
            for record in records:
                raw.append("realsense_rgb", record)
            raw.close()

            report = build_realsense_timeline(session, origin)
            with h5py.File(session / report["path"], "r") as aligned:
                np.testing.assert_array_equal(aligned["time_ns"][:], [10_000_000, 43_000_000])
                np.testing.assert_array_equal(aligned["camera_frame_number"][:], [100, 101])
                np.testing.assert_array_equal(aligned["camera_timestamp_ms"][:], [12.5, 45.8])
                self.assertEqual(aligned["camera_timestamp_domain"].asstr()[:].tolist(),
                                 ["hardware_clock", "hardware_clock"])
                self.assertIn("not exposure time", report["limits"])
            self.assertEqual(json.loads((session / report["path"]).parent.joinpath("report.json").read_text())["status"],
                             "complete")


if __name__ == "__main__":
    unittest.main()
