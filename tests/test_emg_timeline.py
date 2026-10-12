import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from acquisition.processing.emg_timeline import build_emg_timeline
from acquisition.recording.hdf5_writer import Hdf5Writer
from acquisition.session.session import Session


def packet(device, host, rate=1000, count=3):
    return {"sensor_type": "emg", "device_timestamp_raw": device, "host_receive_mono_ns": host,
            "sample_rate_hz": rate, "sample_count": count,
            "data": [[i + channel for channel in range(9)] for i in range(count)]}


class EmgTimelineTests(unittest.TestCase):
    def build(self, path, rows, origin=10**16, rate=1000):
        writer = Hdf5Writer(path / "raw")
        for row in rows:
            writer.append("emg", row)
        writer.close()
        original = (path / "raw" / "emg.h5").read_bytes()
        report = build_emg_timeline(path, origin, rate)
        self.assertEqual(original, (path / "raw" / "emg.h5").read_bytes())
        return report

    def test_live_emg_preview_keeps_rolling_samples_with_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session._append_emg_record(packet(4, 10**16, count=3))
            preview = session.preview["emg"]

            self.assertEqual(preview["sequence_start"], 0)
            self.assertEqual(preview["samples"], packet(4, 10**16, count=3)["data"])
            self.assertEqual(preview["latest"], [2 + channel for channel in range(9)])

    def test_user_line_example_and_equal_timestamp_across_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            origin = 10**16
            rows = [packet(4, origin + 410_000_000, count=1),
                    packet(5, origin + 520_000_000), packet(5, origin + 530_000_000),
                    packet(6, origin + 610_000_000, count=1)]
            report = self.build(path, rows, origin)
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:],
                    np.array([410, 510, 511, 512, 513, 514, 515, 610]) * 1_000_000)
                np.testing.assert_array_equal(result["timestamp_group_offset"][:], [0,0,1,2,3,4,5,0])

    def test_endpoint_mapping_preserves_gap_and_packet_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            origin = 10**16
            rows = [packet(100, origin + 2_000_000), packet(103, origin + 19_000_000),
                    packet(110, origin + 22_000_000)]
            report = self.build(path, rows)
            with h5py.File(path / report["path"], "r") as result:
                # Middle callback jitter does not affect device endpoint interpolation.
                np.testing.assert_array_equal(result["time_ns"][:],
                    [2_000_000, 3_000_000, 4_000_000, 8_000_000, 9_000_000, 10_000_000,
                     22_000_000, 23_000_000, 24_000_000])
                np.testing.assert_array_equal(result["raw_record_index"][:], [0]*3 + [1]*3 + [2]*3)
                np.testing.assert_array_equal(result["packet_offset"][:], [0, 1, 2]*3)
                np.testing.assert_array_equal(result["data"][:], np.concatenate([r["data"] for r in rows]))
                self.assertNotIn("time_valid", result)
            second = build_emg_timeline(path, origin, 1000)
            self.assertNotEqual(second["path"], report["path"])

    def test_nonconsecutive_equal_timestamps_do_not_reuse_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            base = 10**16
            rows = [packet(d, base + i * 100_000_000)
                    for i, d in enumerate((4, 5, 6, 5, 7))]
            report = self.build(path, rows)
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["timestamp_group_offset"][9:12], [0,1,2])
                np.testing.assert_array_equal(result["time_ns"][9:12], result["time_ns"][3:6])
                self.assertNotIn("segment_id", result)
            self.assertEqual(report["mapping"]["repeated_device_timestamp_count"], 0)

    def test_rate_mismatch_fails_instead_of_creating_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "fixed session configuration"):
                self.build(path, [packet(1, 10**16), packet(3, 10**16 + 4_000_000, 500)])
            self.assertFalse((path / "derived").exists())

    def test_fixed_500_hz_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            report = self.build(path, [packet(1, 10**16, 500),
                                      packet(3, 10**16 + 10_000_000, 500)], rate=500)
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(np.diff(result["time_ns"][:3]), [2_000_000]*2)

    def test_device_regression_with_increasing_qpc_keeps_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            base = 10**16
            rows = [packet(d, base + h * 1_000_000, count=1)
                    for d, h in ((100, 0), (110, 8), (105, 12), (120, 20))]
            report = self.build(path, rows)
            self.assertEqual(report["status"], "complete")
            self.assertNotIn("segments", report)
            self.assertEqual(report["mapping"]["device_timestamp_regression_count"], 1)
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:], [0, 10_000_000, 5_000_000, 20_000_000])
                self.assertNotIn("segment_id", result)
                self.assertNotIn("time_valid", result)
                self.assertTrue(json.loads(result["packet_json"].asstr()[2])["device_timestamp_regressed"])

    def test_intermediate_host_regression_does_not_change_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            base = 10**16
            rows = [packet(d, base + h * 1_000_000, count=1)
                    for d, h in ((100, 0), (110, 15), (115, 12), (120, 20))]
            report = self.build(path, rows)
            self.assertNotIn("segments", report)
            self.assertEqual(report["mapping"]["host_timestamp_regression_count"], 1)
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:], [0, 10_000_000, 15_000_000, 20_000_000])

    def test_session_stop_automatically_builds_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session.prepare()
            base = 10**16
            session._manifest["planned_start_mono_ns"] = base
            session._manifest["streams"]["emg"] = {"sample_rate_hz_configured": 1000}
            session.state = "ARMED"
            for row in [packet(1, base + 2_000_000), packet(4, base + 5_000_000)]:
                session._append_emg_record(row)
            session.stop()
            manifest = json.loads((session.path / "manifest.json").read_text())
            self.assertEqual(manifest["emg_timeline"]["status"], "complete")
            self.assertTrue((session.path / manifest["emg_timeline"]["path"]).exists())

    def test_pre_record_emg_preview_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session.prepare()
            row = packet(1, 10**16)
            session._append_emg_record(row)
            self.assertIn("emg", session.preview)
            self.assertEqual(session.sample_count, 0)
            self.assertFalse((session.path / "raw" / "emg.h5").exists())
            session.stop()

    def test_constant_device_clock_does_not_fabricate_valid_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "positive spans"):
                self.build(path, [packet(7, 10**16), packet(7, 10**16 + 10_000_000)])
            self.assertFalse((path / "derived").exists())

    def test_processing_failure_preserves_raw_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session.prepare()
            session._manifest["planned_start_mono_ns"] = 10**16
            session._manifest["streams"]["emg"] = {"sample_rate_hz_configured": 1000}
            session.state = "ARMED"
            session._append_emg_record(packet(7, 10**16))
            session.stop()
            manifest = json.loads((session.path / "manifest.json").read_text())
            self.assertEqual(manifest["emg_timeline"]["status"], "failed")
            self.assertTrue((session.path / "raw" / "emg.h5").exists())
            self.assertFalse((session.path / "derived").exists())



if __name__ == "__main__":
    unittest.main()
