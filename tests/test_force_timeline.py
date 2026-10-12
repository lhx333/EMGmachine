import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from acquisition.processing.force_timeline import ALGORITHM, build_force_timeline


class ForceTimelineTests(unittest.TestCase):
    def build(self, session_path: Path, channels: list[int], rtc_times: list[int],
              host_times: list[int], origin: int, valid: list[int] | None = None):
        raw_dir = session_path / "raw"
        raw_dir.mkdir(parents=True)
        count = len(channels)
        with h5py.File(raw_dir / "force.h5", "w") as raw:
            frames = raw.create_group("frames")
            frames.create_dataset("channel", data=np.asarray(channels, dtype=np.uint8))
            frames.create_dataset("seq_id", data=np.arange(count, dtype=np.uint32))
            frames.create_dataset("device_timestamp_us", data=np.asarray(rtc_times, dtype=np.uint64) + 1_000_000)
            frames.create_dataset("device_rtc_timestamp_us", data=np.asarray(rtc_times, dtype=np.uint64))
            frames.create_dataset("device_rtc_raw", data=np.tile(np.arange(8, dtype=np.uint8), (count, 1)))
            frames.create_dataset("device_rtc_valid", data=np.asarray(valid or [1] * count, dtype=np.uint8))
            frames.create_dataset("host_receive_mono_ns", data=np.asarray(host_times, dtype=np.uint64))
            frames.create_dataset("sensor_id", data=np.asarray([b"sensor"] * count))
        return build_force_timeline(session_path, origin)

    def test_maps_each_channel_on_its_own_endpoint_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            origin = 10**16
            report = self.build(
                path,
                channels=[0, 1, 0, 1, 0],
                rtc_times=[100, 100, 200, 150, 300],
                host_times=[origin + 10, origin + 30, origin + 70, origin + 70, origin + 110],
                origin=origin,
            )
            output_path = path / report["path"]
            with h5py.File(output_path, "r") as result:
                # Same RTC across different channels is allowed; each gets its own line.
                np.testing.assert_array_equal(result["time_ns"][:], [10, 30, 60, 70, 110])
                np.testing.assert_array_equal(result["raw_frame_index"][:], np.arange(5))
                self.assertIn("device_rtc_raw", result)
                self.assertEqual(result.attrs["algorithm"], ALGORITHM)
                mapping = json.loads(result.attrs["mapping_json"])
                self.assertEqual(mapping["channels"]["0"]["first_raw_frame_index"], 0)
                self.assertEqual(mapping["channels"]["0"]["last_raw_frame_index"], 4)
                self.assertEqual(mapping["channels"]["1"]["frame_count"], 2)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["frame_count"], 5)

    def test_intermediate_host_jitter_does_not_change_channel_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            origin = 10**16
            report = self.build(
                path,
                channels=[2, 2, 2],
                rtc_times=[100, 150, 200],
                host_times=[origin + 100, origin + 900, origin + 300],
                origin=origin,
            )
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:], [100, 200, 300])
            self.assertEqual(report["mapping"]["device_rtc_regression_count"], 0)
            self.assertEqual(report["mapping"]["channels"]["2"]["host_timestamp_regression_count"], 1)

    def test_reports_but_does_not_segment_regressed_intermediate_rtc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            origin = 10**16
            report = self.build(
                path,
                channels=[0, 0, 0, 0],
                rtc_times=[100, 110, 105, 120],
                host_times=[origin, origin + 8_000_000, origin + 12_000_000, origin + 20_000_000],
                origin=origin,
            )
            with h5py.File(path / report["path"], "r") as result:
                np.testing.assert_array_equal(result["time_ns"][:], [0, 10_000_000, 5_000_000, 20_000_000])
            self.assertEqual(report["mapping"]["device_rtc_regression_count"], 1)

    def test_invalid_rtc_frame_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "RTC decode is invalid"):
                self.build(path, [0, 0], [10, 20], [100, 200], 0, valid=[1, 0])
            self.assertFalse((path / "derived").exists())

    def test_zero_endpoint_span_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "channel 0.*positive spans"):
                self.build(path, [0, 0], [10, 10], [100, 200], 0)
            self.assertFalse((path / "derived").exists())


if __name__ == "__main__":
    unittest.main()
