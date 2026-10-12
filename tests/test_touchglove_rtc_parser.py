from __future__ import annotations

import datetime
import tempfile
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

VENDOR_PYTHON = Path(__file__).resolve().parents[1] / "vendor" / "touchglove" / "python"
sys.path.insert(0, str(VENDOR_PYTHON))

from touch_glove_ptp._fallback import IMAGE_BYTES, TouchGlove, decode_device_rtc_bcd  # noqa: E402
from acquisition.devices.touchglove import _append_frames  # noqa: E402


class _SerialBuffer:
    is_open = True

    def __init__(self, data: bytes):
        self.data = data

    def read(self, _size: int) -> bytes:
        data, self.data = self.data, b""
        return data


class TouchGloveRtcParserTests(unittest.TestCase):
    def test_decodes_device_bcd_calendar_to_utc_microseconds(self):
        raw = bytes.fromhex("26 08 12 10 12 57 02 18")
        expected = int(datetime.datetime(2026, 8, 12, 10, 12, 57, 218000,
                                         tzinfo=datetime.timezone.utc).timestamp() * 1_000_000)
        self.assertEqual(decode_device_rtc_bcd(raw), expected)

    def test_python_serial_parser_preserves_rtc_bytes_and_decoded_time(self):
        raw_rtc = bytes.fromhex("26 08 12 10 12 57 02 18")
        length = IMAGE_BYTES + 2
        packet = (b"\x55\xaa\x01" + length.to_bytes(2, "little") + b"A\x11"
                  + raw_rtc + bytes(IMAGE_BYTES) + b"\x00\xcc\x33")
        self.assertEqual(len(packet), length + 16)

        reader = TouchGlove(auto_open=False, use_c_lib=False)
        reader._is_open = True
        reader.ser = _SerialBuffer(packet)
        frames = list(reader._poll_pure())

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].device_rtc_raw, raw_rtc)
        self.assertEqual(frames[0].device_rtc_timestamp_us, decode_device_rtc_bcd(raw_rtc))
        self.assertEqual(frames[0].image.shape, (192, 192))

    def test_invalid_bcd_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BCD"):
            decode_device_rtc_bcd(bytes.fromhex("2a 08 12 10 12 57 02 18"))

    def test_raw_rtc_fields_append_to_force_h5_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "force.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("frames")
                group.create_dataset("image", shape=(0, 192, 192), maxshape=(None, 192, 192),
                                     chunks=(1, 192, 192), dtype=np.uint8)
                for name, dtype in (("channel", np.uint8), ("seq_id", np.uint32),
                                    ("device_timestamp_us", np.uint64), ("host_receive_mono_ns", np.uint64),
                                    ("device_rtc_timestamp_us", np.uint64), ("device_rtc_valid", np.uint8)):
                    group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1,), dtype=dtype)
                group.create_dataset("device_rtc_raw", shape=(0, 8), maxshape=(None, 8),
                                     chunks=(1, 8), dtype=np.uint8)
                group.create_dataset("sensor_id", shape=(0,), maxshape=(None,), chunks=(1,),
                                     dtype=h5py.string_dtype("utf-8"))
                row = {"channel": 0, "seq_id": 1, "device_timestamp_us": 2,
                       "host_receive_mono_ns": 3, "device_rtc_raw": list(range(8)),
                       "device_rtc_timestamp_us": 4, "device_rtc_valid": 1, "sensor_id": "CH01"}
                _append_frames(group, [np.zeros((192, 192), dtype=np.uint8)], [row])
                np.testing.assert_array_equal(group["device_rtc_raw"][0], np.arange(8))
                self.assertEqual(int(group["device_rtc_timestamp_us"][0]), 4)
                self.assertEqual(int(group["device_rtc_valid"][0]), 1)


if __name__ == "__main__":
    unittest.main()
