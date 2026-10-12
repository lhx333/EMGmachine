from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from acquisition.session.session import Session


class SessionResponsivenessTests(unittest.TestCase):
    def test_preview_remains_readable_during_slow_timeline_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp))
            session.prepare()
            session._manifest["planned_start_mono_ns"] = 10**16
            session._manifest["streams"]["emg"] = {"sample_rate_hz_configured": 1000}
            session._preview["emg"] = {"latest": [1.0]}
            session.state = "ARMED"

            processing_started = threading.Event()
            release_processing = threading.Event()
            preview_read = threading.Event()

            def slow_timeline(*_args):
                processing_started.set()
                if not release_processing.wait(3):
                    raise TimeoutError("test did not release timeline processing")
                return {"status": "complete", "path": "derived/test/emg.h5"}

            def read_preview():
                self.assertEqual(session.preview["emg"]["latest"], [1.0])
                preview_read.set()

            with patch("acquisition.session.session.build_emg_timeline", side_effect=slow_timeline):
                stop_thread = threading.Thread(target=session.stop, daemon=True)
                stop_thread.start()
                try:
                    self.assertTrue(processing_started.wait(2), "timeline processing did not start")
                    reader_thread = threading.Thread(target=read_preview, daemon=True)
                    reader_thread.start()
                    self.assertTrue(
                        preview_read.wait(0.5),
                        "preview access was blocked while background timeline processing ran",
                    )
                    reader_thread.join(1)
                finally:
                    release_processing.set()
                    stop_thread.join(3)

            self.assertFalse(stop_thread.is_alive(), "session stop did not finish")
            self.assertEqual(session.state, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
