from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from acquisition.visualization.nokov_trajectory import prepare_marker_track
from acquisition.visualization.result_data import load_session_result


class ResultPreviewDataTests(unittest.TestCase):
    def test_nokov_large_missing_point_is_kept_raw_but_cleaned_for_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            (session / "raw").mkdir()
            (session / "derived" / "nokov_host_qpc_v1_test").mkdir(parents=True)
            rows = [
                {"marker_sets": [{"name": "Body0", "markers": [
                    {"marker_index": 0, "position_mm": [10.0, 20.0, 30.0]},
                ]}], "labeled_markers": [{"id": 1, "position_mm": [100.0, 200.0, 300.0]}],
                    "unidentified_markers_mm": [[10.0, 20.0, 30.0]]},
                {"marker_sets": [{"name": "Body0", "markers": [
                    {"marker_index": 0, "position_mm": [9_999_999.0] * 3},
                ]}], "skeletons": [{"id": 0, "segments": [
                    {"id": 1, "position_mm": [4.0, 5.0, 6.0]},
                ]}], "unidentified_markers_mm": [[9_999_999.0] * 3]},
                {"marker_sets": [{"name": "Body0", "markers": [
                    {"marker_index": 0, "position_mm": [12.0, 22.0, 32.0]},
                ]}], "rigid_bodies": [{"id": 1, "position_mm": [7.0, 8.0, 9.0]}],
                    "unidentified_markers_mm": [[12.0, 22.0, 32.0]]},
            ]
            with h5py.File(session / "raw" / "nokov.h5", "w") as raw:
                raw.create_dataset("record_json", data=[json.dumps(row) for row in rows],
                                   dtype=h5py.string_dtype("utf-8"))
                raw.create_dataset("host_receive_mono_ns", data=[100, 200, 300])
            timeline_rel = "derived/nokov_host_qpc_v1_test/nokov.h5"
            with h5py.File(session / timeline_rel, "w") as aligned:
                aligned.create_dataset("time_ns", data=[0, 100, 200])
                aligned.create_dataset("raw_record_index", data=[0, 1, 2])
            (session / "manifest.json").write_text(json.dumps({
                "planned_start_mono_ns": 100,
                "streams": {"nokov": {}},
                "nokov_timeline": {"status": "complete", "path": timeline_rel},
            }), encoding="utf-8")

            result = load_session_result(session)
            self.assertEqual(list(result.nokov_tracks), ["Body0 · 点 0"])
            track = result.nokov_tracks["Body0 · 点 0"]
            self.assertEqual(len(track), 3)
            clean, interpolated, missing = prepare_marker_track(result.nokov_time_s, track)
            self.assertEqual(missing.tolist(), [False, True, False])
            self.assertTrue(np.isnan(clean[1]).all())
            np.testing.assert_allclose(interpolated[1], [11.0, 21.0, 31.0])


if __name__ == "__main__":
    unittest.main()
