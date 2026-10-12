from __future__ import annotations

import unittest

import numpy as np

from acquisition.visualization.nokov_trajectory import gap_runs, prepare_marker_track


class NokovTrajectoryTests(unittest.TestCase):
    def test_large_sentinel_is_hidden_and_interpolated_only_for_display(self):
        times = np.array([0.0, 0.1, 0.4, 0.5])
        raw = np.array([
            [0.0, 10.0, 20.0],
            [9_999_999.0, 9_999_999.0, 9_999_999.0],
            [40.0, 50.0, 60.0],
            [50.0, 60.0, 70.0],
        ])

        clean, interpolated, missing = prepare_marker_track(times, raw)

        self.assertEqual(missing.tolist(), [False, True, False, False])
        self.assertTrue(np.isnan(clean[1]).all())
        np.testing.assert_allclose(interpolated[1], [10.0, 20.0, 30.0])
        self.assertEqual(gap_runs(missing), [(1, 1)])
        self.assertEqual(raw[1, 0], 9_999_999.0)  # input remains untouched

    def test_leading_and_trailing_gaps_use_nearest_valid_value(self):
        times = np.arange(4, dtype=np.float64)
        raw = np.array([
            [9_999_999.0] * 3,
            [1.0, 2.0, 3.0],
            [9_999_999.0] * 3,
            [9_999_999.0] * 3,
        ])

        clean, interpolated, missing = prepare_marker_track(times, raw)

        self.assertTrue(np.isnan(clean[0]).all())
        np.testing.assert_allclose(interpolated[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(interpolated[2:], [[1.0, 2.0, 3.0]] * 2)
        self.assertEqual(gap_runs(missing), [(0, 0), (2, 3)])

    def test_track_with_no_valid_observations_stays_empty(self):
        times = np.arange(3, dtype=np.float64)
        raw = np.full((3, 3), 9_999_999.0)
        clean, interpolated, missing = prepare_marker_track(times, raw)
        self.assertTrue(np.isnan(clean).all())
        self.assertTrue(np.isnan(interpolated).all())
        self.assertTrue(missing.all())


if __name__ == "__main__":
    unittest.main()
