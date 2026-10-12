from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from acquisition.visualization.nokov_trajectory import create_nokov_trajectory_dialog


class NokovTrajectoryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_dialog_fits_available_screen_and_has_minimize_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nokov.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["time", "marker_X_mm", "marker_Y_mm", "marker_Z_mm"])
                writer.writeheader()
                writer.writerow({"time": 0, "marker_X_mm": 1, "marker_Y_mm": 2, "marker_Z_mm": 3})

            dialog = create_nokov_trajectory_dialog(None, path)
            try:
                hints = dialog.windowFlags()
                self.assertTrue(hints & QtCore.Qt.WindowType.WindowMinimizeButtonHint)
                self.assertTrue(hints & QtCore.Qt.WindowType.WindowMaximizeButtonHint)
                available = QtWidgets.QApplication.primaryScreen().availableGeometry()
                self.assertLessEqual(dialog.width(), available.width())
                self.assertLessEqual(dialog.height(), available.height())
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
