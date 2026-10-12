from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6 import QtCore, QtWidgets

from acquisition.visualization.result_preview import ResultPreviewDialog


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ResultPreviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_nokov_points_are_wrapped_in_a_bounded_scroll_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            dialog = ResultPreviewDialog(Path(tmp))
            try:
                dialog.result = SimpleNamespace(
                    emg_data=np.empty((0, 0)),
                    imu_data=np.empty((0, 0)),
                    force={},
                    nokov_tracks={f"Body0 · 点 {index}": None for index in range(40)},
                )
                dialog._populate_controls()

                self.assertEqual(dialog.nokov_check_layout.count(), 40)
                checked = [name for name, box in dialog.nokov_checks.items() if box.isChecked()]
                self.assertEqual(checked, ["Body0 · 点 0"])
                self.assertLessEqual(dialog.nokov_check_scroll.maximumHeight(), 112)
                self.assertTrue(
                    dialog.windowFlags() & QtCore.Qt.WindowType.WindowMinimizeButtonHint
                )
                available = QtWidgets.QApplication.primaryScreen().availableGeometry()
                self.assertLessEqual(dialog.maximumWidth(), available.width())
                self.assertLessEqual(dialog.maximumHeight(), available.height())
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
