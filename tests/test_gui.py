from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from railbound.geometry import GridGeometry, PixelPoint
from railbound.gui import CalibrationWindow


def _lattice_image() -> np.ndarray:
    image = np.zeros((340, 440, 3), dtype=np.uint8)
    for x in range(60, 361, 50):
        cv2.line(image, (x, 50), (x, 275), (255, 255, 255), 2, cv2.LINE_8)
    for y in range(50, 276, 45):
        cv2.line(image, (60, y), (360, y), (255, 255, 255), 2, cv2.LINE_8)
    return image


def test_interactive_gui_adopts_a_confident_detection_as_editable_calibration() -> None:
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow(automatic_detection=True)
    window._image = _lattice_image()

    window._propose_grid()

    assert application is not None
    assert window._grid is not None
    assert window._grid.calibration.method == "automatic_edge_lattice"
    assert len(window._points) == 4
    assert "Accepted proposal" in window._status.text()
    window.close()


def test_gui_extracts_calibrated_cells_without_writing_files() -> None:
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    window._image = np.full((20, 20, 3), 120, dtype=np.uint8)
    window._grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=2,
        image_width=20,
        image_height=20,
        origin=PixelPoint(x=5, y=5),
        next_col=PixelPoint(x=15, y=5),
        next_row=PixelPoint(x=5, y=15),
    )
    window._points = [
        PixelPoint(x=5, y=5),
        PixelPoint(x=15, y=5),
        PixelPoint(x=5, y=15),
        PixelPoint(x=15, y=15),
    ]
    window._update_extraction_controls()

    window._extract_cells()

    assert application is not None
    assert len(window.extracted_crops) == 4
    assert window._contact_sheet is not None
    assert window._show_crops_button.isEnabled()
    assert "No files were written" in window._crop_summary.text()

    if window._crop_preview is not None:
        window._crop_preview.close()
    window.close()
