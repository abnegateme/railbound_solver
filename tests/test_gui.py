from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from railbound.geometry import GridGeometry, PixelPoint
from railbound.gui import CalibrationWindow


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
