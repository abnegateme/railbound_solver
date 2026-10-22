from __future__ import annotations

import numpy as np
import pytest

from railbound.geometry import GridGeometry, PixelPoint
from railbound.rendering import render_grid_overlay


def test_overlay_retains_source_dimensions_and_marks_all_centers() -> None:
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=3,
        image_width=180,
        image_height=160,
        origin=PixelPoint(x=40, y=40),
        next_col=PixelPoint(x=80, y=40),
        next_row=PixelPoint(x=40, y=80),
    )

    overlay = render_grid_overlay(image, grid)

    assert overlay.shape == image.shape
    assert np.any(overlay != image)
    for _, center in grid.cell_centers():
        x, y = round(center.x), round(center.y)
        assert np.any(overlay[y, x] != 0)


def test_overlay_rejects_a_different_source_size() -> None:
    grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=2,
        image_width=100,
        image_height=100,
        origin=PixelPoint(x=30, y=30),
        next_col=PixelPoint(x=50, y=30),
        next_row=PixelPoint(x=30, y=50),
    )

    with pytest.raises(ValueError, match="dimensions"):
        render_grid_overlay(np.zeros((99, 100, 3), dtype=np.uint8), grid)
