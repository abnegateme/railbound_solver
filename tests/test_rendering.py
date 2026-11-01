from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railbound.geometry import GridGeometry, PixelPoint
from railbound.level_io import load_level
from railbound.rendering import render_grid_overlay, render_level_schematic


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
    assert np.any(overlay[60, 60] != 0)


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


def test_level_schematic_uses_formal_grid_dimensions() -> None:
    level = load_level(Path(__file__).parents[1] / "data" / "example-level.json")

    image = render_level_schematic(level, cell_size=48)

    assert image.shape == (24 + level.rows * 48, 24 + level.cols * 48, 3)
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 3
