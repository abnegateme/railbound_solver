from __future__ import annotations

import pytest

from railbound.calibration import (
    grid_from_four_intersections,
    grid_from_three_intersections,
)
from railbound.geometry import PixelPoint


def test_three_intersections_are_converted_to_cell_centers() -> None:
    grid = grid_from_three_intersections(
        rows=8,
        cols=10,
        image_width=1000,
        image_height=800,
        top_left=PixelPoint(x=100, y=150),
        next_col=PixelPoint(x=150, y=150),
        next_row=PixelPoint(x=100, y=200),
        image_sha256=None,
    )

    assert grid.origin_px.x == pytest.approx(125)
    assert grid.origin_px.y == pytest.approx(175)
    assert grid.outer_boundary()[0].x == pytest.approx(100)
    assert grid.outer_boundary()[0].y == pytest.approx(150)
    assert grid.calibration.method == "manual_intersection_affine"


def test_four_intersections_derive_grid_dimensions_from_outer_corner() -> None:
    grid = grid_from_four_intersections(
        image_width=1000,
        image_height=800,
        top_left=PixelPoint(x=100, y=150),
        next_col=PixelPoint(x=150, y=150),
        next_row=PixelPoint(x=100, y=200),
        bottom_right=PixelPoint(x=600, y=550),
        image_sha256=None,
    )

    assert (grid.rows, grid.cols) == (8, 10)
    assert grid.outer_boundary()[2].x == pytest.approx(600)
    assert grid.outer_boundary()[2].y == pytest.approx(550)


def test_fourth_intersection_must_be_near_a_grid_crossing() -> None:
    with pytest.raises(ValueError, match="too far"):
        grid_from_four_intersections(
            image_width=1000,
            image_height=800,
            top_left=PixelPoint(x=100, y=150),
            next_col=PixelPoint(x=150, y=150),
            next_row=PixelPoint(x=100, y=200),
            bottom_right=PixelPoint(x=578, y=550),
            image_sha256=None,
        )
