from __future__ import annotations

import pytest

from railbound.calibration import grid_from_four_points
from railbound.geometry import PixelPoint


def test_four_clicks_derive_grid_dimensions() -> None:
    grid = grid_from_four_points(
        image_width=1000,
        image_height=800,
        origin=PixelPoint(x=100, y=150),
        next_col=PixelPoint(x=150, y=150),
        next_row=PixelPoint(x=100, y=200),
        last_cell=PixelPoint(x=550, y=500),
        image_sha256=None,
    )

    assert (grid.rows, grid.cols) == (8, 10)


def test_fourth_click_refines_the_affine_grid() -> None:
    grid = grid_from_four_points(
        image_width=1000,
        image_height=800,
        origin=PixelPoint(x=0, y=0),
        next_col=PixelPoint(x=60, y=0),
        next_row=PixelPoint(x=0, y=60),
        last_cell=PixelPoint(x=250, y=250),
        image_sha256=None,
    )

    fitted_last = grid.grid_to_pixel(4, 4)
    assert grid.calibration.method == "manual_multi_point_affine"
    assert len(grid.calibration.control_points) == 4
    assert abs(fitted_last.x - 250) < abs(240 - 250)
    assert abs(fitted_last.y - 250) < abs(240 - 250)


def test_fourth_click_must_be_near_a_cell_center() -> None:
    with pytest.raises(ValueError, match="too far"):
        grid_from_four_points(
            image_width=1000,
            image_height=800,
            origin=PixelPoint(x=100, y=150),
            next_col=PixelPoint(x=150, y=150),
            next_row=PixelPoint(x=100, y=200),
            last_cell=PixelPoint(x=578, y=500),
            image_sha256=None,
        )


def test_fourth_click_must_be_lower_right_of_origin() -> None:
    with pytest.raises(ValueError, match="below and to the right"):
        grid_from_four_points(
            image_width=1000,
            image_height=800,
            origin=PixelPoint(x=100, y=150),
            next_col=PixelPoint(x=150, y=150),
            next_row=PixelPoint(x=100, y=200),
            last_cell=PixelPoint(x=100, y=150),
            image_sha256=None,
        )
