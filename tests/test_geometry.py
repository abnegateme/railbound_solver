from __future__ import annotations

from math import isclose

import pytest
from pydantic import ValidationError

from railbound.geometry import CellCoord, GridGeometry, PixelPoint


@pytest.fixture
def rectilinear_grid() -> GridGeometry:
    return GridGeometry.from_manual_calibration(
        rows=3,
        cols=4,
        image_width=400,
        image_height=300,
        origin=PixelPoint(x=100, y=80),
        next_col=PixelPoint(x=140, y=80),
        next_row=PixelPoint(x=100, y=130),
    )


def test_generates_all_expected_centers(rectilinear_grid: GridGeometry) -> None:
    centers = rectilinear_grid.cell_centers()

    assert len(centers) == 12
    assert centers[0] == (CellCoord(row=0, col=0), PixelPoint(x=100, y=80))
    assert centers[-1] == (CellCoord(row=2, col=3), PixelPoint(x=220, y=180))


def test_skewed_grid_round_trips_continuous_coordinates() -> None:
    grid = GridGeometry.from_manual_calibration(
        rows=6,
        cols=5,
        image_width=1000,
        image_height=800,
        origin=PixelPoint(x=320.5, y=140.25),
        next_col=PixelPoint(x=361.75, y=147.5),
        next_row=PixelPoint(x=313.0, y=193.25),
    )

    pixel = grid.grid_to_pixel(3.25, 1.75)
    restored = grid.pixel_to_grid(pixel.x, pixel.y)

    assert isclose(restored.row, 3.25, abs_tol=1e-10)
    assert isclose(restored.col, 1.75, abs_tol=1e-10)


def test_negative_axis_direction_is_supported() -> None:
    grid = GridGeometry.from_manual_calibration(
        rows=3,
        cols=3,
        image_width=500,
        image_height=400,
        origin=PixelPoint(x=300, y=300),
        next_col=PixelPoint(x=250, y=285),
        next_row=PixelPoint(x=315, y=245),
    )

    assert grid.grid_to_pixel(2, 2) == PixelPoint(x=230, y=160)
    assert grid.nearest_cell(230, 160).cell == CellCoord(row=2, col=2)


@pytest.mark.parametrize(
    ("next_col", "next_row"),
    [
        (PixelPoint(x=30, y=10), PixelPoint(x=50, y=10)),
        (PixelPoint(x=11, y=10), PixelPoint(x=10, y=11)),
    ],
)
def test_collinear_or_too_short_calibration_is_rejected(
    next_col: PixelPoint, next_row: PixelPoint
) -> None:
    with pytest.raises(ValidationError):
        GridGeometry.from_manual_calibration(
            rows=2,
            cols=2,
            image_width=100,
            image_height=100,
            origin=PixelPoint(x=10, y=10),
            next_col=next_col,
            next_row=next_row,
        )


def test_nearest_cell_uses_documented_half_up_rounding(rectilinear_grid: GridGeometry) -> None:
    near_half_column = rectilinear_grid.grid_to_pixel(0.49, 0.5)
    exactly_half_column = rectilinear_grid.grid_to_pixel(0.0, 0.5)
    below_origin = rectilinear_grid.grid_to_pixel(-0.51, -0.51)

    assert rectilinear_grid.nearest_cell(near_half_column.x, near_half_column.y).cell == CellCoord(
        row=0, col=1
    )
    assert rectilinear_grid.nearest_cell(
        exactly_half_column.x, exactly_half_column.y
    ).cell == CellCoord(row=0, col=1)
    nearest = rectilinear_grid.nearest_cell(below_origin.x, below_origin.y)
    assert nearest.cell == CellCoord(row=-1, col=-1)
    assert not nearest.in_bounds


def test_out_of_bounds_pixel_reports_distance_and_bounds(rectilinear_grid: GridGeometry) -> None:
    pixel = rectilinear_grid.grid_to_pixel(4, 5)
    nearest = rectilinear_grid.nearest_cell(pixel.x, pixel.y)

    assert nearest.cell == CellCoord(row=4, col=5)
    assert nearest.distance_px == 0
    assert not nearest.in_bounds
