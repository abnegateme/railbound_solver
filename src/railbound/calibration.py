"""Helpers for building a grid from interactive calibration selections."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from math import hypot
from pathlib import Path

from railbound.geometry import CellCoord, GridCalibrationPoint, GridGeometry, PixelPoint

_LAST_CELL_MAX_ERROR_AXIS_FRACTION = 0.4


def image_sha256(path: Path) -> str:
    """Return the SHA-256 digest of an image file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid_from_three_points(
    *,
    rows: int,
    cols: int,
    image_width: int,
    image_height: int,
    origin: PixelPoint,
    next_col: PixelPoint,
    next_row: PixelPoint,
    image_sha256: str | None,
) -> GridGeometry:
    """Build a grid from its first cell and its two adjacent cell centers."""
    return GridGeometry.from_manual_calibration(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        origin=origin,
        next_col=next_col,
        next_row=next_row,
        image_sha256=image_sha256,
    )


def grid_from_control_points(
    *,
    rows: int,
    cols: int,
    image_width: int,
    image_height: int,
    control_points: Sequence[GridCalibrationPoint],
    image_sha256: str | None,
) -> GridGeometry:
    """Fit a grid to three or more known grid-coordinate/pixel pairs."""
    return GridGeometry.from_control_points(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        control_points=control_points,
        image_sha256=image_sha256,
    )


def grid_from_four_points(
    *,
    image_width: int,
    image_height: int,
    origin: PixelPoint,
    next_col: PixelPoint,
    next_row: PixelPoint,
    last_cell: PixelPoint,
    image_sha256: str | None,
) -> GridGeometry:
    """Build a grid, deriving its dimensions from the bottom-right cell center.

    The four selections are centers of ``(0, 0)``, ``(0, 1)``, ``(1, 0)``,
    and the final cell in the grid.  This removes the need to count rows and
    columns outside the calibration UI.
    """
    reference = grid_from_three_points(
        rows=2,
        cols=2,
        image_width=image_width,
        image_height=image_height,
        origin=origin,
        next_col=next_col,
        next_row=next_row,
        image_sha256=image_sha256,
    )
    nearest = reference.nearest_cell(last_cell.x, last_cell.y)
    axis_length = min(
        hypot(reference.col_axis_px.x, reference.col_axis_px.y),
        hypot(reference.row_axis_px.x, reference.row_axis_px.y),
    )
    if nearest.distance_px > axis_length * _LAST_CELL_MAX_ERROR_AXIS_FRACTION:
        raise ValueError("the final cell selection is too far from a cell center")
    if nearest.cell.row < 1 or nearest.cell.col < 1:
        raise ValueError("the final cell must be below and to the right of the origin")

    rows = nearest.cell.row + 1
    cols = nearest.cell.col + 1
    control_points = (
        GridCalibrationPoint(cell=CellCoord(row=0, col=0), pixel=origin),
        GridCalibrationPoint(cell=CellCoord(row=0, col=1), pixel=next_col),
        GridCalibrationPoint(cell=CellCoord(row=1, col=0), pixel=next_row),
        GridCalibrationPoint(cell=CellCoord(row=rows - 1, col=cols - 1), pixel=last_cell),
    )
    return grid_from_control_points(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        control_points=control_points,
        image_sha256=image_sha256,
    )
