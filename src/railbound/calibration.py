"""Helpers for building a grid from interactive calibration selections."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, hypot
from pathlib import Path

from railbound.geometry import CellCoord, GridCalibrationPoint, GridGeometry, PixelPoint

_FINAL_REFERENCE_MAX_ERROR_AXIS_FRACTION = 0.4


@dataclass(frozen=True)
class GridIntersectionPoint:
    """One selected grid-line intersection and its integral boundary indices."""

    row: int
    col: int
    pixel: PixelPoint


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


def grid_from_three_intersections(
    *,
    rows: int,
    cols: int,
    image_width: int,
    image_height: int,
    top_left: PixelPoint,
    next_col: PixelPoint,
    next_row: PixelPoint,
    image_sha256: str | None,
) -> GridGeometry:
    """Build a center-based grid from three adjacent boundary intersections."""
    return grid_from_intersection_points(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        intersections=(
            GridIntersectionPoint(row=0, col=0, pixel=top_left),
            GridIntersectionPoint(row=0, col=1, pixel=next_col),
            GridIntersectionPoint(row=1, col=0, pixel=next_row),
        ),
        image_sha256=image_sha256,
    )


def grid_from_intersection_points(
    *,
    rows: int,
    cols: int,
    image_width: int,
    image_height: int,
    intersections: Sequence[GridIntersectionPoint],
    image_sha256: str | None,
) -> GridGeometry:
    """Fit center-based geometry to three or more selected boundary intersections."""
    points = tuple(intersections)
    if len(points) < 3:
        raise ValueError("at least three grid intersections are required")
    if any(not 0 <= point.row <= rows or not 0 <= point.col <= cols for point in points):
        raise ValueError("grid intersection indices must lie on the board boundary lattice")
    indices = tuple((point.row, point.col) for point in points)
    if len(set(indices)) != len(indices):
        raise ValueError("grid intersections must have unique boundary indices")

    control_points = tuple(
        GridCalibrationPoint(
            cell=CellCoord(row=point.row, col=point.col),
            pixel=point.pixel,
        )
        for point in points
    )
    boundary_grid = grid_from_control_points(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        control_points=control_points,
        image_sha256=image_sha256,
    )
    center_origin = PixelPoint(
        x=(
            boundary_grid.origin_px.x
            + 0.5 * boundary_grid.col_axis_px.x
            + 0.5 * boundary_grid.row_axis_px.x
        ),
        y=(
            boundary_grid.origin_px.y
            + 0.5 * boundary_grid.col_axis_px.y
            + 0.5 * boundary_grid.row_axis_px.y
        ),
    )
    calibration = boundary_grid.calibration.model_copy(
        update={"method": "manual_intersection_affine"}
    )
    return GridGeometry(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        image_sha256=boundary_grid.image_sha256,
        origin_px=center_origin,
        col_axis_px=boundary_grid.col_axis_px,
        row_axis_px=boundary_grid.row_axis_px,
        calibration=calibration,
    )


def grid_from_four_intersections(
    *,
    image_width: int,
    image_height: int,
    top_left: PixelPoint,
    next_col: PixelPoint,
    next_row: PixelPoint,
    bottom_right: PixelPoint,
    image_sha256: str | None,
) -> GridGeometry:
    """Build a grid and derive its dimensions from four boundary intersections."""
    reference = grid_from_three_intersections(
        rows=2,
        cols=2,
        image_width=image_width,
        image_height=image_height,
        top_left=top_left,
        next_col=next_col,
        next_row=next_row,
        image_sha256=image_sha256,
    )
    continuous = reference.pixel_to_grid(bottom_right.x, bottom_right.y)
    rows = floor(continuous.row + 1.0)
    cols = floor(continuous.col + 1.0)
    expected = reference.grid_to_pixel(rows - 0.5, cols - 0.5)
    axis_length = min(
        hypot(reference.col_axis_px.x, reference.col_axis_px.y),
        hypot(reference.row_axis_px.x, reference.row_axis_px.y),
    )
    distance = hypot(bottom_right.x - expected.x, bottom_right.y - expected.y)
    if distance > axis_length * _FINAL_REFERENCE_MAX_ERROR_AXIS_FRACTION:
        raise ValueError("the final selection is too far from a grid intersection")
    if rows < 2 or cols < 2:
        raise ValueError("the final intersection must be below and to the right of the origin")

    return grid_from_intersection_points(
        rows=rows,
        cols=cols,
        image_width=image_width,
        image_height=image_height,
        intersections=(
            GridIntersectionPoint(row=0, col=0, pixel=top_left),
            GridIntersectionPoint(row=0, col=1, pixel=next_col),
            GridIntersectionPoint(row=1, col=0, pixel=next_row),
            GridIntersectionPoint(row=rows, col=cols, pixel=bottom_right),
        ),
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
    if nearest.distance_px > axis_length * _FINAL_REFERENCE_MAX_ERROR_AXIS_FRACTION:
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
