"""Affine grid geometry for manually calibrated Railbound screenshots."""

from __future__ import annotations

from collections.abc import Sequence
from math import floor, hypot, isfinite
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
COORDINATE_CONVENTION = "pixels=(x, y); grid=(row, col); origin=center(row=0, col=0)"
MIN_AXIS_LENGTH_PX = 2.0
MIN_AXIS_SINE = 1e-4


class _FrozenModel(BaseModel):
    """Base class for validated, serializable value objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PixelPoint(_FrozenModel):
    """A pixel coordinate or pixel-space vector, represented as ``(x, y)``."""

    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        """Reject non-finite coordinates before they reach numerical operations."""
        if not isfinite(value):
            raise ValueError("coordinates must be finite")
        return value


class GridPoint(_FrozenModel):
    """A continuous grid coordinate, represented as ``(row, col)``."""

    row: float
    col: float

    @field_validator("row", "col")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        """Reject non-finite coordinates before they reach numerical operations."""
        if not isfinite(value):
            raise ValueError("coordinates must be finite")
        return value


class CellCoord(_FrozenModel):
    """An integral semantic grid coordinate, represented as ``(row, col)``."""

    row: int
    col: int


class GridCalibrationPoint(_FrozenModel):
    """A manually selected cell center with its semantic grid coordinate."""

    cell: CellCoord
    pixel: PixelPoint


class GridCalibration(_FrozenModel):
    """Manual control points and provenance for a fitted affine grid."""

    origin: PixelPoint
    next_col: PixelPoint
    next_row: PixelPoint
    control_points: tuple[GridCalibrationPoint, ...] = ()
    method: Literal["manual_three_point_affine", "manual_multi_point_affine"] = (
        "manual_three_point_affine"
    )
    residual_rms_px: Annotated[float, Field(ge=0.0)] = 0.0
    residual_max_px: Annotated[float, Field(ge=0.0)] = 0.0


class NearestCell(_FrozenModel):
    """The rounded cell for a pixel and its distance from that cell's center."""

    cell: CellCoord
    distance_px: Annotated[float, Field(ge=0.0)]
    in_bounds: bool


class GridGeometry(_FrozenModel):
    """A validated affine mapping between grid cell centers and image pixels.

    Pixel coordinates are always ``(x, y)``. Grid coordinates are always
    ``(row, col)``. ``origin_px`` is the center of cell ``(0, 0)`` and the
    two axes move by exactly one adjacent cell.
    """

    schema_version: Literal[1] = 1
    coordinate_convention: Literal[
        "pixels=(x, y); grid=(row, col); origin=center(row=0, col=0)"
    ] = "pixels=(x, y); grid=(row, col); origin=center(row=0, col=0)"
    rows: Annotated[int, Field(gt=0)]
    cols: Annotated[int, Field(gt=0)]
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    image_sha256: str | None = None
    origin_px: PixelPoint
    col_axis_px: PixelPoint
    row_axis_px: PixelPoint
    calibration: GridCalibration

    @field_validator("image_sha256")
    @classmethod
    def valid_image_hash(cls, value: str | None) -> str | None:
        """Accept an optional SHA-256 hexadecimal digest for the source image."""
        if value is None:
            return value
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError("image_sha256 must be a 64-character hexadecimal SHA-256 digest")
        return value.lower()

    @model_validator(mode="after")
    def valid_affine_basis(self) -> GridGeometry:
        """Reject axes that cannot provide a stable inverse transform."""
        col_length = hypot(self.col_axis_px.x, self.col_axis_px.y)
        row_length = hypot(self.row_axis_px.x, self.row_axis_px.y)
        if col_length < MIN_AXIS_LENGTH_PX or row_length < MIN_AXIS_LENGTH_PX:
            raise ValueError(
                f"grid axes must be at least {MIN_AXIS_LENGTH_PX:g} pixels long; "
                f"got column={col_length:.6g}, row={row_length:.6g}"
            )

        determinant = self._determinant()
        if abs(determinant) / (col_length * row_length) < MIN_AXIS_SINE:
            raise ValueError("grid axes are collinear or too close to collinear")
        return self

    @classmethod
    def from_manual_calibration(
        cls,
        *,
        rows: int,
        cols: int,
        image_width: int,
        image_height: int,
        origin: PixelPoint,
        next_col: PixelPoint,
        next_row: PixelPoint,
        image_sha256: str | None = None,
    ) -> GridGeometry:
        """Build geometry from centers of ``(0, 0)``, ``(0, 1)``, and ``(1, 0)``."""
        control_points = (
            GridCalibrationPoint(cell=CellCoord(row=0, col=0), pixel=origin),
            GridCalibrationPoint(cell=CellCoord(row=0, col=1), pixel=next_col),
            GridCalibrationPoint(cell=CellCoord(row=1, col=0), pixel=next_row),
        )
        return cls(
            rows=rows,
            cols=cols,
            image_width=image_width,
            image_height=image_height,
            image_sha256=image_sha256,
            origin_px=origin,
            col_axis_px=PixelPoint(x=next_col.x - origin.x, y=next_col.y - origin.y),
            row_axis_px=PixelPoint(x=next_row.x - origin.x, y=next_row.y - origin.y),
            calibration=GridCalibration(
                origin=origin,
                next_col=next_col,
                next_row=next_row,
                control_points=control_points,
            ),
        )

    @classmethod
    def from_control_points(
        cls,
        *,
        rows: int,
        cols: int,
        image_width: int,
        image_height: int,
        control_points: Sequence[GridCalibrationPoint],
        image_sha256: str | None = None,
    ) -> GridGeometry:
        """Fit an affine grid to three or more manually selected cell centers."""
        points = tuple(control_points)
        if len(points) < 3:
            raise ValueError("at least three control points are required")

        point_by_cell = {(point.cell.row, point.cell.col): point for point in points}
        required_cells = ((0, 0), (0, 1), (1, 0))
        if any(cell not in point_by_cell for cell in required_cells):
            raise ValueError("control points must include cells (0, 0), (0, 1), and (1, 0)")

        design = np.array(
            [[1.0, point.cell.col, point.cell.row] for point in points], dtype=np.float64
        )
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError("control-point grid coordinates must not be collinear")

        pixel_x = np.array([point.pixel.x for point in points], dtype=np.float64)
        pixel_y = np.array([point.pixel.y for point in points], dtype=np.float64)
        x_solution, *_ = np.linalg.lstsq(design, pixel_x, rcond=None)
        y_solution, *_ = np.linalg.lstsq(design, pixel_y, rcond=None)
        predicted_x = design @ x_solution
        predicted_y = design @ y_solution
        errors = np.hypot(predicted_x - pixel_x, predicted_y - pixel_y)

        origin = point_by_cell[(0, 0)].pixel
        next_col = point_by_cell[(0, 1)].pixel
        next_row = point_by_cell[(1, 0)].pixel
        return cls(
            rows=rows,
            cols=cols,
            image_width=image_width,
            image_height=image_height,
            image_sha256=image_sha256,
            origin_px=PixelPoint(x=float(x_solution[0]), y=float(y_solution[0])),
            col_axis_px=PixelPoint(x=float(x_solution[1]), y=float(y_solution[1])),
            row_axis_px=PixelPoint(x=float(x_solution[2]), y=float(y_solution[2])),
            calibration=GridCalibration(
                origin=origin,
                next_col=next_col,
                next_row=next_row,
                control_points=points,
                method="manual_multi_point_affine",
                residual_rms_px=float(np.sqrt(np.mean(np.square(errors)))),
                residual_max_px=float(np.max(errors)),
            ),
        )

    def grid_to_pixel(self, row: float, col: float) -> PixelPoint:
        """Return the pixel center corresponding to continuous ``(row, col)``."""
        return PixelPoint(
            x=self.origin_px.x + col * self.col_axis_px.x + row * self.row_axis_px.x,
            y=self.origin_px.y + col * self.col_axis_px.y + row * self.row_axis_px.y,
        )

    def pixel_to_grid(self, x: float, y: float) -> GridPoint:
        """Invert a pixel position to an unrounded continuous ``(row, col)``."""
        offset_x = x - self.origin_px.x
        offset_y = y - self.origin_px.y
        determinant = self._determinant()
        col = (offset_x * self.row_axis_px.y - offset_y * self.row_axis_px.x) / determinant
        row = (self.col_axis_px.x * offset_y - self.col_axis_px.y * offset_x) / determinant
        return GridPoint(row=row, col=col)

    def round_grid_point(self, point: GridPoint) -> CellCoord:
        """Round a continuous point, resolving exact half ties toward greater indices."""
        return CellCoord(row=_round_half_up(point.row), col=_round_half_up(point.col))

    def is_in_bounds(self, cell: CellCoord) -> bool:
        """Return whether a cell lies within the semantic grid extents."""
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def nearest_cell(self, x: float, y: float) -> NearestCell:
        """Return the rounded coordinate, center distance, and its grid-bound status."""
        point = self.pixel_to_grid(x, y)
        cell = self.round_grid_point(point)
        expected_center = self.grid_to_pixel(cell.row, cell.col)
        return NearestCell(
            cell=cell,
            distance_px=hypot(x - expected_center.x, y - expected_center.y),
            in_bounds=self.is_in_bounds(cell),
        )

    def cell_centers(self) -> list[tuple[CellCoord, PixelPoint]]:
        """Return every expected center in deterministic row-major order."""
        return [
            (CellCoord(row=row, col=col), self.grid_to_pixel(row, col))
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def outer_boundary(self) -> tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]:
        """Return the four corners enclosing the outer half-cells in clockwise order."""
        return (
            self.grid_to_pixel(-0.5, -0.5),
            self.grid_to_pixel(-0.5, self.cols - 0.5),
            self.grid_to_pixel(self.rows - 0.5, self.cols - 0.5),
            self.grid_to_pixel(self.rows - 0.5, -0.5),
        )

    def is_geometry_inside_image(self) -> bool:
        """Return whether the full outer grid boundary lies inside the source image."""
        return all(
            0 <= point.x < self.image_width and 0 <= point.y < self.image_height
            for point in self.outer_boundary()
        )

    def _determinant(self) -> float:
        return self.col_axis_px.x * self.row_axis_px.y - self.col_axis_px.y * self.row_axis_px.x


def _round_half_up(value: float) -> int:
    """Round to the nearest integer with ties deterministically toward positive infinity."""
    return floor(value + 0.5)
