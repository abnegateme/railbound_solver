"""Normalized, grid-aligned image extraction for calibrated Railbound boards."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Annotated, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from railbound.geometry import CellCoord, GridGeometry, PixelPoint

BgrImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
CropMode = Literal["core", "context"]
MaskMode = Literal["none", "cell"]
Interpolation = Literal["nearest", "linear", "cubic"]

_INTERPOLATION_FLAGS: dict[Interpolation, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}
_SHEET_BACKGROUND = (30, 30, 30)
_SHEET_LABEL_COLOR = (235, 235, 235)


class CropSpec(BaseModel):
    """Settings that make a cell-extraction run reproducible.

    ``core_fraction`` is the side length, in cell units, of a core crop.
    ``context_fraction`` is the side length of a contextual crop; its values
    greater than one intentionally include the neighbouring image area.
    Both variants are rectified into a square BGR image of ``output_size``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_size: Annotated[int, Field(gt=0)] = 96
    core_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    context_fraction: Annotated[float, Field(ge=1.0)] = 1.25
    mode: CropMode = "core"
    mask_mode: MaskMode = "none"
    interpolation: Interpolation = "linear"

    def scale(self) -> float:
        """Return the selected crop's side length in calibrated cell units."""
        if self.mode == "core":
            return self.core_fraction
        return self.context_fraction


@dataclass(frozen=True)
class CellCrop:
    """One normalized cell image and, when requested, its target-cell mask."""

    cell: CellCoord
    image: BgrImage
    source_corners_px: tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]
    mask: GrayImage | None


class CellExtractor:
    """Extract consistently oriented cell images from a calibrated screenshot."""

    def extract(
        self, image: NDArray[np.uint8], grid: GridGeometry, spec: CropSpec
    ) -> list[CellCrop]:
        """Return one normalized crop per in-bounds cell in row-major order.

        The output's top edge always follows decreasing grid-row coordinates,
        and its left edge follows decreasing grid-column coordinates.  That
        convention makes orientation independent of the screenshot's rotation
        or skew.
        """
        _validate_image_dimensions(image, grid)
        source_image = _as_bgr(image)
        scale = spec.scale()
        destination = _destination_corners(spec.output_size)
        mask = _target_cell_mask(spec.output_size, scale) if spec.mask_mode == "cell" else None

        crops: list[CellCrop] = []
        for cell, _ in grid.cell_centers():
            corners = _source_corners(grid, cell, scale)
            transform = cv2.getPerspectiveTransform(
                np.array([_point_tuple(point) for point in corners], dtype=np.float32),
                destination,
            )
            extracted = cast(
                BgrImage,
                cv2.warpPerspective(
                    source_image,
                    transform,
                    (spec.output_size, spec.output_size),
                    flags=_INTERPOLATION_FLAGS[spec.interpolation],
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                ),
            )
            if mask is not None:
                extracted = cast(BgrImage, cv2.bitwise_and(extracted, extracted, mask=mask))
            crops.append(
                CellCrop(
                    cell=cell,
                    image=extracted,
                    source_corners_px=corners,
                    mask=None if mask is None else mask.copy(),
                )
            )
        return crops


def render_crop_contact_sheet(crops: list[CellCrop], *, columns: int | None = None) -> BgrImage:
    """Render a labeled, deterministic contact sheet for extracted cell crops."""
    if not crops:
        raise ValueError("at least one crop is required to render a contact sheet")

    crop_height, crop_width = _validate_crop_images(crops)
    sheet_columns = columns if columns is not None else max(crop.cell.col for crop in crops) + 1
    if sheet_columns <= 0:
        raise ValueError("contact-sheet columns must be positive")

    padding = 4
    label_height = 18
    sheet_rows = ceil(len(crops) / sheet_columns)
    tile_height = crop_height + label_height
    sheet_height = padding + sheet_rows * (tile_height + padding)
    sheet_width = padding + sheet_columns * (crop_width + padding)
    sheet = np.full((sheet_height, sheet_width, 3), _SHEET_BACKGROUND, dtype=np.uint8)

    for index, crop in enumerate(crops):
        row, col = divmod(index, sheet_columns)
        x = padding + col * (crop_width + padding)
        y = padding + row * (tile_height + padding)
        sheet[y : y + crop_height, x : x + crop_width] = crop.image
        cv2.putText(
            sheet,
            f"{crop.cell.row},{crop.cell.col}",
            (x, y + crop_height + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            _SHEET_LABEL_COLOR,
            1,
            cv2.LINE_AA,
        )
    return sheet


def _validate_image_dimensions(image: NDArray[np.uint8], grid: GridGeometry) -> None:
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale, BGR, or BGRA image array")
    if image.shape[:2] != (grid.image_height, grid.image_width):
        raise ValueError(
            "image dimensions do not match grid metadata: "
            f"expected {(grid.image_height, grid.image_width)}, got {image.shape[:2]}"
        )
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("color images must have three (BGR) or four (BGRA) channels")


def _as_bgr(image: NDArray[np.uint8]) -> BgrImage:
    if image.ndim == 2:
        return cast(BgrImage, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if image.shape[2] == 4:
        return cast(BgrImage, cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    return image.copy()


def _source_corners(
    grid: GridGeometry, cell: CellCoord, scale: float
) -> tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]:
    half = scale / 2.0
    return (
        grid.grid_to_pixel(cell.row - half, cell.col - half),
        grid.grid_to_pixel(cell.row - half, cell.col + half),
        grid.grid_to_pixel(cell.row + half, cell.col + half),
        grid.grid_to_pixel(cell.row + half, cell.col - half),
    )


def _destination_corners(output_size: int) -> NDArray[np.float32]:
    end = output_size - 0.5
    return np.array([(-0.5, -0.5), (end, -0.5), (end, end), (-0.5, end)], dtype=np.float32)


def _target_cell_mask(output_size: int, scale: float) -> GrayImage:
    half_cell_width = output_size / (2.0 * scale)
    center = (output_size - 1) / 2.0
    corners = np.array(
        [
            (center - half_cell_width, center - half_cell_width),
            (center + half_cell_width, center - half_cell_width),
            (center + half_cell_width, center + half_cell_width),
            (center - half_cell_width, center + half_cell_width),
        ],
        dtype=np.int32,
    )
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 255, lineType=cv2.LINE_8)
    return mask


def _point_tuple(point: PixelPoint) -> tuple[float, float]:
    return (point.x, point.y)


def _validate_crop_images(crops: list[CellCrop]) -> tuple[int, int]:
    first_image = crops[0].image
    if first_image.ndim != 3 or first_image.shape[2] != 3:
        raise ValueError("crop images must be BGR arrays")
    height, width = first_image.shape[:2]
    if first_image.dtype != np.uint8:
        raise ValueError("crop images must use uint8 pixels")
    if any(crop.image.shape != first_image.shape or crop.image.dtype != np.uint8 for crop in crops):
        raise ValueError("all crop images must have the same BGR shape and uint8 dtype")
    return height, width
