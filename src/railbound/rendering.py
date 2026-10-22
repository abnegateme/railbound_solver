"""OpenCV rendering for inspecting a manually calibrated grid."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from railbound.geometry import GridGeometry, PixelPoint

ColorImage = NDArray[np.uint8]

_WHITE = (255, 255, 255)
_ORANGE = (0, 165, 255)
_GREEN = (0, 220, 0)
_RED = (0, 0, 255)
_MAGENTA = (255, 0, 255)


def render_grid_overlay(image: NDArray[np.uint8], grid: GridGeometry) -> ColorImage:
    """Return an annotated copy of an image with the expected grid geometry drawn.

    The result is BGR, has the same height and width as ``image``, and does not
    modify the input array.
    """
    _validate_image_dimensions(image, grid)
    overlay = _as_bgr(image)

    boundary = np.array([_cv_point(point) for point in grid.outer_boundary()], dtype=np.int32)
    cv2.polylines(
        overlay, [boundary], isClosed=True, color=_ORANGE, thickness=2, lineType=cv2.LINE_AA
    )

    origin = _cv_point(grid.origin_px)
    col_end = _cv_point(grid.grid_to_pixel(0, max(grid.cols - 1, 1)))
    row_end = _cv_point(grid.grid_to_pixel(max(grid.rows - 1, 1), 0))
    cv2.arrowedLine(overlay, origin, col_end, _GREEN, 2, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(overlay, origin, row_end, _RED, 2, cv2.LINE_AA, tipLength=0.08)

    for cell, center in grid.cell_centers():
        position = _cv_point(center)
        cv2.circle(overlay, position, 3, _WHITE, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, position, 5, _ORANGE, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"{cell.row},{cell.col}",
            (position[0] + 6, position[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _WHITE,
            1,
            cv2.LINE_AA,
        )

    cv2.drawMarker(overlay, origin, _MAGENTA, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "origin (0,0)",
        (origin[0] + 8, origin[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _MAGENTA,
        1,
        cv2.LINE_AA,
    )

    if not grid.is_geometry_inside_image():
        cv2.putText(
            overlay,
            "WARNING: grid boundary extends outside image",
            (8, max(20, overlay.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _RED,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _validate_image_dimensions(image: NDArray[np.uint8], grid: GridGeometry) -> None:
    if image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale, BGR, or BGRA image array")
    if image.shape[:2] != (grid.image_height, grid.image_width):
        raise ValueError(
            "image dimensions do not match grid metadata: "
            f"expected {(grid.image_height, grid.image_width)}, got {image.shape[:2]}"
        )
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("color images must have three (BGR) or four (BGRA) channels")


def _as_bgr(image: NDArray[np.uint8]) -> ColorImage:
    if image.ndim == 2:
        return cast(ColorImage, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if image.shape[2] == 4:
        return cast(ColorImage, cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    return image.copy()


def _cv_point(point: PixelPoint) -> tuple[int, int]:
    return (int(round(point.x)), int(round(point.y)))
