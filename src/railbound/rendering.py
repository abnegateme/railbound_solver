"""OpenCV rendering for inspecting a manually calibrated grid."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from railbound.geometry import CellCoord, GridGeometry, PixelPoint
from railbound.level import Direction, Level, Port
from railbound.topology import RailPlacement, rail_piece_transitions, validate_placement

ColorImage = NDArray[np.uint8]

_WHITE = (255, 255, 255)
_ORANGE = (0, 165, 255)
_RED = (0, 0, 255)
_MAGENTA = (255, 0, 255)
_SCHEMATIC_BACKGROUND = (245, 243, 248)
_SCHEMATIC_GRID = (180, 176, 186)
_SCHEMATIC_BUILDABLE = (232, 225, 243)
_SCHEMATIC_BLOCKED = (95, 91, 100)
_SCHEMATIC_TRACK = (55, 52, 58)
_SCHEMATIC_SOLUTION_TRACK = (190, 92, 54)
_SCHEMATIC_OPEN_CELL = (40, 150, 220)
_SCHEMATIC_GOAL = (70, 150, 80)
_SCHEMATIC_ACTOR = (164, 80, 35)


def render_grid_overlay(image: NDArray[np.uint8], grid: GridGeometry) -> ColorImage:
    """Return an annotated copy of an image with the expected grid geometry drawn.

    The result is BGR, has the same height and width as ``image``, and does not
    modify the input array.
    """
    _validate_image_dimensions(image, grid)
    overlay = _as_bgr(image)

    for row in range(grid.rows + 1):
        grid_row = row - 0.5
        cv2.line(
            overlay,
            _cv_point(grid.grid_to_pixel(grid_row, -0.5)),
            _cv_point(grid.grid_to_pixel(grid_row, grid.cols - 0.5)),
            _ORANGE,
            1,
            cv2.LINE_AA,
        )
    for col in range(grid.cols + 1):
        grid_col = col - 0.5
        cv2.line(
            overlay,
            _cv_point(grid.grid_to_pixel(-0.5, grid_col)),
            _cv_point(grid.grid_to_pixel(grid.rows - 0.5, grid_col)),
            _ORANGE,
            1,
            cv2.LINE_AA,
        )

    boundary = np.array([_cv_point(point) for point in grid.outer_boundary()], dtype=np.int32)
    cv2.polylines(
        overlay, [boundary], isClosed=True, color=_ORANGE, thickness=2, lineType=cv2.LINE_AA
    )

    origin = _cv_point(grid.origin_px)
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


def render_level_schematic(level: Level, *, cell_size: int = 72) -> ColorImage:
    """Render a compact graph-oriented view of one validated semantic level."""
    if cell_size < 24:
        raise ValueError("schematic cell size must be at least 24 pixels")
    padding = 12
    height = padding * 2 + level.rows * cell_size
    width = padding * 2 + level.cols * cell_size
    image = np.full((height, width, 3), _SCHEMATIC_BACKGROUND, dtype=np.uint8)
    buildable = {(cell.row, cell.col) for cell in level.buildable_cells}
    fixed_track_cells = {(track.cell.row, track.cell.col) for track in level.fixed_tracks}
    blocked = {
        (terrain.cell.row, terrain.cell.col)
        for terrain in level.terrain
        if (terrain.cell.row, terrain.cell.col) not in buildable
        and (terrain.cell.row, terrain.cell.col) not in fixed_track_cells
    }
    for row in range(level.rows):
        for col in range(level.cols):
            top_left = (padding + col * cell_size, padding + row * cell_size)
            bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)
            if (row, col) in blocked:
                cv2.rectangle(image, top_left, bottom_right, _SCHEMATIC_BLOCKED, thickness=-1)
            elif (row, col) in buildable:
                cv2.rectangle(image, top_left, bottom_right, _SCHEMATIC_BUILDABLE, thickness=-1)
            cv2.rectangle(image, top_left, bottom_right, _SCHEMATIC_GRID, thickness=1)

    for track in level.fixed_tracks:
        center = _schematic_center(track.cell, padding, cell_size)
        for port in track.ports:
            cv2.line(
                image,
                center,
                _schematic_port(track.cell, port, padding, cell_size),
                _SCHEMATIC_TRACK,
                5,
                cv2.LINE_AA,
            )

    _draw_schematic_entities(image, level, padding, cell_size)
    return image


def render_placement_schematic(
    level: Level,
    placement: RailPlacement,
    *,
    cell_size: int = 72,
    open_cells: Iterable[CellCoord] = (),
) -> ColorImage:
    """Render assigned rails and unresolved selected cells for one search state."""
    validate_placement(level, placement)
    image = render_level_schematic(level, cell_size=cell_size)
    padding = 12
    pending = tuple(open_cells)
    unknown = set(pending) - set(level.buildable_cells)
    if unknown:
        raise ValueError("open search cells must belong to the buildable grid")
    inset = max(4, cell_size // 10)
    for cell in pending:
        top_left = (
            padding + cell.col * cell_size + inset,
            padding + cell.row * cell_size + inset,
        )
        bottom_right = (
            padding + (cell.col + 1) * cell_size - inset,
            padding + (cell.row + 1) * cell_size - inset,
        )
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            _SCHEMATIC_OPEN_CELL,
            3,
            cv2.LINE_AA,
        )
    for rail in placement.rails:
        center = _schematic_center(rail.cell, padding, cell_size)
        ports = {
            port
            for transition in rail_piece_transitions(rail.piece_kind, rail.orientation)
            for port in (transition.entry_port, transition.exit_port)
        }
        for port in ports:
            cv2.line(
                image,
                center,
                _schematic_port(rail.cell, port, padding, cell_size),
                _SCHEMATIC_SOLUTION_TRACK,
                7,
                cv2.LINE_AA,
            )
        cv2.circle(image, center, 5, _SCHEMATIC_SOLUTION_TRACK, thickness=-1, lineType=cv2.LINE_AA)
    _draw_schematic_entities(image, level, padding, cell_size)
    return image


def _draw_schematic_entities(
    image: ColorImage,
    level: Level,
    padding: int,
    cell_size: int,
) -> None:
    goal_cells = {arrival.target_cell for arrival in level.win_condition.required_arrivals}
    for cell in goal_cells:
        cv2.circle(
            image,
            _schematic_center(cell, padding, cell_size),
            max(7, cell_size // 6),
            _SCHEMATIC_GOAL,
            3,
            cv2.LINE_AA,
        )
    for actor in level.actors:
        center = _schematic_center(actor.cell, padding, cell_size)
        cv2.fillConvexPoly(
            image,
            _actor_triangle(center, actor.direction, max(10, cell_size // 5)),
            _SCHEMATIC_ACTOR,
            lineType=cv2.LINE_AA,
        )

    for cell, label in (
        *((item.cell, "S") for item in level.switches),
        *((item.cell, "B") for item in level.buttons),
        *((item.cell, "G") for item in level.gates),
        *((item.cell, "T") for item in level.tunnel_endpoints),
    ):
        center = _schematic_center(cell, padding, cell_size)
        cv2.putText(
            image,
            label,
            (center[0] - 8, center[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _MAGENTA,
            2,
            cv2.LINE_AA,
        )


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


def _schematic_center(cell: CellCoord, padding: int, cell_size: int) -> tuple[int, int]:
    return (
        padding + cell.col * cell_size + cell_size // 2,
        padding + cell.row * cell_size + cell_size // 2,
    )


def _schematic_port(cell: CellCoord, port: Port, padding: int, cell_size: int) -> tuple[int, int]:
    center_x, center_y = _schematic_center(cell, padding, cell_size)
    half = cell_size // 2
    offsets = {
        Port.NORTH: (0, -half),
        Port.EAST: (half, 0),
        Port.SOUTH: (0, half),
        Port.WEST: (-half, 0),
    }
    offset_x, offset_y = offsets[port]
    return center_x + offset_x, center_y + offset_y


def _actor_triangle(
    center: tuple[int, int], direction: Direction, radius: int
) -> NDArray[np.int32]:
    x, y = center
    if direction is Direction.NORTH:
        points = ((x, y - radius), (x - radius, y + radius), (x + radius, y + radius))
    elif direction is Direction.EAST:
        points = ((x + radius, y), (x - radius, y - radius), (x - radius, y + radius))
    elif direction is Direction.SOUTH:
        points = ((x, y + radius), (x - radius, y - radius), (x + radius, y - radius))
    else:
        points = ((x - radius, y), (x + radius, y - radius), (x + radius, y + radius))
    return np.array(points, dtype=np.int32)
