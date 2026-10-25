"""Assisted grid proposals from visible, regularly spaced board boundaries.

The detector deliberately treats Hough lines as *candidate generation*, not
as an authoritative calibration.  It fits an affine lattice to independent
line intersections, scores the regularity and edge support of that lattice,
and leaves the caller to accept only sufficiently confident candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, exp, hypot, pi
from typing import Annotated, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from railbound.geometry import (
    CellCoord,
    GridCalibration,
    GridCalibrationPoint,
    GridGeometry,
    PixelPoint,
)

Image = NDArray[np.uint8]

_ANGLE_TOLERANCE = pi / 18.0
_MIN_BOUNDARIES = 3
_EDGE_DILATION_SIZE = 3


class _FrozenModel(BaseModel):
    """Base class for immutable, validated detection values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GridDetectionHints(_FrozenModel):
    """Optional constraints that narrow automatic grid proposals.

    Supplying known dimensions makes a noisy screenshot substantially easier
    to validate.  Without dimensions, the detector chooses the largest regular
    sequence of at least two cells along each visible grid axis.
    """

    rows: Annotated[int | None, Field(ge=2)] = None
    cols: Annotated[int | None, Field(ge=2)] = None
    min_cell_size_px: Annotated[float, Field(gt=2.0)] = 16.0
    max_cell_size_px: Annotated[float | None, Field(gt=2.0)] = None
    max_candidates: Annotated[int, Field(ge=1, le=20)] = 5
    image_sha256: str | None = None

    @model_validator(mode="after")
    def valid_size_range(self) -> GridDetectionHints:
        """Reject a cell-size range whose upper bound cannot be reached."""
        if self.max_cell_size_px is not None and self.max_cell_size_px < self.min_cell_size_px:
            raise ValueError("max_cell_size_px must be at least min_cell_size_px")
        return self


class CalibrationCandidate(_FrozenModel):
    """One scored, non-authoritative affine grid proposal."""

    grid: GridGeometry
    method: Literal["edge_lattice"] = "edge_lattice"
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    residual_rms_px: Annotated[float, Field(ge=0.0)]
    residual_max_px: Annotated[float, Field(ge=0.0)]
    edge_support: Annotated[float, Field(ge=0.0, le=1.0)]
    row_boundary_count: Annotated[int, Field(ge=_MIN_BOUNDARIES)]
    col_boundary_count: Annotated[int, Field(ge=_MIN_BOUNDARIES)]

    def is_confident(self, minimum_confidence: float) -> bool:
        """Return whether this proposal meets a caller-selected confidence bar."""
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        return self.confidence >= minimum_confidence


@dataclass(frozen=True)
class _Segment:
    """A finite Hough line segment in image coordinates."""

    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        """Return the segment length in pixels."""
        return hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def angle(self) -> float:
        """Return an undirected line angle in the half-open range ``[0, pi)``."""
        return atan2(self.end[1] - self.start[1], self.end[0] - self.start[0]) % pi

    @property
    def midpoint(self) -> tuple[float, float]:
        """Return the midpoint of this finite line segment."""
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)


@dataclass(frozen=True)
class _LineCluster:
    """One Hough-supported infinite line, represented by its normal offset."""

    rho: float
    support: int


@dataclass(frozen=True)
class _LineFamily:
    """Parallel visible grid boundaries with a shared unit normal."""

    normal: tuple[float, float]
    clusters: tuple[_LineCluster, ...]


@dataclass(frozen=True)
class _BoundarySequence:
    """A regular contiguous selection of boundaries from one line family."""

    family: _LineFamily
    lines: tuple[_LineCluster, ...]
    spacing_px: float
    spacing_cv: float


class GridDetector:
    """Propose affine board grids from repeated, high-contrast boundaries."""

    def propose(
        self, image: Image, hints: GridDetectionHints | None = None
    ) -> list[CalibrationCandidate]:
        """Return candidates ordered by descending confidence.

        An empty list is an explicit insufficient-evidence result.  In
        particular, no candidate is synthesized from image dimensions alone;
        callers should retain M1 manual calibration as the fallback.
        """
        resolved_hints = hints or GridDetectionHints()
        gray = _as_gray(image)
        edges = cast(Image, cv2.Canny(gray, 60, 180, apertureSize=3))
        segments = _hough_segments(edges, resolved_hints.min_cell_size_px)
        if len(segments) < 4:
            return []

        families = _line_families(segments, resolved_hints)
        candidates: list[CalibrationCandidate] = []
        for first, second in _perpendicular_pairs(families):
            for first_sequence in _regular_sequences(first, resolved_hints):
                for second_sequence in _regular_sequences(second, resolved_hints):
                    candidate = _candidate_from_sequences(
                        image_shape=gray.shape,
                        edges=edges,
                        row_sequence=first_sequence,
                        col_sequence=second_sequence,
                        hints=resolved_hints,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

                    swapped = _candidate_from_sequences(
                        image_shape=gray.shape,
                        edges=edges,
                        row_sequence=second_sequence,
                        col_sequence=first_sequence,
                        hints=resolved_hints,
                    )
                    if swapped is not None:
                        candidates.append(swapped)

        return _unique_candidates(candidates, resolved_hints.max_candidates)


def render_detection_overlay(image: Image, candidates: list[CalibrationCandidate]) -> Image:
    """Render a compact debug overlay for zero or more grid proposals.

    A candidate need not be accepted for this renderer to be useful: yellow
    means a lower-confidence proposal and green denotes confidence at least
    0.70.  The source image is never modified.
    """
    overlay = _as_bgr(image)
    height, width = overlay.shape[:2]
    for index, candidate in enumerate(candidates):
        grid = candidate.grid
        if (grid.image_height, grid.image_width) != (height, width):
            raise ValueError("candidate grid dimensions do not match the source image")
        color = (0, 220, 0) if candidate.confidence >= 0.70 else (0, 220, 220)
        boundary = np.array([_cv_point(point) for point in grid.outer_boundary()], dtype=np.int32)
        cv2.polylines(
            overlay, [boundary], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA
        )
        for _, center in grid.cell_centers():
            cv2.circle(overlay, _cv_point(center), 2, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            (
                f"#{index + 1}: {grid.rows}x{grid.cols} "
                f"confidence={candidate.confidence:.2f} rms={candidate.residual_rms_px:.2f}px"
            ),
            (8, 24 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if not candidates:
        cv2.putText(
            overlay,
            "No regular grid candidate -- use manual calibration",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _as_gray(image: Image) -> Image:
    """Validate a screenshot and return a grayscale copy suitable for Canny."""
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be grayscale, BGR, or BGRA")
    conversion = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
    return cast(Image, cv2.cvtColor(image, conversion))


def _as_bgr(image: Image) -> Image:
    """Return a BGR copy of a valid image for overlay rendering."""
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.ndim == 2:
        return cast(Image, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if image.shape[2] == 4:
        return cast(Image, cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    return image.copy()


def _hough_segments(edges: Image, min_cell_size_px: float) -> list[_Segment]:
    """Generate long edge segments without treating them as a final answer."""
    height, width = edges.shape[:2]
    minimum_dimension = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=pi / 360.0,
        threshold=max(20, round(minimum_dimension * 0.10)),
        minLineLength=max(round(min_cell_size_px * 1.5), round(minimum_dimension * 0.15)),
        maxLineGap=max(2, round(min_cell_size_px * 0.2)),
    )
    if lines is None:
        return []
    segments: list[_Segment] = []
    for values in lines.reshape(-1, 4):
        start = (float(values[0]), float(values[1]))
        end = (float(values[2]), float(values[3]))
        segment = _Segment(start=start, end=end)
        if segment.length >= min_cell_size_px:
            segments.append(segment)
    return segments


def _line_families(segments: list[_Segment], hints: GridDetectionHints) -> list[_LineFamily]:
    """Cluster segment orientations, then merge duplicate Hough edge lines."""
    bin_count = 90
    histogram = np.zeros(bin_count, dtype=np.float64)
    for segment in segments:
        bin_index = min(bin_count - 1, int(segment.angle / pi * bin_count))
        histogram[bin_index] += segment.length

    peak_indices = sorted(range(bin_count), key=lambda index: float(histogram[index]), reverse=True)
    used_angles: list[float] = []
    families: list[_LineFamily] = []
    for peak in peak_indices:
        if histogram[peak] <= 0:
            break
        angle = (peak + 0.5) * pi / bin_count
        if any(_angle_distance(angle, used) < _ANGLE_TOLERANCE for used in used_angles):
            continue
        matching = [
            segment
            for segment in segments
            if _angle_distance(segment.angle, angle) < _ANGLE_TOLERANCE
        ]
        if len(matching) < 2:
            continue
        normal = _normal_from_segments(matching)
        clusters = _cluster_parallel_lines(matching, normal, hints.min_cell_size_px)
        if len(clusters) >= _MIN_BOUNDARIES:
            families.append(_LineFamily(normal=normal, clusters=clusters))
            used_angles.append(angle)
    return families


def _normal_from_segments(segments: list[_Segment]) -> tuple[float, float]:
    """Return a stable unit normal for nearly parallel, undirected segments."""
    reference = segments[0]
    reference_direction = (
        (reference.end[0] - reference.start[0]) / reference.length,
        (reference.end[1] - reference.start[1]) / reference.length,
    )
    direction_x = 0.0
    direction_y = 0.0
    for segment in segments:
        current = (
            (segment.end[0] - segment.start[0]) / segment.length,
            (segment.end[1] - segment.start[1]) / segment.length,
        )
        aligned = current[0] * reference_direction[0] + current[1] * reference_direction[1]
        sign = 1.0 if aligned >= 0 else -1.0
        direction_x += sign * current[0] * segment.length
        direction_y += sign * current[1] * segment.length
    magnitude = hypot(direction_x, direction_y)
    return (-direction_y / magnitude, direction_x / magnitude)


def _cluster_parallel_lines(
    segments: list[_Segment], normal: tuple[float, float], min_cell_size_px: float
) -> tuple[_LineCluster, ...]:
    """Collapse multiple Hough segments on the same visible board boundary."""
    positioned = sorted(
        (
            normal[0] * segment.midpoint[0] + normal[1] * segment.midpoint[1],
            segment.length,
        )
        for segment in segments
    )
    merge_distance = max(3.0, min_cell_size_px * 0.25)
    groups: list[list[tuple[float, float]]] = []
    for rho, weight in positioned:
        if not groups or rho - _weighted_mean(groups[-1]) > merge_distance:
            groups.append([(rho, weight)])
        else:
            groups[-1].append((rho, weight))
    return tuple(
        _LineCluster(rho=_weighted_mean(group), support=len(group))
        for group in groups
    )


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    """Return a weighted arithmetic mean for non-empty ``(value, weight)`` pairs."""
    total_weight = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total_weight


def _perpendicular_pairs(families: list[_LineFamily]) -> list[tuple[_LineFamily, _LineFamily]]:
    """Return distinct near-perpendicular boundary families in a stable order."""
    pairs: list[tuple[_LineFamily, _LineFamily]] = []
    for first_index, first in enumerate(families):
        first_angle = atan2(-first.normal[0], first.normal[1]) % pi
        for second in families[first_index + 1 :]:
            second_angle = atan2(-second.normal[0], second.normal[1]) % pi
            if abs(_angle_distance(first_angle, second_angle) - pi / 2.0) < _ANGLE_TOLERANCE:
                pairs.append((first, second))
    return pairs


def _regular_sequences(
    family: _LineFamily, hints: GridDetectionHints
) -> list[_BoundarySequence]:
    """Return regular contiguous boundary runs, preferring larger observed grids."""
    requested_counts = {
        cells + 1 for cells in (hints.rows, hints.cols) if cells is not None
    }
    if requested_counts:
        counts = sorted({len(family.clusters), *requested_counts})
    else:
        counts = list(range(_MIN_BOUNDARIES, len(family.clusters) + 1))
    sequences: list[_BoundarySequence] = []
    for count in counts:
        if count < _MIN_BOUNDARIES or count > len(family.clusters):
            continue
        for start in range(len(family.clusters) - count + 1):
            selected = family.clusters[start : start + count]
            differences = np.diff(np.array([line.rho for line in selected], dtype=np.float64))
            spacing = float(np.median(differences))
            if spacing < hints.min_cell_size_px:
                continue
            if hints.max_cell_size_px is not None and spacing > hints.max_cell_size_px:
                continue
            coefficient_of_variation = float(np.std(differences) / spacing)
            if coefficient_of_variation > 0.18:
                continue
            sequences.append(
                _BoundarySequence(
                    family=family,
                    lines=selected,
                    spacing_px=spacing,
                    spacing_cv=coefficient_of_variation,
                )
            )
    return sorted(
        sequences,
        key=lambda sequence: (
            len(sequence.lines),
            -sequence.spacing_cv,
            sum(line.support for line in sequence.lines),
        ),
        reverse=True,
    )[:3]


def _candidate_from_sequences(
    *,
    image_shape: tuple[int, int],
    edges: Image,
    row_sequence: _BoundarySequence,
    col_sequence: _BoundarySequence,
    hints: GridDetectionHints,
) -> CalibrationCandidate | None:
    """Fit and score an affine grid from intersecting, independently found lines."""
    rows = len(row_sequence.lines) - 1
    cols = len(col_sequence.lines) - 1
    if hints.rows is not None and rows != hints.rows:
        return None
    if hints.cols is not None and cols != hints.cols:
        return None

    centers = _cell_centers(row_sequence, col_sequence)
    if centers is None:
        return None
    centers = _orient_centers(centers)
    fitted = _fit_grid(
        centers=centers,
        image_shape=image_shape,
        image_sha256=hints.image_sha256,
    )
    if fitted is None:
        return None
    grid, residual_rms, residual_max = fitted
    if grid.row_axis_px.y <= 0.0 or grid.col_axis_px.x <= 0.0:
        return None
    support = _edge_support(edges, row_sequence, col_sequence)
    regularity = max(0.0, 1.0 - (row_sequence.spacing_cv + col_sequence.spacing_cv) * 3.0)
    residual_scale = min(row_sequence.spacing_px, col_sequence.spacing_px)
    residual_score = exp(-4.0 * residual_rms / residual_scale)
    line_count_score = min(1.0, (rows + cols) / 8.0)
    confidence = (
        0.40 * support
        + 0.30 * regularity
        + 0.20 * residual_score
        + 0.10 * line_count_score
    )
    return CalibrationCandidate(
        grid=grid,
        confidence=confidence,
        residual_rms_px=residual_rms,
        residual_max_px=residual_max,
        edge_support=support,
        row_boundary_count=len(row_sequence.lines),
        col_boundary_count=len(col_sequence.lines),
    )


def _cell_centers(
    row_sequence: _BoundarySequence, col_sequence: _BoundarySequence
) -> list[list[tuple[float, float]]] | None:
    """Intersect midlines between adjacent row and column boundaries."""
    row_midpoints = [
        (first.rho + second.rho) / 2.0
        for first, second in zip(row_sequence.lines, row_sequence.lines[1:], strict=False)
    ]
    col_midpoints = [
        (first.rho + second.rho) / 2.0
        for first, second in zip(col_sequence.lines, col_sequence.lines[1:], strict=False)
    ]
    centers: list[list[tuple[float, float]]] = []
    for row_rho in row_midpoints:
        row: list[tuple[float, float]] = []
        for col_rho in col_midpoints:
            point = _intersection(
                row_sequence.family.normal,
                row_rho,
                col_sequence.family.normal,
                col_rho,
            )
            if point is None:
                return None
            row.append(point)
        centers.append(row)
    return centers


def _intersection(
    first_normal: tuple[float, float],
    first_rho: float,
    second_normal: tuple[float, float],
    second_rho: float,
) -> tuple[float, float] | None:
    """Intersect ``normal · point = rho`` forms of two nonparallel lines."""
    determinant = first_normal[0] * second_normal[1] - first_normal[1] * second_normal[0]
    if abs(determinant) < 0.05:
        return None
    return (
        (first_rho * second_normal[1] - first_normal[1] * second_rho) / determinant,
        (first_normal[0] * second_rho - first_rho * second_normal[0]) / determinant,
    )


def _orient_centers(centers: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Orient a lattice so semantic rows tend down and columns tend right."""
    oriented = centers
    row_vector = (
        oriented[-1][0][0] - oriented[0][0][0],
        oriented[-1][0][1] - oriented[0][0][1],
    )
    if row_vector[1] < 0:
        oriented = list(reversed(oriented))
    col_vector = (
        oriented[0][-1][0] - oriented[0][0][0],
        oriented[0][-1][1] - oriented[0][0][1],
    )
    if col_vector[0] < 0:
        oriented = [list(reversed(row)) for row in oriented]
    return oriented


def _fit_grid(
    *,
    centers: list[list[tuple[float, float]]],
    image_shape: tuple[int, int],
    image_sha256: str | None,
) -> tuple[GridGeometry, float, float] | None:
    """Least-squares fit an affine grid and retain every lattice center as evidence."""
    rows = len(centers)
    cols = len(centers[0])
    design = np.array(
        [[1.0, float(col), float(row)] for row in range(rows) for col in range(cols)],
        dtype=np.float64,
    )
    pixels = np.array(
        [point for center_row in centers for point in center_row],
        dtype=np.float64,
    )
    solution, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
    if rank < 3:
        return None
    predicted = design @ solution
    errors = np.hypot(predicted[:, 0] - pixels[:, 0], predicted[:, 1] - pixels[:, 1])
    controls = tuple(
        GridCalibrationPoint(
            cell=CellCoord(row=row, col=col),
            pixel=PixelPoint(x=centers[row][col][0], y=centers[row][col][1]),
        )
        for row in range(rows)
        for col in range(cols)
    )
    origin = PixelPoint(x=float(solution[0, 0]), y=float(solution[0, 1]))
    col_axis = PixelPoint(x=float(solution[1, 0]), y=float(solution[1, 1]))
    row_axis = PixelPoint(x=float(solution[2, 0]), y=float(solution[2, 1]))
    try:
        grid = GridGeometry(
            rows=rows,
            cols=cols,
            image_width=image_shape[1],
            image_height=image_shape[0],
            image_sha256=image_sha256,
            origin_px=origin,
            col_axis_px=col_axis,
            row_axis_px=row_axis,
            calibration=GridCalibration(
                origin=controls[0].pixel,
                next_col=controls[1].pixel,
                next_row=controls[cols].pixel,
                control_points=controls,
                method="automatic_edge_lattice",
                residual_rms_px=float(np.sqrt(np.mean(np.square(errors)))),
                residual_max_px=float(np.max(errors)),
            ),
        )
    except ValueError:
        return None
    return grid, float(np.sqrt(np.mean(np.square(errors)))), float(np.max(errors))


def _edge_support(
    edges: Image, row_sequence: _BoundarySequence, col_sequence: _BoundarySequence
) -> float:
    """Validate Hough-generated lines against the independent Canny edge map."""
    mask = np.zeros_like(edges)
    row_first, row_last = row_sequence.lines[0].rho, row_sequence.lines[-1].rho
    col_first, col_last = col_sequence.lines[0].rho, col_sequence.lines[-1].rho
    for line in row_sequence.lines:
        start = _intersection(
            row_sequence.family.normal, line.rho, col_sequence.family.normal, col_first
        )
        end = _intersection(
            row_sequence.family.normal, line.rho, col_sequence.family.normal, col_last
        )
        if start is not None and end is not None:
            cv2.line(mask, _cv_point_tuple(start), _cv_point_tuple(end), 255, 1, cv2.LINE_AA)
    for line in col_sequence.lines:
        start = _intersection(
            row_sequence.family.normal, row_first, col_sequence.family.normal, line.rho
        )
        end = _intersection(
            row_sequence.family.normal, row_last, col_sequence.family.normal, line.rho
        )
        if start is not None and end is not None:
            cv2.line(mask, _cv_point_tuple(start), _cv_point_tuple(end), 255, 1, cv2.LINE_AA)
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels == 0:
        return 0.0
    kernel = np.ones((_EDGE_DILATION_SIZE, _EDGE_DILATION_SIZE), dtype=np.uint8)
    nearby_edges = cv2.dilate(edges, kernel)
    return float(np.count_nonzero((mask > 0) & (nearby_edges > 0)) / mask_pixels)


def _unique_candidates(
    candidates: list[CalibrationCandidate], maximum: int
) -> list[CalibrationCandidate]:
    """Sort candidates deterministically and discard near-identical lattice fits."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            -candidate.residual_rms_px,
            candidate.grid.rows * candidate.grid.cols,
        ),
        reverse=True,
    )
    unique: list[CalibrationCandidate] = []
    for candidate in ordered:
        if any(_same_grid(candidate.grid, existing.grid) for existing in unique):
            continue
        unique.append(candidate)
        if len(unique) == maximum:
            break
    return unique


def _same_grid(first: GridGeometry, second: GridGeometry) -> bool:
    """Return whether two candidates describe effectively the same grid."""
    if (first.rows, first.cols) != (second.rows, second.cols):
        return False
    origin_distance = hypot(
        first.origin_px.x - second.origin_px.x,
        first.origin_px.y - second.origin_px.y,
    )
    return origin_distance < 2.0


def _angle_distance(first: float, second: float) -> float:
    """Return the smaller distance between two undirected line angles."""
    difference = abs(first - second) % pi
    return min(difference, pi - difference)


def _cv_point(point: PixelPoint) -> tuple[int, int]:
    """Convert a pixel model to an OpenCV integer point."""
    return (round(point.x), round(point.y))


def _cv_point_tuple(point: tuple[float, float]) -> tuple[int, int]:
    """Convert an unmodelled floating point to an OpenCV integer point."""
    return (round(point[0]), round(point[1]))
