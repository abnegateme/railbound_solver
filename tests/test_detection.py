from __future__ import annotations

import cv2
import numpy as np

from railbound.detection import GridDetectionHints, GridDetector, render_detection_overlay


def _lattice_image() -> np.ndarray:
    image = np.zeros((340, 440, 3), dtype=np.uint8)
    for x in range(60, 361, 50):
        cv2.line(image, (x, 50), (x, 275), (255, 255, 255), 2, cv2.LINE_8)
    for y in range(50, 276, 45):
        cv2.line(image, (60, y), (360, y), (255, 255, 255), 2, cv2.LINE_8)
    return image


def test_detector_fits_a_regular_edge_lattice() -> None:
    image = _lattice_image()

    candidates = GridDetector().propose(
        image,
        GridDetectionHints(rows=5, cols=6, min_cell_size_px=30),
    )

    assert candidates
    candidate = candidates[0]
    assert candidate.grid.rows == 5
    assert candidate.grid.cols == 6
    assert candidate.confidence >= 0.70
    center = candidate.grid.grid_to_pixel(2, 3)
    assert abs(center.x - 235) <= 2
    assert abs(center.y - 162.5) <= 2
    assert candidate.residual_rms_px <= 1.0


def test_detector_returns_no_candidate_for_a_negative_image() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)

    candidates = GridDetector().propose(image)

    assert candidates == []


def test_detection_overlay_preserves_image_dimensions_and_marks_candidate() -> None:
    image = _lattice_image()
    candidate = GridDetector().propose(
        image,
        GridDetectionHints(rows=5, cols=6, min_cell_size_px=30),
    )[0]

    overlay = render_detection_overlay(image, [candidate])

    assert overlay.shape == image.shape
    assert np.any(overlay != image)
    assert np.array_equal(image, _lattice_image())
