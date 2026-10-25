from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from railbound.cli import build_parser, main
from railbound.geometry import GridGeometry, PixelPoint
from railbound.grid_io import load_grid, save_grid


def _lattice_image() -> np.ndarray:
    image = np.zeros((340, 440, 3), dtype=np.uint8)
    for x in range(60, 361, 50):
        cv2.line(image, (x, 50), (x, 275), (255, 255, 255), 2, cv2.LINE_8)
    for y in range(50, 276, 45):
        cv2.line(image, (60, y), (360, y), (255, 255, 255), 2, cv2.LINE_8)
    return image


def test_interactive_command_is_a_separate_cli_entrypoint() -> None:
    arguments = build_parser().parse_args(["interactive", "--image", "input.png"])

    assert arguments.command == "interactive"
    assert arguments.image == Path("input.png")


def test_calibrate_command_writes_portable_grid_and_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    grid_path = tmp_path / "artifacts" / "grid.json"
    overlay_path = tmp_path / "artifacts" / "annotated_grid.png"
    assert cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))

    result = main(
        [
            "calibrate",
            "--image",
            str(image_path),
            "--rows",
            "2",
            "--cols",
            "3",
            "--origin",
            "40",
            "40",
            "--next-col",
            "70",
            "40",
            "--next-row",
            "40",
            "70",
            "--grid-output",
            str(grid_path),
            "--overlay-output",
            str(overlay_path),
        ]
    )

    assert result == 0
    assert load_grid(grid_path).image_sha256 is not None
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert overlay.shape == (120, 160, 3)


def test_extract_command_writes_crops_manifest_and_contact_sheet(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    grid_path = tmp_path / "grid.json"
    output_directory = tmp_path / "crops"
    image = np.full((20, 20, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    save_grid(
        GridGeometry.from_manual_calibration(
            rows=2,
            cols=2,
            image_width=20,
            image_height=20,
            origin=PixelPoint(x=5, y=5),
            next_col=PixelPoint(x=15, y=5),
            next_row=PixelPoint(x=5, y=15),
        ),
        grid_path,
    )

    result = main(
        [
            "extract",
            "--image",
            str(image_path),
            "--grid",
            str(grid_path),
            "--output-dir",
            str(output_directory),
            "--mode",
            "context",
            "--mask-mode",
            "cell",
        ]
    )

    assert result == 0
    assert (output_directory / "manifest.json").is_file()
    assert (output_directory / "contact_sheet.png").is_file()
    crop = cv2.imread(str(output_directory / "r001_c001.png"), cv2.IMREAD_COLOR)
    assert crop is not None
    assert crop.shape == (96, 96, 3)


def test_detect_command_writes_a_confident_grid_and_debug_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    grid_path = tmp_path / "artifacts" / "grid.json"
    overlay_path = tmp_path / "artifacts" / "detected_grid.png"
    assert cv2.imwrite(str(image_path), _lattice_image())

    result = main(
        [
            "detect",
            "--image",
            str(image_path),
            "--rows",
            "5",
            "--cols",
            "6",
            "--min-cell-size",
            "30",
            "--grid-output",
            str(grid_path),
            "--overlay-output",
            str(overlay_path),
        ]
    )

    assert result == 0
    assert load_grid(grid_path).calibration.method == "automatic_edge_lattice"
    assert cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) is not None
