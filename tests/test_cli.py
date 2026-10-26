from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from railbound.cli import build_parser, main
from railbound.crop_io import CropManifest, CropRecord, save_crop_manifest
from railbound.extraction import CropSpec
from railbound.geometry import CellCoord, GridGeometry, PixelPoint
from railbound.grid_io import load_grid, save_grid
from railbound.placement_io import load_placement
from railbound.recognition import TemplateCatalog, TemplateDefinition
from railbound.recognition_io import (
    load_level_draft,
    load_recognition_report,
    save_template_catalog,
)


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


def test_recognize_command_writes_report_and_review_draft(tmp_path: Path) -> None:
    crops_directory = tmp_path / "crops"
    templates_directory = tmp_path / "templates"
    crops_directory.mkdir()
    templates_directory.mkdir()
    blank = np.zeros((12, 12, 3), dtype=np.uint8)
    unknown = np.full((12, 12, 3), (0, 0, 255), dtype=np.uint8)
    assert cv2.imwrite(str(crops_directory / "r000_c000.png"), blank)
    assert cv2.imwrite(str(crops_directory / "r000_c001.png"), unknown)
    save_crop_manifest(
        CropManifest(
            image_width=24,
            image_height=12,
            rows=1,
            cols=2,
            spec=CropSpec(output_size=12),
            crops=(
                CropRecord(cell=CellCoord(row=0, col=0), filename="r000_c000.png"),
                CropRecord(cell=CellCoord(row=0, col=1), filename="r000_c001.png"),
            ),
        ),
        crops_directory / "manifest.json",
    )
    assert cv2.imwrite(str(templates_directory / "empty.png"), blank)
    catalog_path = templates_directory / "catalog.json"
    save_template_catalog(
        TemplateCatalog(
            templates=(
                TemplateDefinition(id="empty-reference", label="terrain.empty", image="empty.png"),
            )
        ),
        catalog_path,
    )
    report_path = tmp_path / "recognition_report.json"
    draft_path = tmp_path / "level_draft.json"

    result = main(
        [
            "recognize",
            "--crops-dir",
            str(crops_directory),
            "--template-catalog",
            str(catalog_path),
            "--report-output",
            str(report_path),
            "--draft-output",
            str(draft_path),
        ]
    )

    assert result == 2
    assert load_recognition_report(report_path).review_cells[0].col == 1
    draft = load_level_draft(draft_path)
    assert [(cell.label, cell.source) for cell in draft.resolved_cells] == [
        ("terrain.empty", "recognition")
    ]
    assert draft.unresolved_cells == (CellCoord(row=0, col=1),)
    assert not draft.is_complete


def test_solve_command_writes_a_simulator_verified_placement(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    placement_path = tmp_path / "solutions" / "example-placement.json"

    result = main(
        [
            "solve",
            "--level",
            str(repository_root / "data" / "example-level.json"),
            "--placement-output",
            str(placement_path),
            "--max-placements",
            "10",
        ]
    )

    assert result == 0
    assert load_placement(placement_path).level_id == "example-straight-arrival"


def test_solve_command_returns_no_solution_status_when_its_search_limit_is_reached(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[1]
    placement_path = tmp_path / "solution.json"

    result = main(
        [
            "solve",
            "--level",
            str(repository_root / "data" / "example-level.json"),
            "--placement-output",
            str(placement_path),
            "--max-placements",
            "1",
        ]
    )

    assert result == 2
    assert not placement_path.exists()
