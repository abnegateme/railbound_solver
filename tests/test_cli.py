from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from railbound.cli import build_parser, main
from railbound.crop_io import CropManifest, CropRecord, save_crop_manifest
from railbound.extraction import CropSpec
from railbound.geometry import CellCoord
from railbound.placement_io import load_placement
from railbound.recognition import TemplateCatalog, TemplateDefinition
from railbound.recognition_io import (
    load_level_draft,
    load_recognition_report,
    save_template_catalog,
)


def test_interactive_command_is_a_separate_cli_entrypoint() -> None:
    arguments = build_parser().parse_args(["interactive", "--image", "input.png"])

    assert arguments.command == "interactive"
    assert arguments.image == Path("input.png")


def test_interactive_command_does_not_accept_calibration_output_paths() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["interactive", "--image", "input.png", "--grid-output", "ignored.json"]
        )


@pytest.mark.parametrize("command", ["calibrate", "detect", "extract"])
def test_scripted_image_stage_commands_are_not_available(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


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
