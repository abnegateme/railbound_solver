from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from railbound.extraction import CropSpec
from railbound.geometry import CellCoord, GridGeometry, PixelPoint
from railbound.gui import CalibrationWindow
from railbound.placement_io import load_placement
from railbound.recognition import LevelDraft, RecognitionSource, ResolvedCell


def _lattice_image() -> np.ndarray:
    image = np.zeros((340, 440, 3), dtype=np.uint8)
    for x in range(60, 361, 50):
        cv2.line(image, (x, 50), (x, 275), (255, 255, 255), 2, cv2.LINE_8)
    for y in range(50, 276, 45):
        cv2.line(image, (60, y), (360, y), (255, 255, 255), 2, cv2.LINE_8)
    return image


def test_interactive_gui_adopts_a_confident_detection_as_editable_calibration() -> None:
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow(automatic_detection=True)
    window._image = _lattice_image()

    window._detect_grid()

    assert application is not None
    assert window._grid is not None
    assert window._grid.calibration.method == "automatic_edge_lattice"
    assert len(window._points) == 4
    assert "Accepted proposal" in window._status.text()
    window.close()


def test_gui_automatically_detects_grid_when_loading_a_screenshot(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    image_path = tmp_path / "lattice.png"
    assert cv2.imwrite(str(image_path), _lattice_image())

    window = CalibrationWindow(image_path=image_path)

    assert application is not None
    assert window._grid is not None
    assert (window._rows.value(), window._cols.value()) == (5, 6)
    assert "Accepted proposal" in window._status.text()
    window.close()


def test_gui_keeps_compact_default_dimensions_when_detection_fails(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    image_path = tmp_path / "blank.png"
    assert cv2.imwrite(str(image_path), np.zeros((200, 300, 3), dtype=np.uint8))

    window = CalibrationWindow(image_path=image_path)

    assert application is not None
    assert window._grid is None
    assert (window._rows.value(), window._cols.value()) == (8, 10)
    assert window._rows.width() == 72
    assert window._cols.width() == 72
    assert "current 8 rows × 10 columns are unchanged" in window._status.text()
    window.close()


def test_gui_extracts_authors_catalog_and_exports_portable_artifacts(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    window._image = np.full((20, 20, 3), 120, dtype=np.uint8)
    window._grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=2,
        image_width=20,
        image_height=20,
        origin=PixelPoint(x=5, y=5),
        next_col=PixelPoint(x=15, y=5),
        next_row=PixelPoint(x=5, y=15),
    )
    window._points = [
        PixelPoint(x=5, y=5),
        PixelPoint(x=15, y=5),
        PixelPoint(x=5, y=15),
        PixelPoint(x=15, y=15),
    ]
    window._update_extraction_controls()

    window._extract_cells()

    assert application is not None
    assert len(window.extracted_crops) == 4
    assert window._contact_sheet is not None
    assert window._show_crops_button.isEnabled()
    assert not window._recognize_button.isEnabled()
    assert "No files were written" in window._crop_summary.text()

    window._template_label_editor.setText("terrain.empty")
    assert window._add_template_button.isEnabled()
    window._add_template_from_crop()
    assert window._recognize_button.isEnabled()
    catalog_path = tmp_path / "catalog" / "catalog.json"
    window._save_template_catalog_to_path(catalog_path)
    window._new_template_catalog()
    window._load_template_catalog_from_path(catalog_path)
    window._recognize_crops()
    report_path = tmp_path / "recognition_report.json"
    draft_path = tmp_path / "level_draft.json"
    export_directory = tmp_path / "crops"
    window._save_recognition_report_to_path(report_path)
    window._save_level_draft_to_path(draft_path)
    window._export_crops_to_directory(export_directory)

    assert (catalog_path.parent / "references" / "terrain.empty-r000-c000.png").is_file()
    assert len(window._catalog_templates) == 1
    assert window._recognition_report is not None
    assert window._level_draft is not None
    assert window._create_level_from_draft_button.isEnabled()
    assert report_path.is_file()
    assert draft_path.is_file()
    assert (export_directory / "manifest.json").is_file()
    assert (export_directory / "contact_sheet.png").is_file()

    if window._crop_preview is not None:
        window._crop_preview.close()
    window.close()


def test_gui_creates_semantic_template_from_saved_recognition_draft(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    draft_path = tmp_path / "level_draft.json"
    window = CalibrationWindow()
    window._image = np.full((20, 20, 3), 120, dtype=np.uint8)
    window._grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=2,
        image_width=20,
        image_height=20,
        origin=PixelPoint(x=5, y=5),
        next_col=PixelPoint(x=15, y=5),
        next_row=PixelPoint(x=5, y=15),
    )
    window._points = [
        PixelPoint(x=5, y=5),
        PixelPoint(x=15, y=5),
        PixelPoint(x=5, y=15),
        PixelPoint(x=15, y=15),
    ]
    window._update_extraction_controls()
    window._extract_cells()
    window._template_label_editor.setText("terrain.empty")
    window._add_template_from_crop()
    window._recognize_crops()
    window._save_level_draft_to_path(draft_path)
    window.close()

    restored = CalibrationWindow()
    restored._load_level_from_path(draft_path)

    assert application is not None
    assert restored._level_draft is not None
    assert restored._create_level_from_draft_button.isEnabled()

    restored._create_level_from_draft()

    template = json.loads(restored._level_editor.toPlainText())
    assert template["rows"] == 2
    assert template["cols"] == 2
    assert len(template["terrain"]) == 4
    assert template["actors"] == []
    assert "actors, buildable cells" in restored._semantic_status.text()
    restored.close()


def test_gui_builds_a_valid_level_from_reviewed_standard_labels() -> None:
    application = QApplication.instance() or QApplication([])
    cells = (
        (0, 0, "actor.train"),
        (0, 1, "terrain.empty"),
        (0, 2, "actor.train"),
        (1, 0, "terrain.empty"),
        (1, 1, "terrain.occupied"),
        (1, 2, "goal.arrival"),
    )
    window = CalibrationWindow()
    window._level_draft = LevelDraft(
        source=RecognitionSource(
            image_width=300,
            image_height=200,
            rows=2,
            cols=3,
            crop_spec=CropSpec(),
        ),
        recognition_report_fingerprint="0" * 64,
        resolved_cells=tuple(
            ResolvedCell(
                cell=CellCoord(row=row, col=col),
                label=label,
                source="override",
            )
            for row, col, label in cells
        ),
        unresolved_cells=(),
    )
    window._refresh_semantic_mapping_form()
    window._update_semantic_controls()
    window._semantic_inventory.setValue(4)
    for selector in window._actor_direction_selectors.values():
        selector.setCurrentIndex(2)

    window._build_level_from_labels()

    assert application is not None
    assert window._level is not None
    assert [actor.id for actor in window._level.actors] == ["train-1", "train-2"]
    assert len(window._level.buildable_cells) == 5
    assert window._level.rail_inventory.total_budget == 4
    assert window._level.win_condition.required_arrivals[0].target_cell == CellCoord(row=1, col=2)
    assert "Built and validated level" in window._semantic_status.text()
    window.close()


def test_primary_gui_replays_and_saves_a_verified_placement(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    repository_root = Path(__file__).parents[1]
    level_path = repository_root / "data" / "example-level.json"
    authored_placement_path = repository_root / "data" / "example-placement.json"
    placement_path = tmp_path / "solution.json"
    window = CalibrationWindow()
    window._load_level_from_path(level_path)
    window._load_placement_from_path(authored_placement_path)
    window._inspect_topology()
    assert "Pieces:" in window._placement_inspection_output.toPlainText()
    window._replay_placement()
    assert "Status: success" in window._placement_inspection_output.toPlainText()

    window._solve_level()

    assert application is not None
    assert window.solver_result is not None
    assert window.solver_result.solved
    assert window._save_solution_button.isEnabled()
    assert "Found a verified solution" in window._solver_status.text()

    window._save_solution_to_path(placement_path)

    assert load_placement(placement_path).level_id == "example-straight-arrival"
    window.close()
