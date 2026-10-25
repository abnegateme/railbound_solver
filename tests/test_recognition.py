from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from railbound.crop_io import CropManifest, CropRecord
from railbound.extraction import CellCrop, CropSpec
from railbound.geometry import CellCoord, PixelPoint
from railbound.recognition import (
    RecognitionOverride,
    RecognitionStatus,
    RecognitionThresholds,
    Template,
    TemplateCatalog,
    TemplateDefinition,
    TemplateRecognizer,
    resolve,
)
from railbound.recognition_io import (
    load_level_draft,
    load_recognition_report,
    load_template_catalog,
    load_template_images,
    recognize_crop_directory,
    save_level_draft,
    save_recognition_report,
    save_template_catalog,
)


def _glyph() -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.arrowedLine(image, (5, 25), (25, 7), (230, 180, 80), 3, cv2.LINE_8, tipLength=0.3)
    return image


def _crop(image: np.ndarray, *, row: int = 0, col: int = 0) -> CellCrop:
    origin = PixelPoint(x=0, y=0)
    return CellCrop(
        cell=CellCoord(row=row, col=col),
        image=image,
        source_corners_px=(origin, origin, origin, origin),
        mask=None,
    )


def _recognizer(
    definitions: tuple[TemplateDefinition, ...],
    images: tuple[np.ndarray, ...],
    *,
    thresholds: RecognitionThresholds | None = None,
) -> TemplateRecognizer:
    catalog = TemplateCatalog(
        recognizer_version="test-template-v1",
        thresholds=thresholds or RecognitionThresholds(),
        templates=definitions,
    )
    templates = tuple(
        Template(definition=definition, image=image)
        for definition, image in zip(definitions, images, strict=True)
    )
    return TemplateRecognizer(catalog, templates)


def test_template_recognition_handles_brightness_variation_and_rotation() -> None:
    image = _glyph()
    recognizer = _recognizer(
        (
            TemplateDefinition(
                id="arrow-reference",
                label="actor.arrow",
                image="arrow.png",
            ),
        ),
        (image,),
    )
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    brightened = np.clip(rotated.astype(np.int16) + 18, 0, 255).astype(np.uint8)

    observation = recognizer.recognize(_crop(brightened))

    assert observation.status is RecognitionStatus.ACCEPTED
    assert observation.candidates[0].label == "actor.arrow"
    assert observation.candidates[0].rotation_degrees == 90
    assert observation.confidence > 0.9


def test_template_recognition_rejects_an_unknown_crop() -> None:
    recognizer = _recognizer(
        (TemplateDefinition(id="arrow-reference", label="actor.arrow", image="arrow.png"),),
        (_glyph(),),
    )
    unknown = np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8)

    observation = recognizer.recognize(_crop(unknown))

    assert observation.status is RecognitionStatus.UNKNOWN
    assert observation.candidates == ()
    assert observation.confidence == 0.0


def test_template_recognition_marks_equal_labels_as_uncertain() -> None:
    image = _glyph()
    recognizer = _recognizer(
        (
            TemplateDefinition(id="first-reference", label="actor.first", image="first.png"),
            TemplateDefinition(id="second-reference", label="actor.second", image="second.png"),
        ),
        (image, image.copy()),
    )

    observation = recognizer.recognize(_crop(image))

    assert observation.status is RecognitionStatus.UNCERTAIN
    assert [candidate.label for candidate in observation.candidates] == [
        "actor.first",
        "actor.second",
    ]


def test_confidence_threshold_keeps_near_match_in_the_review_list() -> None:
    image = _glyph()
    recognizer = _recognizer(
        (TemplateDefinition(id="arrow-reference", label="actor.arrow", image="arrow.png"),),
        (image,),
        thresholds=RecognitionThresholds(min_confidence=0.999),
    )
    brightened = np.clip(image.astype(np.int16) + 18, 0, 255).astype(np.uint8)

    observation = recognizer.recognize(_crop(brightened))

    assert observation.status is RecognitionStatus.UNCERTAIN
    assert observation.confidence < 0.999


def test_report_round_trip_and_resolution_preserve_unknown_cells_for_review(tmp_path: Path) -> None:
    crops_directory = tmp_path / "crops"
    templates_directory = tmp_path / "templates"
    crops_directory.mkdir()
    templates_directory.mkdir()
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    unknown = np.full((16, 16, 3), (0, 0, 255), dtype=np.uint8)
    assert cv2.imwrite(str(crops_directory / "r000_c000.png"), blank)
    assert cv2.imwrite(str(crops_directory / "r001_c000.png"), unknown)
    assert cv2.imwrite(str(templates_directory / "empty.png"), blank)
    catalog_path = templates_directory / "catalog.json"
    catalog = TemplateCatalog(
        recognizer_version="test-template-v1",
        templates=(
            TemplateDefinition(id="empty-reference", label="terrain.empty", image="empty.png"),
        ),
    )
    save_template_catalog(catalog, catalog_path)
    manifest = CropManifest(
        image_width=16,
        image_height=32,
        rows=2,
        cols=1,
        spec=CropSpec(output_size=16),
        crops=(
            CropRecord(cell=CellCoord(row=0, col=0), filename="r000_c000.png"),
            CropRecord(cell=CellCoord(row=1, col=0), filename="r001_c000.png"),
        ),
    )
    reloaded_catalog = load_template_catalog(catalog_path)
    recognizer = TemplateRecognizer(
        reloaded_catalog,
        load_template_images(reloaded_catalog, catalog_path),
    )

    report = recognize_crop_directory(manifest, crops_directory, recognizer)
    report_path = tmp_path / "recognition_report.json"
    save_recognition_report(report, report_path)
    reloaded_report = load_recognition_report(report_path)
    draft = resolve(
        reloaded_report,
        [
            RecognitionOverride(
                cell=CellCoord(row=0, col=0),
                label="terrain.confirmed-empty",
                note="manual audit",
            ),
            RecognitionOverride(
                cell=CellCoord(row=1, col=0),
                label="terrain.manual-review",
                note="red marker",
            ),
        ],
    )
    draft_path = tmp_path / "level_draft.json"
    save_level_draft(draft, draft_path)

    assert reloaded_report.review_cells == (CellCoord(row=1, col=0),)
    assert draft.is_complete
    assert [(cell.label, cell.source) for cell in draft.resolved_cells] == [
        ("terrain.confirmed-empty", "override"),
        ("terrain.manual-review", "override"),
    ]
    assert load_level_draft(draft_path) == draft
