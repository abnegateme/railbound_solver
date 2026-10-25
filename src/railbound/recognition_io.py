"""Persistence and image loading for deterministic recognition artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
from pydantic import TypeAdapter

from railbound.crop_io import CropManifest
from railbound.extraction import BgrImage
from railbound.recognition import (
    LevelDraft,
    RecognitionOverride,
    RecognitionReport,
    Template,
    TemplateCatalog,
    TemplateRecognizer,
)

_OVERRIDES_ADAPTER = TypeAdapter(tuple[RecognitionOverride, ...])


def save_template_catalog(catalog: TemplateCatalog, path: Path) -> None:
    """Serialize a template catalog as stable, reviewable JSON."""
    _save_model(catalog, path)


def load_template_catalog(path: Path) -> TemplateCatalog:
    """Load and validate a template catalog without decoding its image files."""
    return TemplateCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def load_template_images(catalog: TemplateCatalog, catalog_path: Path) -> tuple[Template, ...]:
    """Decode every catalog image relative to its portable catalog path."""
    templates: list[Template] = []
    for definition in catalog.templates:
        image_path = catalog_path.parent / definition.image
        image = _read_bgr_image(image_path, f"template {definition.id!r}")
        templates.append(Template(definition=definition, image=image))
    return tuple(templates)


def recognize_crop_directory(
    manifest: CropManifest,
    crop_directory: Path,
    recognizer: TemplateRecognizer,
) -> RecognitionReport:
    """Decode a manifest's crop PNGs and recognize them in manifest order."""
    images = tuple(
        _read_crop_image(crop_directory / record.filename, manifest.spec.output_size)
        for record in manifest.crops
    )
    return recognizer.report_from_manifest(manifest, images)


def save_recognition_report(report: RecognitionReport, path: Path) -> None:
    """Serialize recognition evidence separately from semantic level data."""
    _save_model(report, path)


def load_recognition_report(path: Path) -> RecognitionReport:
    """Load and validate a portable recognition-evidence report."""
    return RecognitionReport.model_validate_json(path.read_text(encoding="utf-8"))


def save_recognition_overrides(overrides: tuple[RecognitionOverride, ...], path: Path) -> None:
    """Write manual override decisions as a JSON array in supplied order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump() for item in overrides], indent=2) + "\n", encoding="utf-8"
    )


def load_recognition_overrides(path: Path) -> tuple[RecognitionOverride, ...]:
    """Load manual override decisions from a JSON array."""
    return _OVERRIDES_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def save_level_draft(draft: LevelDraft, path: Path) -> None:
    """Serialize a review-state draft; this does not create a semantic Level."""
    _save_model(draft, path)


def load_level_draft(path: Path) -> LevelDraft:
    """Load a resolved/unresolved recognition draft for later manual completion."""
    return LevelDraft.model_validate_json(path.read_text(encoding="utf-8"))


def _save_model(model: TemplateCatalog | RecognitionReport | LevelDraft, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _read_crop_image(path: Path, output_size: int) -> BgrImage:
    image = _read_bgr_image(path, "crop")
    if image.shape[:2] != (output_size, output_size):
        raise ValueError(
            f"crop image {path} has shape {image.shape[:2]}, expected {(output_size, output_size)}"
        )
    return image


def _read_bgr_image(path: Path, description: str) -> BgrImage:
    image = cast(BgrImage | None, cv2.imread(str(path), cv2.IMREAD_COLOR))
    if image is None:
        raise OSError(f"could not read {description} image: {path}")
    return image
