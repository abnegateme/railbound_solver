"""Deterministic, reviewable recognition of normalized Railbound cell crops.

Recognition is deliberately evidence rather than level data.  A template
catalog maps labelled reference images to short semantic labels, and this
module records both the matching evidence and any cells that require review.
The resulting :class:`LevelDraft` is useful during annotation, but is not a
validated :class:`railbound.level.Level` and cannot be simulated directly.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from railbound.crop_io import CropManifest
from railbound.extraction import BgrImage, CellCrop, CropSpec
from railbound.geometry import CellCoord
from railbound.level import Identifier

RECOGNITION_SCHEMA_VERSION: Final = 1
TEMPLATE_CATALOG_SCHEMA_VERSION: Final = 1
LEVEL_DRAFT_SCHEMA_VERSION: Final = 1
type RotationDegrees = Literal[0, 90, 180, 270]


class _FrozenModel(BaseModel):
    """Base class for immutable, serializable recognition values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RecognitionStatus(StrEnum):
    """Whether an observation may be used without a manual decision."""

    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class RecognitionThresholds(_FrozenModel):
    """Explicit, persisted thresholds used by template matching.

    ``min_similarity`` rejects images that do not resemble any labelled
    reference.  ``ambiguity_margin`` is the minimum gap between the best two
    *different labels* before the top match keeps its full confidence.
    """

    min_similarity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.84
    min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70
    ambiguity_margin: Annotated[float, Field(gt=0.0, le=1.0)] = 0.08
    max_candidates: Annotated[int, Field(gt=0)] = 3


class TemplateDefinition(_FrozenModel):
    """One labelled reference image in a portable template catalog."""

    id: Identifier
    label: Identifier
    image: str
    rotations: tuple[RotationDegrees, ...] = (0, 90, 180, 270)

    @field_validator("image")
    @classmethod
    def relative_image_path(cls, value: str) -> str:
        """Require an image path relative to the catalog, without traversal."""
        if not value or "\\" in value:
            raise ValueError("template image must be a non-empty relative POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError("template image must be a relative path without '..'")
        return value

    @model_validator(mode="after")
    def at_least_one_rotation(self) -> Self:
        """Require unique rotations so matching and tie-breaking stay stable."""
        if not self.rotations:
            raise ValueError("template rotations must not be empty")
        if len(set(self.rotations)) != len(self.rotations):
            raise ValueError("template rotations must not repeat an angle")
        return self


class TemplateCatalog(_FrozenModel):
    """Versioned set of labelled reference crops for one recognizer version."""

    schema_version: Literal[1] = TEMPLATE_CATALOG_SCHEMA_VERSION
    recognizer_version: Identifier = "template-ncc-v1"
    thresholds: RecognitionThresholds = RecognitionThresholds()
    templates: tuple[TemplateDefinition, ...]

    @model_validator(mode="after")
    def unique_template_ids(self) -> Self:
        """Reject an empty catalog or duplicate stable reference IDs."""
        if not self.templates:
            raise ValueError("template catalog requires at least one template")
        duplicates = _duplicates(template.id for template in self.templates)
        if duplicates:
            raise ValueError("template catalog IDs must be unique: " + ", ".join(duplicates))
        return self


@dataclass(frozen=True)
class Template:
    """A catalog definition paired with the decoded BGR reference pixels."""

    definition: TemplateDefinition
    image: BgrImage


class RecognitionCandidate(_FrozenModel):
    """The strongest reference image for one candidate semantic label."""

    rank: Annotated[int, Field(gt=0)]
    label: Identifier
    template_id: Identifier
    rotation_degrees: RotationDegrees
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]


class RecognitionObservation(_FrozenModel):
    """Evidence and decision state for a single semantic cell."""

    cell: CellCoord
    source_crop: str
    candidates: tuple[RecognitionCandidate, ...] = ()
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    status: RecognitionStatus
    review_reason: str | None = None
    recognizer_version: Identifier

    @model_validator(mode="after")
    def valid_decision_state(self) -> Self:
        """Keep accepted, uncertain, and rejected observations unambiguous."""
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != tuple(range(1, len(self.candidates) + 1)):
            raise ValueError("recognition candidates must have consecutive ranks starting at one")
        if self.status is RecognitionStatus.ACCEPTED:
            if not self.candidates:
                raise ValueError("an accepted observation requires a candidate")
            if self.review_reason is not None:
                raise ValueError("an accepted observation cannot have a review_reason")
        elif self.review_reason is None:
            raise ValueError("an uncertain or unknown observation requires a review_reason")
        if self.status is RecognitionStatus.UNKNOWN and self.candidates:
            raise ValueError("an unknown observation must not retain candidate labels")
        return self


class RecognitionSource(_FrozenModel):
    """Crop-manifest provenance retained without a machine-local path."""

    image_sha256: str | None = None
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    rows: Annotated[int, Field(gt=0)]
    cols: Annotated[int, Field(gt=0)]
    crop_spec: CropSpec

    @field_validator("image_sha256")
    @classmethod
    def valid_image_hash(cls, value: str | None) -> str | None:
        """Accept the optional source-image digest from the crop manifest."""
        if value is None:
            return value
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
        return value


class RecognitionReport(_FrozenModel):
    """Portable evidence from one deterministic recognition pass."""

    schema_version: Literal[1] = RECOGNITION_SCHEMA_VERSION
    source: RecognitionSource
    template_catalog: TemplateCatalog
    template_fingerprint: str
    thresholds: RecognitionThresholds
    observations: tuple[RecognitionObservation, ...]
    review_cells: tuple[CellCoord, ...]

    @field_validator("template_fingerprint")
    @classmethod
    def sha256_fingerprint(cls, value: str) -> str:
        """Validate the digest that identifies reference configuration and pixels."""
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("template_fingerprint must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def complete_ordered_observations(self) -> Self:
        """Require one row-major observation and review decision per crop cell."""
        expected_cells = tuple(
            CellCoord(row=row, col=col)
            for row in range(self.source.rows)
            for col in range(self.source.cols)
        )
        actual_cells = tuple(observation.cell for observation in self.observations)
        if actual_cells != expected_cells:
            raise ValueError("observations must cover every source cell once in row-major order")
        expected_review_cells = tuple(
            observation.cell
            for observation in self.observations
            if observation.status is not RecognitionStatus.ACCEPTED
        )
        if self.review_cells != expected_review_cells:
            raise ValueError("review_cells must list every non-accepted observation in order")
        return self


class RecognitionOverride(_FrozenModel):
    """A manual semantic decision that takes precedence over recognition evidence."""

    cell: CellCoord
    label: Identifier
    note: str | None = None


class ResolvedCell(_FrozenModel):
    """A semantic label selected automatically or by a manual override."""

    cell: CellCoord
    label: Identifier
    source: Literal["recognition", "override"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    template_id: Identifier | None = None
    rotation_degrees: Literal[0, 90, 180, 270] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def valid_provenance(self) -> Self:
        """Require template evidence only for automatic decisions."""
        if self.source == "recognition":
            if self.confidence is None or self.template_id is None or self.rotation_degrees is None:
                raise ValueError("recognized cells require confidence, template_id, and rotation")
        elif any(
            value is not None
            for value in (self.confidence, self.template_id, self.rotation_degrees)
        ):
            raise ValueError("overridden cells must not retain automatic recognition evidence")
        return self


class LevelDraft(_FrozenModel):
    """A review-state document, deliberately separate from authoritative levels.

    Labels are catalog-defined semantic candidates.  A reviewer must still
    map them to complete M4 layers and provide inventory and win conditions
    before constructing a :class:`railbound.level.Level`.
    """

    schema_version: Literal[1] = LEVEL_DRAFT_SCHEMA_VERSION
    source: RecognitionSource
    recognition_report_fingerprint: str
    resolved_cells: tuple[ResolvedCell, ...]
    unresolved_cells: tuple[CellCoord, ...]

    @field_validator("recognition_report_fingerprint")
    @classmethod
    def report_fingerprint(cls, value: str) -> str:
        """Validate the report fingerprint used to tie a draft to its evidence."""
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("recognition_report_fingerprint must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def complete_cell_decisions(self) -> Self:
        """Require a resolved label or an explicit unresolved entry for every cell."""
        expected_cells = tuple(
            CellCoord(row=row, col=col)
            for row in range(self.source.rows)
            for col in range(self.source.cols)
        )
        resolved_cells = tuple(value.cell for value in self.resolved_cells)
        actual_cells = _merge_row_major(resolved_cells, self.unresolved_cells)
        if actual_cells != expected_cells:
            raise ValueError("draft cells must cover every source cell exactly once")
        return self

    @property
    def is_complete(self) -> bool:
        """Return whether every observation has been resolved to a label."""
        return not self.unresolved_cells


class TemplateRecognizer:
    """Match normalized crops to labelled templates with explicit rejection.

    The matcher combines normalized cross-correlation of colour structure with
    a small mean-colour feature.  All ordering, rotations, and thresholds are
    stored in the resulting report, so the same inputs make the same decision.
    """

    def __init__(
        self,
        catalog: TemplateCatalog,
        templates: Sequence[Template],
        *,
        thresholds: RecognitionThresholds | None = None,
    ) -> None:
        """Create a recognizer after checking its decoded templates match its catalog."""
        definitions = {definition.id: definition for definition in catalog.templates}
        template_ids = {template.definition.id for template in templates}
        if len(templates) != len(definitions) or template_ids != set(definitions):
            raise ValueError("decoded templates must match the template catalog exactly")
        prepared_templates: list[Template] = []
        for template in templates:
            if template.definition != definitions[template.definition.id]:
                raise ValueError("decoded template definition does not match the catalog")
            _validate_bgr_image(template.image, f"template {template.definition.id!r}")
            image = template.image.copy()
            image.setflags(write=False)
            prepared_templates.append(Template(definition=template.definition, image=image))
        self._catalog = catalog
        self._templates = tuple(
            sorted(prepared_templates, key=lambda template: template.definition.id)
        )
        self._thresholds = thresholds or catalog.thresholds
        self._fingerprint = _template_fingerprint(self._templates)

    @property
    def catalog(self) -> TemplateCatalog:
        """Return the immutable catalog used by this recognizer."""
        return self._catalog

    @property
    def thresholds(self) -> RecognitionThresholds:
        """Return the effective matching thresholds."""
        return self._thresholds

    @property
    def fingerprint(self) -> str:
        """Return a digest of decoded template definitions and image pixels."""
        return self._fingerprint

    def recognize(self, crop: CellCrop) -> RecognitionObservation:
        """Return an evidence record for one normalized extraction crop."""
        return self.recognize_image(cell=crop.cell, image=crop.image, source_crop="in-memory")

    def recognize_image(
        self,
        *,
        cell: CellCoord,
        image: BgrImage,
        source_crop: str,
    ) -> RecognitionObservation:
        """Recognize decoded crop pixels while retaining their portable filename."""
        _validate_bgr_image(image, "crop")
        candidates = self._rank_candidates(image)
        if not candidates or candidates[0].similarity < self._thresholds.min_similarity:
            best_similarity = candidates[0].similarity if candidates else 0.0
            return RecognitionObservation(
                cell=cell,
                source_crop=source_crop,
                confidence=0.0,
                status=RecognitionStatus.UNKNOWN,
                review_reason=(
                    "no template reached min_similarity "
                    f"{self._thresholds.min_similarity:.2f} (best {best_similarity:.2f})"
                ),
                recognizer_version=self._catalog.recognizer_version,
            )

        confidence = _decision_confidence(candidates, self._thresholds.ambiguity_margin)
        selected = tuple(candidates[: self._thresholds.max_candidates])
        if confidence < self._thresholds.min_confidence:
            return RecognitionObservation(
                cell=cell,
                source_crop=source_crop,
                candidates=selected,
                confidence=confidence,
                status=RecognitionStatus.UNCERTAIN,
                review_reason=(
                    "best candidate is ambiguous or below min_confidence "
                    f"{self._thresholds.min_confidence:.2f}"
                ),
                recognizer_version=self._catalog.recognizer_version,
            )
        return RecognitionObservation(
            cell=cell,
            source_crop=source_crop,
            candidates=selected,
            confidence=confidence,
            status=RecognitionStatus.ACCEPTED,
            recognizer_version=self._catalog.recognizer_version,
        )

    def report_from_manifest(
        self,
        manifest: CropManifest,
        images: Sequence[BgrImage],
    ) -> RecognitionReport:
        """Recognize manifest-ordered crop pixels and return a complete report."""
        if len(images) != len(manifest.crops):
            raise ValueError("crop image count does not match the crop manifest")
        observations = tuple(
            self.recognize_image(cell=record.cell, image=image, source_crop=record.filename)
            for record, image in zip(manifest.crops, images, strict=True)
        )
        source = RecognitionSource(
            image_sha256=manifest.image_sha256,
            image_width=manifest.image_width,
            image_height=manifest.image_height,
            rows=manifest.rows,
            cols=manifest.cols,
            crop_spec=manifest.spec,
        )
        return RecognitionReport(
            source=source,
            template_catalog=self._catalog,
            template_fingerprint=self.fingerprint,
            thresholds=self._thresholds,
            observations=observations,
            review_cells=tuple(
                observation.cell
                for observation in observations
                if observation.status is not RecognitionStatus.ACCEPTED
            ),
        )

    def _rank_candidates(self, image: BgrImage) -> tuple[RecognitionCandidate, ...]:
        best_by_label: dict[str, tuple[float, str, RotationDegrees]] = {}
        for template in self._templates:
            for angle in template.definition.rotations:
                similarity = _image_similarity(image, _rotate(template.image, angle))
                current = best_by_label.get(template.definition.label)
                candidate_key = (similarity, template.definition.id, angle)
                if current is None or _is_better_match(candidate_key, current):
                    best_by_label[template.definition.label] = candidate_key

        ordered = sorted(
            (
                (label, similarity, template_id, angle)
                for label, (similarity, template_id, angle) in best_by_label.items()
            ),
            key=lambda value: (-value[1], value[0], value[2], value[3]),
        )
        return tuple(
            RecognitionCandidate(
                rank=rank,
                label=label,
                template_id=template_id,
                rotation_degrees=angle,
                similarity=similarity,
            )
            for rank, (label, similarity, template_id, angle) in enumerate(ordered, start=1)
        )


def resolve(
    report: RecognitionReport,
    overrides: Iterable[RecognitionOverride] = (),
    *,
    minimum_confidence: float | None = None,
) -> LevelDraft:
    """Turn accepted observations and manual overrides into a reviewable draft.

    Overrides always win, including for already accepted cells.  Values below
    ``minimum_confidence`` remain explicitly unresolved rather than silently
    becoming semantic level data.
    """
    threshold = (
        report.thresholds.min_confidence if minimum_confidence is None else minimum_confidence
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")

    override_by_cell = _overrides_by_cell(report.source, overrides)
    resolved_cells: list[ResolvedCell] = []
    unresolved_cells: list[CellCoord] = []
    for observation in report.observations:
        override = override_by_cell.get(_cell_key(observation.cell))
        if override is not None:
            resolved_cells.append(
                ResolvedCell(
                    cell=observation.cell,
                    label=override.label,
                    source="override",
                    note=override.note,
                )
            )
            continue
        if observation.status is RecognitionStatus.ACCEPTED and observation.confidence >= threshold:
            candidate = observation.candidates[0]
            resolved_cells.append(
                ResolvedCell(
                    cell=observation.cell,
                    label=candidate.label,
                    source="recognition",
                    confidence=observation.confidence,
                    template_id=candidate.template_id,
                    rotation_degrees=candidate.rotation_degrees,
                )
            )
        else:
            unresolved_cells.append(observation.cell)

    return LevelDraft(
        source=report.source,
        recognition_report_fingerprint=_report_fingerprint(report),
        resolved_cells=tuple(resolved_cells),
        unresolved_cells=tuple(unresolved_cells),
    )


def _validate_bgr_image(image: NDArray[np.uint8], name: str) -> None:
    if image.dtype != np.uint8:
        raise ValueError(f"{name} image must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} image must be a three-channel BGR image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} image must not be empty")


def _image_similarity(image: BgrImage, template: BgrImage) -> float:
    """Return deterministic structural and colour similarity in the range zero to one."""
    reference = cast(
        BgrImage,
        cv2.resize(template, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA),
    )
    image_values = image.astype(np.float32)
    reference_values = reference.astype(np.float32)
    structural = _normalized_correlation(image_values, reference_values)
    mean_difference = float(
        np.mean(np.abs(np.mean(image_values, axis=(0, 1)) - np.mean(reference_values, axis=(0, 1))))
    )
    colour = 1.0 - mean_difference / 255.0
    return float(np.clip(0.85 * structural + 0.15 * colour, 0.0, 1.0))


def _normalized_correlation(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    """Compare images after brightness normalization, including constant references."""
    centered_first = first - np.mean(first, axis=(0, 1), keepdims=True)
    centered_second = second - np.mean(second, axis=(0, 1), keepdims=True)
    denominator = float(np.linalg.norm(centered_first) * np.linalg.norm(centered_second))
    if denominator <= np.finfo(np.float32).eps:
        difference = float(np.mean(np.abs(first - second)))
        return float(np.clip(1.0 - difference / 255.0, 0.0, 1.0))
    correlation = float(np.sum(centered_first * centered_second) / denominator)
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def _rotate(image: BgrImage, angle: Literal[0, 90, 180, 270]) -> BgrImage:
    if angle == 0:
        return image
    if angle == 90:
        return cast(BgrImage, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    if angle == 180:
        return cast(BgrImage, cv2.rotate(image, cv2.ROTATE_180))
    return cast(BgrImage, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))


def _is_better_match(candidate: tuple[float, str, int], current: tuple[float, str, int]) -> bool:
    candidate_similarity, candidate_id, candidate_angle = candidate
    current_similarity, current_id, current_angle = current
    if candidate_similarity != current_similarity:
        return candidate_similarity > current_similarity
    return (candidate_id, candidate_angle) < (current_id, current_angle)


def _decision_confidence(
    candidates: Sequence[RecognitionCandidate], ambiguity_margin: float
) -> float:
    top_similarity = candidates[0].similarity
    if len(candidates) == 1:
        return top_similarity
    margin = max(0.0, top_similarity - candidates[1].similarity)
    return top_similarity * min(1.0, margin / ambiguity_margin)


def _template_fingerprint(templates: Sequence[Template]) -> str:
    digest = sha256()
    for template in templates:
        definition = json.dumps(
            template.definition.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(definition.encode("utf-8"))
        digest.update(template.image.shape.__repr__().encode("ascii"))
        digest.update(template.image.tobytes())
    return digest.hexdigest()


def _report_fingerprint(report: RecognitionReport) -> str:
    """Return a stable digest of serialized evidence for draft provenance."""
    payload = report.model_dump_json(exclude={"template_fingerprint"})
    return sha256(payload.encode("utf-8")).hexdigest()


def _overrides_by_cell(
    source: RecognitionSource, overrides: Iterable[RecognitionOverride]
) -> dict[tuple[int, int], RecognitionOverride]:
    override_by_cell: dict[tuple[int, int], RecognitionOverride] = {}
    for override in overrides:
        key = _cell_key(override.cell)
        if not 0 <= override.cell.row < source.rows or not 0 <= override.cell.col < source.cols:
            raise ValueError(
                f"override contains out-of-bounds cell ({override.cell.row}, {override.cell.col})"
            )
        if key in override_by_cell:
            raise ValueError(
                f"overrides contain duplicate cell ({override.cell.row}, {override.cell.col})"
            )
        override_by_cell[key] = override
    return override_by_cell


def _merge_row_major(
    resolved: Sequence[CellCoord], unresolved: Sequence[CellCoord]
) -> tuple[CellCoord, ...]:
    cells = [*resolved, *unresolved]
    keys = [_cell_key(cell) for cell in cells]
    if len(set(keys)) != len(keys):
        return ()
    return tuple(CellCoord(row=row, col=col) for row, col in sorted(keys))


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    return cell.row, cell.col
