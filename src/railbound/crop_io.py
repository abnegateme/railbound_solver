"""Persistence for normalized cell crops and their deterministic manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field, model_validator

from railbound.extraction import CellCrop, CropSpec
from railbound.geometry import CellCoord, GridGeometry

MANIFEST_SCHEMA_VERSION = 1


class _FrozenModel(BaseModel):
    """Base class for validated, serializable crop documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CropRecord(_FrozenModel):
    """The stable location of one crop within an extraction output directory."""

    cell: CellCoord
    filename: str


class CropManifest(_FrozenModel):
    """Portable record of all files created by one extraction run."""

    schema_version: Literal[1] = 1
    image_sha256: str | None = None
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    rows: Annotated[int, Field(gt=0)]
    cols: Annotated[int, Field(gt=0)]
    spec: CropSpec
    crops: tuple[CropRecord, ...]

    @model_validator(mode="after")
    def complete_row_major_coverage(self) -> CropManifest:
        """Require exactly one ordered record for every semantic cell."""
        expected_cells = tuple(
            CellCoord(row=row, col=col) for row in range(self.rows) for col in range(self.cols)
        )
        actual_cells = tuple(record.cell for record in self.crops)
        if actual_cells != expected_cells:
            raise ValueError("crop records must cover every cell once in row-major order")
        return self


def crop_filename(cell: CellCoord) -> str:
    """Return the deterministic PNG filename for a semantic cell coordinate."""
    return f"r{cell.row:03d}_c{cell.col:03d}.png"


def build_crop_manifest(grid: GridGeometry, spec: CropSpec, crops: list[CellCrop]) -> CropManifest:
    """Build a manifest after confirming that the crop set covers the full grid."""
    return CropManifest(
        image_sha256=grid.image_sha256,
        image_width=grid.image_width,
        image_height=grid.image_height,
        rows=grid.rows,
        cols=grid.cols,
        spec=spec,
        crops=tuple(
            CropRecord(cell=crop.cell, filename=crop_filename(crop.cell)) for crop in crops
        ),
    )


def save_crop_manifest(manifest: CropManifest, path: Path) -> None:
    """Serialize a crop manifest as deterministic, reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_crop_manifest(path: Path) -> CropManifest:
    """Load and validate a crop-manifest JSON document."""
    return CropManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_crop_set(
    crops: list[CellCrop],
    *,
    grid: GridGeometry,
    spec: CropSpec,
    output_directory: Path,
    manifest_path: Path | None = None,
) -> CropManifest:
    """Write PNG crops plus their manifest and return the validated manifest."""
    manifest = build_crop_manifest(grid, spec, crops)
    output_directory.mkdir(parents=True, exist_ok=True)
    for crop, record in zip(crops, manifest.crops, strict=True):
        destination = output_directory / record.filename
        if not cv2.imwrite(str(destination), crop.image):
            raise OSError(f"could not write crop image: {destination}")
    save_crop_manifest(manifest, manifest_path or output_directory / "manifest.json")
    return manifest
