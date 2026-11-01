"""Local SQLite storage for reusable cell types and confirmed visual samples."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from pydantic import TypeAdapter

from railbound.cell_annotation import (
    CellAnnotation,
    CellTypeDefinition,
    PropertyValue,
    built_in_cell_types,
)
from railbound.extraction import BgrImage, CellCrop

_PROPERTY_VALUES = TypeAdapter(dict[str, PropertyValue])


@dataclass(frozen=True)
class StoredCellSample:
    """One confirmed image and its instance properties."""

    id: str
    type_id: str
    properties: dict[str, PropertyValue]
    image: BgrImage


class AnnotationStore:
    """Persist annotation knowledge without exposing database details to the GUI."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    def cell_types(self) -> tuple[CellTypeDefinition, ...]:
        """Return every known type in stable display-name order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT definition_json FROM cell_types ORDER BY display_name, id"
            ).fetchall()
        return tuple(CellTypeDefinition.model_validate_json(row[0]) for row in rows)

    def save_cell_type(self, definition: CellTypeDefinition) -> None:
        """Insert or replace one reusable type definition."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cell_types (id, display_name, definition_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = excluded.display_name,
                    definition_json = excluded.definition_json
                """,
                (definition.id, definition.display_name, definition.model_dump_json()),
            )

    def samples(self) -> tuple[StoredCellSample, ...]:
        """Decode every confirmed visual sample in stable ID order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, type_id, properties_json, image_png FROM cell_samples ORDER BY id"
            ).fetchall()
        samples: list[StoredCellSample] = []
        for sample_id, type_id, properties_json, image_png in rows:
            encoded = np.frombuffer(image_png, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"stored sample {sample_id!r} contains an invalid PNG")
            properties = _PROPERTY_VALUES.validate_json(properties_json)
            samples.append(
                StoredCellSample(
                    id=sample_id,
                    type_id=type_id,
                    properties=properties,
                    image=cast(BgrImage, image),
                )
            )
        return tuple(samples)

    def save_manual_sample(self, annotation: CellAnnotation, crop: CellCrop) -> None:
        """Upsert one manually reviewed cell as reusable recognition evidence."""
        if annotation.source != "manual" or annotation.type_id is None:
            raise ValueError("only manually reviewed typed cells may become samples")
        success, encoded = cv2.imencode(".png", crop.image)
        if not success:
            raise OSError(f"could not encode sample {annotation.id!r} as PNG")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cell_samples (id, type_id, properties_json, image_png)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    type_id = excluded.type_id,
                    properties_json = excluded.properties_json,
                    image_png = excluded.image_png
                """,
                (
                    annotation.id,
                    annotation.type_id,
                    json.dumps(annotation.properties, sort_keys=True),
                    encoded.tobytes(),
                ),
            )

    def delete_sample(self, sample_id: str) -> None:
        """Remove superseded recognition evidence for one cell identity."""
        with self._connect() as connection:
            connection.execute("DELETE FROM cell_samples WHERE id = ?", (sample_id,))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS cell_types (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    definition_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cell_samples (
                    id TEXT PRIMARY KEY,
                    type_id TEXT NOT NULL REFERENCES cell_types(id),
                    properties_json TEXT NOT NULL,
                    image_png BLOB NOT NULL
                );
                """
            )
        for definition in built_in_cell_types():
            self.save_cell_type(definition)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
