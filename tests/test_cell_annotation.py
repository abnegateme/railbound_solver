from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railbound.annotation_store import AnnotationStore
from railbound.cell_annotation import (
    CellAnnotation,
    CellPropertyDefinition,
    CellTypeDefinition,
    StateFeature,
    annotation_errors,
    built_in_cell_types,
    cell_type_for_role,
    formalize_annotations,
    make_cell_annotation_id,
    rotate_annotation_properties,
)
from railbound.extraction import CellCrop
from railbound.geometry import CellCoord, PixelPoint
from railbound.level import Direction


def test_train_annotation_requires_direction_and_rotates_sample_property() -> None:
    definitions = {definition.id: definition for definition in built_in_cell_types()}
    cell = CellCoord(row=2, col=3)
    incomplete = CellAnnotation(
        id=make_cell_annotation_id("1-13", cell),
        cell=cell,
        type_id="actor.train",
        source="manual",
    )

    assert annotation_errors(incomplete, definitions) == ("Direction is required",)
    assert rotate_annotation_properties({"direction": "N"}, definitions["actor.train"], 90) == {
        "direction": "E"
    }


def test_reviewed_baseline_cells_compile_to_validated_level() -> None:
    definitions = {definition.id: definition for definition in built_in_cell_types()}
    annotations = (
        CellAnnotation(
            id=make_cell_annotation_id("1-13", CellCoord(row=0, col=0)),
            cell=CellCoord(row=0, col=0),
            type_id="terrain.empty",
            source="manual",
        ),
        CellAnnotation(
            id=make_cell_annotation_id("1-13", CellCoord(row=0, col=1)),
            cell=CellCoord(row=0, col=1),
            type_id="actor.train",
            properties={"direction": "E"},
            source="manual",
        ),
        CellAnnotation(
            id=make_cell_annotation_id("1-13", CellCoord(row=0, col=2)),
            cell=CellCoord(row=0, col=2),
            type_id="goal.arrival",
            source="manual",
        ),
    )

    level = formalize_annotations(
        level_number="1-13",
        rows=1,
        cols=3,
        rail_budget=2,
        annotations=annotations,
        definitions=definitions,
    )

    assert level.id == "level-1-13"
    assert level.actors[0].id == "level-1-13-r000-c001"
    assert level.actors[0].direction is Direction.EAST
    assert level.win_condition.required_arrivals[0].target_cell == CellCoord(row=0, col=2)
    assert level.rail_inventory.total_budget == 2


def test_unmodeled_type_is_recognizable_but_blocks_formalization() -> None:
    definition = cell_type_for_role(
        type_id="custom.signal",
        display_name="Signal",
        role="unmodeled",
    )
    annotation = CellAnnotation(
        id="level-1-13-r000-c000",
        cell=CellCoord(row=0, col=0),
        type_id=definition.id,
        source="manual",
    )

    assert annotation_errors(annotation, {definition.id: definition}) == (
        "the selected type has no complete graph-state compiler yet",
    )


def test_state_feature_requires_explicit_matching_state_values() -> None:
    definition = CellTypeDefinition(
        id="device.gate",
        display_name="Gate",
        properties=(
            CellPropertyDefinition(
                id="initial-state",
                label="Initial state",
                value_kind="text",
                choices=("open", "closed"),
            ),
        ),
        features=(
            StateFeature(
                states=("open", "closed"),
                initial_state_property="initial-state",
            ),
        ),
    )

    assert not definition.formalization_supported
    with pytest.raises(ValueError, match="choices must match"):
        CellTypeDefinition(
            id="device.invalid-gate",
            display_name="Invalid gate",
            properties=(
                CellPropertyDefinition(
                    id="initial-state",
                    label="Initial state",
                    value_kind="text",
                    choices=("closed", "open"),
                ),
            ),
            features=(
                StateFeature(
                    states=("open", "closed"),
                    initial_state_property="initial-state",
                ),
            ),
        )


def test_annotation_store_seeds_types_and_round_trips_manual_sample(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "annotations.sqlite3")
    cell = CellCoord(row=0, col=0)
    annotation = CellAnnotation(
        id=make_cell_annotation_id("1-13", cell),
        cell=cell,
        type_id="actor.train",
        properties={"direction": "N"},
        source="manual",
    )
    image = np.full((12, 12, 3), 127, dtype=np.uint8)
    point = PixelPoint(x=0, y=0)
    crop = CellCrop(
        cell=cell,
        image=image,
        source_corners_px=(point, point, point, point),
        mask=None,
    )

    store.save_manual_sample(annotation, crop)

    assert {definition.id for definition in store.cell_types()} >= {
        "terrain.empty",
        "terrain.occupied",
        "actor.train",
        "goal.arrival",
    }
    sample = store.samples()[0]
    assert sample.id == annotation.id
    assert sample.type_id == "actor.train"
    assert sample.properties == {"direction": "N"}
    assert np.array_equal(sample.image, image)

    store.delete_sample(annotation.id)
    assert store.samples() == ()
