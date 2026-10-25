from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from railbound.level import Level
from railbound.level_io import load_level, save_level


def _level_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "ruleset_version": "railbound-observed-v1",
        "id": "test-level",
        "rows": 2,
        "cols": 3,
        "buildable_cells": [{"row": 0, "col": 1}],
        "fixed_tracks": [
            {
                "id": "goal-track",
                "cell": {"row": 0, "col": 2},
                "ports": ["E", "W"],
                "transitions": [{"entry_port": "E", "exit_port": "W"}],
                "fixed": True,
            }
        ],
        "actors": [
            {
                "id": "train-1",
                "cell": {"row": 0, "col": 0},
                "direction": "E",
            }
        ],
        "rail_inventory": {"total_budget": 1},
        "win_condition": {
            "required_arrivals": [
                {
                    "actor_id": "train-1",
                    "target_cell": {"row": 0, "col": 2},
                    "entry_port": "W",
                }
            ]
        },
    }


def test_level_json_round_trip(tmp_path: Path) -> None:
    level = Level.model_validate(_level_data())
    path = tmp_path / "levels" / "test-level.json"

    save_level(level, path)

    assert load_level(path) == level
    assert json.loads(path.read_text(encoding="utf-8"))["id"] == "test-level"


def test_manually_authored_example_is_valid() -> None:
    example_path = Path(__file__).parents[1] / "data" / "example-level.json"

    level = load_level(example_path)

    assert level.id == "example-straight-arrival"


def test_level_rejects_duplicate_entity_ids() -> None:
    data = _level_data()
    data["actors"] = [
        {
            "id": "goal-track",
            "cell": {"row": 0, "col": 0},
            "direction": "E",
        }
    ]

    with pytest.raises(ValidationError, match="globally unique"):
        Level.model_validate(data)


def test_level_rejects_out_of_bounds_coordinate() -> None:
    data = _level_data()
    data["buildable_cells"] = [{"row": 2, "col": 1}]

    with pytest.raises(ValidationError, match="out-of-bounds"):
        Level.model_validate(data)


def test_level_rejects_inconsistent_tunnel_pair() -> None:
    data = _level_data()
    data["tunnel_endpoints"] = [
        {
            "id": "tunnel-a",
            "cell": {"row": 1, "col": 0},
            "pair_id": "violet",
            "entry_port": "N",
            "exit_port": "S",
        }
    ]

    with pytest.raises(ValidationError, match="exactly two endpoints"):
        Level.model_validate(data)


def test_level_rejects_illegal_track_ports() -> None:
    data = _level_data()
    data["fixed_tracks"] = [
        {
            "id": "invalid-track",
            "cell": {"row": 0, "col": 2},
            "ports": ["N", "N"],
            "transitions": [{"entry_port": "N", "exit_port": "S"}],
            "fixed": True,
        }
    ]

    with pytest.raises(ValidationError, match="ports must be unique"):
        Level.model_validate(data)


def test_level_rejects_unknown_schema_version() -> None:
    data = _level_data()
    data["schema_version"] = 2

    with pytest.raises(ValidationError):
        Level.model_validate(data)


def test_level_requires_inventory() -> None:
    data = _level_data()
    del data["rail_inventory"]

    with pytest.raises(ValidationError):
        Level.model_validate(data)
