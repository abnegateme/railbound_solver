from __future__ import annotations

import pytest

from railbound.geometry import CellCoord
from railbound.level import Level, Port
from railbound.simulation import SimulationStatus, simulate
from railbound.solver import (
    SolverLimits,
    SolverStatus,
    iter_placement_candidates,
    solve,
)
from railbound.topology import PlacedRail, RailPieceKind, RailPlacement


def _track(
    row: int, col: int, ports: list[str], transitions: list[tuple[str, str]]
) -> dict[str, object]:
    return {
        "id": f"track-{row}-{col}",
        "cell": {"row": row, "col": col},
        "ports": ports,
        "transitions": [
            {"entry_port": entry_port, "exit_port": exit_port}
            for entry_port, exit_port in transitions
        ],
    }


def _level(
    *,
    rows: int = 1,
    cols: int = 3,
    buildable_cells: list[tuple[int, int]] | None = None,
    tracks: list[dict[str, object]] | None = None,
    rail_inventory: dict[str, object] | None = None,
    switches: list[dict[str, object]] | None = None,
    buttons: list[dict[str, object]] | None = None,
) -> Level:
    return Level.model_validate(
        {
            "id": "solver-level",
            "rows": rows,
            "cols": cols,
            "buildable_cells": [{"row": row, "col": col} for row, col in (buildable_cells or [])],
            "fixed_tracks": tracks or [],
            "actors": [
                {
                    "id": "train-1",
                    "cell": {"row": 0, "col": 0},
                    "direction": "E",
                }
            ],
            "switches": switches or [],
            "buttons": buttons or [],
            "rail_inventory": rail_inventory or {"total_budget": 0},
            "win_condition": {
                "required_arrivals": [
                    {
                        "actor_id": "train-1",
                        "target_cell": {"row": rows - 1, "col": cols - 1},
                        "entry_port": "W",
                    }
                ]
            },
        }
    )


def _straight_line_level(
    *,
    buildable_cells: list[tuple[int, int]],
    rail_inventory: dict[str, object],
) -> Level:
    return _level(
        buildable_cells=buildable_cells,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 2, ["E", "W"], [("E", "W")]),
        ],
        rail_inventory=rail_inventory,
    )


def test_solver_returns_a_verified_zero_rail_solution() -> None:
    level = _level(
        cols=2,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
        ],
    )

    result = solve(level)

    assert result.status is SolverStatus.SOLVED
    assert result.placement == RailPlacement(level_id=level.id)
    assert result.simulation is not None
    assert result.simulation.status is SimulationStatus.SUCCESS
    assert result.trace == result.simulation.events
    assert result.statistics.placements_considered == 1
    assert result.statistics.runtime_states_visited == 2


def test_solver_finds_the_only_legal_horizontal_placement() -> None:
    level = _straight_line_level(
        buildable_cells=[(0, 1)],
        rail_inventory={
            "total_budget": 1,
            "piece_counts": [{"piece_kind": "straight", "count": 1}],
        },
    )

    result = solve(level)

    assert result.solved
    assert result.placement == RailPlacement(
        level_id=level.id,
        rails=(
            PlacedRail(
                cell=CellCoord(row=0, col=1),
                piece_kind=RailPieceKind.STRAIGHT,
                orientation=Port.EAST,
            ),
        ),
    )
    assert result.simulation is not None
    assert simulate(level, result.placement).succeeded


def test_solver_uses_stable_tie_breaking_for_multiple_solutions() -> None:
    level = _straight_line_level(
        buildable_cells=[(0, 1)],
        rail_inventory={
            "total_budget": 1,
            "piece_counts": [
                {"piece_kind": "straight", "count": 1},
                {"piece_kind": "crossing", "count": 1},
            ],
        },
    )
    alternatives = tuple(iter_placement_candidates(level))

    result = solve(level)
    crossing = RailPlacement(
        level_id=level.id,
        rails=(
            PlacedRail(
                cell=CellCoord(row=0, col=1),
                piece_kind=RailPieceKind.CROSSING,
                orientation=Port.NORTH,
            ),
        ),
    )

    assert result.placement is not None
    assert result.placement.rails[0].piece_kind is RailPieceKind.STRAIGHT
    assert crossing in alternatives
    assert simulate(level, crossing).succeeded
    assert solve(level).placement == result.placement


def test_solver_reports_an_exhaustive_unsatisfiable_search() -> None:
    level = _straight_line_level(
        buildable_cells=[],
        rail_inventory={"total_budget": 0},
    )

    result = solve(level)

    assert result.status is SolverStatus.UNSOLVABLE
    assert result.placement is None
    assert result.simulation is None
    assert result.trace == ()
    assert result.statistics.placements_considered == 1


def test_solver_respects_the_level_rail_budget() -> None:
    level = _straight_line_level(
        buildable_cells=[(0, 1)],
        rail_inventory={
            "total_budget": 0,
            "piece_counts": [{"piece_kind": "straight", "count": 1}],
        },
    )

    result = solve(level)

    assert result.status is SolverStatus.UNSOLVABLE
    assert result.statistics.maximum_rails_considered == 0


def test_solver_reports_search_and_rail_bounds_without_claiming_unsatisfiability() -> None:
    level = _straight_line_level(
        buildable_cells=[(0, 1)],
        rail_inventory={
            "total_budget": 1,
            "piece_counts": [{"piece_kind": "straight", "count": 1}],
        },
    )

    placement_limited = solve(level, SolverLimits(max_placements=1))
    rail_limited = solve(level, SolverLimits(max_rails=0))

    assert placement_limited.status is SolverStatus.LIMIT_REACHED
    assert placement_limited.statistics.placements_considered == 1
    assert rail_limited.status is SolverStatus.LIMIT_REACHED
    assert rail_limited.statistics.maximum_rails_considered == 0


def test_solver_replays_dynamic_switch_state() -> None:
    level = _level(
        rows=2,
        cols=4,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
            _track(1, 2, ["N", "E"], [("N", "E")]),
            _track(1, 3, ["E", "W"], [("E", "W")]),
        ],
        switches=[
            {
                "id": "switch-yellow",
                "cell": {"row": 0, "col": 2},
                "entry_port": "W",
                "output_ports": ["E", "S"],
                "initial_output": "E",
                "channel": "yellow",
            }
        ],
        buttons=[
            {
                "id": "button-yellow",
                "cell": {"row": 0, "col": 1},
                "channel": "yellow",
                "activation_behavior": "toggle",
            }
        ],
    )

    result = solve(level)

    assert result.solved
    assert result.simulation is not None
    assert result.simulation.final_state.switch_states[0].selected_output is Port.SOUTH
    assert result.statistics.runtime_states_visited == 5


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_placements", -1),
        ("max_rails", -1),
        ("max_ticks", 0),
    ],
)
def test_solver_limits_reject_invalid_values(keyword: str, value: int) -> None:
    with pytest.raises(ValueError):
        if keyword == "max_placements":
            SolverLimits(max_placements=value)
        elif keyword == "max_rails":
            SolverLimits(max_rails=value)
        else:
            SolverLimits(max_ticks=value)
