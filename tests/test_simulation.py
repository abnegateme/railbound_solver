from __future__ import annotations

from pathlib import Path

import pytest

from railbound.geometry import CellCoord
from railbound.level import Direction, Level, Port
from railbound.level_io import load_level
from railbound.placement_io import load_placement
from railbound.simulation import (
    ActorState,
    ActorStatus,
    EventKind,
    FailureReason,
    SimulationEvent,
    SimulationOptions,
    SimulationStatus,
    initial_state,
    simulate,
    step,
)
from railbound.topology import RailPlacement, compile_board


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
    rows: int,
    cols: int,
    tracks: list[dict[str, object]],
    actors: list[dict[str, object]],
    arrivals: list[dict[str, object]],
    gates: list[dict[str, object]] | None = None,
    arrival_order: list[str] | None = None,
) -> Level:
    win_condition: dict[str, object] = {"required_arrivals": arrivals}
    if arrival_order is not None:
        win_condition["arrival_order"] = arrival_order
    return Level.model_validate(
        {
            "id": "simulation-level",
            "rows": rows,
            "cols": cols,
            "buildable_cells": [],
            "fixed_tracks": tracks,
            "actors": actors,
            "gates": gates or [],
            "rail_inventory": {"total_budget": 0},
            "win_condition": win_condition,
        }
    )


def _actor(actor_id: str, row: int, col: int, direction: str) -> dict[str, object]:
    return {
        "id": actor_id,
        "cell": {"row": row, "col": col},
        "direction": direction,
    }


def _arrival(actor_id: str, row: int, col: int, entry_port: str | None = None) -> dict[str, object]:
    arrival: dict[str, object] = {
        "actor_id": actor_id,
        "target_cell": {"row": row, "col": col},
    }
    if entry_port is not None:
        arrival["entry_port"] = entry_port
    return arrival


def _placement() -> RailPlacement:
    return RailPlacement(level_id="simulation-level")


def test_example_level_and_placement_replay_to_success() -> None:
    repository_root = Path(__file__).parents[1]
    level = load_level(repository_root / "data" / "example-level.json")
    placement = load_placement(repository_root / "data" / "example-placement.json")

    result = simulate(level, placement)

    assert result.status is SimulationStatus.SUCCESS


def test_simulate_straight_arrival_emits_ordered_trace() -> None:
    level = _level(
        rows=1,
        cols=3,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
            _track(0, 2, ["E", "W"], [("E", "W")]),
        ],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 0, 2, "W")],
    )

    result = simulate(level, _placement())

    assert result.status is SimulationStatus.SUCCESS
    assert result.succeeded
    assert result.final_state.tick == 2
    assert result.final_state.actors == (
        ActorState(
            actor_id="train-1",
            cell=CellCoord(row=0, col=2),
            direction=Direction.EAST,
            status=ActorStatus.ARRIVED,
        ),
    )
    assert [(event.sequence, event.tick, event.kind) for event in result.events] == [
        (0, 1, EventKind.MOVE),
        (1, 2, EventKind.MOVE),
        (2, 2, EventKind.ARRIVAL),
        (3, 2, EventKind.SUCCESS),
    ]
    assert result.events[1].source_cell == CellCoord(row=0, col=1)
    assert result.events[1].target_cell == CellCoord(row=0, col=2)


def test_step_traverses_a_curve_and_changes_travel_direction() -> None:
    level = _level(
        rows=2,
        cols=2,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["S", "W"], [("S", "W")]),
            _track(1, 1, ["N", "S"], [("N", "S")]),
        ],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 1, 1, "N")],
    )
    board = compile_board(level, _placement())

    first = step(initial_state(level), board)
    second = step(first.state, board)

    assert first.state.actors[0].cell == CellCoord(row=0, col=1)
    assert first.state.actors[0].direction is Direction.EAST
    assert second.state.status is SimulationStatus.SUCCESS
    assert second.state.actors[0].cell == CellCoord(row=1, col=1)
    assert second.state.actors[0].direction is Direction.SOUTH
    assert second.events[0].exit_port is Port.SOUTH


def test_simulate_reports_derailment_when_an_actor_has_no_enterable_track() -> None:
    level = _level(
        rows=1,
        cols=1,
        tracks=[],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 0, 0)],
    )

    result = simulate(level, _placement())

    assert result.status is SimulationStatus.FAILURE
    assert result.failure is not None
    assert result.failure.reason is FailureReason.DERAILMENT
    assert result.events == (
        SimulationEvent(
            sequence=0,
            tick=1,
            kind=EventKind.FAILURE,
            actor_ids=("train-1",),
            source_cell=CellCoord(row=0, col=0),
            entry_port=Port.WEST,
            reason=FailureReason.DERAILMENT,
        ),
    )


def test_simulate_reports_dead_end_before_leaving_an_unconnected_port() -> None:
    level = _level(
        rows=1,
        cols=2,
        tracks=[_track(0, 0, ["E", "W"], [("E", "W")])],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 0, 1)],
    )

    result = simulate(level, _placement())

    assert result.failure is not None
    assert result.failure.reason is FailureReason.DEAD_END
    assert result.events[-1].exit_port is Port.EAST


def test_simulate_resolves_same_cell_collision_atomically() -> None:
    level = _level(
        rows=1,
        cols=3,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
            _track(0, 2, ["E", "W"], [("E", "W")]),
        ],
        actors=[_actor("beta", 0, 2, "W"), _actor("alpha", 0, 0, "E")],
        arrivals=[_arrival("alpha", 0, 1), _arrival("beta", 0, 1)],
    )

    result = simulate(level, _placement())

    assert result.failure is not None
    assert result.failure.reason is FailureReason.COLLISION
    assert result.failure.actor_ids == ("alpha", "beta")
    assert tuple(actor.cell for actor in result.final_state.actors) == (
        CellCoord(row=0, col=0),
        CellCoord(row=0, col=2),
    )
    assert result.events[0].kind is EventKind.FAILURE


def test_simulate_rejects_an_arrival_order_violation_atomically() -> None:
    level = _level(
        rows=1,
        cols=4,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
            _track(0, 2, ["E", "W"], [("E", "W")]),
            _track(0, 3, ["E", "W"], [("E", "W")]),
        ],
        actors=[_actor("beta", 0, 3, "W"), _actor("alpha", 0, 0, "E")],
        arrivals=[_arrival("alpha", 0, 1), _arrival("beta", 0, 2)],
        arrival_order=["beta", "alpha"],
    )

    result = simulate(level, _placement())

    assert result.failure is not None
    assert result.failure.reason is FailureReason.ARRIVAL_ORDER
    assert result.failure.actor_ids == ("alpha",)
    assert tuple(actor.cell for actor in result.final_state.actors) == (
        CellCoord(row=0, col=0),
        CellCoord(row=0, col=3),
    )


def test_simulation_trace_is_deterministic_and_tick_limit_is_structured() -> None:
    level = _level(
        rows=1,
        cols=2,
        tracks=[
            _track(0, 0, ["E", "W"], [("E", "W")]),
            _track(0, 1, ["E", "W"], [("E", "W")]),
        ],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 0, 0)],
    )

    first = simulate(level, _placement(), SimulationOptions(max_ticks=1))
    second = simulate(level, _placement(), SimulationOptions(max_ticks=1))

    assert first == second
    assert first.failure is not None
    assert first.failure.reason is FailureReason.STEP_LIMIT
    assert first.events[-1].tick == 1
    assert first.events[-1].reason is FailureReason.STEP_LIMIT


def test_static_simulation_rejects_dynamic_devices() -> None:
    level = _level(
        rows=1,
        cols=1,
        tracks=[_track(0, 0, ["E", "W"], [("E", "W")])],
        actors=[_actor("train-1", 0, 0, "E")],
        arrivals=[_arrival("train-1", 0, 0)],
        gates=[
            {
                "id": "gate-1",
                "cell": {"row": 0, "col": 0},
                "channel": "blue",
                "initial_state": "open",
            }
        ],
    )

    with pytest.raises(ValueError, match="does not support: gates"):
        simulate(level, _placement())
