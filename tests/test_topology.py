from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from railbound.geometry import CellCoord
from railbound.level import Level, Port, TrackTransition
from railbound.placement_io import load_placement, save_placement
from railbound.topology import (
    PlacedRail,
    PortNode,
    RailPieceKind,
    RailPlacement,
    compile_board,
    rail_piece_transitions,
    validate_placement,
)


def _level(
    *,
    rows: int = 1,
    cols: int = 3,
    buildable_cells: list[tuple[int, int]] | None = None,
    total_budget: int | None = None,
    piece_counts: list[dict[str, object]] | None = None,
    fixed_tracks: list[dict[str, object]] | None = None,
) -> Level:
    buildable = buildable_cells
    if buildable is None:
        buildable = [(0, col) for col in range(cols)]
    inventory: dict[str, object] = {}
    if total_budget is not None:
        inventory["total_budget"] = total_budget
    if piece_counts is not None:
        inventory["piece_counts"] = piece_counts
    if not inventory:
        inventory["total_budget"] = len(buildable)
    return Level.model_validate(
        {
            "id": "topology-level",
            "rows": rows,
            "cols": cols,
            "buildable_cells": [{"row": row, "col": col} for row, col in buildable],
            "fixed_tracks": fixed_tracks or [],
            "actors": [
                {
                    "id": "train-1",
                    "cell": {"row": 0, "col": 0},
                    "direction": "E",
                }
            ],
            "rail_inventory": inventory,
            "win_condition": {
                "required_arrivals": [
                    {
                        "actor_id": "train-1",
                        "target_cell": {"row": 0, "col": 0},
                    }
                ]
            },
        }
    )


def _placement(*rails: PlacedRail) -> RailPlacement:
    return RailPlacement(level_id="topology-level", rails=rails)


def _rail(row: int, col: int, kind: RailPieceKind, orientation: Port) -> PlacedRail:
    return PlacedRail(
        cell=CellCoord(row=row, col=col),
        piece_kind=kind,
        orientation=orientation,
    )


def test_compiles_straights_and_matching_adjacency() -> None:
    level = _level(total_budget=3, piece_counts=[{"piece_kind": "straight", "count": 3}])
    board = compile_board(
        level,
        _placement(
            _rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST),
            _rail(0, 1, RailPieceKind.STRAIGHT, Port.EAST),
            _rail(0, 2, RailPieceKind.STRAIGHT, Port.EAST),
        ),
    )

    middle_west = PortNode(cell=CellCoord(row=0, col=1), port=Port.WEST)
    assert board.transition(middle_west) == PortNode(cell=CellCoord(row=0, col=1), port=Port.EAST)
    assert board.adjacent(PortNode(cell=CellCoord(row=0, col=0), port=Port.EAST)) == middle_west
    assert len(board.adjacency_connections) == 2
    assert len(board.connected_components) == 1


def test_curve_transition_is_rotated_from_its_canonical_definition() -> None:
    level = _level(
        rows=2,
        cols=2,
        buildable_cells=[(0, 0)],
        piece_counts=[{"piece_kind": "curve", "count": 1}],
    )
    board = compile_board(
        level,
        _placement(_rail(0, 0, RailPieceKind.CURVE, Port.SOUTH)),
    )

    assert rail_piece_transitions(RailPieceKind.CURVE, Port.SOUTH) == (
        TrackTransition(entry_port=Port.SOUTH, exit_port=Port.WEST),
    )
    assert board.transition(PortNode(CellCoord(row=0, col=0), Port.SOUTH)) == PortNode(
        CellCoord(row=0, col=0), Port.WEST
    )


def test_crossing_preserves_independent_paths() -> None:
    level = _level(
        rows=1,
        cols=1,
        buildable_cells=[(0, 0)],
        piece_counts=[{"piece_kind": "crossing", "count": 1}],
    )
    board = compile_board(
        level,
        _placement(_rail(0, 0, RailPieceKind.CROSSING, Port.NORTH)),
    )

    north = PortNode(CellCoord(row=0, col=0), Port.NORTH)
    east = PortNode(CellCoord(row=0, col=0), Port.EAST)
    assert board.transition(north) == PortNode(CellCoord(row=0, col=0), Port.SOUTH)
    assert board.transition(east) == PortNode(CellCoord(row=0, col=0), Port.WEST)
    assert len(board.connected_components) == 2
    assert all(len(component) == 2 for component in board.connected_components)


def test_neighbouring_rails_without_matching_ports_remain_disconnected() -> None:
    level = _level(
        cols=2,
        total_budget=2,
        piece_counts=[
            {"piece_kind": "straight", "count": 2},
        ],
    )
    board = compile_board(
        level,
        _placement(
            _rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST),
            _rail(0, 1, RailPieceKind.STRAIGHT, Port.NORTH),
        ),
    )

    right_edge = PortNode(CellCoord(row=0, col=0), Port.EAST)
    assert board.adjacent(right_edge) is None
    assert right_edge in board.unconnected_ports
    assert len(board.connected_components) == 2


def test_boundary_exits_identify_ports_facing_out_of_bounds() -> None:
    level = _level(
        rows=1,
        cols=1,
        buildable_cells=[(0, 0)],
        piece_counts=[{"piece_kind": "straight", "count": 1}],
    )
    board = compile_board(
        level,
        _placement(_rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST)),
    )

    assert board.boundary_exits == (
        PortNode(CellCoord(row=0, col=0), Port.EAST),
        PortNode(CellCoord(row=0, col=0), Port.WEST),
    )


def test_fixed_tracks_compile_with_placed_rails() -> None:
    level = _level(
        cols=2,
        buildable_cells=[(0, 1)],
        piece_counts=[{"piece_kind": "straight", "count": 1}],
        fixed_tracks=[
            {
                "id": "fixed-start",
                "cell": {"row": 0, "col": 0},
                "ports": ["E", "W"],
                "transitions": [{"entry_port": "E", "exit_port": "W"}],
            }
        ],
    )
    board = compile_board(
        level,
        _placement(_rail(0, 1, RailPieceKind.STRAIGHT, Port.EAST)),
    )

    fixed = board.piece_at(CellCoord(row=0, col=0))
    assert fixed is not None
    assert fixed.source == "fixed"
    assert board.adjacent(PortNode(CellCoord(row=0, col=0), Port.EAST)) == PortNode(
        CellCoord(row=0, col=1), Port.WEST
    )


@pytest.mark.parametrize(
    ("placement", "message"),
    [
        (
            RailPlacement(level_id="other-level"),
            "targets level",
        ),
        (
            _placement(_rail(0, 2, RailPieceKind.STRAIGHT, Port.EAST)),
            "not buildable",
        ),
    ],
)
def test_validate_placement_rejects_wrong_level_or_illegal_cell(
    placement: RailPlacement, message: str
) -> None:
    level = _level(cols=3, buildable_cells=[(0, 0)])

    with pytest.raises(ValueError, match=message):
        validate_placement(level, placement)


def test_validate_placement_rejects_budget_and_piece_count_overruns() -> None:
    level = _level(
        cols=2,
        total_budget=1,
        piece_counts=[{"piece_kind": "straight", "count": 1}],
    )
    over_budget = _placement(
        _rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST),
        _rail(0, 1, RailPieceKind.STRAIGHT, Port.EAST),
    )
    unavailable_kind = _placement(_rail(0, 0, RailPieceKind.CURVE, Port.NORTH))

    with pytest.raises(ValueError, match="total_budget"):
        validate_placement(level, over_budget)
    with pytest.raises(ValueError, match="absent from rail_inventory"):
        validate_placement(level, unavailable_kind)


def test_placement_rejects_duplicate_cells() -> None:
    with pytest.raises(ValidationError, match="duplicate cell"):
        _placement(
            _rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST),
            _rail(0, 0, RailPieceKind.CURVE, Port.NORTH),
        )


def test_isolated_rail_components_are_reported() -> None:
    level = _level(
        total_budget=2,
        piece_counts=[{"piece_kind": "straight", "count": 2}],
    )
    board = compile_board(
        level,
        _placement(
            _rail(0, 0, RailPieceKind.STRAIGHT, Port.EAST),
            _rail(0, 2, RailPieceKind.STRAIGHT, Port.EAST),
        ),
    )

    assert len(board.connected_components) == 2
    assert {component[0].cell for component in board.connected_components} == {
        CellCoord(row=0, col=0),
        CellCoord(row=0, col=2),
    }


def test_placement_json_round_trip(tmp_path: Path) -> None:
    placement = _placement(_rail(0, 1, RailPieceKind.STRAIGHT, Port.EAST))
    path = tmp_path / "placements" / "candidate.json"

    save_placement(placement, path)

    assert load_placement(path) == placement
