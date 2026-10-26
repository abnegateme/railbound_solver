"""Deterministic baseline search over rail placements.

The solver intentionally treats :func:`railbound.simulation.simulate` as its
only authority for whether a placement wins.  It enumerates legal placements
in a stable, low-rail-count-first order and leaves pruning, transposition
caches, and heuristics to M10.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from railbound.geometry import CellCoord
from railbound.level import Level, Port
from railbound.simulation import SimulationEvent, SimulationOptions, SimulationResult, simulate
from railbound.topology import PlacedRail, RailPieceKind, RailPlacement


class SolverStatus(StrEnum):
    """The reason a baseline search stopped."""

    SOLVED = "solved"
    UNSOLVABLE = "unsolvable"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class SolverLimits:
    """Deterministic bounds for one call to :func:`solve`.

    ``max_placements`` limits simulator replays.  ``max_rails`` limits the
    largest candidate placement considered, independently of the level's
    inventory.  ``max_ticks`` is forwarded to each simulation replay.
    """

    max_placements: int | None = None
    max_rails: int | None = None
    max_ticks: int = 10_000

    def __post_init__(self) -> None:
        """Reject nonsensical bounds before search begins."""
        if self.max_placements is not None and self.max_placements < 0:
            raise ValueError("max_placements must be non-negative or None")
        if self.max_rails is not None and self.max_rails < 0:
            raise ValueError("max_rails must be non-negative or None")
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be greater than zero")


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Counters collected without changing solver or simulator semantics."""

    placements_considered: int
    simulations: int
    runtime_states_visited: int
    events_emitted: int
    maximum_rails_considered: int

    @property
    def generated_placements(self) -> int:
        """Return the number of generated candidates considered by the solver."""
        return self.placements_considered


SolverStatistics = SearchStats


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The placement and replay that ended a deterministic search.

    An unsolved result has no placement or simulation trace because no
    candidate was verified as successful.  The counters still describe every
    attempted candidate.
    """

    status: SolverStatus
    placement: RailPlacement | None
    simulation: SimulationResult | None
    trace: tuple[SimulationEvent, ...]
    statistics: SearchStats

    @property
    def solved(self) -> bool:
        """Return whether this result contains a simulator-verified solution."""
        return self.status is SolverStatus.SOLVED

    @property
    def succeeded(self) -> bool:
        """Return ``True`` when the contained simulation reached success."""
        return self.solved


_PIECE_KINDS: tuple[RailPieceKind, ...] = (
    RailPieceKind.STRAIGHT,
    RailPieceKind.CURVE,
    RailPieceKind.CROSSING,
)
_ORIENTATIONS: dict[RailPieceKind, tuple[Port, ...]] = {
    RailPieceKind.STRAIGHT: (Port.NORTH, Port.EAST),
    RailPieceKind.CURVE: (Port.NORTH, Port.EAST, Port.SOUTH, Port.WEST),
    RailPieceKind.CROSSING: (Port.NORTH,),
}


def iter_placement_candidates(
    level: Level, *, max_rails: int | None = None
) -> Iterator[RailPlacement]:
    """Yield each legal placement in canonical iterative-deepening order.

    Candidates start with the empty placement and then increase rail count by
    one.  Cells are ordered by ``(row, col)``; within a cell, shapes use the
    declared :class:`RailPieceKind` order and orientations use clockwise port
    order.  Topologically duplicate rotations are intentionally omitted:
    straights use ``N`` and ``E``, and crossings use ``N`` only.
    """
    if max_rails is not None and max_rails < 0:
        raise ValueError("max_rails must be non-negative or None")

    cells = tuple(sorted(level.buildable_cells, key=_cell_key))
    kind_limits = _kind_limits(level)
    capacity = _placement_capacity(level, cells=cells, kind_limits=kind_limits)
    if max_rails is not None:
        capacity = min(capacity, max_rails)

    for rail_count in range(capacity + 1):
        yield from _placements_with_rail_count(
            level_id=level.id,
            cells=cells,
            rail_count=rail_count,
            kind_limits=kind_limits,
        )


def solve(level: Level, limits: SolverLimits | None = None) -> SolverResult:
    """Find the first simulator-verified placement within ``limits``.

    The first solution is deterministic and uses the fewest placed rails among
    candidates that can be generated from the supplied inventory.  No runtime
    state is cached or merged: every candidate is replayed from its initial
    state, preserving dynamic switch, gate, arrival, and progress state.
    """
    active_limits = limits or SolverLimits()
    natural_capacity = _placement_capacity(
        level,
        cells=tuple(sorted(level.buildable_cells, key=_cell_key)),
        kind_limits=_kind_limits(level),
    )
    placement_limit = active_limits.max_placements
    placements_considered = 0
    simulations = 0
    runtime_states_visited = 0
    events_emitted = 0
    maximum_rails_considered = 0

    for placement in iter_placement_candidates(level, max_rails=active_limits.max_rails):
        if placement_limit is not None and placements_considered >= placement_limit:
            return _result(
                status=SolverStatus.LIMIT_REACHED,
                placements_considered=placements_considered,
                simulations=simulations,
                runtime_states_visited=runtime_states_visited,
                events_emitted=events_emitted,
                maximum_rails_considered=maximum_rails_considered,
            )

        placements_considered += 1
        maximum_rails_considered = max(maximum_rails_considered, len(placement.rails))
        simulation = simulate(
            level,
            placement,
            SimulationOptions(max_ticks=active_limits.max_ticks),
        )
        simulations += 1
        runtime_states_visited += simulation.final_state.tick + 1
        events_emitted += len(simulation.events)
        if simulation.succeeded:
            return SolverResult(
                status=SolverStatus.SOLVED,
                placement=placement,
                simulation=simulation,
                trace=simulation.events,
                statistics=SearchStats(
                    placements_considered=placements_considered,
                    simulations=simulations,
                    runtime_states_visited=runtime_states_visited,
                    events_emitted=events_emitted,
                    maximum_rails_considered=maximum_rails_considered,
                ),
            )

    status = SolverStatus.UNSOLVABLE
    if active_limits.max_rails is not None and active_limits.max_rails < natural_capacity:
        status = SolverStatus.LIMIT_REACHED
    return _result(
        status=status,
        placements_considered=placements_considered,
        simulations=simulations,
        runtime_states_visited=runtime_states_visited,
        events_emitted=events_emitted,
        maximum_rails_considered=maximum_rails_considered,
    )


def _result(
    *,
    status: SolverStatus,
    placements_considered: int,
    simulations: int,
    runtime_states_visited: int,
    events_emitted: int,
    maximum_rails_considered: int,
) -> SolverResult:
    """Create an unsolved result with consistently named empty outputs."""
    return SolverResult(
        status=status,
        placement=None,
        simulation=None,
        trace=(),
        statistics=SearchStats(
            placements_considered=placements_considered,
            simulations=simulations,
            runtime_states_visited=runtime_states_visited,
            events_emitted=events_emitted,
            maximum_rails_considered=maximum_rails_considered,
        ),
    )


def _placements_with_rail_count(
    *,
    level_id: str,
    cells: tuple[CellCoord, ...],
    rail_count: int,
    kind_limits: dict[RailPieceKind, int | None],
) -> Iterator[RailPlacement]:
    """Enumerate exactly ``rail_count`` rails with ascending cell indices."""
    rails: list[PlacedRail] = []
    used_by_kind: dict[RailPieceKind, int] = {}

    def visit(first_cell_index: int) -> Iterator[RailPlacement]:
        remaining = rail_count - len(rails)
        if remaining == 0:
            yield RailPlacement(level_id=level_id, rails=tuple(rails))
            return

        last_start = len(cells) - remaining + 1
        for cell_index in range(first_cell_index, last_start):
            cell = cells[cell_index]
            for piece_kind in _PIECE_KINDS:
                limit = kind_limits.get(piece_kind)
                if limit is None and piece_kind not in kind_limits:
                    continue
                if limit is not None and used_by_kind.get(piece_kind, 0) >= limit:
                    continue
                for orientation in _ORIENTATIONS[piece_kind]:
                    rails.append(
                        PlacedRail(cell=cell, piece_kind=piece_kind, orientation=orientation)
                    )
                    used_by_kind[piece_kind] = used_by_kind.get(piece_kind, 0) + 1
                    yield from visit(cell_index + 1)
                    used_by_kind[piece_kind] -= 1
                    if used_by_kind[piece_kind] == 0:
                        del used_by_kind[piece_kind]
                    rails.pop()

    yield from visit(0)


def _kind_limits(level: Level) -> dict[RailPieceKind, int | None]:
    """Return the usable inventory count for each supported piece kind."""
    if not level.rail_inventory.piece_counts:
        return {piece_kind: None for piece_kind in _PIECE_KINDS}

    counts_by_name = {
        piece_count.piece_kind: piece_count.count
        for piece_count in level.rail_inventory.piece_counts
    }
    return {
        piece_kind: counts_by_name[piece_kind.value]
        for piece_kind in _PIECE_KINDS
        if piece_kind.value in counts_by_name
    }


def _placement_capacity(
    level: Level,
    *,
    cells: tuple[CellCoord, ...],
    kind_limits: dict[RailPieceKind, int | None],
) -> int:
    """Return the largest legal number of rails before caller-specific bounds."""
    if not kind_limits:
        return 0
    capacity = len(cells)
    if level.rail_inventory.total_budget is not None:
        capacity = min(capacity, level.rail_inventory.total_budget)
    finite_kind_limits = tuple(limit for limit in kind_limits.values() if limit is not None)
    if finite_kind_limits:
        capacity = min(capacity, sum(finite_kind_limits))
    return capacity


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    """Return the stable row-major sort key for a buildable cell."""
    return cell.row, cell.col


__all__ = [
    "SearchStats",
    "SolverLimits",
    "SolverResult",
    "SolverStatistics",
    "SolverStatus",
    "iter_placement_candidates",
    "solve",
]
