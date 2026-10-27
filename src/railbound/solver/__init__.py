"""Deterministic, simulator-authoritative search over rail placements."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from time import perf_counter

from railbound.geometry import CellCoord
from railbound.level import Level, Port
from railbound.simulation import (
    SimulationEvent,
    SimulationOptions,
    SimulationResult,
    SimulationState,
    SimulationStatus,
    initial_state,
    simulate,
    step,
)
from railbound.solver.pruning import (
    ConnectivityPruner,
    ForcedPlacementPruner,
    PruningPredicate,
    arrivals_are_reachable,
    forced_placement_cells,
    minimum_required_rails,
    selected_cells_are_reachable,
)
from railbound.topology import (
    PlacedRail,
    RailPieceKind,
    RailPlacement,
    compile_board,
    rail_piece_transitions,
)


class SolverStatus(StrEnum):
    """The reason a baseline search stopped."""

    SOLVED = "solved"
    UNSOLVABLE = "unsolvable"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class SolverLimits:
    """Deterministic bounds for one call to :func:`solve`.

    ``max_placements`` limits complete candidates considered, including ones
    rejected by complete-placement pruning.  ``max_rails`` limits the largest
    candidate independently of the level inventory.  ``max_ticks`` bounds
    each replay.
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
class SolverOptions:
    """Search optimizations that do not alter simulator semantics.

    Defaults are suited to the interactive workflow.  ``baseline()`` is
    available for benchmarks and diagnostics.  A custom predicate is applied
    only to complete legal placements and must reject only impossible ones.
    """

    connectivity_pruning: bool = True
    forced_placements: bool = True
    heuristic_ordering: bool = True
    transposition_cache: bool = True
    pruning_predicates: tuple[PruningPredicate, ...] = ()

    @classmethod
    def baseline(cls) -> SolverOptions:
        """Return the unoptimized M9 search configuration."""
        return cls(
            connectivity_pruning=False,
            forced_placements=False,
            heuristic_ordering=False,
            transposition_cache=False,
        )


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Counters collected without changing solver or simulator semantics."""

    placements_considered: int
    simulations: int
    runtime_states_visited: int
    events_emitted: int
    maximum_rails_considered: int
    connectivity_branches_pruned: int = 0
    connectivity_placements_pruned: int = 0
    forced_cells: int = 0
    forced_placements_pruned: int = 0
    custom_placements_pruned: int = 0
    transposition_hits: int = 0
    elapsed_seconds: float = 0.0

    @property
    def generated_placements(self) -> int:
        """Return the number of generated candidates considered by the solver."""
        return self.placements_considered

    @property
    def placements_pruned(self) -> int:
        """Return the number of complete candidates removed before replay."""
        return (
            self.connectivity_placements_pruned
            + self.forced_placements_pruned
            + self.custom_placements_pruned
        )


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


@dataclass(frozen=True, slots=True)
class SearchBenchmark:
    """Baseline and optimized results produced under identical limits."""

    baseline: SolverResult
    optimized: SolverResult

    @property
    def simulations_saved(self) -> int:
        """Return how many simulator replays optimization avoided."""
        return self.baseline.statistics.simulations - self.optimized.statistics.simulations

    @property
    def runtime_states_saved(self) -> int:
        """Return how many runtime states optimization avoided visiting."""
        return (
            self.baseline.statistics.runtime_states_visited
            - self.optimized.statistics.runtime_states_visited
        )


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


@dataclass(slots=True)
class _SearchCounters:
    """Mutable counters kept private while the public result stays immutable."""

    placements_considered: int = 0
    simulations: int = 0
    runtime_states_visited: int = 0
    events_emitted: int = 0
    maximum_rails_considered: int = 0
    connectivity_branches_pruned: int = 0
    connectivity_placements_pruned: int = 0
    forced_cells: int = 0
    forced_placements_pruned: int = 0
    custom_placements_pruned: int = 0
    transposition_hits: int = 0


@dataclass(frozen=True, slots=True)
class _ReplaySummary:
    """Only the successful replay and counters needed by placement search."""

    simulation: SimulationResult | None
    runtime_states_visited: int
    events_emitted: int
    transposition_hits: int = 0


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


def solve(
    level: Level,
    limits: SolverLimits | None = None,
    options: SolverOptions | None = None,
) -> SolverResult:
    """Find the first simulator-verified placement within ``limits``.

    Search remains iterative by rail count, so the first solution always uses
    the fewest rails.  Pruning uses optimistic reachability and can therefore
    remove only impossible branches.  Dynamic runtime states are never merged
    across placements; the per-placement cache stops only exact deterministic
    cycles.
    """
    active_limits = limits or SolverLimits()
    active_options = options or SolverOptions()
    started_at = perf_counter()
    natural_capacity = _placement_capacity(
        level,
        cells=tuple(sorted(level.buildable_cells, key=_cell_key)),
        kind_limits=_kind_limits(level),
    )
    placement_limit = active_limits.max_placements
    counters = _SearchCounters()
    predicates = active_options.pruning_predicates
    connectivity_pruner = ConnectivityPruner()
    forced_pruner = ForcedPlacementPruner.for_level(level)
    candidates: Iterator[RailPlacement]
    if (
        active_options.connectivity_pruning
        or active_options.forced_placements
        or active_options.heuristic_ordering
    ):
        candidates = _iter_optimized_candidates(
            level,
            max_rails=active_limits.max_rails,
            options=active_options,
            counters=counters,
        )
    else:
        candidates = iter_placement_candidates(level, max_rails=active_limits.max_rails)

    if placement_limit == 0:
        return _result(
            status=SolverStatus.LIMIT_REACHED,
            counters=counters,
            started_at=started_at,
        )

    for placement in candidates:
        if placement_limit is not None and counters.placements_considered >= placement_limit:
            return _result(
                status=SolverStatus.LIMIT_REACHED,
                counters=counters,
                started_at=started_at,
            )

        counters.placements_considered += 1
        counters.maximum_rails_considered = max(
            counters.maximum_rails_considered, len(placement.rails)
        )
        if active_options.forced_placements and forced_pruner.should_prune(level, placement):
            counters.forced_placements_pruned += 1
            continue
        if active_options.connectivity_pruning and connectivity_pruner.should_prune(
            level, placement
        ):
            counters.connectivity_placements_pruned += 1
            continue
        if any(predicate.should_prune(level, placement) for predicate in predicates):
            counters.custom_placements_pruned += 1
            continue

        replay = _replay_candidate(
            level,
            placement,
            max_ticks=active_limits.max_ticks,
            use_transposition_cache=active_options.transposition_cache,
        )
        counters.simulations += 1
        counters.runtime_states_visited += replay.runtime_states_visited
        counters.events_emitted += replay.events_emitted
        counters.transposition_hits += replay.transposition_hits
        if replay.simulation is not None and replay.simulation.succeeded:
            return SolverResult(
                status=SolverStatus.SOLVED,
                placement=placement,
                simulation=replay.simulation,
                trace=replay.simulation.events,
                statistics=_statistics(counters, started_at),
            )

    status = SolverStatus.UNSOLVABLE
    if active_limits.max_rails is not None and active_limits.max_rails < natural_capacity:
        status = SolverStatus.LIMIT_REACHED
    return _result(
        status=status,
        counters=counters,
        started_at=started_at,
    )


def benchmark_search(
    level: Level,
    limits: SolverLimits | None = None,
) -> SearchBenchmark:
    """Compare M9 baseline and M10 optimized search under the same limits."""
    active_limits = limits or SolverLimits()
    baseline = solve(level, active_limits, SolverOptions.baseline())
    optimized = solve(level, active_limits, SolverOptions())
    return SearchBenchmark(baseline=baseline, optimized=optimized)


def _result(
    *,
    status: SolverStatus,
    counters: _SearchCounters,
    started_at: float,
) -> SolverResult:
    """Create an unsolved result with consistently named empty outputs."""
    return SolverResult(
        status=status,
        placement=None,
        simulation=None,
        trace=(),
        statistics=_statistics(counters, started_at),
    )


def _statistics(counters: _SearchCounters, started_at: float) -> SearchStats:
    """Freeze the counters and elapsed wall time for a public result."""
    return SearchStats(
        placements_considered=counters.placements_considered,
        simulations=counters.simulations,
        runtime_states_visited=counters.runtime_states_visited,
        events_emitted=counters.events_emitted,
        maximum_rails_considered=counters.maximum_rails_considered,
        connectivity_branches_pruned=counters.connectivity_branches_pruned,
        connectivity_placements_pruned=counters.connectivity_placements_pruned,
        forced_cells=counters.forced_cells,
        forced_placements_pruned=counters.forced_placements_pruned,
        custom_placements_pruned=counters.custom_placements_pruned,
        transposition_hits=counters.transposition_hits,
        elapsed_seconds=perf_counter() - started_at,
    )


def _iter_optimized_candidates(
    level: Level,
    *,
    max_rails: int | None,
    options: SolverOptions,
    counters: _SearchCounters,
) -> Iterator[RailPlacement]:
    """Yield complete candidates after safe partial-graph pruning."""
    cells = tuple(sorted(level.buildable_cells, key=_cell_key))
    if options.heuristic_ordering:
        cells = tuple(sorted(cells, key=lambda cell: _heuristic_cell_key(level, cell)))
    kind_limits = _kind_limits(level)
    capacity = _placement_capacity(level, cells=cells, kind_limits=kind_limits)
    if max_rails is not None:
        capacity = min(capacity, max_rails)

    forced = forced_placement_cells(level) if options.forced_placements else frozenset()
    minimum_rails = minimum_required_rails(level) if options.forced_placements else 0
    counters.forced_cells = len(forced)
    optional_cells = tuple(cell for cell in cells if cell not in forced)
    forced_cells = tuple(cell for cell in cells if cell in forced)

    yield RailPlacement(level_id=level.id)
    for rail_count in range(max(1, minimum_rails), capacity + 1):
        optional_count = rail_count - len(forced_cells)
        if optional_count < 0 or optional_count > len(optional_cells):
            continue
        for optional_selection in combinations(optional_cells, optional_count):
            selected = (*forced_cells, *optional_selection)
            if options.connectivity_pruning and not selected_cells_are_reachable(level, selected):
                counters.connectivity_branches_pruned += 1
                continue
            ordered_selection = tuple(
                sorted(selected, key=lambda cell: _assignment_cell_key(level, cell))
            )
            yield from _assign_piece_kinds(
                level,
                selected_cells=ordered_selection,
                kind_limits=kind_limits,
                connectivity_pruning=options.connectivity_pruning,
                heuristic_ordering=options.heuristic_ordering,
                counters=counters,
            )


def _assign_piece_kinds(
    level: Level,
    *,
    selected_cells: tuple[CellCoord, ...],
    kind_limits: dict[RailPieceKind, int | None],
    connectivity_pruning: bool,
    heuristic_ordering: bool,
    counters: _SearchCounters,
) -> Iterator[RailPlacement]:
    """Assign rail topology to a fixed cell set with optimistic look-ahead."""
    rails: list[PlacedRail] = []
    used_by_kind: dict[RailPieceKind, int] = {}

    def visit(cell_index: int) -> Iterator[RailPlacement]:
        open_cells = selected_cells[cell_index:]
        partial = RailPlacement(level_id=level.id, rails=tuple(rails))
        if connectivity_pruning and not arrivals_are_reachable(
            level,
            partial,
            open_cells=open_cells,
        ):
            counters.connectivity_branches_pruned += 1
            return
        if cell_index == len(selected_cells):
            yield RailPlacement(
                level_id=level.id,
                rails=tuple(sorted(rails, key=lambda rail: _cell_key(rail.cell))),
            )
            return

        cell = selected_cells[cell_index]
        options = _rail_options(kind_limits)
        if heuristic_ordering:
            options = tuple(
                sorted(
                    options,
                    key=lambda option: _piece_option_key(level, cell, option),
                )
            )
        for piece_kind, orientation in options:
            limit = kind_limits[piece_kind]
            if limit is not None and used_by_kind.get(piece_kind, 0) >= limit:
                continue
            rails.append(PlacedRail(cell=cell, piece_kind=piece_kind, orientation=orientation))
            used_by_kind[piece_kind] = used_by_kind.get(piece_kind, 0) + 1
            yield from visit(cell_index + 1)
            used_by_kind[piece_kind] -= 1
            if used_by_kind[piece_kind] == 0:
                del used_by_kind[piece_kind]
            rails.pop()

    yield from visit(0)


def _rail_options(
    kind_limits: dict[RailPieceKind, int | None],
) -> tuple[tuple[RailPieceKind, Port], ...]:
    """Return canonical shape/orientation pairs allowed by the inventory."""
    return tuple(
        (piece_kind, orientation)
        for piece_kind in _PIECE_KINDS
        if piece_kind in kind_limits
        for orientation in _ORIENTATIONS[piece_kind]
    )


def _heuristic_cell_key(level: Level, cell: CellCoord) -> tuple[int, int, int, int]:
    """Prefer cells near a shortest actor-to-arrival corridor."""
    actors = {actor.id: actor for actor in level.actors}
    scores = [
        _manhattan(actors[requirement.actor_id].cell, cell)
        + _manhattan(cell, requirement.target_cell)
        - _manhattan(actors[requirement.actor_id].cell, requirement.target_cell)
        for requirement in level.win_condition.required_arrivals
    ]
    distance_from_actor = min(_manhattan(actor.cell, cell) for actor in level.actors)
    return min(scores, default=0), distance_from_actor, cell.row, cell.col


def _assignment_cell_key(level: Level, cell: CellCoord) -> tuple[int, int, int, int, int]:
    """Assign actor starts and goals before less constrained cells."""
    actor_cells = {actor.cell for actor in level.actors}
    goal_cells = {requirement.target_cell for requirement in level.win_condition.required_arrivals}
    priority = 0 if cell in actor_cells else 1 if cell in goal_cells else 2
    return priority, *_heuristic_cell_key(level, cell)


def _piece_option_key(
    level: Level,
    cell: CellCoord,
    option: tuple[RailPieceKind, Port],
) -> tuple[int, int, int]:
    """Prefer a shape containing ports demanded at an actor or exact goal."""
    piece_kind, orientation = option
    ports = {
        port
        for transition in rail_piece_transitions(piece_kind, orientation)
        for port in (transition.entry_port, transition.exit_port)
    }
    demanded = {
        _opposite_port(Port(actor.direction)) for actor in level.actors if actor.cell == cell
    }
    demanded.update(
        requirement.entry_port
        for requirement in level.win_condition.required_arrivals
        if requirement.target_cell == cell and requirement.entry_port is not None
    )
    matches = len(ports & demanded)
    kind_rank = _PIECE_KINDS.index(piece_kind)
    orientation_rank = _ORIENTATIONS[piece_kind].index(orientation)
    return -matches, kind_rank, orientation_rank


def _manhattan(first: CellCoord, second: CellCoord) -> int:
    return abs(first.row - second.row) + abs(first.col - second.col)


def _opposite_port(port: Port) -> Port:
    return {
        Port.NORTH: Port.SOUTH,
        Port.EAST: Port.WEST,
        Port.SOUTH: Port.NORTH,
        Port.WEST: Port.EAST,
    }[port]


def _replay_candidate(
    level: Level,
    placement: RailPlacement,
    *,
    max_ticks: int,
    use_transposition_cache: bool,
) -> _ReplaySummary:
    """Replay one candidate, stopping exact deterministic runtime cycles."""
    if not use_transposition_cache:
        simulation = simulate(level, placement, SimulationOptions(max_ticks=max_ticks))
        return _ReplaySummary(
            simulation=simulation if simulation.succeeded else None,
            runtime_states_visited=simulation.final_state.tick + 1,
            events_emitted=len(simulation.events),
        )

    board = compile_board(level, placement)
    state = initial_state(level)
    events: list[SimulationEvent] = []
    visited = {_runtime_state_key(state)}
    runtime_states_visited = 1
    while state.status is SimulationStatus.RUNNING and state.tick < max_ticks:
        step_result = step(state, board)
        state = step_result.state
        events.extend(step_result.events)
        runtime_states_visited += 1
        if state.status is not SimulationStatus.RUNNING:
            break
        key = _runtime_state_key(state)
        if key in visited:
            return _ReplaySummary(
                simulation=None,
                runtime_states_visited=runtime_states_visited,
                events_emitted=len(events),
                transposition_hits=1,
            )
        visited.add(key)

    if state.status is SimulationStatus.SUCCESS:
        simulation = SimulationResult(final_state=state, events=tuple(events))
        return _ReplaySummary(
            simulation=simulation,
            runtime_states_visited=runtime_states_visited,
            events_emitted=len(events),
        )
    step_limit_event = 1 if state.status is SimulationStatus.RUNNING else 0
    return _ReplaySummary(
        simulation=None,
        runtime_states_visited=runtime_states_visited,
        events_emitted=len(events) + step_limit_event,
    )


def _runtime_state_key(state: SimulationState) -> tuple[object, ...]:
    """Exclude tick and event numbering while retaining every semantic value."""
    return (
        tuple(
            (
                actor.actor_id,
                actor.cell.row,
                actor.cell.col,
                actor.direction,
                actor.status,
            )
            for actor in state.actors
        ),
        state.arrived_actor_ids,
        tuple((switch.switch_id, switch.selected_output) for switch in state.switch_states),
        tuple((gate.gate_id, gate.status) for gate in state.gate_states),
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
    "ConnectivityPruner",
    "ForcedPlacementPruner",
    "PruningPredicate",
    "SearchBenchmark",
    "SearchStats",
    "SolverLimits",
    "SolverOptions",
    "SolverResult",
    "SolverStatistics",
    "SolverStatus",
    "benchmark_search",
    "iter_placement_candidates",
    "solve",
]
