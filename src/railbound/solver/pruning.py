"""Sound, deterministic pruning helpers for the placement search.

The graph in this module is deliberately an over-approximation of dynamic
behaviour.  Switches may use either output and gates are treated as open.  A
failed reachability check is therefore safe to prune; a successful check only
means that the simulator must still decide the candidate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Protocol

from railbound.geometry import CellCoord
from railbound.level import Actor, ArrivalRequirement, Level, Port, TunnelEndpoint
from railbound.topology import RailPlacement, rail_piece_transitions


class PruningPredicate(Protocol):
    """A named, complete-placement predicate used by :func:`solve`.

    Returning ``True`` discards the candidate without a replay.  Custom
    predicates must be conservative: they may reject only placements that
    cannot solve the supplied level.
    """

    name: str

    def should_prune(self, level: Level, placement: RailPlacement) -> bool:
        """Return whether ``placement`` is provably unable to solve ``level``."""
        ...


@dataclass(frozen=True, slots=True)
class ConnectivityPruner:
    """Reject placements with no possible static route to every arrival."""

    name: str = "connectivity"

    def should_prune(self, level: Level, placement: RailPlacement) -> bool:
        """Return whether at least one required actor cannot reach its goal."""
        return not arrivals_are_reachable(level, placement)


@dataclass(frozen=True, slots=True)
class ForcedPlacementPruner:
    """Require buildable cells that every potential route must occupy."""

    required_cells: frozenset[CellCoord]
    name: str = "forced_placement"

    @classmethod
    def for_level(cls, level: Level) -> ForcedPlacementPruner:
        """Build the conservative forced-cell set for ``level``."""
        return cls(required_cells=forced_placement_cells(level))

    def should_prune(self, level: Level, placement: RailPlacement) -> bool:
        """Return whether a required buildable cell has no placed rail."""
        del level
        occupied = frozenset(rail.cell for rail in placement.rails)
        return not self.required_cells <= occupied


@dataclass(frozen=True, slots=True)
class _TraversalIndex:
    """Port transitions for one complete or partially specified placement."""

    exits: dict[tuple[CellCoord, Port], tuple[Port, ...]]
    accepted_ports: dict[CellCoord, frozenset[Port]]
    tunnels: dict[tuple[CellCoord, Port], tuple[CellCoord, Port]]


_PORTS: tuple[Port, ...] = (Port.NORTH, Port.EAST, Port.SOUTH, Port.WEST)
_OPPOSITE_PORT: dict[Port, Port] = {
    Port.NORTH: Port.SOUTH,
    Port.EAST: Port.WEST,
    Port.SOUTH: Port.NORTH,
    Port.WEST: Port.EAST,
}
_PORT_OFFSETS: dict[Port, tuple[int, int]] = {
    Port.NORTH: (-1, 0),
    Port.EAST: (0, 1),
    Port.SOUTH: (1, 0),
    Port.WEST: (0, -1),
}


def arrivals_are_reachable(
    level: Level,
    placement: RailPlacement,
    *,
    open_cells: Iterable[CellCoord] = (),
) -> bool:
    """Return whether an over-approximated route exists for every arrival.

    ``open_cells`` represent not-yet-assigned buildable cells during partial
    search.  They accept every port and may connect any two distinct ports.
    This makes partial checks optimistic and therefore safe.
    """
    index = _build_traversal_index(level, placement, frozenset(open_cells))
    actors = {actor.id: actor for actor in level.actors}
    return all(
        _arrival_is_reachable(level, actors[requirement.actor_id], requirement, index)
        for requirement in level.win_condition.required_arrivals
    )


def forced_placement_cells(level: Level) -> frozenset[CellCoord]:
    """Return buildable cells that every coarse actor-to-goal route needs.

    Actor starts and arrival targets without an existing traversable object
    are always required.  Other cells are included only when removing the
    cell disconnects an actor from its required target in an optimistic cell
    graph.  Tunnel pairs are included as extra graph edges.
    """
    buildable = frozenset(level.buildable_cells)
    existing = {
        *(track.cell for track in level.fixed_tracks),
        *(switch.cell for switch in level.switches),
        *(endpoint.cell for endpoint in level.tunnel_endpoints),
    }
    actors = {actor.id: actor for actor in level.actors}
    required: set[CellCoord] = set()
    for requirement in level.win_condition.required_arrivals:
        actor_cell = actors[requirement.actor_id].cell
        if actor_cell in buildable and actor_cell not in existing:
            required.add(actor_cell)
        if requirement.target_cell in buildable and requirement.target_cell not in existing:
            required.add(requirement.target_cell)

    nodes = frozenset((*buildable, *existing))
    neighbours = _cell_neighbours(level, nodes)
    for requirement in level.win_condition.required_arrivals:
        start = actors[requirement.actor_id].cell
        target = requirement.target_cell
        if start == target or not _cell_path_exists(start, target, neighbours):
            continue
        for cell in buildable - existing - {start, target}:
            if not _cell_path_exists(start, target, neighbours, blocked=cell):
                required.add(cell)
    return frozenset(required)


def selected_cells_are_reachable(
    level: Level,
    selected_cells: Iterable[CellCoord],
) -> bool:
    """Cheaply reject a chosen cell set before assigning rail shapes."""
    selected = frozenset(selected_cells)
    existing = {
        *(track.cell for track in level.fixed_tracks),
        *(switch.cell for switch in level.switches),
        *(endpoint.cell for endpoint in level.tunnel_endpoints),
    }
    nodes = frozenset((*selected, *existing))
    neighbours = _cell_neighbours(level, nodes)
    actors = {actor.id: actor for actor in level.actors}
    return all(
        _cell_path_exists(
            actors[requirement.actor_id].cell,
            requirement.target_cell,
            neighbours,
        )
        for requirement in level.win_condition.required_arrivals
    )


def minimum_required_rails(level: Level) -> int:
    """Return a safe lower bound from optimistic weighted cell paths."""
    buildable = frozenset(level.buildable_cells)
    existing = {
        *(track.cell for track in level.fixed_tracks),
        *(switch.cell for switch in level.switches),
        *(endpoint.cell for endpoint in level.tunnel_endpoints),
    }
    nodes = frozenset((*buildable, *existing))
    neighbours = _cell_neighbours(level, nodes)
    rail_required = buildable - existing
    actors = {actor.id: actor for actor in level.actors}
    lower_bounds = (
        _minimum_buildable_cells(
            actors[requirement.actor_id].cell,
            requirement.target_cell,
            neighbours,
            rail_required,
        )
        for requirement in level.win_condition.required_arrivals
    )
    return max(lower_bounds, default=0)


def _build_traversal_index(
    level: Level,
    placement: RailPlacement,
    open_cells: frozenset[CellCoord],
) -> _TraversalIndex:
    exits: dict[tuple[CellCoord, Port], tuple[Port, ...]] = {}
    accepted: dict[CellCoord, set[Port]] = {}

    for track in level.fixed_tracks:
        accepted.setdefault(track.cell, set()).update(track.ports)
        for transition in track.transitions:
            _add_transition(exits, track.cell, transition.entry_port, transition.exit_port)
    for rail in placement.rails:
        transitions = rail_piece_transitions(rail.piece_kind, rail.orientation)
        ports = {
            port
            for transition in transitions
            for port in (transition.entry_port, transition.exit_port)
        }
        accepted.setdefault(rail.cell, set()).update(ports)
        for transition in transitions:
            _add_transition(exits, rail.cell, transition.entry_port, transition.exit_port)
    for cell in open_cells:
        accepted[cell] = set(_PORTS)
        for entry_port in _PORTS:
            exits[(cell, entry_port)] = tuple(port for port in _PORTS if port is not entry_port)

    for switch in level.switches:
        ports = (switch.entry_port, *switch.output_ports)
        accepted[switch.cell] = set(ports)
        exits[(switch.cell, switch.entry_port)] = switch.output_ports
        for output in switch.output_ports:
            exits[(switch.cell, output)] = (switch.entry_port,)

    endpoints_by_pair: dict[str, list[TunnelEndpoint]] = {}
    for endpoint in level.tunnel_endpoints:
        accepted.setdefault(endpoint.cell, set()).add(endpoint.entry_port)
        endpoints_by_pair.setdefault(endpoint.pair_id, []).append(endpoint)
    tunnels: dict[tuple[CellCoord, Port], tuple[CellCoord, Port]] = {}
    for endpoints in endpoints_by_pair.values():
        first, second = endpoints
        tunnels[(first.cell, first.entry_port)] = (second.cell, second.exit_port)
        tunnels[(second.cell, second.entry_port)] = (first.cell, first.exit_port)

    return _TraversalIndex(
        exits=exits,
        accepted_ports={cell: frozenset(ports) for cell, ports in accepted.items()},
        tunnels=tunnels,
    )


def _add_transition(
    exits: dict[tuple[CellCoord, Port], tuple[Port, ...]],
    cell: CellCoord,
    first: Port,
    second: Port,
) -> None:
    exits[(cell, first)] = (second,)
    exits[(cell, second)] = (first,)


def _arrival_is_reachable(
    level: Level,
    actor: Actor,
    requirement: ArrivalRequirement,
    index: _TraversalIndex,
) -> bool:
    start = (actor.cell, _OPPOSITE_PORT[Port(actor.direction)])
    queue = deque((start,))
    visited = {start}
    while queue:
        state = queue.popleft()
        for successor in _successors(level, state, index):
            cell, entry_port = successor
            if cell == requirement.target_cell and (
                requirement.entry_port is None or entry_port is requirement.entry_port
            ):
                return True
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)
    return False


def _successors(
    level: Level,
    state: tuple[CellCoord, Port],
    index: _TraversalIndex,
) -> tuple[tuple[CellCoord, Port], ...]:
    tunnel_exit = index.tunnels.get(state)
    if tunnel_exit is not None:
        source_cell, exit_port = tunnel_exit
        target = _cross_boundary(level, source_cell, exit_port, index)
        return () if target is None else (target,)

    cell, entry_port = state
    successors: list[tuple[CellCoord, Port]] = []
    for exit_port in index.exits.get((cell, entry_port), ()):
        target = _cross_boundary(level, cell, exit_port, index)
        if target is not None:
            successors.append(target)
    return tuple(successors)


def _cross_boundary(
    level: Level,
    cell: CellCoord,
    exit_port: Port,
    index: _TraversalIndex,
) -> tuple[CellCoord, Port] | None:
    row_delta, col_delta = _PORT_OFFSETS[exit_port]
    target_cell = CellCoord(row=cell.row + row_delta, col=cell.col + col_delta)
    if not level.is_in_bounds(target_cell):
        return None
    entry_port = _OPPOSITE_PORT[exit_port]
    if entry_port not in index.accepted_ports.get(target_cell, frozenset()):
        return None
    return target_cell, entry_port


def _cell_neighbours(
    level: Level,
    nodes: frozenset[CellCoord],
) -> dict[CellCoord, frozenset[CellCoord]]:
    neighbours: dict[CellCoord, set[CellCoord]] = {cell: set() for cell in nodes}
    for cell in nodes:
        for port in _PORTS:
            row_delta, col_delta = _PORT_OFFSETS[port]
            candidate = CellCoord(row=cell.row + row_delta, col=cell.col + col_delta)
            if candidate in nodes:
                neighbours[cell].add(candidate)
    endpoints_by_pair: dict[str, list[CellCoord]] = {}
    for endpoint in level.tunnel_endpoints:
        endpoints_by_pair.setdefault(endpoint.pair_id, []).append(endpoint.cell)
    for endpoints in endpoints_by_pair.values():
        first, second = endpoints
        neighbours[first].add(second)
        neighbours[second].add(first)
    return {cell: frozenset(values) for cell, values in neighbours.items()}


def _cell_path_exists(
    start: CellCoord,
    target: CellCoord,
    neighbours: dict[CellCoord, frozenset[CellCoord]],
    *,
    blocked: CellCoord | None = None,
) -> bool:
    if start == blocked or target == blocked or start not in neighbours or target not in neighbours:
        return False
    queue = deque((start,))
    visited = {start}
    while queue:
        cell = queue.popleft()
        if cell == target:
            return True
        for neighbour in neighbours[cell]:
            if neighbour != blocked and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return False


def _minimum_buildable_cells(
    start: CellCoord,
    target: CellCoord,
    neighbours: dict[CellCoord, frozenset[CellCoord]],
    buildable: frozenset[CellCoord],
) -> int:
    """Find the least buildable-cell cost of an optimistic cell path."""
    if start not in neighbours or target not in neighbours:
        return 0
    initial_cost = int(start in buildable)
    distances = {start: initial_cost}
    queue: list[tuple[int, int, int, CellCoord]] = [(initial_cost, start.row, start.col, start)]
    while queue:
        distance, _row, _col, cell = heappop(queue)
        if distance != distances[cell]:
            continue
        if cell == target:
            return distance
        for neighbour in neighbours[cell]:
            candidate = distance + int(neighbour in buildable)
            if candidate >= distances.get(neighbour, candidate + 1):
                continue
            distances[neighbour] = candidate
            heappush(queue, (candidate, neighbour.row, neighbour.col, neighbour))
    return 0


__all__ = [
    "ConnectivityPruner",
    "ForcedPlacementPruner",
    "PruningPredicate",
    "arrivals_are_reachable",
    "forced_placement_cells",
    "minimum_required_rails",
]
