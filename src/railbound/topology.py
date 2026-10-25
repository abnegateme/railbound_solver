"""Static port-level topology compiled from a level and rail placement.

The semantic :mod:`railbound.level` document says which cells can receive
rails.  This module turns fixed and placed rails into a graph whose vertices
are cell-edge ports.  It intentionally contains no actor movement or dynamic
device behavior; those belong to later simulation milestones.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from railbound.geometry import CellCoord
from railbound.level import (
    Button,
    Gate,
    Identifier,
    Level,
    Port,
    Switch,
    TrackPiece,
    TrackTransition,
    TunnelEndpoint,
)

PLACEMENT_SCHEMA_VERSION: Final = 1


class _FrozenModel(BaseModel):
    """Base class for immutable, validated placement documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailPieceKind(StrEnum):
    """The static, placeable rail shapes currently supported by M5."""

    STRAIGHT = "straight"
    CURVE = "curve"
    CROSSING = "crossing"


class PlacedRail(_FrozenModel):
    """One static rail selected for a buildable cell.

    ``orientation`` rotates the shape's canonical north-facing definition
    clockwise.  Rotations that happen to produce identical topology (such as
    a north- and south-oriented straight) are retained as authored; M5 does
    not yet choose canonical representations for search.
    """

    cell: CellCoord
    piece_kind: RailPieceKind
    orientation: Port = Port.NORTH


class RailPlacement(_FrozenModel):
    """A portable set of proposed rails for one semantic level."""

    schema_version: Literal[1] = PLACEMENT_SCHEMA_VERSION
    level_id: Identifier
    rails: tuple[PlacedRail, ...] = ()

    @model_validator(mode="after")
    def unique_cells(self) -> Self:
        """Require at most one placed rail in any semantic cell."""
        counts = Counter((rail.cell.row, rail.cell.col) for rail in self.rails)
        duplicates = sorted(cell for cell, count in counts.items() if count > 1)
        if duplicates:
            row, col = duplicates[0]
            raise ValueError(f"rail placement contains duplicate cell ({row}, {col})")
        return self


class ConnectionKind(StrEnum):
    """Whether two port nodes connect inside a rail or across cells."""

    INTERNAL = "internal"
    ADJACENT = "adjacent"


@dataclass(frozen=True, slots=True)
class PortNode:
    """One named connection point on the boundary of a semantic cell."""

    cell: CellCoord
    port: Port


@dataclass(frozen=True, slots=True)
class PortConnection:
    """An undirected edge in the compiled port graph."""

    first: PortNode
    second: PortNode
    kind: ConnectionKind


@dataclass(frozen=True, slots=True)
class CompiledPiece:
    """One fixed or placed static rail with resolved internal transitions."""

    cell: CellCoord
    ports: tuple[Port, ...]
    transitions: tuple[TrackTransition, ...]
    source: Literal["fixed", "placement"]
    id: str | None = None
    piece_kind: RailPieceKind | None = None
    orientation: Port | None = None

    def transition(self, entry_port: Port) -> Port | None:
        """Return the paired exit port for an entry, or ``None`` if absent."""
        for transition in self.transitions:
            if transition.entry_port == entry_port:
                return transition.exit_port
            if transition.exit_port == entry_port:
                return transition.entry_port
        return None


@dataclass(frozen=True, slots=True)
class CompiledBoard:
    """Static port graph plus immutable definitions of dynamic devices.

    Each port appears in at most one internal connection and at most one
    adjacency connection.  The two connection kinds stay separate so callers
    can distinguish travelling through a rail from crossing a cell boundary.
    """

    level_id: str
    rows: int
    cols: int
    pieces: tuple[CompiledPiece, ...]
    internal_connections: tuple[PortConnection, ...]
    adjacency_connections: tuple[PortConnection, ...]
    boundary_exits: tuple[PortNode, ...]
    unconnected_ports: tuple[PortNode, ...]
    connected_components: tuple[tuple[PortNode, ...], ...]
    switches: tuple[Switch, ...] = ()
    buttons: tuple[Button, ...] = ()
    gates: tuple[Gate, ...] = ()
    tunnel_endpoints: tuple[TunnelEndpoint, ...] = ()

    def piece_at(self, cell: CellCoord) -> CompiledPiece | None:
        """Return the static rail occupying ``cell``, if any."""
        return next((piece for piece in self.pieces if piece.cell == cell), None)

    def transition(self, entry: PortNode) -> PortNode | None:
        """Follow an internal rail transition from ``entry``, if it exists."""
        piece = self.piece_at(entry.cell)
        if piece is None:
            return None
        exit_port = piece.transition(entry.port)
        if exit_port is None:
            return None
        return PortNode(cell=entry.cell, port=exit_port)

    def adjacent(self, port: PortNode) -> PortNode | None:
        """Return the matching neighbouring port connected across a cell edge."""
        for connection in self.adjacency_connections:
            if connection.first == port:
                return connection.second
            if connection.second == port:
                return connection.first
        return None

    def connections_for(self, port: PortNode) -> tuple[PortConnection, ...]:
        """Return all graph edges touching ``port`` in deterministic order."""
        connections = (
            *self.internal_connections,
            *self.adjacency_connections,
        )
        return tuple(
            connection
            for connection in connections
            if connection.first == port or connection.second == port
        )


_PORTS_CLOCKWISE: Final[tuple[Port, ...]] = (
    Port.NORTH,
    Port.EAST,
    Port.SOUTH,
    Port.WEST,
)
_PORT_RANK: Final[dict[Port, int]] = {port: index for index, port in enumerate(_PORTS_CLOCKWISE)}
_PORT_OFFSETS: Final[dict[Port, tuple[int, int]]] = {
    Port.NORTH: (-1, 0),
    Port.EAST: (0, 1),
    Port.SOUTH: (1, 0),
    Port.WEST: (0, -1),
}
_OPPOSITE_PORT: Final[dict[Port, Port]] = {
    Port.NORTH: Port.SOUTH,
    Port.EAST: Port.WEST,
    Port.SOUTH: Port.NORTH,
    Port.WEST: Port.EAST,
}
_CANONICAL_TRANSITIONS: Final[dict[RailPieceKind, tuple[tuple[Port, Port], ...]]] = {
    RailPieceKind.STRAIGHT: ((Port.NORTH, Port.SOUTH),),
    RailPieceKind.CURVE: ((Port.NORTH, Port.EAST),),
    RailPieceKind.CROSSING: (
        (Port.NORTH, Port.SOUTH),
        (Port.EAST, Port.WEST),
    ),
}


def rail_piece_transitions(
    piece_kind: RailPieceKind, orientation: Port = Port.NORTH
) -> tuple[TrackTransition, ...]:
    """Return the internal transitions for a rotated placeable rail shape."""
    return tuple(
        TrackTransition(
            entry_port=_rotate_port(entry_port, orientation),
            exit_port=_rotate_port(exit_port, orientation),
        )
        for entry_port, exit_port in _CANONICAL_TRANSITIONS[piece_kind]
    )


def validate_placement(level: Level, placement: RailPlacement) -> None:
    """Raise ``ValueError`` when ``placement`` cannot be used on ``level``.

    Validation is deliberately separate from construction because the allowed
    cells and inventory are properties of the selected level, not of a
    portable placement document on its own.
    """
    if placement.level_id != level.id:
        raise ValueError(f"rail placement targets level {placement.level_id!r}, not {level.id!r}")

    buildable_cells = {(cell.row, cell.col) for cell in level.buildable_cells}
    invalid_cells = sorted(
        (rail.cell.row, rail.cell.col)
        for rail in placement.rails
        if (rail.cell.row, rail.cell.col) not in buildable_cells
    )
    if invalid_cells:
        row, col = invalid_cells[0]
        raise ValueError(f"rail placement cell ({row}, {col}) is not buildable")

    rail_count = len(placement.rails)
    if (
        level.rail_inventory.total_budget is not None
        and rail_count > level.rail_inventory.total_budget
    ):
        raise ValueError(
            "rail placement uses "
            f"{rail_count} rails but total_budget is {level.rail_inventory.total_budget}"
        )

    available_by_kind = {
        piece_count.piece_kind: piece_count.count
        for piece_count in level.rail_inventory.piece_counts
    }
    if not available_by_kind:
        return

    used_by_kind = Counter(rail.piece_kind.value for rail in placement.rails)
    unavailable_kinds = sorted(set(used_by_kind) - set(available_by_kind))
    if unavailable_kinds:
        raise ValueError(
            "rail placement uses piece kinds absent from rail_inventory: "
            + ", ".join(unavailable_kinds)
        )
    for piece_kind in sorted(used_by_kind):
        used = used_by_kind[piece_kind]
        available = available_by_kind[piece_kind]
        if used > available:
            raise ValueError(
                f"rail placement uses {used} {piece_kind!r} rails but inventory has {available}"
            )


def compile_board(level: Level, placement: RailPlacement) -> CompiledBoard:
    """Compile validated fixed and placed rails into a deterministic port graph."""
    validate_placement(level, placement)

    pieces = tuple(
        sorted(
            (
                *(_compile_fixed_piece(track) for track in level.fixed_tracks),
                *(_compile_placed_rail(rail) for rail in placement.rails),
            ),
            key=lambda piece: _cell_key(piece.cell),
        )
    )
    nodes = tuple(PortNode(cell=piece.cell, port=port) for piece in pieces for port in piece.ports)
    node_set = set(nodes)
    internal_connections = tuple(
        sorted(
            (
                _connection(
                    PortNode(piece.cell, transition.entry_port),
                    PortNode(piece.cell, transition.exit_port),
                    ConnectionKind.INTERNAL,
                )
                for piece in pieces
                for transition in piece.transitions
            ),
            key=_connection_key,
        )
    )
    adjacency_connections = tuple(
        sorted(
            (
                _connection(node, neighbour, ConnectionKind.ADJACENT)
                for node in nodes
                for neighbour in (_neighbouring_port(node),)
                if neighbour in node_set and _node_key(node) < _node_key(neighbour)
            ),
            key=_connection_key,
        )
    )
    connected_adjacency_nodes = {
        node
        for connection in adjacency_connections
        for node in (connection.first, connection.second)
    }
    boundary_exits = tuple(
        node
        for node in nodes
        if not _is_in_bounds(_neighbouring_port(node).cell, rows=level.rows, cols=level.cols)
    )
    unconnected_ports = tuple(node for node in nodes if node not in connected_adjacency_nodes)
    connected_components = _components(
        nodes,
        (*internal_connections, *adjacency_connections),
    )
    return CompiledBoard(
        level_id=level.id,
        rows=level.rows,
        cols=level.cols,
        pieces=pieces,
        internal_connections=internal_connections,
        adjacency_connections=adjacency_connections,
        boundary_exits=boundary_exits,
        unconnected_ports=unconnected_ports,
        connected_components=connected_components,
        switches=level.switches,
        buttons=level.buttons,
        gates=level.gates,
        tunnel_endpoints=level.tunnel_endpoints,
    )


def _compile_fixed_piece(track: TrackPiece) -> CompiledPiece:
    """Turn a validated M4 fixed track declaration into a compiled piece."""
    return CompiledPiece(
        cell=track.cell,
        ports=tuple(sorted(track.ports, key=_PORT_RANK.__getitem__)),
        transitions=tuple(sorted(track.transitions, key=_transition_key)),
        source="fixed",
        id=track.id,
    )


def _compile_placed_rail(rail: PlacedRail) -> CompiledPiece:
    """Resolve a shape definition and rotation for one placed rail."""
    transitions = rail_piece_transitions(rail.piece_kind, rail.orientation)
    ports = tuple(
        sorted(
            (
                port
                for transition in transitions
                for port in (transition.entry_port, transition.exit_port)
            ),
            key=_PORT_RANK.__getitem__,
        )
    )
    return CompiledPiece(
        cell=rail.cell,
        ports=ports,
        transitions=transitions,
        source="placement",
        piece_kind=rail.piece_kind,
        orientation=rail.orientation,
    )


def _rotate_port(port: Port, orientation: Port) -> Port:
    """Rotate a canonical port clockwise so canonical north maps to orientation."""
    return _PORTS_CLOCKWISE[(_PORT_RANK[port] + _PORT_RANK[orientation]) % len(_PORTS_CLOCKWISE)]


def _neighbouring_port(node: PortNode) -> PortNode:
    """Return the oppositely facing port in the cell adjacent to ``node``."""
    row_delta, col_delta = _PORT_OFFSETS[node.port]
    return PortNode(
        cell=CellCoord(row=node.cell.row + row_delta, col=node.cell.col + col_delta),
        port=_OPPOSITE_PORT[node.port],
    )


def _connection(first: PortNode, second: PortNode, kind: ConnectionKind) -> PortConnection:
    """Create an edge whose endpoint ordering is stable for inspection."""
    if _node_key(second) < _node_key(first):
        first, second = second, first
    return PortConnection(first=first, second=second, kind=kind)


def _components(
    nodes: tuple[PortNode, ...], connections: tuple[PortConnection, ...]
) -> tuple[tuple[PortNode, ...], ...]:
    """Return deterministically ordered connected components without a graph dependency."""
    neighbours: dict[PortNode, set[PortNode]] = {node: set() for node in nodes}
    for connection in connections:
        neighbours[connection.first].add(connection.second)
        neighbours[connection.second].add(connection.first)

    unseen = set(nodes)
    components: list[tuple[PortNode, ...]] = []
    while unseen:
        first = min(unseen, key=_node_key)
        unseen.remove(first)
        queue = deque((first,))
        component = [first]
        while queue:
            current = queue.popleft()
            for neighbour in sorted(neighbours[current], key=_node_key):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(tuple(sorted(component, key=_node_key)))
    return tuple(sorted(components, key=lambda component: _node_key(component[0])))


def _is_in_bounds(cell: CellCoord, *, rows: int, cols: int) -> bool:
    """Return whether a coordinate lies inside compiled board dimensions."""
    return 0 <= cell.row < rows and 0 <= cell.col < cols


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    """Return a coordinate key used for deterministic piece ordering."""
    return cell.row, cell.col


def _node_key(node: PortNode) -> tuple[int, int, int]:
    """Return a deterministic sort key for one port graph vertex."""
    return node.cell.row, node.cell.col, _PORT_RANK[node.port]


def _transition_key(transition: TrackTransition) -> tuple[int, int]:
    """Return an undirected deterministic sort key for an internal transition."""
    first = _PORT_RANK[transition.entry_port]
    second = _PORT_RANK[transition.exit_port]
    return min(first, second), max(first, second)


def _connection_key(connection: PortConnection) -> tuple[int, int, int, int, int, int]:
    """Return a deterministic sort key for a graph edge."""
    return (*_node_key(connection.first), *_node_key(connection.second))
