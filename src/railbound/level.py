"""Validated semantic level documents independent of screenshots.

This module deliberately describes *what* is on a Railbound board without
implementing movement or solving.  Pixels and recognition confidence end at
the parsing boundary; :class:`Level` is the stable input contract for later
board compilation and simulation milestones.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from railbound.geometry import CellCoord

LEVEL_SCHEMA_VERSION: Final = 1
RULESET_VERSION: Final = "railbound-observed-v1"

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
Identifier = Annotated[str, Field(min_length=1, pattern=_IDENTIFIER_PATTERN)]
Label = Annotated[str, Field(min_length=1)]
Value = TypeVar("Value")


class _FrozenModel(BaseModel):
    """Base class for immutable, validated level schema values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Port(StrEnum):
    """One of the four track connection points on a cell boundary."""

    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"


class Direction(StrEnum):
    """A cardinal actor travel direction."""

    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"


class ButtonActivation(StrEnum):
    """The observed state effect emitted when an actor enters a button."""

    TOGGLE = "toggle"


class LevelMetadata(_FrozenModel):
    """Optional human-readable provenance that does not affect rules."""

    title: Label | None = None
    world: Label | None = None
    source: Label | None = None


class TerrainCell(_FrozenModel):
    """A terrain label at one cell, kept separate from track and devices.

    Terrain labels are descriptive in M4.  Their movement or placement effects
    are introduced only when a later milestone has observed and documented
    them.
    """

    cell: CellCoord
    kind: Identifier


class TrackTransition(_FrozenModel):
    """A bidirectional internal connection between two ports of one track."""

    entry_port: Port
    exit_port: Port

    @model_validator(mode="after")
    def distinct_ports(self) -> Self:
        """Reject a transition that enters and leaves through the same port."""
        if self.entry_port == self.exit_port:
            raise ValueError("a track transition must use two distinct ports")
        return self


class TrackPiece(_FrozenModel):
    """A fixed semantic rail piece and its explicit internal transitions."""

    id: Identifier
    cell: CellCoord
    ports: tuple[Port, ...]
    transitions: tuple[TrackTransition, ...]
    fixed: bool = True

    @model_validator(mode="after")
    def valid_topology(self) -> Self:
        """Require every declared port to belong to exactly one transition."""
        if len(self.ports) < 2:
            raise ValueError("a track piece must declare at least two ports")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError("track piece ports must be unique")
        if not self.transitions:
            raise ValueError("a track piece must declare at least one transition")

        transition_ports = tuple(
            port
            for transition in self.transitions
            for port in (transition.entry_port, transition.exit_port)
        )
        if any(port not in self.ports for port in transition_ports):
            raise ValueError("track transitions may only reference declared ports")
        if len(set(transition_ports)) != len(transition_ports):
            raise ValueError("each track port must belong to exactly one transition")
        if set(transition_ports) != set(self.ports):
            raise ValueError("track transitions must cover every declared port")
        return self


class Actor(_FrozenModel):
    """The initial state and stable identity of one movable actor."""

    id: Identifier
    cell: CellCoord
    direction: Direction
    actor_type: Identifier = "train"
    number: int | None = None
    color: Label | None = None


class Switch(_FrozenModel):
    """A three-port junction whose selected branch may share a button channel.

    A train entering ``entry_port`` leaves through ``initial_output``.  A
    train entering either output always merges onto ``entry_port``.  A switch
    with no ``channel`` is fixed for the duration of a replay; one with a
    channel toggles when an actor enters a matching button.
    """

    id: Identifier
    cell: CellCoord
    entry_port: Port
    output_ports: tuple[Port, Port]
    initial_output: Port
    channel: Identifier | None = None

    @model_validator(mode="after")
    def valid_outputs(self) -> Self:
        """Require one entry and two distinct selectable outputs."""
        if self.entry_port in self.output_ports:
            raise ValueError("a switch entry port cannot also be an output port")
        if self.output_ports[0] == self.output_ports[1]:
            raise ValueError("a switch must have two distinct output ports")
        if self.initial_output not in self.output_ports:
            raise ValueError("a switch initial_output must be one of its output_ports")
        return self


class Button(_FrozenModel):
    """A channelled device activated when an actor enters its cell."""

    id: Identifier
    cell: CellCoord
    channel: Identifier
    activation_behavior: ButtonActivation


class Gate(_FrozenModel):
    """A channelled gate with its initial observed state."""

    id: Identifier
    cell: CellCoord
    channel: Identifier
    initial_state: Literal["open", "closed"]


class TunnelEndpoint(_FrozenModel):
    """One end of a paired tunnel with its local traversal ports."""

    id: Identifier
    cell: CellCoord
    pair_id: Identifier
    entry_port: Port
    exit_port: Port

    @model_validator(mode="after")
    def distinct_ports(self) -> Self:
        """Reject a tunnel endpoint that has no local crossing direction."""
        if self.entry_port == self.exit_port:
            raise ValueError("a tunnel endpoint must use distinct entry and exit ports")
        return self


class RailPieceCount(_FrozenModel):
    """The available count for one named placeable rail-piece kind."""

    piece_kind: Identifier
    count: Annotated[int, Field(ge=0)]


class RailInventory(_FrozenModel):
    """A total placement budget, per-kind counts, or both."""

    total_budget: Annotated[int, Field(ge=0)] | None = None
    piece_counts: tuple[RailPieceCount, ...] = ()

    @model_validator(mode="after")
    def has_a_limit(self) -> Self:
        """Reject an inventory that leaves permitted placement unconstrained."""
        if self.total_budget is None and not self.piece_counts:
            raise ValueError("rail inventory requires a total_budget or piece_counts")
        piece_kinds = [piece_count.piece_kind for piece_count in self.piece_counts]
        if _duplicates(piece_kinds):
            raise ValueError("rail inventory piece_counts must not repeat a piece_kind")
        return self


class ArrivalRequirement(_FrozenModel):
    """Require one actor to enter a target cell, optionally through one port."""

    actor_id: Identifier
    target_cell: CellCoord
    entry_port: Port | None = None


class CouplingRequirement(_FrozenModel):
    """Require two distinct actors to be coupled by the successful end state."""

    leading_actor_id: Identifier
    trailing_actor_id: Identifier

    @model_validator(mode="after")
    def distinct_actors(self) -> Self:
        """Reject a coupling requirement that refers to one actor twice."""
        if self.leading_actor_id == self.trailing_actor_id:
            raise ValueError("a coupling requirement must reference two distinct actors")
        return self


class WinCondition(_FrozenModel):
    """Explicit completion requirements, without any simulator behavior yet."""

    required_arrivals: tuple[ArrivalRequirement, ...] = ()
    arrival_order: tuple[Identifier, ...] = ()
    required_couplings: tuple[CouplingRequirement, ...] = ()

    @model_validator(mode="after")
    def has_a_requirement(self) -> Self:
        """Require at least one observable condition for success."""
        if not (self.required_arrivals or self.arrival_order or self.required_couplings):
            raise ValueError("win_condition requires an arrival, order, or coupling requirement")
        arrival_actor_ids = [arrival.actor_id for arrival in self.required_arrivals]
        if _duplicates(arrival_actor_ids):
            raise ValueError("win_condition required_arrivals must not repeat an actor_id")
        if _duplicates(self.arrival_order):
            raise ValueError("win_condition arrival_order must not repeat an actor_id")
        if any(actor_id not in arrival_actor_ids for actor_id in self.arrival_order):
            raise ValueError("win_condition arrival_order actors must have required arrivals")
        return self


class Level(_FrozenModel):
    """A complete, image-independent semantic Railbound level document."""

    schema_version: Literal[1] = LEVEL_SCHEMA_VERSION
    ruleset_version: Literal["railbound-observed-v1"] = RULESET_VERSION
    id: Identifier
    rows: Annotated[int, Field(gt=0)]
    cols: Annotated[int, Field(gt=0)]
    metadata: LevelMetadata | None = None
    terrain: tuple[TerrainCell, ...] = ()
    buildable_cells: tuple[CellCoord, ...]
    fixed_tracks: tuple[TrackPiece, ...] = ()
    actors: tuple[Actor, ...]
    switches: tuple[Switch, ...] = ()
    buttons: tuple[Button, ...] = ()
    gates: tuple[Gate, ...] = ()
    tunnel_endpoints: tuple[TunnelEndpoint, ...] = ()
    rail_inventory: RailInventory
    win_condition: WinCondition

    @model_validator(mode="after")
    def valid_references_and_coordinates(self) -> Self:
        """Validate cross-object references and every level-relative coordinate."""
        self._validate_cell_collection(self.terrain, "terrain", lambda value: value.cell)
        self._validate_cell_collection(self.buildable_cells, "buildable_cells", lambda value: value)
        self._validate_cell_collection(self.fixed_tracks, "fixed_tracks", lambda value: value.cell)
        self._validate_cell_collection(self.actors, "actors", lambda value: value.cell)
        self._validate_cell_collection(self.switches, "switches", lambda value: value.cell)
        self._validate_cell_collection(self.buttons, "buttons", lambda value: value.cell)
        self._validate_cell_collection(self.gates, "gates", lambda value: value.cell)
        self._validate_cell_collection(
            self.tunnel_endpoints, "tunnel_endpoints", lambda value: value.cell
        )
        self._validate_cells(
            self.win_condition.required_arrivals,
            "win_condition.required_arrivals",
            lambda value: value.target_cell,
        )

        fixed_track_cells = _cell_keys(track.cell for track in self.fixed_tracks)
        buildable_cells = _cell_keys(self.buildable_cells)
        if fixed_track_cells & buildable_cells:
            raise ValueError("buildable_cells cannot contain a fixed track")
        if any(not track.fixed for track in self.fixed_tracks):
            raise ValueError("fixed_tracks must contain only fixed track pieces")

        entity_ids = [
            *[track.id for track in self.fixed_tracks],
            *[actor.id for actor in self.actors],
            *[switch.id for switch in self.switches],
            *[button.id for button in self.buttons],
            *[gate.id for gate in self.gates],
            *[endpoint.id for endpoint in self.tunnel_endpoints],
        ]
        duplicates = _duplicates(entity_ids)
        if duplicates:
            raise ValueError(f"level entity IDs must be globally unique: {', '.join(duplicates)}")

        actor_ids = {actor.id for actor in self.actors}
        if not actor_ids:
            raise ValueError("a level requires at least one actor")
        self._validate_win_condition_references(actor_ids)
        self._validate_tunnel_pairs()
        self._validate_button_targets()
        return self

    def is_in_bounds(self, cell: CellCoord) -> bool:
        """Return whether a semantic cell lies within this level's dimensions."""
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def _validate_cells(
        self,
        values: tuple[Value, ...],
        name: str,
        coordinate: Callable[[Value], CellCoord],
    ) -> None:
        for value in values:
            cell = coordinate(value)
            if not self.is_in_bounds(cell):
                raise ValueError(f"{name} contains out-of-bounds cell ({cell.row}, {cell.col})")

    def _validate_cell_collection(
        self,
        values: tuple[Value, ...],
        name: str,
        coordinate: Callable[[Value], CellCoord],
    ) -> None:
        self._validate_cells(values, name, coordinate)
        cell_counts = Counter(_cell_key(coordinate(value)) for value in values)
        duplicates = sorted(cell for cell, count in cell_counts.items() if count > 1)
        if duplicates:
            row, col = duplicates[0]
            raise ValueError(f"{name} contains duplicate cell ({row}, {col})")

    def _validate_win_condition_references(self, actor_ids: set[str]) -> None:
        referenced_actor_ids = [
            *[arrival.actor_id for arrival in self.win_condition.required_arrivals],
            *self.win_condition.arrival_order,
            *[
                actor_id
                for coupling in self.win_condition.required_couplings
                for actor_id in (coupling.leading_actor_id, coupling.trailing_actor_id)
            ],
        ]
        unknown_actor_ids = sorted(set(referenced_actor_ids) - actor_ids)
        if unknown_actor_ids:
            raise ValueError(
                "win_condition references unknown actor IDs: " + ", ".join(unknown_actor_ids)
            )

    def _validate_tunnel_pairs(self) -> None:
        endpoints_by_pair: defaultdict[str, list[TunnelEndpoint]] = defaultdict(list)
        for endpoint in self.tunnel_endpoints:
            endpoints_by_pair[endpoint.pair_id].append(endpoint)
        for pair_id, endpoints in endpoints_by_pair.items():
            if len(endpoints) != 2:
                raise ValueError(f"tunnel pair {pair_id!r} must contain exactly two endpoints")
            if endpoints[0].cell == endpoints[1].cell:
                raise ValueError(f"tunnel pair {pair_id!r} endpoints must occupy different cells")

    def _validate_button_targets(self) -> None:
        """Require every simulated button channel to control a device."""
        controlled_channels = {
            *(gate.channel for gate in self.gates),
            *(switch.channel for switch in self.switches if switch.channel is not None),
        }
        unused_channels = sorted({button.channel for button in self.buttons} - controlled_channels)
        if unused_channels:
            raise ValueError(
                "button channels do not control a gate or switch: " + ", ".join(unused_channels)
            )


def _duplicates(values: list[str] | tuple[str, ...]) -> list[str]:
    """Return duplicate identifiers in deterministic lexical order."""
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    """Return a hashable, readable representation of a coordinate."""
    return cell.row, cell.col


def _cell_keys(cells: Iterable[CellCoord]) -> set[tuple[int, int]]:
    """Return coordinate keys for an iterable of cells without exposing mutability."""
    return {_cell_key(cell) for cell in cells}
