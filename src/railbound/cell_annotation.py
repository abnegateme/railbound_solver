"""Composable cell annotations that compile into formal level state and topology."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from railbound.geometry import CellCoord
from railbound.level import (
    Actor,
    ArrivalRequirement,
    Direction,
    Identifier,
    Level,
    LevelMetadata,
    Port,
    RailInventory,
    RailPieceCount,
    Switch,
    TerrainCell,
    TrackPiece,
    TrackTransition,
    WinCondition,
)

PropertyValue = str | int | bool
PropertyKind = Literal["direction", "text", "integer", "boolean"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CellPropertyDefinition(_FrozenModel):
    """One instance value requested when a cell uses a type definition."""

    id: Identifier
    label: str
    value_kind: PropertyKind
    required: bool = True
    choices: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def valid_constraints(self) -> Self:
        if self.choices and self.value_kind != "text":
            raise ValueError("only text properties may declare choices")
        if (self.minimum is not None or self.maximum is not None) and self.value_kind != "integer":
            raise ValueError("only integer properties may declare numeric bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("property minimum cannot exceed maximum")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("property choices must be unique")
        return self


class TerrainFeature(_FrozenModel):
    kind: Literal["terrain"] = "terrain"
    terrain_kind: Identifier
    buildable: bool


class TopologyFeature(_FrozenModel):
    """A fixed local graph fragment expressed as ports and internal transitions."""

    kind: Literal["topology"] = "topology"
    ports: tuple[Port, ...]
    transitions: tuple[TrackTransition, ...]
    fixed: Literal[True] = True

    @model_validator(mode="after")
    def valid_graph_fragment(self) -> Self:
        TrackPiece(
            id="validation-track",
            cell=CellCoord(row=0, col=0),
            ports=self.ports,
            transitions=self.transitions,
            fixed=self.fixed,
        )
        return self


class FixedTrackFeature(_FrozenModel):
    """A fixed two-port rail whose sides are annotation properties."""

    kind: Literal["fixed_track"] = "fixed_track"
    first_port_property: Identifier = "first_port"
    second_port_property: Identifier = "second_port"


class FixedSwitchFeature(_FrozenModel):
    """A fixed three-port switch whose current route is annotation state."""

    kind: Literal["fixed_switch"] = "fixed_switch"
    branch_port_property: Identifier = "branch_port"
    selected_output_property: Identifier = "selected_output"


class ActorFeature(_FrozenModel):
    kind: Literal["actor"] = "actor"
    actor_type: Identifier = "train"
    direction_property: Identifier = "direction"
    number_property: Identifier | None = None
    color_property: Identifier | None = None
    arrival_order_property: Identifier | None = None


class GoalFeature(_FrozenModel):
    kind: Literal["goal"] = "goal"
    goal_type: Literal["arrival"] = "arrival"
    entry_port_property: Identifier | None = None


class StateFeature(_FrozenModel):
    """State declaration available to future graph-state compiler handlers."""

    kind: Literal["state"] = "state"
    states: tuple[Identifier, ...]
    initial_state_property: Identifier

    @model_validator(mode="after")
    def valid_states(self) -> Self:
        if not self.states:
            raise ValueError("a state feature requires at least one state")
        if len(set(self.states)) != len(self.states):
            raise ValueError("state feature states must be unique")
        return self


class TriggerFeature(_FrozenModel):
    """A state transition emitted by an observed event."""

    kind: Literal["trigger"] = "trigger"
    event: Literal["enter", "exit", "tick"]
    operation: Literal["toggle", "set"]
    channel_property: Identifier | None = None


class LinkFeature(_FrozenModel):
    """A graph link whose group is supplied by a cell instance property."""

    kind: Literal["link"] = "link"
    link_kind: Identifier
    group_property: Identifier
    entry_port_property: Identifier | None = None
    exit_port_property: Identifier | None = None


class UnmodeledFeature(_FrozenModel):
    """Recognizable evidence deliberately blocked from formalization."""

    kind: Literal["unmodeled"] = "unmodeled"
    description: str


CellFeature = Annotated[
    TerrainFeature
    | TopologyFeature
    | FixedTrackFeature
    | FixedSwitchFeature
    | ActorFeature
    | GoalFeature
    | StateFeature
    | TriggerFeature
    | LinkFeature
    | UnmodeledFeature,
    Field(discriminator="kind"),
]


class CellTypeDefinition(_FrozenModel):
    """Reusable visual type plus the features needed to formalize its cells."""

    id: Identifier
    display_name: str
    properties: tuple[CellPropertyDefinition, ...] = ()
    features: tuple[CellFeature, ...]

    @model_validator(mode="after")
    def valid_feature_property_references(self) -> Self:
        property_ids = [definition.id for definition in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("cell type property IDs must be unique")
        property_map = {definition.id: definition for definition in self.properties}
        if not self.features:
            raise ValueError("a cell type requires at least one formal feature")
        if sum(isinstance(feature, TerrainFeature) for feature in self.features) > 1:
            raise ValueError("a cell type may declare at most one terrain feature")
        known = set(property_ids)
        for feature in self.features:
            referenced = _feature_property_references(feature)
            missing = sorted(referenced - known)
            if missing:
                raise ValueError(
                    f"feature {feature.kind!r} references unknown properties: {', '.join(missing)}"
                )
            _validate_feature_property_kinds(feature, property_map)
        return self

    @property
    def formalization_supported(self) -> bool:
        """Return whether every feature has a current Level compiler handler."""
        supported = (
            TerrainFeature,
            TopologyFeature,
            FixedTrackFeature,
            FixedSwitchFeature,
            ActorFeature,
            GoalFeature,
        )
        return all(isinstance(feature, supported) for feature in self.features)


class CellAnnotation(_FrozenModel):
    """One reviewed cell type and its type-defined instance values."""

    id: Identifier
    cell: CellCoord
    type_id: Identifier | None = None
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    source: Literal["automatic", "manual"] | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def consistent_source(self) -> Self:
        if self.type_id is None and self.properties:
            raise ValueError("an untyped cell cannot have properties")
        if self.type_id is None and self.source == "automatic":
            raise ValueError("an automatic annotation requires a type")
        if self.source == "automatic" and self.confidence is None:
            raise ValueError("an automatic annotation requires confidence")
        if self.source != "automatic" and self.confidence is not None:
            raise ValueError("only automatic annotations retain recognition confidence")
        return self


def built_in_cell_types() -> tuple[CellTypeDefinition, ...]:
    """Return the initial reusable vocabulary for the supported baseline profile."""
    return (
        CellTypeDefinition(
            id="terrain.empty",
            display_name="Empty",
            features=(TerrainFeature(terrain_kind="terrain.empty", buildable=True),),
        ),
        CellTypeDefinition(
            id="terrain.occupied",
            display_name="Obstacle",
            features=(TerrainFeature(terrain_kind="terrain.occupied", buildable=False),),
        ),
        CellTypeDefinition(
            id="track.fixed.two-port",
            display_name="Fixed rail · 2 ports",
            properties=(
                CellPropertyDefinition(
                    id="first_port",
                    label="Side A",
                    value_kind="direction",
                ),
                CellPropertyDefinition(
                    id="second_port",
                    label="Side B",
                    value_kind="direction",
                ),
            ),
            features=(
                TerrainFeature(terrain_kind="terrain.empty", buildable=False),
                FixedTrackFeature(),
            ),
        ),
        CellTypeDefinition(
            id="track.fixed.three-port",
            display_name="Fixed switch · 3 ports",
            properties=(
                CellPropertyDefinition(
                    id="branch_port",
                    label="Side branch",
                    value_kind="direction",
                ),
                CellPropertyDefinition(
                    id="selected_output",
                    label="Selected output",
                    value_kind="direction",
                ),
            ),
            features=(
                TerrainFeature(terrain_kind="terrain.empty", buildable=False),
                FixedSwitchFeature(),
            ),
        ),
        CellTypeDefinition(
            id="actor.train",
            display_name="Train",
            properties=(
                CellPropertyDefinition(id="direction", label="Direction", value_kind="direction"),
                CellPropertyDefinition(
                    id="arrival_order",
                    label="Arrival order",
                    value_kind="integer",
                    required=False,
                    minimum=1,
                ),
            ),
            features=(
                TerrainFeature(terrain_kind="terrain.empty", buildable=False),
                ActorFeature(arrival_order_property="arrival_order"),
            ),
        ),
        CellTypeDefinition(
            id="goal.arrival",
            display_name="Arrival",
            properties=(
                CellPropertyDefinition(
                    id="entry_port",
                    label="Arrival direction",
                    value_kind="direction",
                ),
            ),
            features=(
                TerrainFeature(terrain_kind="terrain.empty", buildable=False),
                GoalFeature(entry_port_property="entry_port"),
            ),
        ),
    )


def cell_type_for_role(
    *,
    type_id: str,
    display_name: str,
    role: Literal["empty", "obstacle", "train", "goal", "unmodeled"],
) -> CellTypeDefinition:
    """Create a reusable type from one currently supported authoring role."""
    if role == "empty":
        features: tuple[CellFeature, ...] = (TerrainFeature(terrain_kind=type_id, buildable=True),)
        properties: tuple[CellPropertyDefinition, ...] = ()
    elif role == "obstacle":
        features = (TerrainFeature(terrain_kind=type_id, buildable=False),)
        properties = ()
    elif role == "train":
        features = (
            TerrainFeature(terrain_kind="terrain.empty", buildable=False),
            ActorFeature(arrival_order_property="arrival_order"),
        )
        properties = (
            CellPropertyDefinition(id="direction", label="Direction", value_kind="direction"),
            CellPropertyDefinition(
                id="arrival_order",
                label="Arrival order",
                value_kind="integer",
                required=False,
                minimum=1,
            ),
        )
    elif role == "goal":
        features = (
            TerrainFeature(terrain_kind="terrain.empty", buildable=False),
            GoalFeature(entry_port_property="entry_port"),
        )
        properties = (
            CellPropertyDefinition(
                id="entry_port",
                label="Arrival direction",
                value_kind="direction",
            ),
        )
    else:
        features = (UnmodeledFeature(description="Awaiting a graph-state feature definition"),)
        properties = ()
    return CellTypeDefinition(
        id=type_id,
        display_name=display_name,
        properties=properties,
        features=features,
    )


def make_cell_annotation_id(level_number: str, cell: CellCoord) -> str:
    """Build a stable level-and-coordinate identity for one visual sample."""
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", level_number.strip()).strip("-.")
    if not normalized:
        raise ValueError("level number must contain at least one identifier character")
    return f"level-{normalized}-r{cell.row:03d}-c{cell.col:03d}"


def annotation_errors(
    annotation: CellAnnotation,
    definitions: Mapping[str, CellTypeDefinition],
) -> tuple[str, ...]:
    """Return every reason an annotation is not ready for formalization."""
    errors = list(annotation_value_errors(annotation, definitions))
    if annotation.type_id is None:
        return tuple(errors)
    definition = definitions.get(annotation.type_id)
    if definition is None:
        return tuple(errors)
    if not definition.formalization_supported:
        errors.append("the selected type has no complete graph-state compiler yet")
    for feature in definition.features:
        if isinstance(feature, FixedTrackFeature):
            first = annotation.properties.get(feature.first_port_property)
            second = annotation.properties.get(feature.second_port_property)
            if first is not None and first == second:
                errors.append("the two-port rail sides must be distinct")
        elif isinstance(feature, FixedSwitchFeature):
            branch = annotation.properties.get(feature.branch_port_property)
            output = annotation.properties.get(feature.selected_output_property)
            if (
                isinstance(branch, str)
                and isinstance(output, str)
                and branch in {port.value for port in Port}
                and output in {port.value for port in Port}
                and Port(output) not in _perpendicular_ports(Port(branch))
            ):
                errors.append("the switch output must be perpendicular to its side branch")
    return tuple(errors)


def annotation_value_errors(
    annotation: CellAnnotation,
    definitions: Mapping[str, CellTypeDefinition],
) -> tuple[str, ...]:
    """Return type/property errors independently from compiler availability."""
    if annotation.type_id is None:
        return ("type is not selected",)
    definition = definitions.get(annotation.type_id)
    if definition is None:
        return (f"unknown type {annotation.type_id!r}",)
    errors: list[str] = []
    for property_definition in definition.properties:
        value = annotation.properties.get(property_definition.id)
        if property_definition.required and value in (None, ""):
            errors.append(f"{property_definition.label} is required")
            continue
        if value is not None and not _valid_property_value(value, property_definition):
            errors.append(f"{property_definition.label} has an invalid value")
    unknown = sorted(set(annotation.properties) - {item.id for item in definition.properties})
    if unknown:
        errors.append("unknown properties: " + ", ".join(unknown))
    return tuple(errors)


def rotate_annotation_properties(
    properties: Mapping[str, PropertyValue],
    definition: CellTypeDefinition,
    rotation_degrees: Literal[0, 90, 180, 270],
) -> dict[str, PropertyValue]:
    """Rotate directional sample properties into an observed crop orientation."""
    rotated = dict(properties)
    for property_definition in definition.properties:
        if property_definition.value_kind != "direction":
            continue
        value = rotated.get(property_definition.id)
        if isinstance(value, str) and value in {direction.value for direction in Direction}:
            rotated[property_definition.id] = _rotate_direction(
                Direction(value), rotation_degrees
            ).value
    return rotated


def formalize_annotations(
    *,
    level_number: str,
    rows: int,
    cols: int,
    rail_budget: int,
    annotations: Sequence[CellAnnotation],
    definitions: Mapping[str, CellTypeDefinition],
    goal_assignments: Mapping[str, CellCoord] | None = None,
) -> Level:
    """Compile reviewed cell features into the existing validated Level contract."""
    expected_cells = tuple(
        CellCoord(row=row, col=col) for row in range(rows) for col in range(cols)
    )
    actual_cells = tuple(annotation.cell for annotation in annotations)
    if actual_cells != expected_cells:
        raise ValueError("cell annotations must cover the complete grid in row-major order")
    errors = [
        f"({annotation.cell.row}, {annotation.cell.col}): {error}"
        for annotation in annotations
        for error in annotation_errors(annotation, definitions)
    ]
    if errors:
        raise ValueError("incomplete cell annotations: " + "; ".join(errors))

    terrain: list[TerrainCell] = []
    buildable_cells: list[CellCoord] = []
    fixed_tracks: list[TrackPiece] = []
    switches: list[Switch] = []
    actors: list[Actor] = []
    arrival_orders: dict[str, int] = {}
    goals: dict[CellCoord, Port] = {}
    for annotation in annotations:
        if annotation.type_id is None:
            raise AssertionError("validated annotation must have a type")
        definition = definitions[annotation.type_id]
        has_explicit_topology = any(
            isinstance(feature, (TopologyFeature, FixedTrackFeature))
            for feature in definition.features
        )
        for feature in definition.features:
            if isinstance(feature, TerrainFeature):
                terrain.append(TerrainCell(cell=annotation.cell, kind=feature.terrain_kind))
                if feature.buildable:
                    buildable_cells.append(annotation.cell)
            elif isinstance(feature, TopologyFeature):
                fixed_tracks.append(
                    TrackPiece(
                        id=annotation.id,
                        cell=annotation.cell,
                        ports=feature.ports,
                        transitions=feature.transitions,
                        fixed=feature.fixed,
                    )
                )
            elif isinstance(feature, FixedTrackFeature):
                first = Port(str(annotation.properties[feature.first_port_property]))
                second = Port(str(annotation.properties[feature.second_port_property]))
                _require_in_bounds_ports(
                    annotation.cell,
                    (first, second),
                    rows=rows,
                    cols=cols,
                    description="fixed two-port rail",
                )
                fixed_tracks.append(
                    TrackPiece(
                        id=annotation.id,
                        cell=annotation.cell,
                        ports=(first, second),
                        transitions=(TrackTransition(entry_port=first, exit_port=second),),
                        fixed=True,
                    )
                )
            elif isinstance(feature, FixedSwitchFeature):
                branch = Port(str(annotation.properties[feature.branch_port_property]))
                selected_output = Port(str(annotation.properties[feature.selected_output_property]))
                other_output = _opposite_port(selected_output)
                _require_in_bounds_ports(
                    annotation.cell,
                    (branch, selected_output, other_output),
                    rows=rows,
                    cols=cols,
                    description="fixed three-port switch",
                )
                switches.append(
                    Switch(
                        id=annotation.id,
                        cell=annotation.cell,
                        entry_port=branch,
                        output_ports=(selected_output, other_output),
                        initial_output=selected_output,
                    )
                )
            elif isinstance(feature, ActorFeature):
                direction = annotation.properties[feature.direction_property]
                if not isinstance(direction, str):
                    raise ValueError(f"actor {annotation.id!r} has no direction")
                actor_direction = Direction(direction)
                direction_port = Port(actor_direction.value)
                entry_port = _opposite_port(direction_port)
                if not has_explicit_topology:
                    _require_in_bounds_ports(
                        annotation.cell,
                        (direction_port,),
                        rows=rows,
                        cols=cols,
                        description="train departure",
                    )
                    fixed_tracks.append(
                        TrackPiece(
                            id=f"{annotation.id}:track",
                            cell=annotation.cell,
                            ports=(entry_port, direction_port),
                            transitions=(
                                TrackTransition(
                                    entry_port=entry_port,
                                    exit_port=direction_port,
                                ),
                            ),
                            fixed=True,
                        )
                    )
                number = (
                    annotation.properties.get(feature.number_property)
                    if feature.number_property is not None
                    else None
                )
                color = (
                    annotation.properties.get(feature.color_property)
                    if feature.color_property is not None
                    else None
                )
                actors.append(
                    Actor(
                        id=annotation.id,
                        cell=annotation.cell,
                        direction=actor_direction,
                        actor_type=feature.actor_type,
                        number=number if isinstance(number, int) else None,
                        color=color if isinstance(color, str) else None,
                    )
                )
                if feature.arrival_order_property is not None:
                    arrival_order = annotation.properties.get(feature.arrival_order_property)
                    if arrival_order is not None:
                        if (
                            not isinstance(arrival_order, int)
                            or isinstance(arrival_order, bool)
                            or arrival_order < 1
                        ):
                            raise ValueError(
                                f"actor {annotation.id!r} has an invalid arrival order"
                            )
                        arrival_orders[annotation.id] = arrival_order
            elif isinstance(feature, GoalFeature):
                entry_port: Port | None = None
                if feature.entry_port_property is not None:
                    value = annotation.properties[feature.entry_port_property]
                    if not isinstance(value, str):
                        raise ValueError(f"goal {annotation.id!r} has no orientation")
                    property_definition = next(
                        item
                        for item in definition.properties
                        if item.id == feature.entry_port_property
                    )
                    observed_port = Port(value)
                    entry_port = (
                        _opposite_port(observed_port)
                        if property_definition.value_kind == "direction"
                        else observed_port
                    )
                if entry_port is None:
                    raise ValueError(f"goal {annotation.id!r} requires an orientation")
                if not has_explicit_topology:
                    _require_in_bounds_ports(
                        annotation.cell,
                        (entry_port,),
                        rows=rows,
                        cols=cols,
                        description="goal track",
                    )
                    fixed_tracks.append(
                        TrackPiece(
                            id=f"{annotation.id}:track",
                            cell=annotation.cell,
                            ports=(entry_port,),
                            transitions=(),
                            fixed=True,
                        )
                    )
                goals[annotation.cell] = entry_port

    if not actors:
        raise ValueError("formal level requires at least one actor cell")
    if not goals:
        raise ValueError("formal level requires at least one goal cell")
    assignments = goal_assignments or {}
    arrivals: list[ArrivalRequirement] = []
    for actor in actors:
        if len(goals) == 1:
            target = next(iter(goals))
        else:
            target = assignments.get(actor.id)
            if target not in goals:
                raise ValueError(f"choose one of the recognized goals for actor {actor.id!r}")
        arrivals.append(
            ArrivalRequirement(
                actor_id=actor.id,
                target_cell=target,
                entry_port=goals[target],
            )
        )

    level_id = make_level_id(level_number)
    order_counts = {
        value: tuple(actor_id for actor_id, order in arrival_orders.items() if order == value)
        for value in set(arrival_orders.values())
    }
    duplicate_orders = sorted(
        value for value, actor_ids in order_counts.items() if len(actor_ids) > 1
    )
    if duplicate_orders:
        raise ValueError(
            "train arrival order values must be unique: "
            + ", ".join(str(value) for value in duplicate_orders)
        )
    arrival_order = tuple(
        actor_id for actor_id, _order in sorted(arrival_orders.items(), key=lambda item: item[1])
    )
    fixed_track_cells = {track.cell for track in fixed_tracks}
    return Level(
        id=level_id,
        rows=rows,
        cols=cols,
        metadata=LevelMetadata(source="interactive-cell-annotations"),
        terrain=tuple(terrain),
        buildable_cells=tuple(
            cell for cell in dict.fromkeys(buildable_cells) if cell not in fixed_track_cells
        ),
        fixed_tracks=tuple(fixed_tracks),
        actors=tuple(actors),
        switches=tuple(switches),
        rail_inventory=RailInventory(
            total_budget=rail_budget,
            piece_counts=(
                RailPieceCount(piece_kind="straight", count=rail_budget),
                RailPieceCount(piece_kind="curve", count=rail_budget),
                RailPieceCount(piece_kind="switch", count=rail_budget),
            ),
        ),
        win_condition=WinCondition(
            required_arrivals=tuple(arrivals),
            arrival_order=arrival_order,
        ),
    )


def make_level_id(level_number: str) -> str:
    """Return the stable Level identifier corresponding to a user-facing number."""
    sample_id = make_cell_annotation_id(level_number, CellCoord(row=0, col=0))
    return sample_id.rsplit("-r", maxsplit=1)[0]


def _feature_property_references(feature: CellFeature) -> set[str]:
    if isinstance(feature, FixedTrackFeature):
        return {feature.first_port_property, feature.second_port_property}
    if isinstance(feature, FixedSwitchFeature):
        return {feature.branch_port_property, feature.selected_output_property}
    if isinstance(feature, ActorFeature):
        return {
            value
            for value in (
                feature.direction_property,
                feature.number_property,
                feature.color_property,
                feature.arrival_order_property,
            )
            if value is not None
        }
    if isinstance(feature, GoalFeature):
        return {feature.entry_port_property} if feature.entry_port_property else set()
    if isinstance(feature, StateFeature):
        return {feature.initial_state_property}
    if isinstance(feature, TriggerFeature):
        return {feature.channel_property} if feature.channel_property else set()
    if isinstance(feature, LinkFeature):
        return {
            value
            for value in (
                feature.group_property,
                feature.entry_port_property,
                feature.exit_port_property,
            )
            if value is not None
        }
    return set()


def _validate_feature_property_kinds(
    feature: CellFeature,
    properties: Mapping[str, CellPropertyDefinition],
) -> None:
    if isinstance(feature, FixedTrackFeature):
        _require_port_property(properties, feature.first_port_property, feature.kind)
        _require_port_property(properties, feature.second_port_property, feature.kind)
    elif isinstance(feature, FixedSwitchFeature):
        _require_port_property(properties, feature.branch_port_property, feature.kind)
        _require_port_property(properties, feature.selected_output_property, feature.kind)
    elif isinstance(feature, ActorFeature):
        _require_property_kind(properties, feature.direction_property, "direction", feature.kind)
        if feature.number_property is not None:
            _require_property_kind(properties, feature.number_property, "integer", feature.kind)
        if feature.color_property is not None:
            _require_property_kind(properties, feature.color_property, "text", feature.kind)
        if feature.arrival_order_property is not None:
            _require_property_kind(
                properties,
                feature.arrival_order_property,
                "integer",
                feature.kind,
            )
    elif isinstance(feature, GoalFeature) and feature.entry_port_property is not None:
        _require_port_property(properties, feature.entry_port_property, feature.kind)
    elif isinstance(feature, StateFeature):
        definition = _require_property_kind(
            properties, feature.initial_state_property, "text", feature.kind
        )
        if tuple(definition.choices) != tuple(feature.states):
            raise ValueError("state initial-state choices must match the declared states")
    elif isinstance(feature, TriggerFeature) and feature.channel_property is not None:
        _require_property_kind(properties, feature.channel_property, "text", feature.kind)
    elif isinstance(feature, LinkFeature):
        _require_property_kind(properties, feature.group_property, "text", feature.kind)
        if feature.entry_port_property is not None:
            _require_port_property(properties, feature.entry_port_property, feature.kind)
        if feature.exit_port_property is not None:
            _require_port_property(properties, feature.exit_port_property, feature.kind)


def _require_property_kind(
    properties: Mapping[str, CellPropertyDefinition],
    property_id: str,
    expected: PropertyKind,
    feature_kind: str,
) -> CellPropertyDefinition:
    definition = properties[property_id]
    if definition.value_kind != expected:
        raise ValueError(
            f"{feature_kind} feature property {property_id!r} must have kind {expected!r}"
        )
    return definition


def _require_port_property(
    properties: Mapping[str, CellPropertyDefinition],
    property_id: str,
    feature_kind: str,
) -> None:
    definition = properties[property_id]
    if definition.value_kind == "direction":
        return
    definition = _require_property_kind(properties, property_id, "text", feature_kind)
    allowed = tuple(port.value for port in Port)
    if definition.choices != allowed:
        raise ValueError(f"{feature_kind} port property choices must be {allowed!r}")


def _opposite_port(port: Port) -> Port:
    return {
        Port.NORTH: Port.SOUTH,
        Port.EAST: Port.WEST,
        Port.SOUTH: Port.NORTH,
        Port.WEST: Port.EAST,
    }[port]


def _perpendicular_ports(port: Port) -> frozenset[Port]:
    """Return the two cardinal ports perpendicular to ``port``."""
    if port in (Port.NORTH, Port.SOUTH):
        return frozenset((Port.EAST, Port.WEST))
    return frozenset((Port.NORTH, Port.SOUTH))


def _require_in_bounds_ports(
    cell: CellCoord,
    ports: tuple[Port, ...],
    *,
    rows: int,
    cols: int,
    description: str,
) -> None:
    offsets = {
        Port.NORTH: (-1, 0),
        Port.EAST: (0, 1),
        Port.SOUTH: (1, 0),
        Port.WEST: (0, -1),
    }
    for port in ports:
        row_delta, col_delta = offsets[port]
        neighbour_row = cell.row + row_delta
        neighbour_col = cell.col + col_delta
        if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols):
            raise ValueError(
                f"{description} at ({cell.row}, {cell.col}) points outside the grid through {port.value}"
            )


def _valid_property_value(value: PropertyValue, definition: CellPropertyDefinition) -> bool:
    if definition.value_kind == "direction":
        return isinstance(value, str) and value in {direction.value for direction in Direction}
    if definition.value_kind == "text":
        return isinstance(value, str) and (not definition.choices or value in definition.choices)
    if definition.value_kind == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (definition.minimum is None or value >= definition.minimum)
            and (definition.maximum is None or value <= definition.maximum)
        )
    return isinstance(value, bool)


def _rotate_direction(
    direction: Direction, rotation_degrees: Literal[0, 90, 180, 270]
) -> Direction:
    directions = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
    return directions[(directions.index(direction) + rotation_degrees // 90) % len(directions)]
