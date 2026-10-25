"""Deterministic replay for levels that use only static rail topology.

M6 deliberately keeps runtime state separate from the semantic :mod:`level`
document.  An actor occupies the centre of a cell and its direction is its
current travel direction.  On a tick it enters the cell's rail through the
port opposite that direction, follows the rail's internal transition, then
crosses the selected boundary into the next cell.  All intents are calculated
from one immutable snapshot before any state changes are applied.

Switches, buttons, gates, tunnels, and coupling are M7 mechanics.  This
module rejects them up front rather than silently assigning invented rules to
them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from railbound.geometry import CellCoord
from railbound.level import ArrivalRequirement, Direction, Level, Port
from railbound.topology import CompiledBoard, PortNode, RailPlacement, compile_board


class SimulationStatus(StrEnum):
    """The terminal status of a deterministic simulation state."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class ActorStatus(StrEnum):
    """Whether an actor still participates in movement ticks."""

    ACTIVE = "active"
    ARRIVED = "arrived"


class FailureReason(StrEnum):
    """A machine-readable reason for an unsuccessful replay."""

    DERAILMENT = "derailment"
    DEAD_END = "dead_end"
    BOUNDARY_EXIT = "boundary_exit"
    COLLISION = "collision"
    ARRIVAL_ORDER = "arrival_order"
    STEP_LIMIT = "step_limit"


class EventKind(StrEnum):
    """The kinds of basic, structured events emitted by M6."""

    MOVE = "move"
    ARRIVAL = "arrival"
    FAILURE = "failure"
    SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class ActorState:
    """One actor's immutable runtime position and travel direction."""

    actor_id: str
    cell: CellCoord
    direction: Direction
    status: ActorStatus = ActorStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class SimulationFailure:
    """Structured context for one terminal simulation failure."""

    reason: FailureReason
    actor_ids: tuple[str, ...]
    source_cell: CellCoord | None = None
    target_cell: CellCoord | None = None
    entry_port: Port | None = None
    exit_port: Port | None = None


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """One deterministically sequenced state-transition event."""

    sequence: int
    tick: int
    kind: EventKind
    actor_ids: tuple[str, ...] = ()
    source_cell: CellCoord | None = None
    target_cell: CellCoord | None = None
    entry_port: Port | None = None
    exit_port: Port | None = None
    reason: FailureReason | None = None


@dataclass(frozen=True, slots=True)
class SimulationState:
    """Complete immutable M6 runtime state used as the input to :func:`step`."""

    level_id: str
    actors: tuple[ActorState, ...]
    arrival_requirements: tuple[ArrivalRequirement, ...]
    arrival_order: tuple[str, ...]
    tick: int = 0
    arrived_actor_ids: tuple[str, ...] = ()
    status: SimulationStatus = SimulationStatus.RUNNING
    failure: SimulationFailure | None = None
    next_event_sequence: int = 0

    def __post_init__(self) -> None:
        """Reject malformed manually assembled state before it can be stepped."""
        actor_ids = tuple(actor.actor_id for actor in self.actors)
        if self.tick < 0:
            raise ValueError("simulation tick must be non-negative")
        if self.next_event_sequence < 0:
            raise ValueError("next_event_sequence must be non-negative")
        if actor_ids != tuple(sorted(actor_ids)):
            raise ValueError("simulation actors must be ordered by actor_id")
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("simulation actors must have unique actor_id values")
        if any(actor_id not in actor_ids for actor_id in self.arrived_actor_ids):
            raise ValueError("arrived_actor_ids must refer to simulation actors")
        if len(set(self.arrived_actor_ids)) != len(self.arrived_actor_ids):
            raise ValueError("arrived_actor_ids must not repeat an actor_id")


@dataclass(frozen=True, slots=True)
class StepResult:
    """The next immutable state and all events emitted by one attempted tick."""

    state: SimulationState
    events: tuple[SimulationEvent, ...]


@dataclass(frozen=True, slots=True)
class SimulationOptions:
    """Bounds for a complete deterministic replay."""

    max_ticks: int = 10_000

    def __post_init__(self) -> None:
        """Require a positive finite tick limit for cyclic rail networks."""
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be greater than zero")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """The terminal state and complete event trace from :func:`simulate`."""

    final_state: SimulationState
    events: tuple[SimulationEvent, ...]

    @property
    def state(self) -> SimulationState:
        """Return the terminal state as a concise alias for ``final_state``."""
        return self.final_state

    @property
    def status(self) -> SimulationStatus:
        """Return the terminal simulation status."""
        return self.final_state.status

    @property
    def succeeded(self) -> bool:
        """Return whether replay completed every supported win condition."""
        return self.status is SimulationStatus.SUCCESS

    @property
    def failure(self) -> SimulationFailure | None:
        """Return terminal failure context, if replay was unsuccessful."""
        return self.final_state.failure


@dataclass(frozen=True, slots=True)
class _MoveIntent:
    """One actor's proposed move, derived from an immutable input snapshot."""

    actor_id: str
    source_cell: CellCoord
    target_cell: CellCoord
    entry_port: Port
    exit_port: Port
    direction: Direction


_OPPOSITE_PORT: dict[Port, Port] = {
    Port.NORTH: Port.SOUTH,
    Port.EAST: Port.WEST,
    Port.SOUTH: Port.NORTH,
    Port.WEST: Port.EAST,
}


def initial_state(level: Level) -> SimulationState:
    """Create the immutable tick-zero state for a statically simulated level.

    Actors are sorted by stable ID so movement, collision selection, and event
    order do not depend on their order in JSON.
    """
    _validate_static_level(level)
    actors = tuple(
        sorted(
            (
                ActorState(
                    actor_id=actor.id,
                    cell=actor.cell,
                    direction=actor.direction,
                )
                for actor in level.actors
            ),
            key=lambda actor: actor.actor_id,
        )
    )
    return SimulationState(
        level_id=level.id,
        actors=actors,
        arrival_requirements=level.win_condition.required_arrivals,
        arrival_order=level.win_condition.arrival_order,
    )


def simulate(
    level: Level,
    placement: RailPlacement,
    options: SimulationOptions | None = None,
) -> SimulationResult:
    """Compile and replay a static placement until success, failure, or its limit.

    ``max_ticks`` is a deterministic safety bound for cycles.  A limit is a
    terminal failure rather than an exception so callers can inspect it in the
    same way as a derailment or collision.
    """
    active_options = options or SimulationOptions()
    state = initial_state(level)
    board = compile_board(level, placement)
    events: tuple[SimulationEvent, ...] = ()
    while state.status is SimulationStatus.RUNNING:
        if state.tick >= active_options.max_ticks:
            limit_result = _step_limit(state)
            state = limit_result.state
            events = (*events, *limit_result.events)
            break
        step_result = step(state, board)
        state = step_result.state
        events = (*events, *step_result.events)
    return SimulationResult(final_state=state, events=events)


def step(state: SimulationState, board: CompiledBoard) -> StepResult:
    """Advance one atomic static-track tick from ``state`` over ``board``.

    A tick first derives all active actors' intents.  Per-actor movement
    failures take precedence over collisions; otherwise all collisions are
    resolved from the snapshot.  Accepted moves apply together, after which
    arrivals and success are evaluated in stable actor-ID order.
    """
    if state.level_id != board.level_id:
        raise ValueError(
            f"simulation state targets {state.level_id!r}, not board {board.level_id!r}"
        )
    if state.status is not SimulationStatus.RUNNING:
        return StepResult(state=state, events=())

    tick = state.tick + 1
    intents: list[_MoveIntent] = []
    failures: list[SimulationFailure] = []
    for actor in state.actors:
        if actor.status is ActorStatus.ARRIVED:
            continue
        result = _movement_intent(actor, board)
        if isinstance(result, SimulationFailure):
            failures.append(result)
        else:
            intents.append(result)

    if failures:
        failure = failures[0]
        return _terminal_failure(state, tick=tick, failure=failure)

    collision = _collision_failure(intents)
    if collision is not None:
        return _terminal_failure(state, tick=tick, failure=collision)

    arrivals = [intent for intent in intents if _matching_requirement(state, intent) is not None]
    order_failure = _arrival_order_failure(state, arrivals)
    if order_failure is not None:
        return _terminal_failure(state, tick=tick, failure=order_failure)

    intent_by_actor = {intent.actor_id: intent for intent in intents}
    arriving_actor_ids = {intent.actor_id for intent in arrivals}
    arrived_actor_ids = list(state.arrived_actor_ids)
    moved_actors: list[ActorState] = []
    for actor in state.actors:
        intent = intent_by_actor.get(actor.actor_id)
        if intent is None:
            moved_actors.append(actor)
            continue
        if actor.actor_id not in arriving_actor_ids:
            moved_actors.append(
                ActorState(
                    actor_id=actor.actor_id,
                    cell=intent.target_cell,
                    direction=intent.direction,
                )
            )
            continue
        arrived_actor_ids.append(actor.actor_id)
        moved_actors.append(
            ActorState(
                actor_id=actor.actor_id,
                cell=intent.target_cell,
                direction=intent.direction,
                status=ActorStatus.ARRIVED,
            )
        )

    events = [
        _event(
            state,
            offset=index,
            tick=tick,
            kind=EventKind.MOVE,
            actor_ids=(intent.actor_id,),
            source_cell=intent.source_cell,
            target_cell=intent.target_cell,
            entry_port=intent.entry_port,
            exit_port=intent.exit_port,
        )
        for index, intent in enumerate(intents)
    ]
    for intent in arrivals:
        events.append(
            _event(
                state,
                offset=len(events),
                tick=tick,
                kind=EventKind.ARRIVAL,
                actor_ids=(intent.actor_id,),
                target_cell=intent.target_cell,
                entry_port=intent.entry_port,
            )
        )

    updated = replace(
        state,
        actors=tuple(moved_actors),
        arrived_actor_ids=tuple(arrived_actor_ids),
        tick=tick,
        next_event_sequence=state.next_event_sequence + len(events),
    )
    if _is_success(updated):
        events.append(
            _event(
                updated,
                offset=0,
                tick=tick,
                kind=EventKind.SUCCESS,
                actor_ids=updated.arrived_actor_ids,
            )
        )
        updated = replace(
            updated,
            status=SimulationStatus.SUCCESS,
            next_event_sequence=updated.next_event_sequence + 1,
        )
    return StepResult(state=updated, events=tuple(events))


def _validate_static_level(level: Level) -> None:
    """Reject mechanics that would otherwise receive invented M6 semantics."""
    unsupported: list[str] = []
    if level.switches:
        unsupported.append("switches")
    if level.buttons:
        unsupported.append("buttons")
    if level.gates:
        unsupported.append("gates")
    if level.tunnel_endpoints:
        unsupported.append("tunnel_endpoints")
    if level.win_condition.required_couplings:
        unsupported.append("required_couplings")
    if unsupported:
        raise ValueError("M6 static simulation does not support: " + ", ".join(unsupported))


def _movement_intent(actor: ActorState, board: CompiledBoard) -> _MoveIntent | SimulationFailure:
    """Derive one actor's move or its precise static-topology failure."""
    entry_port = _OPPOSITE_PORT[Port(actor.direction)]
    exit_node = board.transition(PortNode(actor.cell, entry_port))
    if exit_node is None:
        return SimulationFailure(
            reason=FailureReason.DERAILMENT,
            actor_ids=(actor.actor_id,),
            source_cell=actor.cell,
            entry_port=entry_port,
        )
    target_entry = board.adjacent(exit_node)
    if target_entry is None:
        reason = (
            FailureReason.BOUNDARY_EXIT
            if exit_node in board.boundary_exits
            else FailureReason.DEAD_END
        )
        return SimulationFailure(
            reason=reason,
            actor_ids=(actor.actor_id,),
            source_cell=actor.cell,
            entry_port=entry_port,
            exit_port=exit_node.port,
        )
    return _MoveIntent(
        actor_id=actor.actor_id,
        source_cell=actor.cell,
        target_cell=target_entry.cell,
        entry_port=target_entry.port,
        exit_port=exit_node.port,
        direction=Direction(exit_node.port),
    )


def _collision_failure(intents: list[_MoveIntent]) -> SimulationFailure | None:
    """Return a deterministic collision selected from simultaneous intents."""
    by_target: dict[CellCoord, list[_MoveIntent]] = {}
    for intent in intents:
        by_target.setdefault(intent.target_cell, []).append(intent)
    shared_targets = [
        (target, tuple(sorted(intent.actor_id for intent in target_intents)))
        for target, target_intents in by_target.items()
        if len(target_intents) > 1
    ]
    if shared_targets:
        target, actor_ids = min(shared_targets, key=lambda value: _cell_key(value[0]))
        return SimulationFailure(
            reason=FailureReason.COLLISION,
            actor_ids=actor_ids,
            target_cell=target,
        )

    intents_by_actor = {intent.actor_id: intent for intent in intents}
    for actor_id in sorted(intents_by_actor):
        intent = intents_by_actor[actor_id]
        for other_id in sorted(intents_by_actor):
            if actor_id >= other_id:
                continue
            other = intents_by_actor[other_id]
            if intent.target_cell == other.source_cell and other.target_cell == intent.source_cell:
                return SimulationFailure(
                    reason=FailureReason.COLLISION,
                    actor_ids=(actor_id, other_id),
                    source_cell=min((intent.source_cell, other.source_cell), key=_cell_key),
                )
    return None


def _matching_requirement(state: SimulationState, intent: _MoveIntent) -> ArrivalRequirement | None:
    """Return the unsatisfied arrival requirement matched by a proposed entry."""
    if intent.actor_id in state.arrived_actor_ids:
        return None
    return next(
        (
            requirement
            for requirement in state.arrival_requirements
            if requirement.actor_id == intent.actor_id
            and requirement.target_cell == intent.target_cell
            and (requirement.entry_port is None or requirement.entry_port == intent.entry_port)
        ),
        None,
    )


def _is_success(state: SimulationState) -> bool:
    """Return whether all arrivals and their optional relative order are satisfied."""
    required_actor_ids = {requirement.actor_id for requirement in state.arrival_requirements}
    if set(state.arrived_actor_ids) != required_actor_ids:
        return False
    ordered_arrivals = tuple(
        actor_id for actor_id in state.arrived_actor_ids if actor_id in state.arrival_order
    )
    return ordered_arrivals == state.arrival_order


def _arrival_order_failure(
    state: SimulationState, arrivals: list[_MoveIntent]
) -> SimulationFailure | None:
    """Reject a newly completed arrival that violates the required relative order."""
    observed = tuple(
        actor_id
        for actor_id in (*state.arrived_actor_ids, *(intent.actor_id for intent in arrivals))
        if actor_id in state.arrival_order
    )
    expected = state.arrival_order[: len(observed)]
    if observed == expected:
        return None
    for actor_id, expected_actor_id in zip(observed, expected, strict=True):
        if actor_id != expected_actor_id:
            intent = next(intent for intent in arrivals if intent.actor_id == actor_id)
            return SimulationFailure(
                reason=FailureReason.ARRIVAL_ORDER,
                actor_ids=(actor_id,),
                source_cell=intent.source_cell,
                target_cell=intent.target_cell,
                entry_port=intent.entry_port,
            )
    raise RuntimeError("arrival order comparison did not identify its mismatch")


def _terminal_failure(
    state: SimulationState,
    *,
    tick: int,
    failure: SimulationFailure,
) -> StepResult:
    """Create an atomic failed state and its final failure event."""
    event = _event(
        state,
        offset=0,
        tick=tick,
        kind=EventKind.FAILURE,
        actor_ids=failure.actor_ids,
        source_cell=failure.source_cell,
        target_cell=failure.target_cell,
        entry_port=failure.entry_port,
        exit_port=failure.exit_port,
        reason=failure.reason,
    )
    return StepResult(
        state=replace(
            state,
            tick=tick,
            status=SimulationStatus.FAILURE,
            failure=failure,
            next_event_sequence=state.next_event_sequence + 1,
        ),
        events=(event,),
    )


def _step_limit(state: SimulationState) -> StepResult:
    """Turn an exhausted replay bound into the normal terminal failure form."""
    return _terminal_failure(
        state,
        tick=state.tick,
        failure=SimulationFailure(reason=FailureReason.STEP_LIMIT, actor_ids=()),
    )


def _event(
    state: SimulationState,
    *,
    offset: int,
    tick: int,
    kind: EventKind,
    actor_ids: tuple[str, ...] = (),
    source_cell: CellCoord | None = None,
    target_cell: CellCoord | None = None,
    entry_port: Port | None = None,
    exit_port: Port | None = None,
    reason: FailureReason | None = None,
) -> SimulationEvent:
    """Build one event with a sequence number local to a state snapshot."""
    return SimulationEvent(
        sequence=state.next_event_sequence + offset,
        tick=tick,
        kind=kind,
        actor_ids=actor_ids,
        source_cell=source_cell,
        target_cell=target_cell,
        entry_port=entry_port,
        exit_port=exit_port,
        reason=reason,
    )


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    """Return a deterministic coordinate key for simultaneous conflict selection."""
    return cell.row, cell.col
