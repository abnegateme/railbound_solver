"""Deterministic replay for static rails and observed dynamic devices.

M7 keeps the semantic level immutable and stores mutable switch and gate
values only in :class:`SimulationState`.  Every tick reads one state snapshot:
movement uses the gate and switch values at the start of the tick, then button
effects update matching devices for the *next* tick.  Tunnels transfer an
actor to their paired endpoint's exit in the same movement tick.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from railbound.geometry import CellCoord
from railbound.level import ArrivalRequirement, Direction, Level, Port, Switch, TunnelEndpoint
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


class GateStatus(StrEnum):
    """The runtime state of a gate that may block entry to its cell."""

    OPEN = "open"
    CLOSED = "closed"


class FailureReason(StrEnum):
    """A machine-readable reason for an unsuccessful replay."""

    DERAILMENT = "derailment"
    DEAD_END = "dead_end"
    BOUNDARY_EXIT = "boundary_exit"
    GATE_CLOSED = "gate_closed"
    COLLISION = "collision"
    ARRIVAL_ORDER = "arrival_order"
    STEP_LIMIT = "step_limit"


class EventPhase(StrEnum):
    """The deterministic phase in which an event occurred."""

    MOVEMENT = "movement"
    INTERACTION = "interaction"
    STATE_CHANGE = "state_change"
    OUTCOME = "outcome"
    TERMINAL = "terminal"


class EventKind(StrEnum):
    """Structured events that together explain a replay trace."""

    MOVE = "move"
    TUNNEL_TRANSIT = "tunnel_transit"
    BUTTON_PRESS = "button_press"
    GATE_STATE_CHANGE = "gate_state_change"
    SWITCH_STATE_CHANGE = "switch_state_change"
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
class SwitchState:
    """The selected output of one dynamic switch."""

    switch_id: str
    selected_output: Port


@dataclass(frozen=True, slots=True)
class GateState:
    """The current open or closed value of one gate."""

    gate_id: str
    status: GateStatus


@dataclass(frozen=True, slots=True)
class SimulationFailure:
    """Structured context for one terminal simulation failure."""

    reason: FailureReason
    actor_ids: tuple[str, ...]
    source_cell: CellCoord | None = None
    target_cell: CellCoord | None = None
    entry_port: Port | None = None
    exit_port: Port | None = None
    device_ids: tuple[str, ...] = ()


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
    phase: EventPhase = EventPhase.MOVEMENT
    device_ids: tuple[str, ...] = ()
    before_value: str | None = None
    after_value: str | None = None
    causal_event_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SimulationState:
    """Complete immutable M7 runtime state used as the input to :func:`step`."""

    level_id: str
    actors: tuple[ActorState, ...]
    arrival_requirements: tuple[ArrivalRequirement, ...]
    arrival_order: tuple[str, ...]
    switch_states: tuple[SwitchState, ...] = ()
    gate_states: tuple[GateState, ...] = ()
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
        switch_ids = tuple(switch_state.switch_id for switch_state in self.switch_states)
        if switch_ids != tuple(sorted(switch_ids)) or len(set(switch_ids)) != len(switch_ids):
            raise ValueError("simulation switch states must be unique and ordered by switch_id")
        gate_ids = tuple(gate_state.gate_id for gate_state in self.gate_states)
        if gate_ids != tuple(sorted(gate_ids)) or len(set(gate_ids)) != len(gate_ids):
            raise ValueError("simulation gate states must be unique and ordered by gate_id")
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
    tunnel_transit: _TunnelTransit | None = None


@dataclass(frozen=True, slots=True)
class _TunnelTransit:
    """The paired endpoint traversal folded into one movement tick."""

    pair_id: str
    entry_endpoint: TunnelEndpoint
    exit_endpoint: TunnelEndpoint


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


def initial_state(level: Level) -> SimulationState:
    """Create the immutable tick-zero state for a level supported by M7.

    Actors and dynamic devices use stable ID order, so parsing order in a JSON
    document cannot affect movement, channel effects, or event sequencing.
    """
    _validate_supported_level(level)
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
        switch_states=tuple(
            SwitchState(switch_id=switch.id, selected_output=switch.initial_output)
            for switch in sorted(level.switches, key=lambda switch: switch.id)
        ),
        gate_states=tuple(
            GateState(gate_id=gate.id, status=GateStatus(gate.initial_state))
            for gate in sorted(level.gates, key=lambda gate: gate.id)
        ),
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
    """Advance one atomic tick using the documented M7 precedence.

    Movement samples the state at the beginning of the tick.  A movement
    failure wins over collisions; collisions win over button effects; accepted
    moves trigger buttons, whose toggles apply only after movement resolves.
    Arrivals and terminal success are evaluated last.
    """
    if state.level_id != board.level_id:
        raise ValueError(
            f"simulation state targets {state.level_id!r}, not board {board.level_id!r}"
        )
    if state.status is not SimulationStatus.RUNNING:
        return StepResult(state=state, events=())
    _validate_runtime_state(state, board)

    tick = state.tick + 1
    switch_outputs = {
        switch_state.switch_id: switch_state.selected_output for switch_state in state.switch_states
    }
    gate_statuses = {gate_state.gate_id: gate_state.status for gate_state in state.gate_states}
    intents: list[_MoveIntent] = []
    failures: list[SimulationFailure] = []
    for actor in state.actors:
        if actor.status is ActorStatus.ARRIVED:
            continue
        result = _movement_intent(actor, board, switch_outputs, gate_statuses)
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

    events: list[SimulationEvent] = []
    move_events: dict[str, SimulationEvent] = {}
    for intent in intents:
        move_event = _event(
            state,
            offset=len(events),
            tick=tick,
            kind=EventKind.MOVE,
            phase=EventPhase.MOVEMENT,
            actor_ids=(intent.actor_id,),
            source_cell=intent.source_cell,
            target_cell=intent.target_cell,
            entry_port=intent.entry_port,
            exit_port=intent.exit_port,
        )
        events.append(move_event)
        move_events[intent.actor_id] = move_event
        if intent.tunnel_transit is not None:
            transit = intent.tunnel_transit
            events.append(
                _event(
                    state,
                    offset=len(events),
                    tick=tick,
                    kind=EventKind.TUNNEL_TRANSIT,
                    phase=EventPhase.INTERACTION,
                    actor_ids=(intent.actor_id,),
                    source_cell=transit.entry_endpoint.cell,
                    target_cell=transit.exit_endpoint.cell,
                    entry_port=transit.entry_endpoint.entry_port,
                    exit_port=transit.exit_endpoint.exit_port,
                    device_ids=(transit.entry_endpoint.id, transit.exit_endpoint.id),
                    causal_event_ids=(move_event.sequence,),
                )
            )

    button_events = _button_events(intents, move_events, state, board, tick, events)
    _apply_button_effects(
        button_events,
        state,
        board,
        tick,
        events,
        switch_outputs,
        gate_statuses,
    )
    for intent in arrivals:
        events.append(
            _event(
                state,
                offset=len(events),
                tick=tick,
                kind=EventKind.ARRIVAL,
                phase=EventPhase.OUTCOME,
                actor_ids=(intent.actor_id,),
                target_cell=intent.target_cell,
                entry_port=intent.entry_port,
                causal_event_ids=(move_events[intent.actor_id].sequence,),
            )
        )

    updated = _updated_state(
        state,
        intents,
        arrivals,
        switch_outputs=switch_outputs,
        gate_statuses=gate_statuses,
        tick=tick,
        emitted_event_count=len(events),
    )
    if _is_success(updated):
        events.append(
            _event(
                updated,
                offset=0,
                tick=tick,
                kind=EventKind.SUCCESS,
                phase=EventPhase.TERMINAL,
                actor_ids=updated.arrived_actor_ids,
            )
        )
        updated = replace(
            updated,
            status=SimulationStatus.SUCCESS,
            next_event_sequence=updated.next_event_sequence + 1,
        )
    return StepResult(state=updated, events=tuple(events))


def _validate_supported_level(level: Level) -> None:
    """Reject rules outside M7 instead of silently assigning semantics."""
    if level.win_condition.required_couplings:
        raise ValueError("M7 simulation does not support: required_couplings")


def _validate_runtime_state(state: SimulationState, board: CompiledBoard) -> None:
    """Ensure a manually assembled dynamic state matches the compiled board."""
    expected_switch_ids = tuple(
        switch.id for switch in sorted(board.switches, key=lambda value: value.id)
    )
    if tuple(value.switch_id for value in state.switch_states) != expected_switch_ids:
        raise ValueError("simulation switch states do not match board switches")
    expected_gate_ids = tuple(gate.id for gate in sorted(board.gates, key=lambda value: value.id))
    if tuple(value.gate_id for value in state.gate_states) != expected_gate_ids:
        raise ValueError("simulation gate states do not match board gates")
    switches_by_id = {switch.id: switch for switch in board.switches}
    for switch_state in state.switch_states:
        if switch_state.selected_output not in switches_by_id[switch_state.switch_id].output_ports:
            raise ValueError("simulation switch state selects an unavailable output")


def _movement_intent(
    actor: ActorState,
    board: CompiledBoard,
    switch_outputs: dict[str, Port],
    gate_statuses: dict[str, GateStatus],
) -> _MoveIntent | SimulationFailure:
    """Derive one snapshot-based move, including an immediate tunnel transfer."""
    entry_port = _OPPOSITE_PORT[Port(actor.direction)]
    starting_tunnel = _tunnel_at(actor.cell, board)
    if starting_tunnel is not None and entry_port == starting_tunnel.entry_port:
        return _through_tunnel(actor, starting_tunnel, board, gate_statuses)

    exit_port = _internal_exit_port(actor.cell, entry_port, board, switch_outputs)
    if exit_port is None:
        return SimulationFailure(
            reason=FailureReason.DERAILMENT,
            actor_ids=(actor.actor_id,),
            source_cell=actor.cell,
            entry_port=entry_port,
        )
    target_entry = _cross_boundary(actor.cell, exit_port, board, actor.actor_id)
    if isinstance(target_entry, SimulationFailure):
        return target_entry
    gate_failure = _gate_failure(target_entry, actor.actor_id, board, gate_statuses)
    if gate_failure is not None:
        return gate_failure
    endpoint = _tunnel_at(target_entry.cell, board)
    if endpoint is not None and target_entry.port == endpoint.entry_port:
        return _through_tunnel(actor, endpoint, board, gate_statuses)
    return _MoveIntent(
        actor_id=actor.actor_id,
        source_cell=actor.cell,
        target_cell=target_entry.cell,
        entry_port=target_entry.port,
        exit_port=exit_port,
        direction=Direction(exit_port),
    )


def _internal_exit_port(
    cell: CellCoord,
    entry_port: Port,
    board: CompiledBoard,
    switch_outputs: dict[str, Port],
) -> Port | None:
    """Follow a dynamic switch or static rail transition inside one cell."""
    switch = _switch_at(cell, board)
    if switch is not None:
        if entry_port == switch.entry_port:
            return switch_outputs[switch.id]
        if entry_port in switch.output_ports:
            return switch.entry_port
        return None
    exit_node = board.transition(PortNode(cell, entry_port))
    return None if exit_node is None else exit_node.port


def _through_tunnel(
    actor: ActorState,
    entry_endpoint: TunnelEndpoint,
    board: CompiledBoard,
    gate_statuses: dict[str, GateStatus],
) -> _MoveIntent | SimulationFailure:
    """Exit through a paired tunnel endpoint without consuming another tick."""
    exit_endpoint = _tunnel_peer(entry_endpoint, board)
    target_entry = _cross_boundary(
        exit_endpoint.cell,
        exit_endpoint.exit_port,
        board,
        actor.actor_id,
        device_ids=(entry_endpoint.id, exit_endpoint.id),
    )
    if isinstance(target_entry, SimulationFailure):
        return target_entry
    gate_failure = _gate_failure(target_entry, actor.actor_id, board, gate_statuses)
    if gate_failure is not None:
        return replace(
            gate_failure,
            device_ids=(entry_endpoint.id, exit_endpoint.id, *gate_failure.device_ids),
        )
    return _MoveIntent(
        actor_id=actor.actor_id,
        source_cell=actor.cell,
        target_cell=target_entry.cell,
        entry_port=target_entry.port,
        exit_port=exit_endpoint.exit_port,
        direction=Direction(exit_endpoint.exit_port),
        tunnel_transit=_TunnelTransit(
            pair_id=entry_endpoint.pair_id,
            entry_endpoint=entry_endpoint,
            exit_endpoint=exit_endpoint,
        ),
    )


def _cross_boundary(
    source_cell: CellCoord,
    exit_port: Port,
    board: CompiledBoard,
    actor_id: str,
    *,
    device_ids: tuple[str, ...] = (),
) -> PortNode | SimulationFailure:
    """Cross one edge while preserving M6 boundary and dead-end failures."""
    row_delta, col_delta = _PORT_OFFSETS[exit_port]
    target_cell = CellCoord(row=source_cell.row + row_delta, col=source_cell.col + col_delta)
    if not (0 <= target_cell.row < board.rows and 0 <= target_cell.col < board.cols):
        return SimulationFailure(
            reason=FailureReason.BOUNDARY_EXIT,
            actor_ids=(actor_id,),
            source_cell=source_cell,
            exit_port=exit_port,
            device_ids=device_ids,
        )
    target_entry = PortNode(cell=target_cell, port=_OPPOSITE_PORT[exit_port])
    if not _has_entry_port(target_entry, board):
        return SimulationFailure(
            reason=FailureReason.DEAD_END,
            actor_ids=(actor_id,),
            source_cell=source_cell,
            target_cell=target_cell,
            exit_port=exit_port,
            device_ids=device_ids,
        )
    return target_entry


def _has_entry_port(entry: PortNode, board: CompiledBoard) -> bool:
    """Return whether a static or dynamic cell accepts a boundary entry."""
    switch = _switch_at(entry.cell, board)
    if switch is not None:
        return entry.port in (switch.entry_port, *switch.output_ports)
    endpoint = _tunnel_at(entry.cell, board)
    if endpoint is not None and entry.port == endpoint.entry_port:
        return True
    piece = board.piece_at(entry.cell)
    return piece is not None and entry.port in piece.ports


def _gate_failure(
    entry: PortNode,
    actor_id: str,
    board: CompiledBoard,
    gate_statuses: dict[str, GateStatus],
) -> SimulationFailure | None:
    """Return a gate-blocked movement failure from the start-of-tick snapshot."""
    gate = next((gate for gate in board.gates if gate.cell == entry.cell), None)
    if gate is None or gate_statuses[gate.id] is GateStatus.OPEN:
        return None
    return SimulationFailure(
        reason=FailureReason.GATE_CLOSED,
        actor_ids=(actor_id,),
        target_cell=entry.cell,
        entry_port=entry.port,
        device_ids=(gate.id,),
    )


def _switch_at(cell: CellCoord, board: CompiledBoard) -> Switch | None:
    """Return the switch overlay at a cell, if it has one."""
    return next((switch for switch in board.switches if switch.cell == cell), None)


def _tunnel_at(cell: CellCoord, board: CompiledBoard) -> TunnelEndpoint | None:
    """Return the tunnel endpoint overlay at a cell, if it has one."""
    return next((endpoint for endpoint in board.tunnel_endpoints if endpoint.cell == cell), None)


def _tunnel_peer(endpoint: TunnelEndpoint, board: CompiledBoard) -> TunnelEndpoint:
    """Return the validated opposite endpoint of one tunnel pair."""
    return next(
        candidate
        for candidate in board.tunnel_endpoints
        if candidate.pair_id == endpoint.pair_id and candidate.id != endpoint.id
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

    by_tunnel_pair: dict[str, list[_MoveIntent]] = {}
    for intent in intents:
        if intent.tunnel_transit is not None:
            by_tunnel_pair.setdefault(intent.tunnel_transit.pair_id, []).append(intent)
    simultaneous_tunnels = [
        (pair_id, tuple(sorted(intent.actor_id for intent in pair_intents)), pair_intents[0])
        for pair_id, pair_intents in by_tunnel_pair.items()
        if len(pair_intents) > 1
    ]
    if simultaneous_tunnels:
        _pair_id, actor_ids, intent = min(simultaneous_tunnels, key=lambda value: value[0])
        assert intent.tunnel_transit is not None
        return SimulationFailure(
            reason=FailureReason.COLLISION,
            actor_ids=actor_ids,
            source_cell=intent.tunnel_transit.entry_endpoint.cell,
            target_cell=intent.tunnel_transit.exit_endpoint.cell,
            device_ids=(
                intent.tunnel_transit.entry_endpoint.id,
                intent.tunnel_transit.exit_endpoint.id,
            ),
        )
    return None


def _button_events(
    intents: list[_MoveIntent],
    move_events: dict[str, SimulationEvent],
    state: SimulationState,
    board: CompiledBoard,
    tick: int,
    events: list[SimulationEvent],
) -> list[tuple[str, SimulationEvent]]:
    """Emit button entries in stable actor-ID order after accepted movement."""
    presses = sorted(
        (
            intent.actor_id,
            button.id,
            intent.target_cell,
        )
        for intent in intents
        for button in board.buttons
        if button.cell == intent.target_cell
    )
    button_events: list[tuple[str, SimulationEvent]] = []
    for actor_id, button_id, cell in presses:
        event = _event(
            state,
            offset=len(events),
            tick=tick,
            kind=EventKind.BUTTON_PRESS,
            phase=EventPhase.INTERACTION,
            actor_ids=(actor_id,),
            target_cell=cell,
            device_ids=(button_id,),
            causal_event_ids=(move_events[actor_id].sequence,),
        )
        events.append(event)
        button_events.append((button_id, event))
    return button_events


def _apply_button_effects(
    button_events: list[tuple[str, SimulationEvent]],
    state: SimulationState,
    board: CompiledBoard,
    tick: int,
    events: list[SimulationEvent],
    switch_outputs: dict[str, Port],
    gate_statuses: dict[str, GateStatus],
) -> None:
    """Toggle every same-channel device after all movement has succeeded."""
    buttons_by_id = {button.id: button for button in board.buttons}
    gates_by_id = {gate.id: gate for gate in board.gates}
    switches_by_id = {switch.id: switch for switch in board.switches}
    for button_id, button_event in button_events:
        button = buttons_by_id[button_id]
        target_ids = sorted(
            [gate.id for gate in board.gates if gate.channel == button.channel]
            + [switch.id for switch in board.switches if switch.channel == button.channel]
        )
        for target_id in target_ids:
            if target_id in gates_by_id:
                gate_before = gate_statuses[target_id]
                gate_after = (
                    GateStatus.CLOSED if gate_before is GateStatus.OPEN else GateStatus.OPEN
                )
                gate_statuses[target_id] = gate_after
                events.append(
                    _event(
                        state,
                        offset=len(events),
                        tick=tick,
                        kind=EventKind.GATE_STATE_CHANGE,
                        phase=EventPhase.STATE_CHANGE,
                        target_cell=gates_by_id[target_id].cell,
                        device_ids=(target_id,),
                        before_value=gate_before.value,
                        after_value=gate_after.value,
                        causal_event_ids=(button_event.sequence,),
                    )
                )
                continue
            switch = switches_by_id[target_id]
            switch_before = switch_outputs[target_id]
            first, second = switch.output_ports
            switch_after = second if switch_before == first else first
            switch_outputs[target_id] = switch_after
            events.append(
                _event(
                    state,
                    offset=len(events),
                    tick=tick,
                    kind=EventKind.SWITCH_STATE_CHANGE,
                    phase=EventPhase.STATE_CHANGE,
                    target_cell=switch.cell,
                    device_ids=(target_id,),
                    before_value=switch_before.value,
                    after_value=switch_after.value,
                    causal_event_ids=(button_event.sequence,),
                )
            )


def _updated_state(
    state: SimulationState,
    intents: list[_MoveIntent],
    arrivals: list[_MoveIntent],
    *,
    switch_outputs: dict[str, Port],
    gate_statuses: dict[str, GateStatus],
    tick: int,
    emitted_event_count: int,
) -> SimulationState:
    """Apply accepted movement and deferred device effects atomically."""
    intent_by_actor = {intent.actor_id: intent for intent in intents}
    arriving_actor_ids = {intent.actor_id for intent in arrivals}
    arrived_actor_ids = [*state.arrived_actor_ids]
    moved_actors: list[ActorState] = []
    for actor in state.actors:
        intent = intent_by_actor.get(actor.actor_id)
        if intent is None:
            moved_actors.append(actor)
            continue
        status = ActorStatus.ARRIVED if actor.actor_id in arriving_actor_ids else ActorStatus.ACTIVE
        if status is ActorStatus.ARRIVED:
            arrived_actor_ids.append(actor.actor_id)
        moved_actors.append(
            ActorState(
                actor_id=actor.actor_id,
                cell=intent.target_cell,
                direction=intent.direction,
                status=status,
            )
        )
    return replace(
        state,
        actors=tuple(moved_actors),
        switch_states=tuple(
            SwitchState(switch_id=switch_id, selected_output=selected_output)
            for switch_id, selected_output in sorted(switch_outputs.items())
        ),
        gate_states=tuple(
            GateState(gate_id=gate_id, status=status)
            for gate_id, status in sorted(gate_statuses.items())
        ),
        arrived_actor_ids=tuple(arrived_actor_ids),
        tick=tick,
        next_event_sequence=state.next_event_sequence + emitted_event_count,
    )


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
        phase=EventPhase.TERMINAL,
        actor_ids=failure.actor_ids,
        source_cell=failure.source_cell,
        target_cell=failure.target_cell,
        entry_port=failure.entry_port,
        exit_port=failure.exit_port,
        reason=failure.reason,
        device_ids=failure.device_ids,
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
    phase: EventPhase,
    actor_ids: tuple[str, ...] = (),
    source_cell: CellCoord | None = None,
    target_cell: CellCoord | None = None,
    entry_port: Port | None = None,
    exit_port: Port | None = None,
    reason: FailureReason | None = None,
    device_ids: tuple[str, ...] = (),
    before_value: str | None = None,
    after_value: str | None = None,
    causal_event_ids: tuple[int, ...] = (),
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
        phase=phase,
        device_ids=device_ids,
        before_value=before_value,
        after_value=after_value,
        causal_event_ids=causal_event_ids,
    )


def _cell_key(cell: CellCoord) -> tuple[int, int]:
    """Return a deterministic coordinate key for simultaneous conflict selection."""
    return cell.row, cell.col
