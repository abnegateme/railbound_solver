# Deterministic simulation (M6–M7)

M6 replays static rails; M7 adds the observed World 2–4 device rules without
changing the static port convention. A `Level` and `RailPlacement` compile to
one immutable board, while `SimulationState` keeps the selected switch outputs
and gate states. Coupling remains unsupported and is rejected before replay.

## Entry and movement convention

An `ActorState` records the actor's current cell and the direction in which it
travelled into that cell. Its next tick uses the port opposite that direction
as the rail entry port. It follows the cell's internal transition, crosses the
selected boundary into the matching neighbour, and adopts the selected exit
port as its next travel direction. Therefore an initial actor must begin on a
fixed or placed track whose transition accepts the port opposite its declared
`Actor.direction`.

For example, an actor at `(0, 0)` with direction `E` enters that cell through
its west (`W`) port. An east-west straight takes it to `E`; if the neighbour at
`(0, 1)` has a matching `W` port, the actor enters that cell in one tick with
direction `E`.

## API

```python
from railbound import SimulationOptions, simulate

result = simulate(level, placement, SimulationOptions(max_ticks=100))
assert result.succeeded
```

`simulate()` compiles the board, creates `initial_state(level)`, then calls
`step(state, board)` until it reaches a terminal result. `SimulationState`,
`ActorState`, all event records, and result objects are frozen dataclasses, so
replaying equal inputs is reproducible and previous states are safe to retain.

`SimulationOptions.max_ticks` prevents an otherwise valid cycle from running
forever. Reaching it produces the regular terminal failure reason
`step_limit`, including a final failure event.

## Tick order and device timing

Every `step()` uses one immutable input state:

1. Derive a move intent for each active actor, sorted by actor ID, using gate
   and switch values from the beginning of the tick.
2. A switch routes an actor from its `entry_port` to its selected output;
   either output may merge back into the entry port. A tunnel transfers an
   actor from its entered endpoint to the paired endpoint's `exit_port` in the
   same tick.
3. If an intent cannot enter a track, is blocked by a closed gate, reaches an
   unconnected port, or exits the board, finish atomically with the first
   actor-ID-ordered failure.
4. Detect simultaneous same-cell, head-on edge-swap, and shared-tunnel
   collisions. A collision is terminal and no button effects occur.
5. Apply every accepted move together. Each actor entering a button emits a
   press event; press events are sorted by actor ID, then button ID.
6. Each press toggles every same-channel gate and switch in global device-ID
   order. Multiple presses on a channel therefore toggle its targets once per
   press and can leave them unchanged after an even number of presses.
7. Mark matching arrival requirements as complete, then evaluate their
   optional relative order and emit success if all are met.

Consequently, a button cannot make a closed gate passable in the same tick;
its state change becomes visible to movement in the next tick. Buttons have
the observed `toggle` behavior. A switch with no `channel` remains fixed at
its `initial_output`.

An actor that meets an arrival requirement becomes `arrived` and no longer
moves or occupies a collision destination. This lets independently arriving
actors wait for the rest of a multi-actor condition.

## Events and failures

`SimulationResult.events` is a tuple of `SimulationEvent` records. Event
sequences start at zero, ticks start at one, and order is deterministic. Move
events identify source/target cells and ports; arrival events identify their
target entry; failure events carry one of these `FailureReason` values:

- `derailment`: the actor cannot follow an internal transition from its entry
  port.
- `dead_end`: its selected exit has no matching in-bounds neighbour port.
- `boundary_exit`: its selected exit leaves the board.
- `gate_closed`: its selected destination is blocked by a closed gate.
- `collision`: simultaneous intents share a target cell or swap an edge.
- `arrival_order`: a newly completed arrival contradicts `arrival_order`.
- `step_limit`: the configured replay bound was exhausted.

Each event also records a deterministic `phase`, any `device_ids`, optional
`before_value`/`after_value`, and `causal_event_ids`. A move causes a tunnel
transit or button press; a button press causes each resulting gate or switch
state change. These links permit an explanation layer to trace every dynamic
state update back to the movement that caused it.

A shared tunnel produces the existing `collision` reason and identifies both
tunnel endpoints in `device_ids`.
