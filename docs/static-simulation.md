# Deterministic static simulation (M6)

M6 replays a `Level` and `RailPlacement` over the static port graph compiled
by M5. It has no guessed rules for switches, buttons, gates, tunnels, or
coupling: passing any of those mechanics raises `ValueError` before replay.

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

## Tick order

Every `step()` uses one immutable input state:

1. Derive a move intent for each active actor, sorted by actor ID.
2. If any intent cannot enter a track, reaches an unconnected port, or exits
   the board, finish atomically with the first actor-ID-ordered failure.
3. Detect simultaneous same-cell and head-on edge-swap collisions. A collision
   is terminal and no actor moves.
4. Apply every accepted move together.
5. Mark matching arrival requirements as complete, then evaluate their
   optional relative order and emit success if all are met.

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
- `collision`: simultaneous intents share a target cell or swap an edge.
- `arrival_order`: a newly completed arrival contradicts `arrival_order`.
- `step_limit`: the configured replay bound was exhausted.

This is a deliberately small trace model. M7 will extend it with dynamic
device state, phases, and causal links without changing the static port
semantics above.
