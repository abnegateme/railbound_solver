# Deterministic simulation (M6–M7)

M6 replays rails; M7 adds observed World 2–4 devices. The board is immutable;
`SimulationState` holds selected switch outputs and gate states. Coupling is
unsupported.

## Entry and movement convention

An actor enters its current cell through the port opposite its travel
direction, follows its internal transition, crosses the matching neighbour
port, and adopts the exit port as its next direction. An actor at `(0, 0)`
travelling `E`, for example, enters through `W`.

## API

```python
from railbound import SimulationOptions, simulate

result = simulate(level, placement, SimulationOptions(max_ticks=100))
assert result.succeeded
```

`simulate()` compiles the board and repeatedly calls `step()`. State, result,
and event objects are immutable. `max_ticks` turns a cycle into `step_limit`.

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

A button cannot open a gate for the same tick. Buttons toggle; an unchannelled
switch stays at `initial_output`. Arrived actors no longer move or collide.

## Events and failures

`SimulationResult.events` is deterministic: sequences start at zero and ticks
at one. Failures are:

- `derailment`: the actor cannot follow an internal transition from its entry
  port.
- `dead_end`: its selected exit has no matching in-bounds neighbour port.
- `boundary_exit`: its selected exit leaves the board.
- `gate_closed`: its selected destination is blocked by a closed gate.
- `collision`: simultaneous intents share a target cell or swap an edge.
- `arrival_order`: a newly completed arrival contradicts `arrival_order`.
- `step_limit`: the configured replay bound was exhausted.

Events also contain `phase`, `device_ids`, optional before/after values, and
causal IDs. Moves cause tunnel/button events; button events cause state
changes. Shared tunnels report `collision` and both endpoint IDs.
