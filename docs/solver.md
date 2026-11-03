# Exhaustive optimized solver

`solve()` generates legal connected `RailPlacement` values and replays each
one through the deterministic simulator. A placement is a solution only when
that replay finishes with `SimulationStatus.SUCCESS`. The default API call
still returns after the first solution; `find_all=True` exhausts the finite
board search and returns every verified solution in `result.solutions`.

```python
from pathlib import Path

from railbound import SolverLimits, load_level, solve

level = load_level(Path("data/example-level.json"))
result = solve(level, SolverLimits(max_placements=1_000, max_ticks=100))
assert result.solved
assert result.placement is not None
assert result.simulation is not None
```

## Command-line use

Use the stage command with an authoritative, manually authored `level.json`:

```bash
python3 -m railbound solve \
  --level data/example-level.json \
  --placement-output data/solved-placement.json \
  --max-placements 10000
```

It writes a portable placement only after replay reaches success. Exit status
`2` means that the search exhausted a requested bound or the level has no
valid placement in its inventory; no output placement is written in either
case. `--max-rails` and `--max-ticks` expose the other `SolverLimits` bounds.

## Primary interactive workflow

The Windows-host `interactive` command opens the one primary session, rather
than a separate solver window. The visual **Cells** workspace continues from
the calibrated screenshot and in-memory crops. Once every cell is reviewed,
**Formalize level** compiles the annotations into the validated in-memory
`Level` after asking for the level's available rail count. **Solve level**
remains on the same **Workflow** tab and exhausts all legal connected
placements within that gameplay budget unless the user presses **Stop**.
Formalization opens the schematic **Level / solutions** workspace. Only
simulator-verified solutions are visualized there, as soon as they are found;
arrow buttons move through every solution retained so far. Statistics are
appended to the activity log. The optional
**Placement** tab can load or paste a manual `placement.json`, inspect its M5
port graph, and replay it through M6–M7.

Each item in `result.solutions` includes its placement, `SimulationResult`,
event trace, and the candidate number at which it was found. Legacy
`placement`, `simulation`, and `trace` fields retain the first solution. A
stopped exhaustive search has status `stopped` and keeps all solutions found
before the request. When no placement can be found, the solution fields are
empty and status is `unsolvable`.
One placement is shared by every actor, so replay accepts sequential rail reuse.
Simultaneous same-cell and head-on conflicts remain collisions regardless of
the cell's internal topology. Optional `arrival_order` is part of the
authoritative replay state and therefore filters solver results.
Actor starts and arrival targets are fixed oriented topology rather than
placement cells: starts use an aligned straight and targets use a one-port
terminal. They may be visited again on a later tick. Candidate generation
uses only `buildable_cells`, so obstacles cannot receive rails or participate
in a route. It also rejects every rail orientation with a port facing outside
the grid or into an obstacle.
The rear port beneath an actor already standing at the edge is only a virtual
initial entry and is exempt; the actor's first movement must point in bounds.
Interactive formalization exposes the observed straight, curved, and
three-port switch rails. A switch candidate has one perpendicular branch and
two mutually opposite main-line ports. The branch connects bidirectionally to
the selected port, while the other port merges one-way into it. The generic
topology model retains four-way crossings for future explicit level
inventories, not as an implicit interactive default.

The search constructs placements recursively from the actor starts. At every
step it follows the currently reachable port graph and may add a rail only at
an unoccupied buildable frontier cell through a compatible entry port. It
never selects disconnected cell sets for later rejection. Partial placements
reached through different construction orders share one canonical signature.
This restriction preserves every minimum solution: each rail used by a replay
is reachable from a start before its first traversal, while a disconnected
unused rail can be removed to produce a smaller equivalent placement.

The interactive session additionally uses forced buildable cells, optimistic
port-connectivity pruning, canonical rail rotations, deterministic heuristic
ordering, and a per-placement runtime-state cache. These optimizations can
only discard candidates that cannot succeed; every reported solution is still
replayed by the simulator. The GUI reports replay, pruning, cache, and
elapsed-time totals in the activity log.

## Current construction rules

- The finite grid is a hard boundary; no rail port may point outside it.
- A buildable cell holds at most one placed rail.
- Rails connect only through matching north, east, south, and west ports of
  adjacent cells.
- Actors must reach their assigned goal cells in the declared arrival order.
- Observed fixed rails cannot be changed. New rails may join only their
  existing exposed ports.
- Fixed two-port rails declare any two distinct sides. Fixed three-port rails
  declare a perpendicular side branch and selected longitudinal output; the
  opposite longitudinal side is inferred.
- Actor starts use fixed aligned straight track; goal cells use fixed oriented
  terminal track.
- A three-port switch has two longitudinal ports and one perpendicular branch.
  The branch and one selected longitudinal port form the selected route; the
  other longitudinal port merges toward that output.
- Obstacles are not buildable and no placed rail may expose a port toward one.

## Deterministic candidate order

The generator searches zero rails, then one rail, then successively larger
connected placements up to the level inventory. At each recursive state it
orders reachable frontier cells by `(row, col)`, piece kinds as `straight`,
`curve`, `switch`, `crossing`, and orientations clockwise. Equivalent
rotations are omitted: straights use `N` and `E`; switches use four branch
directions and two opposite selections; crossings use `N`. Per-cell variants
that point beyond the grid or into an obstacle are omitted. Consequently the
first solution has the fewest rails and search order is stable between runs.

The interactive workflow does not set the search-only `max_placements` or
`max_rails` overrides; the level's rail inventory still limits every legal
placement as part of the game rules. These overrides remain available to
bounded API and CLI callers, while `max_ticks` bounds each replay. The GUI can
show the current candidate number, but not a precomputed exact total: obtaining
the total with recursive connectivity pruning would require performing the
same exhaustive traversal a second time. For measurements or regressions,
`SolverOptions.baseline()` disables optional pruning, heuristic ordering, and
runtime caching while retaining constructive frontier generation. Custom
conservative predicates can be supplied through
`SolverOptions.pruning_predicates`.

`benchmark_search(level, limits)` runs the baseline and optimized policies
under identical bounds and returns both results plus replay/state reductions.
Use explicit small limits when benchmarking a large unsolved level.
