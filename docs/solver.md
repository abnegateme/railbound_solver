# Optimized solver (M10)

`solve()` generates legal `RailPlacement` values and replays each one through
the deterministic simulator. A placement is a solution only when that replay
finishes with `SimulationStatus.SUCCESS`.

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
`Level`. **Solve level** remains on the same **Workflow** tab and uses the
interactive search bounds. Formalization opens the schematic **Level /
solution** workspace; while solving, that workspace samples the real stream of
partial branches, failed candidates, and successful placements. Assigned rails
are drawn directly, while outlined cells show the unresolved remainder of a
partial branch. The verified solution remains after playback.
The search runs outside the Qt GUI thread and sends sampled progress through
queued signals. A bounded GUI-side queue plays those samples at a visible pace,
including when the search itself finishes faster than Qt can repaint. Statistics
and the replay trace are appended to the activity log. The optional
**Placement** tab can load or paste a manual `placement.json`, inspect its M5
port graph, and replay it through M6–M7.

The result includes the successful placement, its `SimulationResult`, the
same ordered event trace in `result.trace`, and `SearchStats`. When no
placement can be found, `placement`, `simulation`, and `trace` are empty and
`status` distinguishes an exhaustive `unsolvable` search from `limit_reached`.
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

The interactive session enables the M10 search policy automatically. It uses
forced buildable cells, optimistic port-connectivity pruning, canonical rail
rotations, deterministic heuristic ordering, and a per-placement runtime-state
cache. These optimizations can only discard candidates that cannot succeed;
every reported solution is still replayed by the simulator. The GUI reports
replay, pruning, cache, and elapsed-time totals in the activity log.

## Deterministic candidate order

The baseline enumerator searches zero rails, then one rail, then successively
larger placements up to the level inventory. It orders cells by `(row, col)`,
piece kinds as `straight`, `curve`, `switch`, `crossing`, and orientations
clockwise. Equivalent rotations are omitted: straights use `N` and `E`;
switches use four branch directions and two opposite selections; crossings
use `N`. Per-cell variants that point beyond the grid are omitted.
Consequently the first solution has the fewest rails and is stable between
runs.

`SolverLimits.max_placements` bounds complete candidates considered,
`SolverLimits.max_rails` bounds placement size, and `SolverLimits.max_ticks`
bounds each replay. For measurements or regressions,
`SolverOptions.baseline()` restores the M9 enumeration and replay policy;
custom conservative predicates can be supplied through
`SolverOptions.pruning_predicates`.

`benchmark_search(level, limits)` runs the baseline and optimized policies
under identical bounds and returns both results plus replay/state reductions.
Use explicit small limits when benchmarking a large unsolved level.
