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
than a separate solver window. Its **Review and solve** tab continues from the
same calibrated screenshot and in-memory crops: recognition review, semantic
`level.json` validation, limits, solve, trace inspection, and placement save
all happen there. A semantic level may also be opened directly when solving an
authored level without a screenshot. The same tab can load or paste a manual
`placement.json`, inspect its M5 port graph, and replay it through M6–M7
before invoking the search.

The result includes the successful placement, its `SimulationResult`, the
same ordered event trace in `result.trace`, and `SearchStats`. When no
placement can be found, `placement`, `simulation`, and `trace` are empty and
`status` distinguishes an exhaustive `unsolvable` search from `limit_reached`.

The interactive session enables the M10 search policy automatically. It uses
forced buildable cells, optimistic port-connectivity pruning, canonical rail
rotations, deterministic heuristic ordering, and a per-placement runtime-state
cache. These optimizations can only discard candidates that cannot succeed;
every reported solution is still replayed by the simulator. The GUI keeps the
controls unchanged and reports replay, pruning, cache, and elapsed-time totals
in the result pane.

## Deterministic candidate order

The baseline enumerator searches zero rails, then one rail, then successively
larger placements up to the level inventory. It orders cells by `(row, col)`,
piece kinds as `straight`, `curve`, `crossing`, and orientations clockwise.
Equivalent rotations are omitted: straights use `N` and `E`; crossings use
`N`. Consequently the first solution has the fewest rails and is stable
between runs.

`SolverLimits.max_placements` bounds complete candidates considered,
`SolverLimits.max_rails` bounds placement size, and `SolverLimits.max_ticks`
bounds each replay. For measurements or regressions,
`SolverOptions.baseline()` restores the M9 enumeration and replay policy;
custom conservative predicates can be supplied through
`SolverOptions.pruning_predicates`.

`benchmark_search(level, limits)` runs the baseline and optimized policies
under identical bounds and returns both results plus replay/state reductions.
Use explicit small limits when benchmarking a large unsolved level.
