# Railbound Solver

This project provides assisted, reproducible grid calibration and cell
extraction for a Railbound screenshot, a manually authored semantic level
model, rail-topology compilation, and deterministic replay of static and
observed World 2–4 dynamic devices. It does not yet recognize tiles or solve
levels.

The Windows interactive workflow is the primary user journey. Its individual
CLI stages remain available for reproducible batch work, debugging, and tests.

## Calibrate a grid

The devcontainer uses the official Python 3.14 image and installs the core
runtime dependencies and development tools during image build. The checkout is
bind-mounted at runtime, and the development stage exposes `src/` through
`PYTHONPATH`; it does not copy or install the workspace. After pulling these
changes, use **Dev Containers: Rebuild Container** once.

```bash
python3 -m railbound --help
```

The Dockerfile also provides a headless `release` target for scripted use. Its
builder copies the package, installs it non-editably with the core dependencies
declared in `pyproject.toml` into `/opt/railbound-venv`, and the final image
contains that environment rather than the source checkout or development tools.

```bash
docker build -f .devcontainer/Dockerfile --target release -t railbound-solver .
docker run --rm railbound-solver --help
```

## Run the GUI on Windows

Development and automated checks stay in the devcontainer. The native GUI and
future Windows window capture run on the Windows host, where they can access
the desktop. A VS Code task is unsuitable for this: in a devcontainer it would
run inside Linux rather than Windows.

In PowerShell at the repository root, create or refresh the GUI runtime
environment:

```powershell
.\scripts\setup-windows-gui.ps1
```

This creates `.venv-gui` and installs the current source non-editably with its
`gui` extra. Run it after changing application code, then start the calibration
window to check the same runtime-style environment used for local release:

```powershell
.\.venv-gui\Scripts\python.exe -m railbound interactive --image data\Railbound-1-13.PNG
```

The window accepts proposals at confidence `0.70` or above. Otherwise click
four centers in this order: `(0, 0)`, `(0, 1)`, `(1, 0)`, then the
bottom-right cell. Extra clicks refine the fit; **Fit image** resets zoom.
Saving writes `grid.json` and `annotated_grid.png`; **Extract cells in memory**
opens a crop preview for the next stage.

Use the host environment only to run the GUI. Edit and verify in the
devcontainer. The scripted form below takes centers for `(0, 0)`, `(0, 1)`,
and `(1, 0)`.

```bash
python3 -m railbound calibrate \
  --image data/screenshot.png \
  --rows 8 --cols 10 \
  --origin 120 240 \
  --next-col 172 240 \
  --next-row 120 292 \
  --grid-output output/grid.json \
  --overlay-output output/annotated_grid.png
```

Both modes write portable output. `grid.json` records dimensions and a SHA-256
digest, never the local path. Pixels are `(x, y)`; cells are `(row, col)`.

## Propose a grid automatically

M3 proposes an affine grid from visible boundaries. It writes a grid only when
the best proposal meets the confidence threshold; otherwise review the overlay
and calibrate manually.

Known dimensions are optional but useful on visually busy screenshots:

```bash
python3 -m railbound detect \
  --image data/screenshot.png \
  --rows 8 --cols 10 \
  --grid-output output/grid.json \
  --overlay-output output/detected_grid.png
```

The default threshold is `0.70`; exit status `2` means no accepted candidate.
Use `--min-cell-size` and `--max-cell-size` only when display scale is known.
`interactive` provides the same proposal, correction, extraction, and save
flow on Windows.

## Extract normalized cell crops

M2 writes one oriented square BGR PNG per cell, a deterministic manifest, and
a contact sheet. `core` excludes borders; `context` retains nearby pixels.

```bash
python3 -m railbound extract \
  --image data/screenshot.png \
  --grid output/grid.json \
  --output-dir output/crops \
  --mode context --mask-mode cell
```

`--mask-mode cell` blacks out non-cell pixels in contextual crops. The manifest
records settings, and extraction rejects a screenshot whose digest differs
from its calibration.

## Author a semantic level

`level.json` is the validated, screenshot-independent contract for board
dimensions, terrain, tracks, actors, devices, inventory, and goals. Use
`load_level()` / `save_level()`; see [the example](data/example-level.json)
and [schema reference](docs/level-schema.md). Recognition evidence stays out
of the level.

## Compile static rail topology

M5 keeps a candidate solution separate from the level. A
[`RailPlacement`](src/railbound/topology.py) names its target `level_id` and
lists static `straight`, `curve`, or `crossing` rails by cell and cardinal
orientation. [`data/example-placement.json`](data/example-placement.json)
completes the example level. Load both documents and compile a port graph with
`load_placement()` and `compile_board()`:

```python
from pathlib import Path

from railbound import compile_board, load_level, load_placement

level = load_level(Path("data/example-level.json"))
placement = load_placement(Path("data/example-placement.json"))
board = compile_board(level, placement)
```

The graph uses `(cell, N|E|S|W)` nodes. `transition()` stays within a rail and
`adjacent()` crosses a validated edge, preserving independent crossing paths.
Placement validation checks its level ID, cells, and inventory. See
[topology details](docs/board-topology.md).

## Replay a placement

M6 replays fixed and placed rails with `simulate()`. The supplied example now
includes a fixed start track, so its matching placement reaches the goal:

```python
from pathlib import Path

from railbound import load_level, load_placement, simulate

level = load_level(Path("data/example-level.json"))
placement = load_placement(Path("data/example-placement.json"))
result = simulate(level, placement)
assert result.succeeded
```

The result contains immutable actor/device state and an ordered trace. Movement
is simultaneous; failures and collisions are atomic. M7 supports toggle
buttons, channel-matched gates and switches, and same-tick tunnel transfers.
Button effects apply next tick; coupling is unsupported. Use
`SimulationOptions(max_ticks=...)` for cycles. See [simulation rules](docs/static-simulation.md).

## Development

Use `Railbound: Verify changes` for checks and `Railbound: Run all tests` for
the full suite.
