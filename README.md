# Railbound Solver

This project provides assisted, reproducible grid calibration and cell
extraction for a Railbound screenshot, a manually authored semantic level
model, and static rail-topology compilation. It does not yet recognize tiles,
simulate movement, or solve levels.

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

The interactive workflow first proposes a grid and draws its candidates over
the screenshot. A proposal at confidence at least `0.70` becomes editable;
otherwise the window visibly asks for manual calibration. For that fallback,
click four cell centers in the shown order: the upper-left cell `(0, 0)`, its
right neighbor `(0, 1)`, its lower neighbor `(1, 0)`, and the bottom-right
cell. The final click derives the number of rows and columns and becomes a
fourth calibration reference; it refines the affine grid rather than merely
setting its dimensions. Then click the centers of any labeled cells whose
markers do not align with the screenshot. Each additional reference refits the
grid and shows RMS and maximum residual error. Zoom with the mouse wheel and
use **Fit image** to restore the full screenshot. Save chooses `grid.json` and
writes an `annotated_grid.png` next to it. Once the grid is fitted, choose a
crop mode and click **Extract cells in memory**. The application opens a
zoomable contact sheet and retains the normalized cells for the following
recognition stage; no crop or calibration files are required for this flow.

The host virtual environment is only for running and checking the application;
use the devcontainer for editing, linting, types, and tests. It is a local
runtime environment, not a Windows installer; packaging an installer is a later
distribution step.

The scripted form remains useful for tests and batch processing. Use the
centers of three adjacent cells: the origin `(0, 0)`, the cell to its right
`(0, 1)`, and the cell below it `(1, 0)`.

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

Both modes write a portable, validated `grid.json` and an
`annotated_grid.png`. The JSON records image dimensions and its SHA-256 digest,
but not the local source path. Make one calibration per screenshot or level;
the digest records which image it belongs to.

Pixels are always `(x, y)`; semantic grid coordinates are always `(row, col)`.
The calibration points are cell centers, not corners. The overlay marks every
cell center, row/column coordinate, grid boundary, origin, and both grid axes.

## Propose a grid automatically

M3 can propose a grid from regularly spaced visible board boundaries. It uses
Hough lines only to generate hypotheses, then scores the fitted affine lattice
for geometric regularity, residual error, and edge support. It never silently
turns a weak proposal into a calibration: inspect the debug overlay and fall
back to the manual command above when its reported confidence is insufficient.

Known dimensions are optional but useful on visually busy screenshots:

```bash
python3 -m railbound detect \
  --image data/screenshot.png \
  --rows 8 --cols 10 \
  --grid-output output/grid.json \
  --overlay-output output/detected_grid.png
```

The command writes `grid.json` only when the strongest proposal reaches the
default confidence of `0.70`; adjust it with `--min-confidence` only after
reviewing the overlay. `--min-cell-size` and `--max-cell-size` narrow the
search where display scale is known. An exit status of 2 means that no
candidate was confident enough, and does not alter any manual calibration.

## Assisted interactive workflow

Use the dedicated interactive command to open a screenshot, propose a grid,
review the overlay, manually refine any cell centers, extract cells in memory,
and save when satisfied:

```bash
python3 -m railbound interactive --image data/screenshot.png
```

The proposal runs when an image is opened and can be rerun with **Propose grid
from visible edges**. The UI accepts only confidence at least `0.70`; lower
confidence candidates remain visible in the debug overlay and the UI asks for
the original four manual points. An accepted proposal seeds those same four
editable references, so any click-based correction switches naturally to the
existing manual affine fit. `calibrate`, `detect`, and `extract` remain the
separate, scriptable stage commands. On Windows, install the `gui` extra as
described above before using this workflow.

## Extract normalized cell crops

Given the screenshot used for calibration and its `grid.json`, M2 rectifies
every grid cell into a consistently oriented, square BGR PNG. The default
`core` mode excludes cell borders; `context` retains neighbouring image area.
The command also writes a deterministic `manifest.json` and a labeled contact
sheet for inspection.

```bash
python3 -m railbound extract \
  --image data/screenshot.png \
  --grid output/grid.json \
  --output-dir output/crops \
  --mode context --mask-mode cell
```

`--mask-mode cell` makes pixels beyond the target cell black in a contextual
crop. This is useful where extra context helps review but decorative neighbours
would otherwise affect later recognition. Crop settings are stored in the
manifest, so the exact output can be recreated from the screenshot and grid
JSON. The command rejects a screenshot whose stored SHA-256 does not match the
calibration.

## Author a semantic level

M4 introduces a screenshot-independent, versioned `level.json` contract. It
contains the board dimensions, terrain/buildability, fixed tracks, actors,
devices, rail inventory, and explicit win conditions. It intentionally does
not implement movement or solving yet. Use `load_level()` and `save_level()`
from `railbound.level_io` (or the package root) to validate and persist these
documents.

[`data/example-level.json`](data/example-level.json) is a small manually
authored example. The complete field reference and validation rules are in
[`docs/level-schema.md`](docs/level-schema.md). A level has no source image or
recognition confidence: those remain separate evidence for a later review
stage.

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

The graph uses `(cell, N|E|S|W)` port nodes. `board.transition(port)` follows
only the rail inside a cell; `board.adjacent(port)` crosses only into a
neighbour with a matching facing port. `boundary_exits`, `unconnected_ports`,
and `connected_components` expose static topology for inspection. This keeps
crossing paths independent rather than treating a crossing as one connected
cell. Placement validation rejects a wrong level ID, non-buildable cells,
exceeding the total rail budget, and unavailable kinds or counts. See
[`docs/board-topology.md`](docs/board-topology.md) for the full format.

## Development

Run **Tasks: Run Task** → `Railbound: Verify changes` for linting, type
checking, and whitespace validation. The equivalent commands from the
repository root are:

```bash
ruff check .
mypy src
```

Use the `Railbound: Run all tests` VS Code task for the full test suite.
