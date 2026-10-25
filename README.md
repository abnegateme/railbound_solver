# Railbound Solver

This project begins with assisted, reproducible grid calibration and cell
extraction for a Railbound screenshot. It does not yet recognize tiles or solve
levels.

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
.\.venv-gui\Scripts\python.exe -m railbound calibrate --interactive --image data\Railbound-1-13.PNG
```

Click four cell centers in the shown order: the upper-left cell `(0, 0)`, its
right neighbor `(0, 1)`, its lower neighbor `(1, 0)`, and the bottom-right
cell. The final click derives the number of rows and columns and becomes a
fourth calibration reference; it refines the affine grid rather than merely
setting its dimensions. Then click the centers of any labeled cells whose
markers do not align with the screenshot. Each additional reference refits the
grid and shows RMS and maximum residual error. Zoom with the mouse wheel and
use **Fit image** to restore the full screenshot. Save chooses `grid.json` and
writes an `annotated_grid.png` next to it. Once the four-point grid is fitted,
choose a crop mode and click **Extract cells in memory**. The application opens
a zoomable contact sheet and retains the normalized cells for the following
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

## Development

Run **Tasks: Run Task** → `Railbound: Verify changes` for linting, type
checking, and whitespace validation. The equivalent commands from the
repository root are:

```bash
ruff check .
mypy src
```

Use the `Railbound: Run all tests` VS Code task for the full test suite.
