# Railbound Solver Implementation Plan

## 1. Current repository assessment

- Git is clean on `master`, synchronized with `origin/master`.
- The repository contains nine tracked files: devcontainer and VS Code configuration, `.gitignore`, `AGENTS.md`, and `pyproject.toml`.
- There is currently no application code, test suite, README, Makefile, or data directory. This matches the intentional cleanup and is not a defect.
- The devcontainer uses the official Python 3.14 Debian-based image, `pip`, user `vscode`, and working directory `/workspace`. Docker Compose is only the existing mechanism used by VS Code to start the development container.
- VS Code is configured to use `/usr/local/bin/python3`. Project commands consistently use `python3` unless an entry-point executable is later installed.
- The ignored `.venv` is not the configured interpreter and does not have a working `pip`; it should not become part of the project design.
- Dependencies are installed from their current compatible releases during the devcontainer image build.
- Installed development tools are pytest 9.1.1, pytest-cov 7.1.0, Ruff 0.17.0, and mypy 2.4.0.
- `pyproject.toml` currently contains tool configuration only. It does not yet define package metadata, build configuration, or application entry points.
- `ruff check .` passes.
- `pytest` discovers no tests and exits with code 5. This reflects the intentionally empty application, not a failing test suite.
- Use the newest stable Python version supported by all required binary dependencies. Python 3.14 is currently selected because OpenCV publishes wheels through 3.14; upgrade when its 3.15 wheels are available.

## 2. Architectural principles

The proposed subsystem boundary is sound, with one refinement: recognition results should not immediately become authoritative level data.

```text
Screenshot
    │
    ├── Grid geometry ── Cell extraction ── Recognition observations
    │                                           │
    │                                validation/manual correction
    │                                           │
    └─────────────────────────────── Formal Level Model
                                                │
                                      Board compilation
                                                │
                                  Simulator + Event Trace
                                                │
                                             Solver
                                                │
                                Explanation and visualization
```

Core principles:

1. **Pixels stop at the parsing boundary.** The solver and simulator accept semantic level data, never images or recognition confidence.
2. **Recognition is evidence, not truth.** A recognizer should return candidates, confidence, and provenance. A validation step turns those observations into a complete `Level`.
3. **The formal level is the stable integration contract.** It must be serializable, manually authorable, schema-versioned, and usable without any screenshot.
4. **Board topology is derived data.** Store semantic pieces and devices in JSON; compile them into a port graph rather than treating a serialized graph as authoritative.
5. **Rail placement is separate from the level.** The level describes fixed objects, buildable cells, inventory, actors, devices, and goals. A `RailPlacement` describes the player's proposed solution.
6. **Simulation precedes search.** A solver is only trustworthy if every candidate solution can be replayed through a deterministic simulator.
7. **Events are first-class output.** The simulator should produce structured events from the beginning. Debugging, explanations, hints, and solution overlays can all consume the same trace.
8. **Determinism is mandatory.** Coordinate rounding, candidate ordering, simultaneous movement resolution, solver tie-breaking, and event ordering must be explicitly defined.
9. **Automation must fail visibly.** Low-confidence grid or object recognition should request manual correction rather than silently producing a plausible but incorrect level.
10. **Start with observed mechanics.** Do not create a universal Railbound ontology before examining representative levels. Add semantic types as mechanics require them.
11. **The interactive session is the primary product workflow.** A user should be able to open a screenshot, review and correct recognition, progress to a semantic level, simulate, solve, and inspect the explanation without leaving the interactive session. CLI stage commands remain for deterministic batch work, debugging, and tests; they are not the primary user journey.

### Execution boundary

- The Linux devcontainer is the development environment for editing, linting,
  type checking, and automated tests.
- The native GUI and Windows screen capture run in a Windows-host virtual
  environment. A devcontainer cannot directly access Windows windows.
- The GUI dependency is an optional package extra installed by the host setup
  command. The setup command always creates a non-editable runtime installation
  from the current source, which is checked immediately on the host.
- A future host GUI may send frames and commands to solver services in the
  devcontainer, but desktop access must stay on the host side of that boundary.

## 3. Proposed domain model

### Geometry and parsing

- `PixelPoint(x, y)`
- `CellCoord(row, col)`
- `GridGeometry`
  - `rows`, `cols`
  - `origin_px`
  - `col_axis_px`
  - `row_axis_px`
  - image dimensions
  - calibration method and residual/error metrics
- `GridCalibration`
  - supplied control points
  - method name
  - source image identifier or hash
- `CellCrop`
  - cell coordinate
  - normalized image
  - source polygon
  - crop/mask parameters
- `RecognitionObservation`
  - cell coordinate
  - candidate semantic values
  - confidence scores
  - recognizer/template version
  - evidence or crop reference

The forward transform is:

```text
pixel(row, col) = origin + col × col_axis + row × row_axis
```

The inverse should solve the two-axis linear system and return continuous grid coordinates. Rounding and bounds checking should be separate operations so callers can inspect distance from the nearest cell.

### Semantic level

A preliminary `Level` should contain:

- `schema_version`
- level identifier and optional human-readable metadata
- grid dimensions
- terrain/buildability
- fixed track pieces
- actors and their initial state
- devices
- available rail inventory or rail budget
- goals/win conditions
- a ruleset version

A cell should not be forced into one large mutually exclusive enum. Terrain, track, actor, and device can coexist conceptually and should be modeled as separate layers.

Likely entities include:

- `TrackPiece`: set of ports, internal port transitions, and fixed/placeable status
- `Actor`: stable ID, position, direction, type, and optional number/color
- `Switch`: entry/output ports and initial state
- `Button`: channel/color and activation behavior
- `Gate`: channel/color and initial state
- `TunnelEndpoint`: pair ID and local entry/exit port
- `RailInventory`: total budget or counts by permitted piece type
- `WinCondition`: required arrivals, ordering, coupling, or other observed rules

The first schema should cover only mechanics demonstrated by selected reference levels.

### Recognition boundary

Use two related documents:

- `level.json`: authoritative semantic input for simulator and solver.
- `recognition_report.json`: candidates, confidence, source crops, and provenance.

A parser may produce a `LevelDraft` combining both, but the simulator should receive only a validated `Level`. Manual corrections change the semantic value and record that it was overridden.

### Rail placement

`RailPlacement` should map buildable coordinates to selected track pieces. It should be serializable separately from `Level`, allowing multiple candidate solutions for one level.

### Port graph

Use cell ports as graph vertices rather than whole cells:

```text
(row, col, N|E|S|W)
```

This representation handles straight and curved tracks as internal port transitions, crossings as independent transition pairs, adjacent cells as boundary connections, switches as state-dependent transitions, and tunnels as long-distance edges.

A cell-only graph loses important information for crossings, entry direction, and switches. The simulator should use a transition interface conceptually equivalent to `transition(entry_port, dynamic_state) -> exit_port or failure`.

A compiled NetworkX graph may be useful for validation, reachability, and solver heuristics, but it should not replace the explicit transition rules used by simulation.

### Scaling mechanics beyond the static graph

The port graph is the stable topology layer, not a universal representation of
all game rules. It scales naturally to new rail shapes and static topology,
but a button, station, semaphore, passenger, decoy, or other future object
may also require state, timing, or an interaction with another object.

Treat every supported mechanic as four explicit additions:

1. A validated semantic definition in `Level`, independent of pixels.
2. Immutable compiled data attached to the board when it affects topology or
   location-based lookup.
3. A typed runtime-state field plus deterministic transition/effect function
   when it can change during replay.
4. Structured events, rule-focused microboards, and documented tick
   precedence for every externally visible effect.

The simulator should retain one common transition boundary:

```text
next_state, events = step(state, compiled_board)
```

Before adding the next qualitatively different dynamic mechanic after M7,
extract device-specific code behind a common mechanics interface or registry.
It must expose topology/entry handling, post-movement effects, state
validation, and deterministic event generation without making pixels or
recognition data visible to the simulator. This prevents `simulation.py` from
becoming a growing collection of unrelated conditional branches while keeping
observed rules explicit.

The eventual solver searches a state space layered over the port graph, not
only graph paths. A runtime search node must be able to include at least:

```text
(actor positions and travel directions,
 dynamic device states,
 goal/progress state,
 tick or equivalent timing state)
```

Rail placement remains a separate candidate design variable. The simulator is
the authority for edges in this runtime state space: a solver may use static
graph reachability for pruning, but must replay every proposed solution using
the same `step()` semantics. Add inventory, consist/coupling, waiting, or
other mechanic-specific state to the node only when an observed rule requires
it.

### Simulation and events

A simulation state will eventually include:

- tick number
- actor/consist positions
- entry or travel direction
- wagon ordering and coupling
- switch states
- gate states
- active button/channel state
- tunnel transit state, if traversal consumes time
- terminal status

Suggested tick phases:

1. Read an immutable snapshot of the current state.
2. Compute movement intents for all actors.
3. Identify interactions triggered by those intents.
4. Resolve collisions, coupling, and mutually conflicting moves.
5. Apply accepted moves atomically.
6. Apply device state changes at the empirically established time.
7. Emit deterministically ordered events.
8. Evaluate failure and success conditions.

The exact placement of steps 3–6 must be confirmed against the game.

Each event should contain a stable sequence number, tick and phase, event kind, actor/device IDs, cell and optional port, before/after values, causal event IDs, and structured details rather than preformatted prose.

## 4. Milestone roadmap

Dependency summary:

```text
M0 → M1 → M2
      │     └──────────────┐
      └→ M3                │
                           ↓
M4 → M5 → M6 → M7 → M9 → M10 → M12
│                   ↑              ↑
└────────→ M8 ──────┘              │
M1 + M8 + M9 ───────────────→ M11 ─┘
```

M3 and M8 improve automation but do not block manually authored levels, simulation, or the baseline solver.

### M0 — Repository and environment verification

**Goal:** Establish the existing environment as the supported baseline.

**Inputs:** Current repository and devcontainer.

**Deliverables:** Documented commands, interpreter convention, and confirmed library availability.

**Proposed interfaces:** None.

**Dependencies:** Existing tools only.

**Tests:** Import smoke checks, Ruff, and pytest discovery.

**Acceptance criteria:** Python 3.14 and required libraries import successfully; lint runs; the absence of tests is explicitly understood.

**Risks:** Confusing the ignored `.venv` with the configured system interpreter.

**What we explicitly do not implement yet:** Source tree, package metadata, application code, or environment changes.

**Status:** Verified by the current assessment.

### M1 — Assisted calibrated grid vertical slice

**Goal:** Convert one screenshot plus manual calibration into validated grid geometry, `grid.json`, and `annotated_grid.png`.

**Inputs:** Screenshot and four ordered cell centers: the origin, its right and
lower neighbours, and the bottom-right cell. The latter derives the row and
column counts, which remain editable for correction.

**Deliverables:** Grid geometry model, transforms, JSON serialization, scripted
CLI, native calibration window, overlay renderer, and unit tests.

**Proposed interfaces:** `GridGeometry.grid_to_pixel()`, `pixel_to_grid()`,
`nearest_cell()`, `load_grid()`, `save_grid()`, a scripted CLI command, and a
native window reusable by the future solver UI.

**Dependencies:** NumPy, OpenCV, Pydantic, standard-library JSON/path
handling, and the host-only `gui` extra containing PySide6.

**Tests:** Synthetic affine grids, four-click dimension derivation and affine
refinement, additional control-point fitting, round trips, invalid/collinear
calibration, bounds behavior, JSON round-trip, and overlay dimensions.

**Acceptance criteria:** All expected centers are generated; inverse mapping is accurate within an agreed tolerance; invalid calibration is rejected; output JSON reloads unchanged; the overlay visibly identifies rows, columns, and axes.

**Risks:** Incorrect point ordering, ambiguous grid extents, rounding near boundaries, and affine geometry being insufficient for some screenshots.

**What we explicitly do not implement yet:** Automatic detection, cell classification, game objects, or solver behavior.

### M2 — Cell extraction

**Goal:** Produce consistently oriented normalized images for every calibrated cell.

**Inputs:** Screenshot and `GridGeometry`.

**Deliverables:** Core and contextual crop modes, optional polygon mask,
normalization settings, manifest, and debug contact sheet. The native GUI keeps
the extracted cells and contact sheet in memory after calibration; writing
files is an optional export/debug path rather than a prerequisite for the next
interactive stage.

**Proposed interfaces:** `CellExtractor.extract(image, grid, spec) -> list[CellCrop]` and `CropSpec(output_size, core_fraction, context_fraction, mask_mode)`.

**Dependencies:** NumPy and OpenCV. Pillow only if it provides a concrete format or text-rendering advantage.

**Tests:** Synthetic colored cells, border cells, orientation markers, crop dimensions, mask behavior, and deterministic manifests.

**Acceptance criteria:** Every in-bounds cell yields one indexed crop of identical shape and orientation; extraction can be reproduced solely from image, grid JSON, and crop settings. In the GUI, a calibrated grid can proceed directly to an in-memory crop set without requiring intermediate files.

**Risks:** Landscape leaking into crops, insufficient context for large objects, interpolation artifacts, and perspective distortion.

**What we explicitly do not implement yet:** Semantic classification or automatic grid detection.

### M3 — Assisted and automatic grid detection

**Goal:** Begin an interactive session with an automatic grid proposal, reducing manual calibration while retaining a reliable fallback.

**Inputs:** Reference screenshots and M1 geometry/error metrics.

**Deliverables:** Calibration candidates with confidence, residuals, and debug overlays; an interactive proposal/review step that adopts only a confident candidate and falls back visibly to M1 manual calibration.

**Proposed interfaces:** `GridDetector.propose(image, hints) -> list[CalibrationCandidate]`.

**Dependencies:** NumPy and OpenCV.

**Tests:** Reference screenshots with expected centers, perturbation tests, and negative images that must produce low confidence.

**Acceptance criteria:** Opening a screenshot in the interactive workflow automatically proposes a grid within the agreed pixel tolerance or explicitly reports insufficient confidence; manual M1 calibration always remains available. The separate stage commands remain reproducible support tools.

**Risks:** Artwork and rails create false edges; aspect ratios and display scaling vary; perspective may violate the affine model.

**What we explicitly do not implement yet:** Object recognition or a requirement for fully unattended detection.

Recommended method order:

1. Three manually selected centers.
2. More than three manual centers with least-squares affine fitting and residual reporting.
3. Local template/correlation refinement around predicted centers.
4. Lattice fitting from multiple visual candidates.
5. Edge/Hough detection as candidate generation, not as the sole decision-maker.
6. Feature matching if stable reusable visual landmarks are demonstrated.
7. Homography or local warping only if fixture residuals prove the affine model insufficient.

Hough-first detection is likely fragile because game artwork contains many strong non-grid lines. Template or feature methods require evidence that visual anchors remain stable across worlds, resolutions, and visual themes.

### M4 — Manually authored Level Model and JSON

**Goal:** Represent selected levels without images.

**Inputs:** Several manually inspected levels and documented mechanics.

**Deliverables:** Versioned semantic schema, validators, JSON load/save, manually authored examples, and schema documentation.

**Proposed interfaces:** `Level`, `CellCoord`, `TrackPiece`, `Actor`, device models, `RailInventory`, `WinCondition`, `load_level()`, and `save_level()`.

**Dependencies:** Pydantic and the standard library.

**Tests:** JSON round-trip, duplicate IDs, invalid coordinates, inconsistent tunnel pairs, illegal ports, unknown schema version, and missing required fields.

**Acceptance criteria:** At least one small representative level can be authored, validated, serialized, reloaded, and inspected without any PNG.

**Risks:** Prematurely encoding incorrect mechanics or forcing all content into a single cell enum.

**What we explicitly do not implement yet:** Recognition, movement, or solving.

### M5 — Static rail semantics and board compilation

**Goal:** Compile a level and placement into validated port-level topology.

**Inputs:** M4 `Level` and `RailPlacement`.

**Deliverables:** Port model, piece transition definitions, adjacency compilation, placement validation, and topology inspection.

**Proposed interfaces:** `compile_board(level, placement) -> CompiledBoard`, `validate_placement()`, and `transition()` for static pieces.

**Dependencies:** Standard library; NetworkX only if it materially simplifies validation or reachability tools.

**Tests:** Straights, curves, disconnected neighbors, crossings, boundary exits, illegal placement, and isolated components.

**Acceptance criteria:** Compiled connectivity matches hand-drawn tiny boards and preserves independent crossing paths.

**Risks:** Confusing geometric adjacency with actual connectivity and mishandling direction at crossings.

**What we explicitly do not implement yet:** Time, moving actors, switches, buttons, gates, or tunnels.

### M6 — Deterministic static simulator

**Goal:** Replay movement on fixed track without interactive devices.

**Inputs:** Compiled board and initial actor state.

**Deliverables:** Immutable simulation state, tick loop, terminal results, basic event trace, and failure reasons.

**Proposed interfaces:** `simulate(level, placement, options) -> SimulationResult` and `step(state, board) -> StepResult`.

**Dependencies:** Standard library and existing model validation.

**Tests:** Straight arrival, curve traversal, derailment, dead end, collision, coupling if already confirmed, and deterministic trace replay.

**Acceptance criteria:** Identical inputs produce identical states and events; every success or failure has a structured reason; synthetic scenarios match manually computed traces.

**Risks:** Incorrect simultaneous movement and collision ordering.

**What we explicitly do not implement yet:** Dynamic devices, search, or recognition.

### M7 — Dynamic mechanics and full event model

**Goal:** Add buttons, switches, gates, tunnels, and any required World 4 mechanics.

**Inputs:** M6 simulator plus empirical rule observations.

**Deliverables:** Dynamic state fields, explicit tick precedence, device events, causal links, and rule-focused regression scenarios.

**Proposed interfaces:** Device-specific transition/effect functions and typed `Event` records.

**Dependencies:** No new libraries expected.

**Tests:** One-mechanic microboards, same-tick interactions, multiple actors activating one channel, tunnel timing, switch state changes, and device/collision combinations.

**Acceptance criteria:** Each supported rule has a minimal reproducible test and documented timing semantics; event traces fully explain state changes.

**Risks:** The game may use unintuitive same-tick precedence, animation timing may not equal logical timing, and multiple devices may share a channel.

**What we explicitly do not implement yet:** Solver optimization or unsupported mechanics inferred only from appearance.

### M8 — Deterministic object recognition

**Goal:** Convert cell crops into semantic candidates suitable for review and manual correction.

**Inputs:** M2 crops, M4 vocabulary, and labeled reference cells.

**Deliverables:** Template/heuristic recognizers, confidence scoring, recognition report, uncertain-cell review list, and conversion to `LevelDraft`.

**Proposed interfaces:** `recognize(crop) -> RecognitionObservation` and `resolve(observations, overrides) -> Level`.

**Dependencies:** NumPy and OpenCV. No ML framework.

**Tests:** Labeled crop fixtures, rotations/orientations, visual variations, unknown-object rejection, confidence thresholds, and end-to-end draft generation.

**Acceptance criteria:** Supported classes meet per-class fixture thresholds; uncertain inputs remain unresolved instead of being silently guessed; all results can be overridden.

**Risks:** Decorative variation, object overlap, orientation ambiguity, and poorly calibrated confidence.

**What we explicitly do not implement yet:** Neural networks or an exhaustive object catalog.

Recognition approach order:

1. Manual semantic annotation.
2. Exact or normalized template matching.
3. Simple deterministic features: masks, contours, color/channel sampling, symmetry, and line/port structure.
4. Feature matching for visually stable but scaled assets.
5. A small classical classifier only if deterministic rules become unmaintainable.
6. ML image classification only after a labeled dataset and measured deterministic baseline demonstrate the need.

### M9 — Baseline solver

**Goal:** Find valid rail placements using the simulator as the final authority.

**Inputs:** M4–M7; recognition is optional because levels can be authored manually.

**Deliverables:** Candidate placement generator, deterministic search over
placement candidates and simulator-reachable runtime states, solver result,
trace, and search statistics.

**Proposed interfaces:** `solve(level, limits) -> SolverResult`, containing placement, simulation result, event trace, and statistics.

**Dependencies:** Standard library; NetworkX may support reachability heuristics.

**Tests:** Synthetic puzzles with zero, one, and multiple solutions;
unsatisfiable puzzles; budget limits; deterministic tie-breaking; dynamic-state
revisits do not conflate distinct switch/gate/progress configurations; every
returned solution replayed by the simulator.

**Acceptance criteria:** The solver resolves a small agreed puzzle set within explicit limits, and every reported solution independently simulates to success.

**Risks:** Combinatorial growth and unclear rail inventory rules.

**What we explicitly do not implement yet:** Z3, OR-Tools, SAT, CP-SAT, or aggressive optimization.

The simplest useful strategy is iterative deepening depth-first search over a canonical ordering of buildable cells and legal track pieces. It has lower memory use than BFS, can search increasing rail counts, and supports early pruning. Plain brute force is useful only as a correctness oracle on tiny boards. A* requires a proven admissible heuristic; constraint solvers should wait for evidence that ordinary search cannot meet target sizes.

### M10 — Search pruning and performance

**Goal:** Solve representative levels efficiently without changing semantics.

**Inputs:** Instrumented M9 solver and benchmark levels.

**Deliverables:** Benchmarks, transposition cache, connectivity pruning, forced-placement rules, symmetry reduction, and optional heuristic ordering.

**Proposed interfaces:** Pluggable pruning predicates and structured `SearchStats`.

**Dependencies:** No new dependency by default.

**Tests:** Optimized and baseline solvers return equally valid results; benchmark limits; cache-key correctness; pruning never removes known solutions.

**Acceptance criteria:** Measured improvement on representative levels with unchanged simulator-verified correctness.

**Risks:** Unsound pruning and memory-heavy caches.

**What we explicitly do not implement yet:** External constraint solvers unless benchmarks establish a concrete need.

### M11 — Solution and recognition overlay

**Goal:** Render debugging and solution information over the original screenshot.

**Inputs:** Screenshot, grid geometry, optional recognition report, solution, and event trace.

**Deliverables:** Layered renderer for coordinates, confidence, pieces, routes, and event order.

**Proposed interfaces:** `render_overlay(image, grid, layers) -> image`.

**Dependencies:** OpenCV; Pillow only if needed for higher-quality text.

**Tests:** Output dimensions, coordinate alignment, layer enable/disable behavior, deterministic rendering, and selected visual golden files.

**Acceptance criteria:** A solution placement and its route align with the calibrated cells and can be inspected without affecting solver behavior.

**Risks:** Clutter, unreadable labels, and brittle pixel-exact golden tests.

**What we explicitly do not implement yet:** Interactive UI or video animation.

### M12 — Explanation and hint generation

**Goal:** Turn a validated placement and causal event trace into human-readable explanations at several disclosure levels.

**Inputs:** Solver result, trace, and device relationships.

**Deliverables:** Structured explanation model, deterministic text renderer, dependency ordering, and weak/medium/full hint policies.

**Proposed interfaces:** `explain(result, detail) -> Explanation` and `generate_hint(result, strength) -> Hint`.

**Dependencies:** Standard library only.

**Tests:** Stable ordering, causal references, no unsupported claims, and hint-strength disclosure tests.

**Acceptance criteria:** Full explanations match the replayed trace, and weaker hints reveal strictly less information than stronger hints.

**Risks:** A chronological trace alone may not establish causality; causal references must therefore be captured during simulation.

**What we explicitly do not implement yet:** Generative AI, localization, or natural-language variation.

## 5. Detailed first vertical slice

The first implementation should offer a native assisted-calibration window and
retain a reproducible scripted command:

```text
screenshot + four center selections
    → validated GridGeometry
    → grid.json
    → annotated_grid.png
```

### Inputs

- Image path.
- Four ordered cell centers:
  - `origin`: center of `(0, 0)`
  - `next_col`: center of `(0, 1)`
  - `next_row`: center of `(1, 0)`
  - `last_cell`: center of the bottom-right cell; its grid coordinate derives
    `rows` and `cols`.
- Editable row/column controls and a live overlay in the native window. The
  bottom-right point and further selected labeled cells are retained as
  control points in a least-squares affine fit.
- Explicit output paths for scripted operation; a save dialog for interactive
  operation.

The native interface is the primary user path because it eliminates external
pixel-coordinate measurement and row/column counting. The saved calibration
JSON remains the reviewable, repeatable integration artifact, while the
scripted command remains available for tests and batch processing.

### Geometry behavior

- `col_axis = next_col - origin`
- `row_axis = next_row - origin`
- Reject nearly collinear or unrealistically short axes.
- Convert grid coordinates to pixel centers using the affine formula.
- Invert the 2×2 basis matrix for pixel-to-grid conversion.
- Return continuous coordinates before rounding.
- When selecting a cell, return the nearest coordinate, distance from its expected center, and whether it is within grid bounds.
- Define pixel coordinates as `(x, y)` and semantic coordinates as `(row, col)` everywhere.

### Proposed `grid.json` content

- Schema version
- Source image dimensions and optional content hash
- Row/column counts
- Origin, column axis, and row axis
- Original calibration points
- Calibration method
- Residual/error metrics
- Coordinate convention

The source path should not be required for portability.

### Overlay behavior

The annotated image should display every expected cell center, row and column labels, the outer grid boundary, a distinct origin marker, colored row/column axes, and warnings for invalid or out-of-image geometry.

### Test set

- Pure horizontal/vertical grid.
- Skewed affine grid.
- Negative axis direction.
- Forward/inverse round-trip.
- Points near cell boundaries.
- Out-of-bounds pixels.
- Collinear calibration rejection.
- JSON save/load equality.
- Overlay retains source dimensions and covers all expected centers.

### Completion boundary

This slice is complete when one real screenshot can be calibrated manually and reproduced from saved inputs, while synthetic tests prove the transform independently. It does not include crop extraction or any knowledge of Railbound objects.

## 6. Testing strategy

- **Geometry tests:** Exact synthetic transforms, inverse mapping, degeneracy validation, rounding, and bounds.
- **Parser tests:** Screenshot and calibration to expected grid/crops, using tolerances for coordinates rather than fragile full-image equality.
- **Model tests:** JSON round-trip, schema versioning, semantic validation, and useful error messages.
- **Topology tests:** Tiny hand-drawn port networks, especially crossings and disconnected neighboring cells.
- **Simulator tests:** One rule per tiny board, followed by carefully selected interaction tests.
- **Solver tests:** Synthetic puzzles small enough to reason about manually. Every returned placement must be replayed by the simulator.
- **Regression tests:** A curated set of real levels spanning visual styles and mechanics.
- **Visualization tests:** Structural assertions first; a small number of reviewed golden images only where visual regression matters.

Each reference level should eventually contain:

```text
screenshot.png
expected_grid.json
expected_level.json
expected_solution.json
recognition_report.json
fixture_metadata.json
```

Fixture progression:

- M1: one synthetic grid and, if redistribution is permitted, one real screenshot.
- M2–M3: several screenshots covering resolution, scale, and grid appearance.
- M4–M7: manually authored semantic levels independent of screenshots.
- M8: labeled cell crops and expected recognition candidates.
- M9 onward: expected placements, traces, and solver limits.

Generated debug artifacts should normally go to an ignored output directory. Only deliberately reviewed fixtures and golden files should be committed.

## 7. Dependency roadmap

| Dependency | First justified use | Purpose |
|---|---:|---|
| Standard library | M1 | CLI, JSON support, enums, dataclasses, paths |
| NumPy | M1 | Affine geometry, matrices, image arrays |
| OpenCV headless | M1 | Image I/O, drawing, transforms, CV |
| Pydantic | M1/M4 | Validated persisted schemas and JSON |
| Pillow | M11 or earlier only if justified | Better text or format handling |
| NetworkX | M5 or M9, optional | Reachability, topology inspection, heuristics |
| pytest | M1 | Deterministic tests |
| pytest-cov | After meaningful code exists | Coverage feedback |
| Ruff | M1 | Existing lint and formatting policy |
| mypy | M1 | Existing static type policy |

No additional dependency is currently justified. In particular, ML frameworks and constraint solvers should not be added until measurements demonstrate the need.

The current requirements are unpinned. Exact pinning or a constraints file should be considered when reproducible builds become necessary, but it is not required to begin M1 and should not trigger an environment redesign.

## 8. Proposed project structure evolution

- **M0:** Keep the repository minimal.
- **M1:** Introduce `src/railbound/` because the first executable domain code now exists. Introduce `tests/` at the same time; production code should not appear without tests.
- **M1:** Extend the existing `pyproject.toml` with project/package metadata and a CLI entry point. Do not replace it.
- **M1:** Add a concise README documenting setup, command usage, coordinate conventions, and generated artifacts.
- **M1–M2:** Begin with focused modules rather than empty packages, for example geometry, grid serialization, CLI, and grid rendering.
- **M2–M3:** Add vision modules for extraction and grid detection. Avoid a generic `parser` package that mixes unrelated stages.
- **M4:** Add the semantic model and schema documentation.
- **M5–M7:** Add topology and simulation modules. Split simulation into a package only when rules and events make a single module unwieldy.
- **M8:** Add recognition as a separate vision component.
- **M9:** Add a solver package only when there is a working simulator to consume.
- **Fixtures:** Create `tests/fixtures/` when M1 has its first stable input/output case. A separate top-level dataset directory is justified only if research data becomes too large or unsuitable for the normal test suite.
- **README:** Expand milestone by milestone; do not attempt comprehensive documentation before behavior exists.
- **Makefile:** Not required. The commands are already short, and another command layer currently solves no problem.

The eventual conceptual layout should follow responsibilities such as `geometry`, `vision`, `model`, `topology`, `simulation`, `solver`, and `visualization`, but these should appear only as their milestones are implemented.

## 9. Key technical risks

1. Railbound's exact update rules are not yet formalized.
2. Three-point affine calibration may not cover perspective-distorted or rescaled screenshots.
3. Cell artwork can obscure grid edges and semantic features.
4. Recognition confidence can be misleading without explicit unknown/reject behavior.
5. Crossings, switches, and tunnels are awkward in a cell-only graph.
6. Simultaneous events can make a superficially plausible simulator incorrect.
7. Solver state space may grow rapidly with buildable cells and piece choices.
8. An event sequence may be chronological without being causally explanatory.
9. The level schema may churn if designed before representative mechanics are collected.
10. Real screenshots may have copyright or redistribution constraints; fixture policy must be decided before committing them.
11. Unpinned dependencies may eventually reduce reproducibility, though they do not block the first milestone.
12. Row/column and `(x, y)` confusion can propagate through every later subsystem unless fixed in M1.

## 10. Open questions / assumptions

- Grid dimensions will initially be supplied manually.
- The three calibration points are centers of adjacent cells, not cell corners.
- Screenshots initially have one affine grid and no severe local distortion.
- Track placement occurs before simulation and is not changed while trains move.
- The rail budget may be either a total count or an inventory by shape; representative levels must determine this.
- It remains to confirm whether track orientation is purely port-based or whether visually identical pieces have directional behavior.
- The exact timing of buttons, switches, gates, tunnels, collision, and coupling must be observed in-game.
- It remains to determine whether actors move simultaneously and how conflicts are resolved.
- Tunnel traversal duration and exit direction need confirmation.
- Win conditions, wagon numbering, ordering, and coupling rules need representative examples.
- The required solver objective must be clarified: any valid placement, minimum rails, unique solution, or game-equivalent constraints.
- Permission to store real screenshots in the repository must be established.
- The first reference-level set should deliberately include visual variation and small isolated examples of each mechanic.

## 11. Recommended immediate next task

Implement M1 as one end-to-end assisted grid calibration flow for a single
screenshot: a native four-click interface that derives dimensions, plus a
scripted three-point command for automation. Both produce the affine
`GridGeometry`, validated `grid.json`, `annotated_grid.png`, and deterministic
geometry/serialization tests.
