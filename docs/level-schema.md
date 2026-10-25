# Semantic level schema (M4)

`level.json` is the authoritative, image-independent input to future board
compilation, simulation, and solving. Recognition output, crop provenance,
and M8 `LevelDraft` labels must stay in a separate report; a level contains no
filenames, pixels, or confidence scores.

The current schema is version `1` and accepts only the ruleset
`railbound-observed-v1`. Unknown fields and unknown schema versions are
rejected, so a reader cannot silently reinterpret a newer document.

See [`data/example-level.json`](../data/example-level.json) for a complete,
small manually authored document.

## Use

Create and validate a semantic level without a screenshot, then persist it
with the public functions:

```python
from pathlib import Path

from railbound import load_level, save_level

level = load_level(Path("data/example-level.json"))
save_level(level, Path("output/level.json"))
```

## Top-level fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Format version. Must be `1`. |
| `ruleset_version` | Observed game-rule vocabulary. Must be `railbound-observed-v1`. |
| `id` | Stable document identifier, using letters, digits, `.`, `_`, `:`, or `-`. |
| `rows`, `cols` | Positive board dimensions. All coordinates are zero-based `(row, col)`. |
| `metadata` | Optional human-readable `title`, `world`, and `source`. |
| `terrain` | Optional descriptive terrain labels; these do not imply simulator behavior yet. |
| `buildable_cells` | Cells in which a future `RailPlacement` may propose a rail. |
| `fixed_tracks` | Immutable track pieces, specified by ports and internal transitions. |
| `actors` | Required initial actors, with position, travel direction, type, and optional number/color. |
| `switches`, `buttons`, `gates`, `tunnel_endpoints` | Optional device layers with observed M7 runtime behavior. |
| `rail_inventory` | A non-negative `total_budget`, per-`piece_kind` counts, or both. |
| `win_condition` | Required arrivals, their optional ordering, and/or required actor couplings. |

`CellCoord` is shared with grid calibration but has a different role here: it
is a semantic cell, not an image measurement. Coordinates must be in bounds
wherever they appear.

## Track topology

Ports are the cell-edge strings `N`, `E`, `S`, and `W`. A `fixed_tracks` item
lists each used port exactly once and contains transitions that pair those
ports. A straight or curve has one pair; a crossing has two independent pairs.
The validator rejects duplicated ports, ports missing from transitions, and
transitions that mention undeclared ports. `fixed_tracks` must have `fixed` set
to `true`. Placeable rails live in the separate M5 `RailPlacement` document,
which can be paired with this level without changing its authoritative
fixed-object data. See [`board-topology.md`](board-topology.md).

## Cross-object validation

The loader rejects duplicate entity IDs across tracks, actors, and devices;
duplicate cells within a single layer; fixed tracks on buildable cells;
out-of-bounds coordinates; unknown actor IDs in a win condition; and tunnel
pair IDs that do not have exactly two endpoints in distinct cells. A switch
has one entry and two distinct outputs, and its initial output must be one of
them. A switch may declare a button `channel`; it is otherwise fixed at its
initial output. Buttons use the observed `toggle` activation behavior, and
every button channel must control at least one same-channel gate or switch.

At runtime, entering a button toggles every same-channel gate and switch after
the tick's movements resolve. A closed gate blocks entry to its cell. Entering
a tunnel endpoint through its `entry_port` transfers the actor through its
pair and out of the paired endpoint's `exit_port` in that same tick. Full
precedence and event-trace semantics are documented in
[`static-simulation.md`](static-simulation.md).

Use `load_level(path)` to parse and validate an existing document and
`save_level(level, path)` to write readable JSON. Both are available from
`railbound.level_io` and the package root.
