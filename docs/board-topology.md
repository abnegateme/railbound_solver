# Static board topology (M5)

M5 compiles a validated `Level` and a separate `RailPlacement` into a
deterministic port graph. It does not move actors or mutate device state. The
compiled board retains immutable switch, button, gate, and tunnel definitions
for M7 simulation; their runtime behavior remains outside the static graph.

## Compile a placement

```python
from pathlib import Path

from railbound import compile_board, load_level, load_placement

level = load_level(Path("data/example-level.json"))
placement = load_placement(Path("data/example-placement.json"))
board = compile_board(level, placement)
```

## Placement document

`RailPlacement` is a versioned JSON document that names the level it belongs
to and lists zero or more rails:

```json
{
  "schema_version": 1,
  "level_id": "example-straight-arrival",
  "rails": [
    {
      "cell": {"row": 0, "col": 1},
      "piece_kind": "straight",
      "orientation": "E"
    }
  ]
}
```

Use `load_placement(path)` and `save_placement(placement, path)` to read and
write the document. A placement may be empty, but it cannot list the same cell
twice. Its `level_id` must match the `Level.id` passed to
`validate_placement(level, placement)` or `compile_board(level, placement)`.

Each rail must be on one of the level's `buildable_cells`. The level's
`total_budget`, when present, limits all rails. A non-empty `piece_counts`
list both limits the named kinds and excludes unnamed kinds from the placement.

## Static rail definitions

The supported `piece_kind` values are `straight`, `curve`, and `crossing`.
`orientation` is one of `N`, `E`, `S`, or `W`; it rotates the canonical
north-facing definition clockwise.

| Kind | Canonical internal transitions |
| --- | --- |
| `straight` | `N` ↔ `S` |
| `curve` | `N` ↔ `E` |
| `crossing` | `N` ↔ `S` and independently `E` ↔ `W` |

For example, an east-oriented straight has the `E` ↔ `W` transition. Rotation
is retained exactly as authored even when it produces an equivalent topology,
such as `N` and `S` straights; solver canonicalisation is a later concern.
`rail_piece_transitions(kind, orientation)` exposes the definitions directly.

## Compiled graph

Every used cell edge is a `PortNode(cell, port)`. The compiler adds two
separate kinds of undirected `PortConnection`:

- `internal` joins the two ports paired by one rail inside its cell.
- `adjacent` joins facing ports of rails in neighbouring cells only when both
  ports exist.

Consequently geometric neighbours are not automatically connected. A rail
facing a neighbour with no matching port remains in `unconnected_ports`; a
port that faces beyond the level dimensions is also listed in `boundary_exits`.
`connected_components` is a deterministic tuple of port-node components for
inspection. A crossing contains two independent internal components unless
other rails join them outside that cell.

`CompiledBoard.transition(port)` traverses a rail within one cell and
`CompiledBoard.adjacent(port)` traverses one validated cell boundary. Both
return `None` when the requested edge does not exist. This separation gives a
later simulator unambiguous movement stages.
