# Deterministic recognition (M8)

M8 turns normalized M2 crops into evidence for review. It does not make pixels
authoritative level data: `recognition_report.json` records candidate labels,
scores, template provenance, and every cell needing a decision; `level_draft.json`
is an explicitly incomplete review document, not a simulator input.

## Label templates

Create a portable catalog beside labelled reference PNGs. The `image` value is
a POSIX path relative to the catalog and cannot escape its directory.

```json
{
  "schema_version": 1,
  "recognizer_version": "template-ncc-v1",
  "thresholds": {
    "min_similarity": 0.84,
    "min_confidence": 0.70,
    "ambiguity_margin": 0.08,
    "max_candidates": 3
  },
  "templates": [
    {
      "id": "grass-empty-01",
      "label": "terrain.empty",
      "image": "references/grass-empty-01.png",
      "rotations": [0, 90, 180, 270]
    }
  ]
}
```

Each label is a short M4-vocabulary candidate, such as `terrain.empty` or
`track.curve`. A catalog may have several images for one label. The matcher
uses the strongest reference for that label, so visual variants do not create
false ambiguity. The recorded rotation is clockwise from the reference image
to the observed crop.

In the primary Windows GUI, **Review and solve** can create this catalog
without leaving the screenshot session: select an extracted crop, give it a
label, then save the catalog. The UI generates the template ID unless a custom
one is supplied. Saving writes the portable JSON and `references/*.png` files
together. The initial labels are necessarily manual; the recognizer cannot
infer the meaning of its first visual reference. Start with labels that
describe a visible class rather than a full game rule: `terrain.empty`,
`track.straight`, `track.curve`, `track.crossing`, `actor.train`,
`goal.arrival`, and `device.*` are useful conventions.

Matching combines normalized colour-structure correlation with a small
mean-colour feature. A top match must exceed `min_similarity`; its gap from
the strongest *different* label scales confidence. A low score becomes
`unknown`; a close competition becomes `uncertain`. Both are retained in the
report and never converted automatically.

## Run recognition

First export the in-memory crops and manifest from the GUI, then provide the
catalog:

```bash
python3 -m railbound recognize \
  --crops-dir output/crops \
  --template-catalog references/catalog.json \
  --report-output output/recognition_report.json \
  --draft-output output/level_draft.json
```

The command exits with `0` only when every cell is accepted. It still writes
the report and optional draft before returning `2` for unresolved cells, and
prints their coordinates. Threshold flags override only that recognition run;
the effective values are included in the report.

## Review and manual overrides

Pass a JSON array of manual decisions with `--overrides` and `--draft-output`:

```json
[
  {
    "cell": {"row": 2, "col": 5},
    "label": "track.curve",
    "note": "confirmed from the screenshot"
  }
]
```

An override wins over both an uncertain and an accepted result. `resolve()`
creates a `LevelDraft` containing every resolved label plus an ordered list of
unresolved cells. It intentionally cannot be passed to `simulate()` or
`compile_board()`: a reviewer must map candidate labels into M4 layers and add
the required inventory and win condition to create a validated `Level`.

The report includes the crop-manifest provenance, full catalog configuration,
and a digest of the template definitions and decoded pixel values. This makes
recognition evidence reviewable without embedding screenshot pixels in a
level document.
