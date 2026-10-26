# Windows GUI

The native GUI runs on the Windows host; editing and verification remain in the
Linux devcontainer. Create or refresh the host runtime from the repository
root:

```powershell
.\scripts\setup-windows-gui.ps1
```

Start the primary interactive session:

```powershell
.\.venv-gui\Scripts\python.exe -m railbound interactive --image data\Railbound-1-13.PNG
```

The window accepts a proposal at confidence `0.70` or higher. Otherwise click
the centres of `(0, 0)`, `(0, 1)`, `(1, 0)`, and the bottom-right cell, in that
order. Extra labelled-cell clicks refine the affine fit; **Fit image** resets
zoom. Saving writes `grid.json` and `annotated_grid.png`; **Extract cells in
memory** opens the crop preview for the next stage.

Use the same window's **Review and solve** tab for every completed stage:

- Create a template catalog from selected in-memory crops, giving each
  reference a semantic candidate label. The template ID is optional and is
  generated from the label and source cell; enter one only when a custom stable
  name is useful. **Save template catalog** writes `catalog.json` and its
  `references/*.png` images; alternatively, **Open template catalog** imports
  an existing portable catalog.
- Run recognition with the in-memory catalog, correct labels through explicit
  overrides, and optionally save the report, overrides, or review-state draft.
- Open or paste the authoritative `level.json`, validate and save it.
  Recognition labels remain evidence rather than simulator input: the semantic
  level still records reviewed terrain, actors, inventory, and win conditions.
- Open or paste a `placement.json` to inspect its compiled port topology and
  replay its M6–M7 event trace. The baseline solver can then search the same
  level, show its verified trace, and save its resulting placement.

The **Grid and crops** tab can also export optional PNG crops, a manifest, and
a contact sheet for batch workflows. No intermediate artifacts are required
for the primary interactive path. A template catalog has a top-level
`templates` list; do not select calibrated `grid.json` or a crop manifest as
its catalog.

The host environment is only a local GUI runtime, not an installer. Its
desktop access must remain outside the devcontainer.

