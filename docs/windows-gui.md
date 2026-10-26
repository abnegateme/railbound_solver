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

Use the same window's **Review and solve** tab to choose a template catalog,
inspect recognition evidence, and apply any explicit review overrides. Then
open or paste the authoritative `level.json`, validate it, choose solver
limits, inspect the verified placement and replay trace, and save the
placement. Recognition labels remain evidence rather than simulator input:
the semantic level still records the reviewed terrain, actors, inventory, and
win conditions. No intermediate crop files are required for this path.

The host environment is only a local GUI runtime, not an installer. Its
desktop access must remain outside the devcontainer.

