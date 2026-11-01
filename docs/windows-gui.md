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

Every screenshot is inspected when it is loaded, including an image supplied
at startup. The window applies a grid proposal at confidence `0.70` or higher
and fills its detected row and column counts. If no reliable grid is found, the
compact controls keep their `8 × 10` defaults and the next-step guidance asks
for manual dimensions and four visible grid intersections: the upper-left
corner, its adjacent intersections to the right and below, and the outer
bottom-right corner. Extra intersection clicks refine the affine fit. The
screenshot is fitted automatically whenever the window changes size.

The lower part of the side panel, visible below both tabs, keeps an activity log
and a current **Next step** recommendation. It occupies roughly one fifth of
the panel and includes manual calibration guidance when automatic detection
cannot establish a reliable grid. GUI calibration belongs to the current
screenshot session and is consumed directly by **Extract cells**; it
is not saved separately.

The **Grid and cells** tab is the complete parsing workflow. Enter the level
number first; a conventional number such as `1-13` is inferred from the
screenshot filename when possible. Cell IDs then combine that number with
zero-based row and column coordinates.

**Extract cells** uses one fixed normalization profile and opens the visual
**Cells** workspace. Recognition runs immediately against the locally confirmed
sample library. Green cards are accepted automatic matches, purple cards were
reviewed manually, and red cards still need a decision. Choose a known type or
create a reusable type directly on a card. Type-specific fields appear only
when needed, for example the direction of a train. A completed manual decision
is stored as recognition evidence; crops and review files do not need to be
exported.

**Formalize level** becomes available only when every cell has a complete type
whose graph-state compiler is implemented. It asks for the rail budget and,
only when necessary, the target goal for each train. The resulting validated
`Level` remains in memory and opens in the schematic **Level** workspace.
Recognizable new mechanics may be recorded without data loss, but they remain
visibly blocked from formalization until their topology/state compiler exists;
they are never silently ignored.

The **Solve** tab consumes that in-memory formal level. It can inspect or replay
a `placement.json`, run the solver, show the verified trace, and save the
resulting placement. Pruning and runtime caching are automatic and add no extra
controls.

The host environment is only a local GUI runtime, not an installer. Its
desktop access must remain outside the devcontainer.

