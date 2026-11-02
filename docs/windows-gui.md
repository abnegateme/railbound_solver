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

The **Workflow** tab is the complete interactive flow. Enter the level
number first; a conventional number such as `1-13` is inferred from the
screenshot filename when possible. Cell IDs then combine that number with
zero-based row and column coordinates.

**Extract cells** uses one fixed normalization profile and opens the visual
**Cells** workspace. Recognition runs immediately against the locally confirmed
sample library. Green cards are accepted automatic matches, purple cards were
reviewed manually, and red cards still need a decision. Choose a known type or
create a reusable type directly on a card. Type-specific fields appear only
when needed. A train has a direction and an optional positive **Arrival order**;
filled order values must be unique and become the relative order in which those
trains must reach their goals. An arrival cell requires its fixed **Arrival
direction**: this is the wagon's direction of travel, so its actual entry port
is on the opposite side. Train and arrival cells are fixed topology and cannot
receive a placed rail or switch. A completed manual decision is stored as
recognition evidence; crops and review files do not need to be exported.

**Formalize level** becomes available only when every cell has a complete type
whose graph-state compiler is implemented. It asks for the rail budget and,
only when necessary, the target goal for each train. The resulting validated
`Level` remains in memory and immediately opens as a schematic in the third
workspace, **Level / solution**; it has no separate text editor.
The current observed interactive inventory permits straight, curved, and
three-port switch rails. Each switch candidate chooses one of four branch
directions and one of its two mutually opposite outputs. That choice fixes a
bidirectional branch-to-output route plus a one-way merge from the other
output. The abstract
four-way `crossing` type is not offered until that mechanic is observed
explicitly.
Recognizable new mechanics may be recorded without data loss, but they remain
visibly blocked from formalization until their topology/state compiler exists;
they are never silently ignored.

**Solve level** is on the same **Workflow** tab. During search, the third
workspace redraws sampled states from the actual search tree: assigned rails,
outlined cells still awaiting a piece assignment, complete failures, and the
verified solution. This includes partial branches rejected before they become
complete placements. The GUI plays a bounded sample queue at a visible rate,
so fast searches still animate without making large searches take minutes.
Search runs in a worker thread; images are delivered back to Qt through queued
signals, so the window remains responsive. Search statistics and the replay
trace go to the activity log; the placement can be saved beside the button.
The optional **Placement** tab can inspect or replay a `placement.json`, and a
successful replay uses the same visual workspace. Pruning, bounds, and runtime
caching use the interactive defaults and add no extra controls.

The host environment is only a local GUI runtime, not an installer. Its
desktop access must remain outside the devcontainer.

