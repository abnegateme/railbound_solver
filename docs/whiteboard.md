# Whiteboard

The whiteboard is a standalone native editor for composing a level directly
from semantic primitives. Install the GUI extra, then launch it without
arguments:

```powershell
.\.venv-gui\Scripts\railbound-whiteboard.exe
```

The first interface increment provides a resizable sketch-style grid. Rows and
columns may be changed immediately while the board is empty. Once primitives
are introduced, the grid may only shrink as far as its last occupied row and
column, so resizing never discards level objects.

The editor draft stores solver-facing level objects rather than pixels or
recognition results. The primitive palette, level formalization, calculation,
and interactive solution playback will be added on top of that boundary.
