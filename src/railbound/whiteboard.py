"""Mutable semantic draft used by the whiteboard level editor."""

from __future__ import annotations

from dataclasses import dataclass, field

from railbound.geometry import CellCoord
from railbound.level import Actor, Button, Gate, Switch, TerrainCell, TrackPiece, TunnelEndpoint

type BoardItem = Actor | Button | Gate | Switch | TerrainCell | TrackPiece | TunnelEndpoint


@dataclass(slots=True)
class BoardDraft:
    """An editable board whose contents already use solver-facing level types."""

    rows: int = 8
    cols: int = 10
    items: dict[CellCoord, list[BoardItem]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_dimensions(self.rows, self.cols)
        if any(not self.contains(cell) for cell in self.items):
            raise ValueError("board items must be inside the grid")

    @property
    def occupied_cells(self) -> frozenset[CellCoord]:
        """Return cells containing at least one semantic item."""
        return frozenset(cell for cell, items in self.items.items() if items)

    @property
    def minimum_rows(self) -> int:
        """Return the smallest row count that retains all placed items."""
        return max((cell.row + 1 for cell in self.occupied_cells), default=1)

    @property
    def minimum_cols(self) -> int:
        """Return the smallest column count that retains all placed items."""
        return max((cell.col + 1 for cell in self.occupied_cells), default=1)

    def contains(self, cell: CellCoord) -> bool:
        """Return whether a coordinate lies inside the current grid."""
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def resize(self, rows: int, cols: int) -> None:
        """Resize without discarding items beyond the new lower or right edge."""
        self._validate_dimensions(rows, cols)
        if rows < self.minimum_rows or cols < self.minimum_cols:
            raise ValueError("the grid cannot be smaller than its last occupied row or column")
        self.rows = rows
        self.cols = cols

    @staticmethod
    def _validate_dimensions(rows: int, cols: int) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("board dimensions must be positive")
