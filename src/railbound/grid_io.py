"""Persistence for the portable, validated grid geometry document."""

from __future__ import annotations

from pathlib import Path

from railbound.geometry import GridGeometry


def save_grid(grid: GridGeometry, path: Path) -> None:
    """Serialize a grid as deterministic, human-reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(grid.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_grid(path: Path) -> GridGeometry:
    """Load and validate a grid geometry JSON document."""
    return GridGeometry.model_validate_json(path.read_text(encoding="utf-8"))
