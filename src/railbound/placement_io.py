"""Persistence for portable rail placements independent of a level document."""

from __future__ import annotations

from pathlib import Path

from railbound.topology import RailPlacement


def save_placement(placement: RailPlacement, path: Path) -> None:
    """Serialize a placement as deterministic, human-reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(placement.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_placement(path: Path) -> RailPlacement:
    """Load and validate a portable rail placement JSON document."""
    return RailPlacement.model_validate_json(path.read_text(encoding="utf-8"))
