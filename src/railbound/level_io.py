"""Persistence for portable, validated semantic Railbound level documents."""

from __future__ import annotations

from pathlib import Path

from railbound.level import Level


def save_level(level: Level, path: Path) -> None:
    """Serialize a level as deterministic, human-reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(level.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_level(path: Path) -> Level:
    """Load and validate a semantic level JSON document."""
    return Level.model_validate_json(path.read_text(encoding="utf-8"))
