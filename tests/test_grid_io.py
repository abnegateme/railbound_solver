from __future__ import annotations

from pathlib import Path

from railbound.geometry import GridGeometry, PixelPoint
from railbound.grid_io import load_grid, save_grid


def test_grid_json_round_trip(tmp_path: Path) -> None:
    grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=3,
        image_width=123,
        image_height=456,
        image_sha256="a" * 64,
        origin=PixelPoint(x=20, y=30),
        next_col=PixelPoint(x=50, y=32),
        next_row=PixelPoint(x=17, y=70),
    )
    path = tmp_path / "nested" / "grid.json"

    save_grid(grid, path)

    assert load_grid(path) == grid
    payload = path.read_text(encoding="utf-8")
    assert '"coordinate_convention"' in payload
    assert '"calibration"' in payload
