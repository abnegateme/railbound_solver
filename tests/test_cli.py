from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from railbound.cli import main
from railbound.grid_io import load_grid


def test_calibrate_command_writes_portable_grid_and_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    grid_path = tmp_path / "artifacts" / "grid.json"
    overlay_path = tmp_path / "artifacts" / "annotated_grid.png"
    assert cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))

    result = main(
        [
            "calibrate",
            "--image",
            str(image_path),
            "--rows",
            "2",
            "--cols",
            "3",
            "--origin",
            "40",
            "40",
            "--next-col",
            "70",
            "40",
            "--next-row",
            "40",
            "70",
            "--grid-output",
            str(grid_path),
            "--overlay-output",
            str(overlay_path),
        ]
    )

    assert result == 0
    assert load_grid(grid_path).image_sha256 is not None
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert overlay.shape == (120, 160, 3)
