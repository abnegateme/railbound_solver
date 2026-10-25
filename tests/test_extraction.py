from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railbound.crop_io import load_crop_manifest, write_crop_set
from railbound.extraction import CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import GridGeometry, PixelPoint


def _grid(*, rows: int = 2, cols: int = 3) -> GridGeometry:
    return GridGeometry.from_manual_calibration(
        rows=rows,
        cols=cols,
        image_width=cols * 10,
        image_height=rows * 10,
        origin=PixelPoint(x=5, y=5),
        next_col=PixelPoint(x=15, y=5),
        next_row=PixelPoint(x=5, y=15),
    )


def test_extracts_one_normalized_crop_per_colored_cell() -> None:
    grid = _grid()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    for row in range(grid.rows):
        for col in range(grid.cols):
            image[row * 10 : (row + 1) * 10, col * 10 : (col + 1) * 10] = (
                20 + row,
                40 + col,
                60 + row + col,
            )

    crops = CellExtractor().extract(
        image,
        grid,
        CropSpec(output_size=10, core_fraction=1.0, interpolation="nearest"),
    )

    assert [(crop.cell.row, crop.cell.col) for crop in crops] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    for crop in crops:
        assert crop.image.shape == (10, 10, 3)
        expected = image[crop.cell.row * 10 + 5, crop.cell.col * 10 + 5]
        assert np.array_equal(crop.image[5, 5], expected)


def test_extraction_rectifies_grid_orientation() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    red = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)
    blue = np.array([255, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    image[0:10, 0:10] = red
    image[0:10, 10:20] = green
    image[10:20, 0:10] = blue
    image[10:20, 10:20] = white
    grid = GridGeometry.from_manual_calibration(
        rows=2,
        cols=2,
        image_width=40,
        image_height=40,
        origin=PixelPoint(x=10, y=10),
        next_col=PixelPoint(x=10, y=30),
        next_row=PixelPoint(x=30, y=10),
    )

    crop = CellExtractor().extract(
        image,
        grid,
        CropSpec(output_size=10, core_fraction=1.0, interpolation="nearest"),
    )[0]

    assert np.array_equal(crop.image[2, 2], red)
    assert np.array_equal(crop.image[2, 8], blue)
    assert np.array_equal(crop.image[8, 2], green)
    assert np.array_equal(crop.image[8, 8], white)


def test_context_mask_excludes_pixels_outside_the_target_cell() -> None:
    grid = GridGeometry.from_manual_calibration(
        rows=1,
        cols=1,
        image_width=30,
        image_height=30,
        origin=PixelPoint(x=15, y=15),
        next_col=PixelPoint(x=25, y=15),
        next_row=PixelPoint(x=15, y=25),
    )
    image = np.full((30, 30, 3), 80, dtype=np.uint8)
    base_settings = {"output_size": 12, "mode": "context", "context_fraction": 1.5}

    unmasked = CellExtractor().extract(image, grid, CropSpec(**base_settings))[0]
    masked = CellExtractor().extract(image, grid, CropSpec(**base_settings, mask_mode="cell"))[0]

    assert np.all(unmasked.image[0, 0] == 80)
    assert np.all(masked.image[0, 0] == 0)
    assert np.all(masked.image[6, 6] == 80)
    assert masked.mask is not None
    assert masked.mask.shape == (12, 12)


def test_border_context_crop_has_a_stable_shape() -> None:
    grid = _grid(rows=2, cols=2)
    image = np.full((20, 20, 3), 100, dtype=np.uint8)

    crops = CellExtractor().extract(
        image,
        grid,
        CropSpec(output_size=16, mode="context", context_fraction=1.5),
    )

    assert all(crop.image.shape == (16, 16, 3) for crop in crops)
    assert np.all(crops[0].image[0, 0] == 0)
    assert np.all(crops[-1].image[-1, -1] == 0)


def test_crop_set_writes_a_deterministic_manifest_and_contact_sheet(tmp_path: Path) -> None:
    grid = _grid(rows=2, cols=2)
    image = np.full((20, 20), 120, dtype=np.uint8)
    spec = CropSpec(output_size=8)
    crops = CellExtractor().extract(image, grid, spec)

    manifest = write_crop_set(crops, grid=grid, spec=spec, output_directory=tmp_path / "crops")
    reloaded = load_crop_manifest(tmp_path / "crops" / "manifest.json")
    contact_sheet = render_crop_contact_sheet(crops, columns=grid.cols)

    assert reloaded == manifest
    assert [record.filename for record in manifest.crops] == [
        "r000_c000.png",
        "r000_c001.png",
        "r001_c000.png",
        "r001_c001.png",
    ]
    assert (tmp_path / "crops" / "r001_c001.png").is_file()
    assert contact_sheet.shape == (64, 28, 3)


def test_extraction_rejects_an_image_with_different_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        CellExtractor().extract(
            np.zeros((21, 20, 3), dtype=np.uint8),
            _grid(rows=2, cols=2),
            CropSpec(),
        )
