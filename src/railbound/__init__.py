"""Railbound board calibration and solver tools."""

from railbound.crop_io import (
    CropManifest,
    CropRecord,
    build_crop_manifest,
    load_crop_manifest,
    save_crop_manifest,
    write_crop_set,
)
from railbound.detection import (
    CalibrationCandidate,
    GridDetectionHints,
    GridDetector,
    render_detection_overlay,
)
from railbound.extraction import CellCrop, CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import (
    CellCoord,
    GridCalibration,
    GridCalibrationPoint,
    GridGeometry,
    GridPoint,
    NearestCell,
    PixelPoint,
)
from railbound.grid_io import load_grid, save_grid

__all__ = [
    "CellCoord",
    "CellCrop",
    "CellExtractor",
    "CalibrationCandidate",
    "CropManifest",
    "CropRecord",
    "CropSpec",
    "GridCalibration",
    "GridCalibrationPoint",
    "GridDetectionHints",
    "GridDetector",
    "GridGeometry",
    "GridPoint",
    "NearestCell",
    "PixelPoint",
    "build_crop_manifest",
    "load_crop_manifest",
    "load_grid",
    "render_crop_contact_sheet",
    "render_detection_overlay",
    "save_crop_manifest",
    "save_grid",
    "write_crop_set",
]
