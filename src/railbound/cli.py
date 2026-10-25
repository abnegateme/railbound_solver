"""Commands for manual and interactive Railbound screenshot calibration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
from pydantic import ValidationError

from railbound.calibration import grid_from_three_points, image_sha256
from railbound.crop_io import write_crop_set
from railbound.detection import GridDetectionHints, GridDetector, render_detection_overlay
from railbound.extraction import CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import PixelPoint
from railbound.grid_io import load_grid, save_grid
from railbound.rendering import ColorImage, render_grid_overlay


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading input or changing state."""
    parser = argparse.ArgumentParser(
        description="Calibrate an affine Railbound grid from three cell centers."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="calibrate a grid from a screenshot")
    calibrate.add_argument(
        "--interactive", action="store_true", help="open the native calibration window"
    )
    calibrate.add_argument("--image", type=Path, help="source screenshot")
    calibrate.add_argument("--rows", type=int, help="number of grid rows")
    calibrate.add_argument("--cols", type=int, help="number of grid columns")
    calibrate.add_argument(
        "--origin",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="center of cell (0, 0)",
    )
    calibrate.add_argument(
        "--next-col",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="center of cell (0, 1)",
    )
    calibrate.add_argument(
        "--next-row",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="center of cell (1, 0)",
    )
    calibrate.add_argument("--grid-output", type=Path, help="destination grid.json")
    calibrate.add_argument("--overlay-output", type=Path, help="destination annotated_grid.png")

    extract = commands.add_parser("extract", help="extract normalized crops from a calibrated grid")
    extract.add_argument("--image", type=Path, help="source screenshot")
    extract.add_argument("--grid", type=Path, help="calibrated grid.json")
    extract.add_argument("--output-dir", type=Path, help="directory for normalized crop PNGs")
    extract.add_argument("--manifest-output", type=Path, help="destination crop manifest JSON")
    extract.add_argument("--contact-sheet-output", type=Path, help="destination contact-sheet PNG")
    extract.add_argument("--mode", choices=("core", "context"), default="core")
    extract.add_argument("--output-size", type=int, default=96, help="square crop size in pixels")
    extract.add_argument(
        "--core-fraction", type=float, default=0.8, help="core crop side length in cell units"
    )
    extract.add_argument(
        "--context-fraction",
        type=float,
        default=1.25,
        help="contextual crop side length in cell units",
    )
    extract.add_argument(
        "--mask-mode",
        choices=("none", "cell"),
        default="none",
        help="mask contextual pixels outside the target cell",
    )

    detect = commands.add_parser(
        "detect", help="propose a grid from repeated visible board boundaries"
    )
    detect.add_argument("--image", type=Path, help="source screenshot")
    detect.add_argument("--rows", type=int, help="known number of grid rows, if available")
    detect.add_argument("--cols", type=int, help="known number of grid columns, if available")
    detect.add_argument(
        "--min-cell-size",
        type=float,
        default=16.0,
        help="smallest plausible cell side length in pixels",
    )
    detect.add_argument(
        "--max-cell-size",
        type=float,
        help="largest plausible cell side length in pixels",
    )
    detect.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="minimum confidence required before writing a grid",
    )
    detect.add_argument(
        "--grid-output",
        type=Path,
        help="destination grid.json for the strongest accepted candidate",
    )
    detect.add_argument(
        "--overlay-output",
        type=Path,
        help="destination debug overlay PNG, including low-confidence candidates",
    )

    interactive = commands.add_parser(
        "interactive", help="open the assisted detection and calibration window"
    )
    interactive.add_argument("--image", type=Path, help="source screenshot to open")
    interactive.add_argument("--grid-output", type=Path, help="destination grid.json")
    interactive.add_argument(
        "--overlay-output", type=Path, help="destination annotated_grid.png"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a requested non-interactive command and return a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "calibrate":
        if arguments.interactive:
            return _interactive_calibrate(arguments, parser)
        return _calibrate(arguments, parser)
    if arguments.command == "extract":
        return _extract(arguments, parser)
    if arguments.command == "detect":
        return _detect(arguments, parser)
    if arguments.command == "interactive":
        return _interactive(arguments, parser)
    parser.error(f"unsupported command: {arguments.command}")


def _calibrate(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_scripted_arguments(arguments, parser)
    image = cast(ColorImage | None, cv2.imread(str(arguments.image), cv2.IMREAD_COLOR))
    if image is None:
        parser.error(f"could not read image: {arguments.image}")

    try:
        grid = grid_from_three_points(
            rows=arguments.rows,
            cols=arguments.cols,
            image_width=image.shape[1],
            image_height=image.shape[0],
            origin=_pixel_point(arguments.origin),
            next_col=_pixel_point(arguments.next_col),
            next_row=_pixel_point(arguments.next_row),
            image_sha256=image_sha256(arguments.image),
        )
    except ValidationError as error:
        parser.error(str(error))

    overlay = render_grid_overlay(image, grid)
    arguments.overlay_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(arguments.overlay_output), overlay):
        parser.error(f"could not write overlay image: {arguments.overlay_output}")
    save_grid(grid, arguments.grid_output)
    print(f"Wrote {arguments.grid_output}")
    print(f"Wrote {arguments.overlay_output}")
    return 0


def _interactive_calibrate(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    return _interactive(arguments, parser, automatic_detection=False)


def _interactive(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    automatic_detection: bool = True,
) -> int:
    if (arguments.grid_output is None) != (arguments.overlay_output is None):
        parser.error(
            "--grid-output and --overlay-output must be supplied together in interactive mode"
        )
    try:
        from railbound.gui import run_calibration_gui
    except ModuleNotFoundError as error:
        if error.name == "PySide6":
            parser.error(
                "the interactive workflow requires the GUI extra; install the project with .[gui]"
            )
        raise

    return run_calibration_gui(
        image_path=arguments.image,
        grid_output=arguments.grid_output,
        overlay_output=arguments.overlay_output,
        automatic_detection=automatic_detection,
    )


def _extract(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_extraction_arguments(arguments, parser)
    image = cast(ColorImage | None, cv2.imread(str(arguments.image), cv2.IMREAD_UNCHANGED))
    if image is None:
        parser.error(f"could not read image: {arguments.image}")

    try:
        grid = load_grid(arguments.grid)
        _validate_source_image_hash(arguments.image, grid.image_sha256)
        spec = CropSpec(
            output_size=arguments.output_size,
            core_fraction=arguments.core_fraction,
            context_fraction=arguments.context_fraction,
            mode=arguments.mode,
            mask_mode=arguments.mask_mode,
        )
        crops = CellExtractor().extract(image, grid, spec)
        manifest_path = arguments.manifest_output or arguments.output_dir / "manifest.json"
        manifest = write_crop_set(
            crops,
            grid=grid,
            spec=spec,
            output_directory=arguments.output_dir,
            manifest_path=manifest_path,
        )
        contact_sheet = render_crop_contact_sheet(crops, columns=grid.cols)
        contact_sheet_path = (
            arguments.contact_sheet_output or arguments.output_dir / "contact_sheet.png"
        )
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
            raise OSError(f"could not write contact sheet: {contact_sheet_path}")
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote {len(manifest.crops)} crops to {arguments.output_dir}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {contact_sheet_path}")
    return 0


def _detect(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_detection_arguments(arguments, parser)
    image = cast(ColorImage | None, cv2.imread(str(arguments.image), cv2.IMREAD_UNCHANGED))
    if image is None:
        parser.error(f"could not read image: {arguments.image}")

    try:
        hints = GridDetectionHints(
            rows=arguments.rows,
            cols=arguments.cols,
            min_cell_size_px=arguments.min_cell_size,
            max_cell_size_px=arguments.max_cell_size,
            image_sha256=image_sha256(arguments.image),
        )
        candidates = GridDetector().propose(image, hints)
        if arguments.overlay_output is not None:
            overlay = render_detection_overlay(image, candidates)
            arguments.overlay_output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(arguments.overlay_output), overlay):
                raise OSError(f"could not write detection overlay: {arguments.overlay_output}")
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))

    for index, candidate in enumerate(candidates, start=1):
        print(
            f"Candidate {index}: {candidate.grid.rows} rows x {candidate.grid.cols} columns, "
            f"confidence {candidate.confidence:.2f}, "
            f"RMS residual {candidate.residual_rms_px:.2f} px"
        )
    if arguments.overlay_output is not None:
        print(f"Wrote {arguments.overlay_output}")

    accepted = next(
        (candidate for candidate in candidates if candidate.is_confident(arguments.min_confidence)),
        None,
    )
    if accepted is None:
        print(
            "No grid candidate reached the requested confidence; "
            "use the manual 'calibrate' command instead."
        )
        return 2
    if arguments.grid_output is not None:
        save_grid(accepted.grid, arguments.grid_output)
        print(f"Wrote {arguments.grid_output}")
    return 0


def _validate_scripted_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    required = (
        "image",
        "rows",
        "cols",
        "origin",
        "next_col",
        "next_row",
        "grid_output",
        "overlay_output",
    )
    missing = [
        f"--{name.replace('_', '-')}" for name in required if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"scripted calibration requires: {', '.join(missing)}")


def _validate_extraction_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    required = ("image", "grid", "output_dir")
    missing = [
        f"--{name.replace('_', '-')}" for name in required if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"crop extraction requires: {', '.join(missing)}")


def _validate_detection_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if arguments.image is None:
        parser.error("grid detection requires: --image")
    if not 0.0 <= arguments.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")


def _validate_source_image_hash(image_path: Path, expected_hash: str | None) -> None:
    if expected_hash is not None and image_sha256(image_path) != expected_hash:
        raise ValueError("source image SHA-256 does not match the calibrated grid")


def _pixel_point(values: Sequence[float]) -> PixelPoint:
    return PixelPoint(x=values[0], y=values[1])
