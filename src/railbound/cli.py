"""Commands for manual and interactive Railbound screenshot calibration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
from pydantic import ValidationError

from railbound.calibration import grid_from_three_points, image_sha256
from railbound.geometry import PixelPoint
from railbound.grid_io import save_grid
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a requested non-interactive command and return a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "calibrate":
        parser.error(f"unsupported command: {arguments.command}")
    if arguments.interactive:
        return _interactive_calibrate(arguments, parser)
    return _calibrate(arguments, parser)


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
    if (arguments.grid_output is None) != (arguments.overlay_output is None):
        parser.error(
            "--grid-output and --overlay-output must be supplied together in interactive mode"
        )
    try:
        from railbound.gui import run_calibration_gui
    except ModuleNotFoundError as error:
        if error.name == "PySide6":
            parser.error(
                "interactive calibration requires the GUI extra; install the project with .[gui]"
            )
        raise

    return run_calibration_gui(
        image_path=arguments.image,
        grid_output=arguments.grid_output,
        overlay_output=arguments.overlay_output,
    )


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


def _pixel_point(values: Sequence[float]) -> PixelPoint:
    return PixelPoint(x=values[0], y=values[1])
