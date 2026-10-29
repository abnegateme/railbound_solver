"""Commands for Railbound calibration, recognition, and solving."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
from pydantic import ValidationError

from railbound.calibration import grid_from_three_points, image_sha256
from railbound.crop_io import load_crop_manifest, write_crop_set
from railbound.detection import GridDetectionHints, GridDetector, render_detection_overlay
from railbound.extraction import CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import PixelPoint
from railbound.grid_io import load_grid, save_grid
from railbound.level_io import load_level
from railbound.placement_io import save_placement
from railbound.recognition import RecognitionThresholds, TemplateRecognizer, resolve
from railbound.recognition_io import (
    load_recognition_overrides,
    load_template_catalog,
    load_template_images,
    recognize_crop_directory,
    save_level_draft,
    save_recognition_report,
)
from railbound.rendering import ColorImage, render_grid_overlay
from railbound.solver import SolverLimits, SolverStatus, solve


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading input or changing state."""
    parser = argparse.ArgumentParser(
        description="Calibrate, extract, recognize, and solve Railbound boards."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="calibrate a grid from a screenshot")
    calibrate.add_argument(
        "--interactive", action="store_true", help="open the primary interactive workflow"
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

    recognize = commands.add_parser(
        "recognize", help="match normalized crops to labelled template references"
    )
    recognize.add_argument(
        "--crops-dir", type=Path, help="directory containing extracted crop PNGs"
    )
    recognize.add_argument(
        "--manifest", type=Path, help="crop manifest JSON (default: crops-dir/manifest.json)"
    )
    recognize.add_argument("--template-catalog", type=Path, help="labelled template catalog JSON")
    recognize.add_argument("--report-output", type=Path, help="destination recognition_report.json")
    recognize.add_argument(
        "--min-similarity", type=float, help="override the catalog rejection threshold"
    )
    recognize.add_argument(
        "--min-confidence", type=float, help="override the catalog acceptance threshold"
    )
    recognize.add_argument(
        "--ambiguity-margin", type=float, help="override the catalog ambiguity margin"
    )
    recognize.add_argument(
        "--max-candidates", type=int, help="override the number of ranked labels retained"
    )
    recognize.add_argument(
        "--overrides", type=Path, help="JSON array of manual cell-label overrides"
    )
    recognize.add_argument(
        "--draft-output", type=Path, help="optional destination level_draft.json for review"
    )

    solve_command = commands.add_parser(
        "solve", help="find and write a simulator-verified rail placement"
    )
    solve_command.add_argument("--level", type=Path, help="authoritative semantic level.json")
    solve_command.add_argument(
        "--placement-output", type=Path, help="destination solution placement JSON"
    )
    solve_command.add_argument(
        "--max-placements",
        type=int,
        default=10_000,
        help="maximum placement candidates to replay (default: 10000)",
    )
    solve_command.add_argument(
        "--max-rails", type=int, help="maximum rails in a candidate placement"
    )
    solve_command.add_argument(
        "--max-ticks",
        type=int,
        default=10_000,
        help="maximum simulation ticks per candidate (default: 10000)",
    )

    interactive = commands.add_parser(
        "interactive", help="open the primary screenshot parsing and solver window"
    )
    interactive.add_argument("--image", type=Path, help="source screenshot to open")
    interactive.add_argument("--grid-output", type=Path, help="destination grid.json")
    interactive.add_argument("--overlay-output", type=Path, help="destination annotated_grid.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a requested command and return its process exit code."""
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
    if arguments.command == "recognize":
        return _recognize(arguments, parser)
    if arguments.command == "solve":
        return _solve(arguments, parser)
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
    return _interactive(arguments, parser, automatic_detection=True)


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


def _recognize(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_recognition_arguments(arguments, parser)
    manifest_path = arguments.manifest or arguments.crops_dir / "manifest.json"
    try:
        manifest = load_crop_manifest(manifest_path)
        catalog = load_template_catalog(arguments.template_catalog)
        thresholds = _recognition_thresholds(arguments, catalog.thresholds)
        recognizer = TemplateRecognizer(
            catalog,
            load_template_images(catalog, arguments.template_catalog),
            thresholds=thresholds,
        )
        report = recognize_crop_directory(manifest, arguments.crops_dir, recognizer)
        save_recognition_report(report, arguments.report_output)
        if arguments.draft_output is not None:
            overrides = (
                load_recognition_overrides(arguments.overrides)
                if arguments.overrides is not None
                else ()
            )
            save_level_draft(resolve(report, overrides), arguments.draft_output)
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))

    accepted = len(report.observations) - len(report.review_cells)
    print(f"Wrote {arguments.report_output}")
    print(f"Accepted {accepted} of {len(report.observations)} cells")
    if arguments.draft_output is not None:
        print(f"Wrote {arguments.draft_output}")
    if report.review_cells:
        cells = ", ".join(f"({cell.row}, {cell.col})" for cell in report.review_cells)
        print(f"Manual review required for: {cells}")
        return 2
    return 0


def _solve(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Search a semantic level and persist its first verified placement."""
    _validate_solver_arguments(arguments, parser)
    try:
        level = load_level(arguments.level)
        result = solve(
            level,
            SolverLimits(
                max_placements=arguments.max_placements,
                max_rails=arguments.max_rails,
                max_ticks=arguments.max_ticks,
            ),
        )
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))

    stats = result.statistics
    print(
        "Searched "
        f"{stats.placements_considered} placements with {stats.simulations} replays; "
        f"pruned {stats.placements_pruned} placements and "
        f"{stats.connectivity_branches_pruned} partial branches; visited "
        f"{stats.runtime_states_visited} runtime states in {stats.elapsed_seconds:.3f}s."
    )
    if result.solved:
        if result.placement is None:
            raise AssertionError("a solved result must include a placement")
        save_placement(result.placement, arguments.placement_output)
        print(f"Wrote {arguments.placement_output}")
        return 0
    if result.status is SolverStatus.LIMIT_REACHED:
        print("No solution found before the requested search limit.")
    else:
        print("No valid placement exists within the level's inventory.")
    return 2


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


def _validate_recognition_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    required = ("crops_dir", "template_catalog", "report_output")
    missing = [
        f"--{name.replace('_', '-')}" for name in required if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"recognition requires: {', '.join(missing)}")
    if arguments.overrides is not None and arguments.draft_output is None:
        parser.error("--overrides requires --draft-output")


def _validate_solver_arguments(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Require solver input and output paths before loading level data."""
    required = ("level", "placement_output")
    missing = [
        f"--{name.replace('_', '-')}" for name in required if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"solve requires: {', '.join(missing)}")


def _recognition_thresholds(
    arguments: argparse.Namespace, defaults: RecognitionThresholds
) -> RecognitionThresholds:
    return RecognitionThresholds(
        min_similarity=(
            defaults.min_similarity
            if arguments.min_similarity is None
            else arguments.min_similarity
        ),
        min_confidence=(
            defaults.min_confidence
            if arguments.min_confidence is None
            else arguments.min_confidence
        ),
        ambiguity_margin=(
            defaults.ambiguity_margin
            if arguments.ambiguity_margin is None
            else arguments.ambiguity_margin
        ),
        max_candidates=(
            defaults.max_candidates
            if arguments.max_candidates is None
            else arguments.max_candidates
        ),
    )


def _validate_source_image_hash(image_path: Path, expected_hash: str | None) -> None:
    if expected_hash is not None and image_sha256(image_path) != expected_hash:
        raise ValueError("source image SHA-256 does not match the calibrated grid")


def _pixel_point(values: Sequence[float]) -> PixelPoint:
    return PixelPoint(x=values[0], y=values[1])
