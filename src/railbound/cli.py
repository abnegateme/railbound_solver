"""Commands for the interactive workflow, recognition, and solving."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from railbound.crop_io import load_crop_manifest
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
from railbound.solver import SolverLimits, SolverStatus, solve


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading input or changing state."""
    parser = argparse.ArgumentParser(
        description="Open the interactive Railbound workflow or run semantic tools."
    )
    commands = parser.add_subparsers(dest="command", required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a requested command and return its process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "recognize":
        return _recognize(arguments, parser)
    if arguments.command == "solve":
        return _solve(arguments, parser)
    if arguments.command == "interactive":
        return _interactive(arguments, parser)
    parser.error(f"unsupported command: {arguments.command}")


def _interactive(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        from railbound.gui import run_calibration_gui
    except ModuleNotFoundError as error:
        if error.name == "PySide6":
            parser.error(
                "the interactive workflow requires the GUI extra; install the project with .[gui]"
            )
        raise

    return run_calibration_gui(image_path=arguments.image)


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
