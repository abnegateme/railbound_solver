"""Native Qt interface for parsing screenshots and solving semantic levels."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import cv2
from pydantic import TypeAdapter
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from railbound.calibration import (
    grid_from_control_points,
    grid_from_four_points,
    grid_from_three_points,
    image_sha256,
)
from railbound.detection import (
    CalibrationCandidate,
    GridDetectionHints,
    GridDetector,
    render_detection_overlay,
)
from railbound.extraction import CellCrop, CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import CellCoord, GridCalibrationPoint, GridGeometry, PixelPoint
from railbound.grid_io import save_grid
from railbound.level import Level
from railbound.level_io import save_level
from railbound.placement_io import save_placement
from railbound.recognition import (
    LevelDraft,
    RecognitionOverride,
    RecognitionReport,
    RecognitionSource,
    RecognitionStatus,
    TemplateRecognizer,
    resolve,
)
from railbound.recognition_io import load_template_catalog, load_template_images
from railbound.rendering import ColorImage, render_grid_overlay
from railbound.simulation import SimulationEvent
from railbound.solver import SolverLimits, SolverResult, SolverStatus, solve

_POINT_COLORS = ((255, 0, 255), (0, 220, 0), (0, 0, 255), (0, 165, 255))
_REFERENCE_POINT_COLOR = (255, 255, 0)
_POINT_LABELS = (
    "1. Center of the upper-left cell (0, 0)",
    "2. Center of the cell immediately to its right (0, 1)",
    "3. Center of the cell immediately below it (1, 0)",
    "4. Center of the bottom-right cell",
)
_OVERRIDES_ADAPTER = TypeAdapter(tuple[RecognitionOverride, ...])


class CalibrationCanvas(QGraphicsView):
    """A zoomable image canvas that reports clicks in source-image pixels."""

    point_selected = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#1f1f1f"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, image: QImage) -> None:
        self._image_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._image_item.boundingRect())

    def fit_in_view(self) -> None:
        if not self._image_item.pixmap().isNull():
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = self.mapToScene(event.position().toPoint())
            if self._scene.sceneRect().contains(position):
                self.point_selected.emit(position.x(), position.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
        else:
            self.scale(1 / 1.15, 1 / 1.15)
        event.accept()


class CropPreviewWindow(QMainWindow):
    """Zoomable contact-sheet view for in-memory normalized cell crops."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Railbound Solver — Extracted cells")
        self.resize(1000, 800)
        self._canvas = CalibrationCanvas()
        self.setCentralWidget(self._canvas)

        toolbar = QToolBar("Cell previews", self)
        fit_action = QAction("Fit cells", self)
        fit_action.triggered.connect(self._canvas.fit_in_view)
        toolbar.addAction(fit_action)
        self.addToolBar(toolbar)

    def set_contact_sheet(self, contact_sheet: ColorImage) -> None:
        """Replace the displayed contact sheet without persisting it to disk."""
        self._canvas.set_image(_bgr_to_qimage(contact_sheet))
        self._canvas.fit_in_view()


class CalibrationWindow(QMainWindow):
    """One interactive workflow from screenshot parsing through baseline solving."""

    def __init__(
        self,
        *,
        image_path: Path | None = None,
        grid_output: Path | None = None,
        overlay_output: Path | None = None,
        automatic_detection: bool = False,
    ) -> None:
        super().__init__()
        self._image: ColorImage | None = None
        self._image_path: Path | None = None
        self._image_hash: str | None = None
        self._points: list[PixelPoint] = []
        self._reference_points: list[GridCalibrationPoint] = []
        self._grid: GridGeometry | None = None
        self._crops: list[CellCrop] = []
        self._crop_spec: CropSpec | None = None
        self._contact_sheet: ColorImage | None = None
        self._crop_preview: CropPreviewWindow | None = None
        self._grid_output = grid_output
        self._overlay_output = overlay_output
        self._automatic_detection = automatic_detection
        self._detection_candidates: list[CalibrationCandidate] = []
        self._detect_button: QPushButton | None = None
        self._recognition_report: RecognitionReport | None = None
        self._level_draft: LevelDraft | None = None
        self._level: Level | None = None
        self._level_path: Path | None = None
        self._solver_result: SolverResult | None = None
        self._placement_output: Path | None = None

        self.setWindowTitle("Railbound Solver — Parse, review, and solve")
        self.resize(1240, 820)
        self._canvas = CalibrationCanvas()
        self._canvas.point_selected.connect(self._select_point)
        self._rows = QSpinBox()
        self._cols = QSpinBox()
        for control in (self._rows, self._cols):
            control.setRange(2, 200)
        self._rows.setValue(8)
        self._cols.setValue(10)
        for control in (self._rows, self._cols):
            control.valueChanged.connect(self._dimensions_changed)
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._build_ui()
        self._update_semantic_controls()
        if image_path is not None:
            self.load_image(image_path)
        else:
            self._update_instruction()

    def _build_ui(self) -> None:
        open_action = QAction("Open screenshot…", self)
        open_action.triggered.connect(self._open_image_dialog)
        fit_action = QAction("Fit image", self)
        fit_action.triggered.connect(self._canvas.fit_in_view)
        toolbar = QToolBar("Calibration", self)
        toolbar.addAction(open_action)
        toolbar.addAction(fit_action)
        self.addToolBar(toolbar)

        calibration_controls = QWidget()
        calibration_controls.setMinimumWidth(300)
        layout = QVBoxLayout(calibration_controls)
        layout.addWidget(QLabel("Click the marked cell centers in order."))
        layout.addWidget(self._instruction)
        form = QFormLayout()
        form.addRow("Rows", self._rows)
        form.addRow("Columns", self._cols)
        layout.addLayout(form)

        if self._automatic_detection:
            self._detect_button = QPushButton("Propose grid from visible edges")
            self._detect_button.clicked.connect(self._propose_grid)
            layout.addWidget(self._detect_button)

        layout.addWidget(QLabel("Cell extraction"))
        self._crop_mode = QComboBox()
        self._crop_mode.addItem("Core (inside cell)")
        self._crop_mode.addItem("Context (includes neighbours)")
        self._crop_size = QSpinBox()
        self._crop_size.setRange(16, 512)
        self._crop_size.setValue(96)
        self._cell_mask = QCheckBox("Mask outside target cell")
        crop_form = QFormLayout()
        crop_form.addRow("Mode", self._crop_mode)
        crop_form.addRow("Output size", self._crop_size)
        crop_form.addRow("", self._cell_mask)
        layout.addLayout(crop_form)
        self._extract_button = QPushButton("Extract cells in memory")
        self._extract_button.clicked.connect(self._extract_cells)
        self._show_crops_button = QPushButton("Show extracted cells")
        self._show_crops_button.clicked.connect(self._show_crop_preview)
        self._crop_summary = QLabel()
        self._crop_summary.setWordWrap(True)
        layout.addWidget(self._extract_button)
        layout.addWidget(self._show_crops_button)
        layout.addWidget(self._crop_summary)
        self._crop_mode.currentIndexChanged.connect(self._crop_settings_changed)
        self._crop_size.valueChanged.connect(self._crop_settings_changed)
        self._cell_mask.toggled.connect(self._crop_settings_changed)

        undo_button = QPushButton("Undo last point")
        undo_button.clicked.connect(self._undo_last_point)
        reset_button = QPushButton("Start over")
        reset_button.clicked.connect(self._reset_points)
        save_button = QPushButton("Save calibration…")
        save_button.clicked.connect(self._save_calibration)
        layout.addWidget(undo_button)
        layout.addWidget(reset_button)
        layout.addWidget(save_button)
        layout.addWidget(self._status)
        layout.addStretch()

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.addWidget(self._canvas, stretch=1)
        tabs = QTabWidget()
        tabs.setMinimumWidth(360)
        tabs.addTab(calibration_controls, "Grid and crops")
        tabs.addTab(self._build_review_and_solver_tab(), "Review and solve")
        content_layout.addWidget(tabs)
        self.setCentralWidget(content)
        self._update_extraction_controls()

    def _build_review_and_solver_tab(self) -> QScrollArea:
        """Build the semantic review and solver controls in the main session."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel("Recognition review"))
        self._recognize_button = QPushButton("Recognize extracted cells…")
        self._recognize_button.clicked.connect(self._recognize_crops)
        self._recognition_summary = QLabel()
        self._recognition_summary.setWordWrap(True)
        self._recognition_view = QPlainTextEdit()
        self._recognition_view.setReadOnly(True)
        self._recognition_view.setMinimumHeight(130)
        self._overrides_editor = QPlainTextEdit("[]")
        self._overrides_editor.setPlaceholderText(
            "Optional JSON array of RecognitionOverride objects, for example:\n"
            '[{"cell": {"row": 0, "col": 1}, "label": "track.curve"}]'
        )
        self._apply_overrides_button = QPushButton("Apply recognition overrides")
        self._apply_overrides_button.clicked.connect(self._apply_recognition_overrides)
        layout.addWidget(self._recognize_button)
        layout.addWidget(self._recognition_summary)
        layout.addWidget(self._recognition_view)
        layout.addWidget(QLabel("Recognition overrides (JSON)"))
        layout.addWidget(self._overrides_editor)
        layout.addWidget(self._apply_overrides_button)

        layout.addWidget(QLabel("Authoritative semantic level"))
        self._load_level_button = QPushButton("Open level.json…")
        self._load_level_button.clicked.connect(self._open_level_dialog)
        self._validate_level_button = QPushButton("Validate semantic level")
        self._validate_level_button.clicked.connect(self._validate_level_editor)
        self._save_level_button = QPushButton("Save level.json…")
        self._save_level_button.clicked.connect(self._save_level_editor)
        self._level_editor = QPlainTextEdit()
        self._level_editor.setPlaceholderText(
            "Open or paste the complete authoritative level.json here. Recognition labels are "
            "review evidence; confirm their semantic mapping, inventory, actors, and goals here."
        )
        self._level_editor.setMinimumHeight(220)
        self._level_editor.textChanged.connect(self._level_editor_changed)
        self._semantic_status = QLabel()
        self._semantic_status.setWordWrap(True)
        layout.addWidget(self._load_level_button)
        layout.addWidget(self._validate_level_button)
        layout.addWidget(self._save_level_button)
        layout.addWidget(self._level_editor)
        layout.addWidget(self._semantic_status)

        layout.addWidget(QLabel("Baseline solver"))
        self._max_placements = QSpinBox()
        self._max_placements.setRange(0, 10_000_000)
        self._max_placements.setValue(10_000)
        self._max_ticks = QSpinBox()
        self._max_ticks.setRange(1, 10_000_000)
        self._max_ticks.setValue(10_000)
        self._limit_rails = QCheckBox("Limit candidate rail count")
        self._max_rails = QSpinBox()
        self._max_rails.setRange(0, 10_000)
        self._max_rails.setEnabled(False)
        self._limit_rails.toggled.connect(self._max_rails.setEnabled)
        solver_form = QFormLayout()
        solver_form.addRow("Maximum placements", self._max_placements)
        solver_form.addRow("Maximum ticks per placement", self._max_ticks)
        solver_form.addRow("", self._limit_rails)
        solver_form.addRow("Maximum rails", self._max_rails)
        self._solve_button = QPushButton("Solve validated level")
        self._solve_button.clicked.connect(self._solve_level)
        self._save_solution_button = QPushButton("Save verified placement…")
        self._save_solution_button.setEnabled(False)
        self._save_solution_button.clicked.connect(self._save_solution)
        self._solver_status = QLabel()
        self._solver_status.setWordWrap(True)
        self._solver_trace = QPlainTextEdit()
        self._solver_trace.setReadOnly(True)
        self._solver_trace.setMinimumHeight(220)
        layout.addLayout(solver_form)
        layout.addWidget(self._solve_button)
        layout.addWidget(self._save_solution_button)
        layout.addWidget(self._solver_status)
        layout.addWidget(self._solver_trace)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    @property
    def solver_result(self) -> SolverResult | None:
        """Return the latest result produced inside this interactive session."""
        return self._solver_result

    def _recognize_crops(self) -> None:
        """Recognize the current in-memory crops without writing intermediate files."""
        if self._image is None or self._grid is None or self._crop_spec is None or not self._crops:
            self._show_error("Extract calibrated cells before starting recognition.")
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open template catalog",
            str(self._image_path.parent if self._image_path else Path.cwd()),
            "Template catalog JSON (*.json)",
        )
        if not selected:
            return
        catalog_path = Path(selected)
        try:
            catalog = load_template_catalog(catalog_path)
            recognizer = TemplateRecognizer(catalog, load_template_images(catalog, catalog_path))
            observations = tuple(recognizer.recognize(crop) for crop in self._crops)
            source = RecognitionSource(
                image_sha256=self._image_hash,
                image_width=self._image.shape[1],
                image_height=self._image.shape[0],
                rows=self._grid.rows,
                cols=self._grid.cols,
                crop_spec=self._crop_spec,
            )
            self._recognition_report = RecognitionReport(
                source=source,
                template_catalog=catalog,
                template_fingerprint=recognizer.fingerprint,
                thresholds=recognizer.thresholds,
                observations=observations,
                review_cells=tuple(
                    observation.cell
                    for observation in observations
                    if observation.status is not RecognitionStatus.ACCEPTED
                ),
            )
        except (OSError, ValueError) as error:
            self._show_error(f"Could not recognize extracted cells: {error}")
            return

        self._overrides_editor.setPlainText("[]")
        self._apply_recognition_overrides()

    def _apply_recognition_overrides(self) -> None:
        """Apply visible manual recognition choices to the current report."""
        if self._recognition_report is None:
            self._show_error("Run recognition before applying overrides.")
            return
        try:
            overrides = _OVERRIDES_ADAPTER.validate_json(
                self._overrides_editor.toPlainText().strip() or "[]"
            )
            self._level_draft = resolve(self._recognition_report, overrides)
        except ValueError as error:
            self._show_error(f"Could not apply recognition overrides: {error}")
            return

        self._render_recognition_review(self._recognition_report, self._level_draft)
        self._update_semantic_controls()

    def _render_recognition_review(self, report: RecognitionReport, draft: LevelDraft) -> None:
        """Show candidates, confidence, and unresolved cells in row-major order."""
        lines: list[str] = []
        for observation in report.observations:
            candidate = observation.candidates[0].label if observation.candidates else "-"
            lines.append(
                f"({observation.cell.row}, {observation.cell.col}) "
                f"{observation.status.value}: {candidate} "
                f"({observation.confidence:.2f})"
            )
        if draft.unresolved_cells:
            unresolved = ", ".join(f"({cell.row}, {cell.col})" for cell in draft.unresolved_cells)
            self._recognition_summary.setText(
                f"{len(draft.unresolved_cells)} cell(s) still require a manual label: {unresolved}"
            )
        else:
            self._recognition_summary.setText(
                "Every cell has a reviewed label. Complete the semantic level below before solving."
            )
        self._recognition_view.setPlainText("\n".join(lines))

    def _open_level_dialog(self) -> None:
        """Load an authoritative semantic level into the in-window editor."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open semantic level",
            str(self._level_path.parent if self._level_path else Path.cwd()),
            "Level JSON (*.json)",
        )
        if selected:
            self._load_level_from_path(Path(selected))

    def _load_level_from_path(self, path: Path) -> None:
        """Read, validate, and display a semantic level without changing its data."""
        try:
            raw_level = path.read_text(encoding="utf-8")
            level = Level.model_validate_json(raw_level)
            self._validate_level_grid_dimensions(level)
        except (OSError, ValueError) as error:
            self._show_error(f"Could not load semantic level: {error}")
            return
        self._level_editor.setPlainText(raw_level)
        self._set_validated_level(level, path)

    def _validate_level_editor(self) -> Level | None:
        """Validate the editable semantic document before simulation or solving."""
        try:
            level = Level.model_validate_json(self._level_editor.toPlainText())
            self._validate_level_grid_dimensions(level)
        except ValueError as error:
            self._show_error(f"Semantic level is invalid: {error}")
            return None
        self._set_validated_level(level, self._level_path)
        return level

    def _level_editor_changed(self) -> None:
        """Invalidate results whenever the authoritative document changes."""
        self._level = None
        self._solver_result = None
        self._save_solution_button.setEnabled(False)
        self._solver_trace.clear()
        self._semantic_status.setText("Semantic level changed; validate it before solving.")
        self._solver_status.clear()
        self._update_semantic_controls()

    def _set_validated_level(self, level: Level, path: Path | None) -> None:
        """Adopt one validated level and invalidate any prior solver result."""
        self._level = level
        self._level_path = path
        self._solver_result = None
        self._save_solution_button.setEnabled(False)
        self._solver_trace.clear()
        self._semantic_status.setText(
            f"Validated level {level.id!r}: {level.rows} rows × {level.cols} columns."
        )
        self._solver_status.setText("Level is ready for deterministic replay and solving.")
        self._update_semantic_controls()

    def _validate_level_grid_dimensions(self, level: Level) -> None:
        """Keep the semantic document aligned with the calibrated screenshot grid."""
        if self._grid is None:
            return
        if (level.rows, level.cols) != (self._grid.rows, self._grid.cols):
            raise ValueError(
                "semantic level dimensions "
                f"({level.rows}, {level.cols}) do not match calibrated grid "
                f"({self._grid.rows}, {self._grid.cols})"
            )

    def _save_level_editor(self) -> None:
        """Persist the currently edited, validated semantic level from this window."""
        level = self._validate_level_editor()
        if level is None:
            return
        suggested = self._level_path or Path("level.json")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save semantic level",
            str(suggested),
            "Level JSON (*.json)",
        )
        if not selected:
            return
        output_path = Path(selected).with_suffix(".json")
        try:
            save_level(level, output_path)
        except OSError as error:
            self._show_error(f"Could not save semantic level: {error}")
            return
        self._set_validated_level(level, output_path)
        self._semantic_status.setText(f"Saved and validated semantic level to {output_path}")

    def _solve_level(self) -> None:
        """Search the in-window semantic level and render its verified replay."""
        level = self._validate_level_editor()
        if level is None:
            return
        max_rails = self._max_rails.value() if self._limit_rails.isChecked() else None
        try:
            self._solver_result = solve(
                level,
                SolverLimits(
                    max_placements=self._max_placements.value(),
                    max_rails=max_rails,
                    max_ticks=self._max_ticks.value(),
                ),
            )
        except ValueError as error:
            self._show_error(f"Could not solve semantic level: {error}")
            return

        self._save_solution_button.setEnabled(self._solver_result.solved)
        self._render_solver_result(self._solver_result)

    def _render_solver_result(self, result: SolverResult) -> None:
        """Render placement and trace from immutable simulator-verified output."""
        stats = result.statistics
        lines = [
            f"Status: {result.status.value}",
            f"Placements considered: {stats.placements_considered}",
            f"Runtime states visited: {stats.runtime_states_visited}",
            f"Events emitted: {stats.events_emitted}",
        ]
        if result.solved and result.placement is not None:
            self._solver_status.setText(
                f"Found a verified solution with {len(result.placement.rails)} rail(s)."
            )
            lines.extend(("", "Placement:"))
            lines.extend(
                f"  ({rail.cell.row}, {rail.cell.col}) "
                f"{rail.piece_kind.value} {rail.orientation.value}"
                for rail in result.placement.rails
            )
        elif result.status is SolverStatus.LIMIT_REACHED:
            self._solver_status.setText("No solution found before the requested search limit.")
        else:
            self._solver_status.setText("No valid placement exists within the level's inventory.")

        if result.trace:
            lines.extend(("", "Trace:"))
            lines.extend(_event_summary(event) for event in result.trace)
        self._solver_trace.setPlainText("\n".join(lines))

    def _save_solution(self) -> None:
        """Save the current solution only after its replay reached success."""
        if (
            self._solver_result is None
            or not self._solver_result.solved
            or self._solver_result.placement is None
        ):
            self._show_error("Solve the level successfully before saving a placement.")
            return
        suggested = self._placement_output or (
            self._level_path.with_name(f"{self._level.id}-solution.json")
            if self._level_path is not None and self._level is not None
            else Path("solution.json")
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save verified rail placement",
            str(suggested),
            "Placement JSON (*.json)",
        )
        if not selected:
            return
        output_path = Path(selected).with_suffix(".json")
        self._save_solution_to_path(output_path)

    def _save_solution_to_path(self, output_path: Path) -> None:
        """Persist the current verified solution to an explicit destination."""
        if self._solver_result is None or self._solver_result.placement is None:
            self._show_error("Solve the level successfully before saving a placement.")
            return
        try:
            save_placement(self._solver_result.placement, output_path)
        except OSError as error:
            self._show_error(f"Could not save verified placement: {error}")
            return
        self._placement_output = output_path
        self._solver_status.setText(f"Saved verified placement to {output_path}")

    def _update_semantic_controls(self) -> None:
        """Enable each in-window stage only when its in-memory input exists."""
        has_crops = bool(self._crops)
        has_report = self._recognition_report is not None
        has_level_text = bool(self._level_editor.toPlainText().strip())
        self._recognize_button.setEnabled(has_crops)
        self._apply_overrides_button.setEnabled(has_report)
        self._validate_level_button.setEnabled(has_level_text)
        self._save_level_button.setEnabled(has_level_text)
        self._solve_button.setEnabled(has_level_text)

    def load_image(self, path: Path) -> None:
        image = cast(ColorImage | None, cv2.imread(str(path), cv2.IMREAD_COLOR))
        if image is None:
            self._show_error(f"Could not read image: {path}")
            return
        self._image = image
        self._image_path = path
        self._image_hash = image_sha256(path)
        self._reset_points()
        self._canvas.fit_in_view()
        self._status.setText(f"Loaded {path.name} ({image.shape[1]} × {image.shape[0]} px)")
        if self._automatic_detection:
            self._propose_grid()

    def _open_image_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Railbound screenshot",
            str(self._image_path.parent if self._image_path else Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)",
        )
        if selected:
            self.load_image(Path(selected))

    def _select_point(self, x: float, y: float) -> None:
        if self._image is None:
            self._show_error("Open a screenshot before selecting cells.")
            return
        point = PixelPoint(x=x, y=y)
        if len(self._points) < len(_POINT_LABELS):
            self._detection_candidates.clear()
            self._points.append(point)
            if len(self._points) == len(_POINT_LABELS):
                self._derive_dimensions_from_last_cell()
            else:
                self._update_grid_from_dimensions()
        else:
            self._add_reference_point(point)
        self._refresh_canvas()
        self._update_instruction()

    def _add_reference_point(self, point: PixelPoint) -> None:
        if self._grid is None:
            self._show_error("Finish the first four points before adding a reference cell.")
            return
        nearest = self._grid.nearest_cell(point.x, point.y)
        if not nearest.in_bounds:
            self._show_error("Choose a labeled cell inside the grid.")
            return
        if any(control.cell == nearest.cell for control in self._control_points()):
            self._show_error(
                f"Cell ({nearest.cell.row}, {nearest.cell.col}) is already a reference."
            )
            return
        self._reference_points.append(GridCalibrationPoint(cell=nearest.cell, pixel=point))
        self._detection_candidates.clear()
        self._update_grid_from_dimensions()

    def _propose_grid(self) -> None:
        """Propose and visibly score a grid, retaining manual calibration as fallback."""
        if self._image is None:
            self._show_error("Open a screenshot before proposing a grid.")
            return
        self._clear_extraction()
        try:
            candidates = GridDetector().propose(
                self._image,
                GridDetectionHints(image_sha256=self._image_hash),
            )
        except ValueError as error:
            self._show_error(f"Could not inspect this image: {error}")
            return

        self._detection_candidates = candidates
        if not candidates:
            self._grid = None
            self._points.clear()
            self._reference_points.clear()
            self._status.setText(
                "No regular grid was detected. Select the four manual calibration points."
            )
            self._refresh_canvas()
            self._update_extraction_controls()
            self._update_instruction()
            return

        candidate = candidates[0]
        if not candidate.is_confident(0.70):
            self._grid = None
            self._points.clear()
            self._reference_points.clear()
            self._status.setText(
                f"Best proposal has confidence {candidate.confidence:.2f}; "
                "manual calibration is required below 0.70."
            )
            self._refresh_canvas()
            self._update_extraction_controls()
            self._update_instruction()
            return

        self._adopt_detection(candidate)
        self._status.setText(
            f"Accepted proposal 1 of {len(candidates)}: {candidate.grid.rows} rows × "
            f"{candidate.grid.cols} columns, confidence {candidate.confidence:.2f}, "
            f"RMS error {candidate.residual_rms_px:.2f} px. "
            "Click labeled cell centers to refine it manually."
        )
        self._refresh_canvas()
        self._update_extraction_controls()
        self._update_instruction()

    def _adopt_detection(self, candidate: CalibrationCandidate) -> None:
        """Make one accepted automatic proposal editable by the M1 controls."""
        self._grid = candidate.grid
        self._reference_points.clear()
        self._points = [
            candidate.grid.grid_to_pixel(0, 0),
            candidate.grid.grid_to_pixel(0, 1),
            candidate.grid.grid_to_pixel(1, 0),
            candidate.grid.grid_to_pixel(candidate.grid.rows - 1, candidate.grid.cols - 1),
        ]
        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(candidate.grid.rows)
        self._cols.setValue(candidate.grid.cols)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)

    def _derive_dimensions_from_last_cell(self) -> None:
        if self._image is None or len(self._points) != len(_POINT_LABELS):
            return
        self._clear_extraction()
        try:
            grid = grid_from_four_points(
                image_width=self._image.shape[1],
                image_height=self._image.shape[0],
                origin=self._points[0],
                next_col=self._points[1],
                next_row=self._points[2],
                last_cell=self._points[3],
                image_sha256=self._image_hash,
            )
        except ValueError as error:
            self._grid = None
            self._status.setText(f"Choose the fourth point again: {error}")
            self._update_extraction_controls()
            return

        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(grid.rows)
        self._cols.setValue(grid.cols)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)
        self._grid = grid
        self._update_extraction_controls()
        self._status.setText(
            f"Detected {grid.rows} rows × {grid.cols} columns from 4 references; "
            f"RMS error {grid.calibration.residual_rms_px:.2f} px. "
            "Click any labeled cell center to add another reference."
        )

    def _dimensions_changed(self) -> None:
        """Discard an automatic proposal once a user edits its assumed dimensions."""
        self._detection_candidates.clear()
        self._update_grid_from_dimensions()

    def _update_grid_from_dimensions(self) -> None:
        self._clear_extraction()
        if self._image is None or len(self._points) < 3:
            self._grid = None
            self._refresh_canvas()
            self._update_extraction_controls()
            return
        try:
            if len(self._points) == len(_POINT_LABELS):
                self._grid = grid_from_control_points(
                    rows=self._rows.value(),
                    cols=self._cols.value(),
                    image_width=self._image.shape[1],
                    image_height=self._image.shape[0],
                    control_points=self._control_points(),
                    image_sha256=self._image_hash,
                )
                self._status.setText(
                    f"Fitted {len(self._control_points())} references; "
                    f"RMS error {self._grid.calibration.residual_rms_px:.2f} px, "
                    f"max {self._grid.calibration.residual_max_px:.2f} px."
                )
            else:
                self._grid = grid_from_three_points(
                    rows=self._rows.value(),
                    cols=self._cols.value(),
                    image_width=self._image.shape[1],
                    image_height=self._image.shape[0],
                    origin=self._points[0],
                    next_col=self._points[1],
                    next_row=self._points[2],
                    image_sha256=self._image_hash,
                )
        except ValueError as error:
            self._grid = None
            self._status.setText(str(error))
        self._refresh_canvas()
        self._update_extraction_controls()

    def _undo_last_point(self) -> None:
        self._detection_candidates.clear()
        if self._reference_points:
            self._reference_points.pop()
        elif self._points:
            self._points.pop()
        self._grid = None
        self._update_grid_from_dimensions()
        self._update_instruction()

    def _reset_points(self) -> None:
        self._points.clear()
        self._reference_points.clear()
        self._detection_candidates.clear()
        self._grid = None
        self._clear_extraction()
        self._refresh_canvas()
        self._update_extraction_controls()
        self._update_instruction()

    def _update_instruction(self) -> None:
        if self._image is None:
            self._instruction.setText("Open a screenshot to begin.")
        elif len(self._points) < len(_POINT_LABELS):
            self._instruction.setText(_POINT_LABELS[len(self._points)])
        else:
            self._instruction.setText(
                "Grid is fitted. Click any labeled cell center to add a reference point, "
                "extract cells in memory, then review and solve in this window."
            )

    def _control_points(self) -> tuple[GridCalibrationPoint, ...]:
        if len(self._points) < 3:
            return ()
        points = [
            GridCalibrationPoint(cell=CellCoord(row=0, col=0), pixel=self._points[0]),
            GridCalibrationPoint(cell=CellCoord(row=0, col=1), pixel=self._points[1]),
            GridCalibrationPoint(cell=CellCoord(row=1, col=0), pixel=self._points[2]),
        ]
        if len(self._points) == len(_POINT_LABELS):
            points.append(
                GridCalibrationPoint(
                    cell=CellCoord(row=self._rows.value() - 1, col=self._cols.value() - 1),
                    pixel=self._points[3],
                )
            )
        return tuple(points + self._reference_points)

    def _refresh_canvas(self) -> None:
        if self._image is None:
            return
        preview = self._image.copy()
        if self._detection_candidates:
            preview = render_detection_overlay(preview, self._detection_candidates)
        if self._grid is not None:
            preview = render_grid_overlay(preview, self._grid)
        for index, point in enumerate(self._points):
            color = _POINT_COLORS[index]
            center = (int(round(point.x)), int(round(point.y)))
            cv2.drawMarker(preview, center, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(
                preview,
                str(index + 1),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        for reference in self._reference_points:
            center = (int(round(reference.pixel.x)), int(round(reference.pixel.y)))
            cv2.drawMarker(
                preview,
                center,
                _REFERENCE_POINT_COLOR,
                cv2.MARKER_DIAMOND,
                16,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"{reference.cell.row},{reference.cell.col}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _REFERENCE_POINT_COLOR,
                2,
                cv2.LINE_AA,
            )
        self._canvas.set_image(_bgr_to_qimage(preview))

    def _save_calibration(self) -> None:
        if self._image is None or self._grid is None:
            self._show_error("Select a valid grid before saving.")
            return
        output_paths = self._output_paths()
        if output_paths is None:
            return
        grid_path, overlay_path = output_paths
        overlay = render_grid_overlay(self._image, self._grid)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(overlay_path), overlay):
            self._show_error(f"Could not write overlay image: {overlay_path}")
            return
        save_grid(self._grid, grid_path)
        self._status.setText(f"Saved {grid_path}\nSaved {overlay_path}")

    @property
    def extracted_crops(self) -> tuple[CellCrop, ...]:
        """Return the current in-memory cells for future recognition stages."""
        return tuple(self._crops)

    def _crop_settings_changed(self) -> None:
        self._clear_extraction()
        self._update_extraction_controls()

    def _update_extraction_controls(self) -> None:
        ready = (
            self._image is not None
            and self._grid is not None
            and len(self._points) == len(_POINT_LABELS)
        )
        self._crop_mode.setEnabled(ready)
        self._crop_size.setEnabled(ready)
        self._cell_mask.setEnabled(ready and self._crop_mode.currentIndex() == 1)
        self._extract_button.setEnabled(ready)
        self._show_crops_button.setEnabled(bool(self._crops))
        if self._detect_button is not None:
            self._detect_button.setEnabled(self._image is not None)

    def _clear_extraction(self) -> None:
        had_extraction = self._contact_sheet is not None
        self._crop_summary.setText(
            "Cell previews are out of date; extract them again." if had_extraction else ""
        )
        self._crops = []
        self._crop_spec = None
        self._contact_sheet = None
        self._recognition_report = None
        self._level_draft = None
        self._solver_result = None
        self._recognition_summary.clear()
        self._recognition_view.clear()
        self._overrides_editor.setPlainText("[]")
        self._save_solution_button.setEnabled(False)
        self._solver_trace.clear()
        if self._crop_preview is not None:
            self._crop_preview.hide()
        self._update_semantic_controls()

    def _extract_cells(self) -> None:
        if self._image is None or self._grid is None:
            self._show_error("Finish a valid grid before extracting cells.")
            return
        mode: Literal["core", "context"] = (
            "core" if self._crop_mode.currentIndex() == 0 else "context"
        )
        mask_mode: Literal["none", "cell"] = (
            "cell" if mode == "context" and self._cell_mask.isChecked() else "none"
        )
        spec = CropSpec(
            output_size=self._crop_size.value(),
            mode=mode,
            mask_mode=mask_mode,
        )
        try:
            crops = CellExtractor().extract(self._image, self._grid, spec)
            contact_sheet = render_crop_contact_sheet(crops, columns=self._grid.cols)
        except ValueError as error:
            self._show_error(f"Could not extract cells: {error}")
            return

        self._crops = crops
        self._crop_spec = spec
        self._contact_sheet = contact_sheet
        self._crop_summary.setText(
            f"Extracted {len(crops)} cells in memory ({spec.mode}, {spec.output_size} × "
            f"{spec.output_size} px). No files were written."
        )
        self._status.setText("Grid calibrated and cells extracted in memory.")
        self._update_extraction_controls()
        self._update_semantic_controls()
        self._show_crop_preview()

    def _show_crop_preview(self) -> None:
        if self._contact_sheet is None:
            self._show_error("Extract cells before opening their preview.")
            return
        if self._crop_preview is None:
            self._crop_preview = CropPreviewWindow(self)
        self._crop_preview.set_contact_sheet(self._contact_sheet)
        self._crop_preview.show()
        self._crop_preview.raise_()
        self._crop_preview.activateWindow()

    def _output_paths(self) -> tuple[Path, Path] | None:
        if self._grid_output is not None and self._overlay_output is not None:
            return self._grid_output, self._overlay_output
        suggested = (
            self._image_path.with_name("grid.json") if self._image_path else Path("grid.json")
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibrated grid",
            str(suggested),
            "Grid JSON (*.json)",
        )
        if not selected:
            return None
        grid_path = Path(selected).with_suffix(".json")
        return grid_path, grid_path.with_name("annotated_grid.png")

    def _show_error(self, message: str) -> None:
        self._status.setText(message)
        QMessageBox.warning(self, "Railbound Solver", message)


def run_calibration_gui(
    *,
    image_path: Path | None,
    grid_output: Path | None,
    overlay_output: Path | None,
    automatic_detection: bool = False,
) -> int:
    """Open the primary native parse, review, and solve window."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
        window = CalibrationWindow(
            image_path=image_path,
            grid_output=grid_output,
            overlay_output=overlay_output,
            automatic_detection=automatic_detection,
        )
        window.show()
        return cast(int, application.exec())

    window = CalibrationWindow(
        image_path=image_path,
        grid_output=grid_output,
        overlay_output=overlay_output,
        automatic_detection=automatic_detection,
    )
    window.show()
    return 0


def _event_summary(event: SimulationEvent) -> str:
    """Render one structured replay event as compact inspectable text."""
    actors = ", ".join(event.actor_ids) if event.actor_ids else "-"
    cell = event.target_cell or event.source_cell
    location = "" if cell is None else f" at ({cell.row}, {cell.col})"
    reason = "" if event.reason is None else f" [{event.reason.value}]"
    return (
        f"{event.sequence:04d}  tick {event.tick}  {event.kind.value}  {actors}{location}{reason}"
    )


def _bgr_to_qimage(image: ColorImage) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = width * 3
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
