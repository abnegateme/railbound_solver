"""Native Qt interface for parsing screenshots and solving semantic levels."""

from __future__ import annotations

import re
import sqlite3
from math import floor, hypot
from pathlib import Path
from typing import Literal, cast

import cv2
from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from railbound.calibration import (
    GridIntersectionPoint,
    grid_from_four_intersections,
    grid_from_intersection_points,
    grid_from_three_intersections,
    image_sha256,
)
from railbound.annotation_gui import CellAnnotationGrid
from railbound.annotation_store import AnnotationStore
from railbound.cell_annotation import (
    ActorFeature,
    CellAnnotation,
    CellTypeDefinition,
    GoalFeature,
    annotation_errors,
    annotation_value_errors,
    cell_type_for_role,
    formalize_annotations,
    make_cell_annotation_id,
    rotate_annotation_properties,
)
from railbound.detection import (
    CalibrationCandidate,
    GridDetectionHints,
    GridDetector,
    render_detection_overlay,
)
from railbound.extraction import CellCrop, CellExtractor, CropSpec
from railbound.geometry import CellCoord, GridGeometry, PixelPoint
from railbound.level import Level
from railbound.placement_io import load_placement, save_placement
from railbound.recognition import (
    RecognitionStatus,
    Template,
    TemplateCatalog,
    TemplateDefinition,
    TemplateRecognizer,
)
from railbound.rendering import ColorImage, render_grid_overlay, render_level_schematic
from railbound.simulation import SimulationEvent, SimulationOptions, simulate
from railbound.solver import SolverLimits, SolverResult, SolverStatus, solve
from railbound.topology import (
    CompiledBoard,
    PortNode,
    RailPlacement,
    compile_board,
    validate_placement,
)

_POINT_COLORS = ((255, 0, 255), (0, 220, 0), (0, 0, 255), (0, 165, 255))
_REFERENCE_POINT_COLOR = (255, 255, 0)
_DEFAULT_ROWS = 8
_DEFAULT_COLS = 10
_DIMENSION_CONTROL_WIDTH = 72
_INTERSECTION_SELECTION_MAX_AXIS_FRACTION = 0.4
_INTERACTIVE_CROP_SPEC = CropSpec(
    output_size=96,
    core_fraction=0.8,
    mode="core",
    mask_mode="none",
)
_CALIBRATION_CARD_STYLE = """
QFrame#calibrationCard, QFrame#cellsCard {
    background-color: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: 12px;
}
QLabel#calibrationTitle {
    color: #1b1b1f;
    font-size: 15px;
    font-weight: 600;
}
QLabel#dimensionLabel {
    color: #3f4249;
    font-size: 11px;
}
QSpinBox#dimensionInput {
    min-height: 22px;
    padding: 3px 8px;
    color: #1b1b1f;
    background-color: #f8f7fc;
    border: 1px solid #c9c6d0;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    selection-background-color: #6750a4;
}
QSpinBox#dimensionInput:hover {
    background-color: #f4f1fa;
    border-color: #79747e;
}
QSpinBox#dimensionInput:focus {
    background-color: #ffffff;
    border: 2px solid #6750a4;
}
QToolButton#calibrationAction {
    border: none;
    border-radius: 16px;
    background-color: transparent;
}
QToolButton#calibrationAction:hover {
    background-color: #f0ebf7;
}
QToolButton#calibrationAction:pressed {
    background-color: #e4d9f2;
}
QToolButton#calibrationAction:disabled {
    background-color: transparent;
}
QPushButton#primaryAction {
    min-height: 36px;
    padding: 0 18px;
    color: #ffffff;
    background-color: #6750a4;
    border: none;
    border-radius: 18px;
    font-weight: 600;
}
QPushButton#primaryAction:hover {
    background-color: #5b4497;
}
QPushButton#primaryAction:pressed {
    background-color: #4f378b;
}
QPushButton#primaryAction:disabled {
    color: #9c97a4;
    background-color: #e2dfe7;
}
QPushButton#secondaryAction {
    min-height: 32px;
    padding: 0 14px;
    color: #6750a4;
    background-color: transparent;
    border: 1px solid #79747e;
    border-radius: 16px;
    font-weight: 600;
}
QPushButton#secondaryAction:hover {
    background-color: #f0ebf7;
}
QPushButton#secondaryAction:disabled {
    color: #9c97a4;
    border-color: #c9c5cf;
}
QLabel#cellsSummary {
    color: #6f7480;
    font-size: 11px;
}
"""
_POINT_LABELS = (
    "Click the upper-left grid intersection.",
    "Click the adjacent grid intersection to its right.",
    "Click the adjacent grid intersection below it.",
    "Click the outer bottom-right grid intersection.",
)


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

    def clear_image(self) -> None:
        """Remove the current image and reset the scene bounds."""
        self._image_item.setPixmap(QPixmap())
        self._scene.setSceneRect(self._image_item.boundingRect())

    def fit_in_view(self) -> None:
        if not self._image_item.pixmap().isNull():
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the complete image fitted when the available canvas size changes."""
        super().resizeEvent(event)
        self.fit_in_view()

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


class CalibrationWindow(QMainWindow):
    """One interactive workflow from screenshot parsing through baseline solving."""

    def __init__(
        self,
        *,
        image_path: Path | None = None,
        automatic_detection: bool = True,
    ) -> None:
        super().__init__()
        self._image: ColorImage | None = None
        self._image_path: Path | None = None
        self._image_hash: str | None = None
        self._points: list[PixelPoint] = []
        self._reference_points: list[GridIntersectionPoint] = []
        self._grid: GridGeometry | None = None
        self._crops: list[CellCrop] = []
        self._crop_spec: CropSpec | None = None
        self._annotation_store = AnnotationStore(_annotation_store_path())
        self._cell_types: dict[str, CellTypeDefinition] = {
            definition.id: definition for definition in self._annotation_store.cell_types()
        }
        self._rail_budget: int | None = None
        self._automatic_detection = automatic_detection
        self._detection_candidates: list[CalibrationCandidate] = []
        self._level: Level | None = None
        self._level_path: Path | None = None
        self._placement: RailPlacement | None = None
        self._placement_path: Path | None = None
        self._solver_result: SolverResult | None = None
        self._solution_output_path: Path | None = None

        self.setWindowTitle("Railbound Solver — Parse, review, and solve")
        self.resize(1240, 820)
        self._canvas = CalibrationCanvas()
        self._canvas.point_selected.connect(self._select_point)
        self._rows = QSpinBox()
        self._cols = QSpinBox()
        for control in (self._rows, self._cols):
            control.setRange(2, 200)
        self._rows.setFixedWidth(_DIMENSION_CONTROL_WIDTH)
        self._cols.setFixedWidth(_DIMENSION_CONTROL_WIDTH)
        self._rows.setValue(_DEFAULT_ROWS)
        self._cols.setValue(_DEFAULT_COLS)
        for control in (self._rows, self._cols):
            control.valueChanged.connect(self._dimensions_changed)
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        self._activity_log = QPlainTextEdit()
        self._activity_log.setReadOnly(True)
        self._activity_log.setPlaceholderText("Screenshot activity will appear here.")
        self._build_ui()
        self._update_semantic_controls()
        if image_path is not None:
            self.load_image(image_path)
        else:
            self._update_instruction()

    def _build_ui(self) -> None:
        open_action = QAction("Open screenshot…", self)
        open_action.triggered.connect(self._open_image_dialog)
        toolbar = QToolBar("Calibration", self)
        toolbar.addAction(open_action)
        self.addToolBar(toolbar)

        calibration_controls = QWidget()
        calibration_controls.setMinimumWidth(300)
        calibration_controls.setStyleSheet(_CALIBRATION_CARD_STYLE)
        layout = QVBoxLayout(calibration_controls)

        calibration_card = QFrame()
        calibration_card.setObjectName("calibrationCard")
        calibration_card.setAccessibleName("Calibration")
        calibration_layout = QVBoxLayout(calibration_card)
        calibration_layout.setContentsMargins(12, 10, 12, 12)
        calibration_layout.setSpacing(8)

        title = QLabel("Calibration")
        title.setObjectName("calibrationTitle")

        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(title, stretch=1)
        self._undo_calibration_button = QToolButton()
        self._undo_calibration_button.setObjectName("calibrationAction")
        self._undo_calibration_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self._undo_calibration_button.setFixedSize(32, 32)
        self._undo_calibration_button.setToolTip("Undo last grid intersection")
        self._undo_calibration_button.setAccessibleName("Undo last grid intersection")
        self._undo_calibration_button.clicked.connect(self._undo_last_point)
        self._reset_calibration_button = QToolButton()
        self._reset_calibration_button.setObjectName("calibrationAction")
        self._reset_calibration_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self._reset_calibration_button.setFixedSize(32, 32)
        self._reset_calibration_button.setToolTip("Start calibration over")
        self._reset_calibration_button.setAccessibleName("Start calibration over")
        self._reset_calibration_button.clicked.connect(self._reset_points)
        header.addWidget(self._undo_calibration_button)
        header.addWidget(self._reset_calibration_button)
        calibration_layout.addLayout(header)

        dimensions = QHBoxLayout()
        dimensions.setSpacing(8)
        for label_text, control in (("Rows", self._rows), ("Columns", self._cols)):
            label = QLabel(label_text)
            label.setObjectName("dimensionLabel")
            control.setObjectName("dimensionInput")
            dimensions.addWidget(label)
            dimensions.addWidget(control)
        dimensions.addStretch()
        calibration_layout.addLayout(dimensions)
        layout.addWidget(calibration_card)

        cells_card = QFrame()
        cells_card.setObjectName("cellsCard")
        cells_card.setAccessibleName("Cells")
        cells_layout = QVBoxLayout(cells_card)
        cells_layout.setContentsMargins(12, 10, 12, 12)
        cells_layout.setSpacing(8)
        cells_title = QLabel("Cells")
        cells_title.setObjectName("calibrationTitle")
        cells_layout.addWidget(cells_title)
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level"))
        self._level_number = QLineEdit()
        self._level_number.setPlaceholderText("e.g. 1-13")
        self._level_number.textChanged.connect(self._level_number_changed)
        level_row.addWidget(self._level_number, stretch=1)
        cells_layout.addLayout(level_row)
        self._extract_button = QPushButton("Extract cells")
        self._extract_button.setObjectName("secondaryAction")
        self._extract_button.clicked.connect(self._extract_cells)
        self._formalize_button = QPushButton("Formalize level")
        self._formalize_button.setObjectName("primaryAction")
        self._formalize_button.clicked.connect(self._formalize_level)
        actions = QHBoxLayout()
        actions.addWidget(self._extract_button)
        actions.addWidget(self._formalize_button)
        cells_layout.addLayout(actions)
        self._crop_summary = QLabel()
        self._crop_summary.setObjectName("cellsSummary")
        self._crop_summary.setWordWrap(True)
        self._crop_summary.hide()
        cells_layout.addWidget(self._crop_summary)
        layout.addWidget(cells_card)

        layout.addStretch()

        activity = QWidget()
        activity_layout = QVBoxLayout(activity)
        activity_layout.setContentsMargins(9, 0, 9, 9)
        activity_layout.addWidget(QLabel("Activity and next step"))
        activity_layout.addWidget(self._activity_log, stretch=1)
        activity_layout.addWidget(self._instruction)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        self._cell_editor = CellAnnotationGrid()
        self._cell_editor.annotation_changed.connect(self._cell_annotation_changed)
        self._cell_editor.new_type_requested.connect(self._create_cell_type)
        self._level_canvas = CalibrationCanvas()
        self._workspace_tabs = QTabWidget()
        self._workspace_tabs.addTab(self._canvas, "Screenshot")
        self._cells_tab_index = self._workspace_tabs.addTab(self._cell_editor, "Cells")
        self._level_tab_index = self._workspace_tabs.addTab(self._level_canvas, "Level")
        self._workspace_tabs.setTabEnabled(self._cells_tab_index, False)
        self._workspace_tabs.setTabEnabled(self._level_tab_index, False)
        content_layout.addWidget(self._workspace_tabs, stretch=1)
        tabs = QTabWidget()
        tabs.addTab(calibration_controls, "Grid and cells")
        tabs.addTab(self._build_review_and_solver_tab(), "Solve")

        sidebar = QWidget()
        sidebar.setMinimumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(tabs, stretch=4)
        sidebar_layout.addWidget(activity, stretch=1)
        content_layout.addWidget(sidebar)
        self.setCentralWidget(content)
        self._update_extraction_controls()

    def _build_review_and_solver_tab(self) -> QScrollArea:
        """Build only controls that consume an already formalized in-memory level."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel("Formal level"))
        self._level_editor = QPlainTextEdit()
        self._level_editor.hide()
        self._level_editor.textChanged.connect(self._level_editor_changed)
        self._semantic_status = QLabel("Formalize the reviewed cells before solving.")
        self._semantic_status.setWordWrap(True)
        layout.addWidget(self._semantic_status)

        layout.addWidget(QLabel("Placement inspection and replay"))
        self._load_placement_button = QPushButton("Open placement.json…")
        self._load_placement_button.clicked.connect(self._open_placement_dialog)
        self._validate_placement_button = QPushButton("Validate placement")
        self._validate_placement_button.clicked.connect(self._validate_placement_editor)
        self._save_placement_button = QPushButton("Save placement.json…")
        self._save_placement_button.clicked.connect(self._save_placement_editor)
        self._placement_editor = QPlainTextEdit()
        self._placement_editor.setPlaceholderText(
            "Open or paste a RailPlacement JSON document to inspect its topology and replay it."
        )
        self._placement_editor.setMinimumHeight(160)
        self._placement_editor.textChanged.connect(self._placement_editor_changed)
        self._inspect_topology_button = QPushButton("Inspect compiled topology")
        self._inspect_topology_button.clicked.connect(self._inspect_topology)
        self._replay_placement_button = QPushButton("Replay placement")
        self._replay_placement_button.clicked.connect(self._replay_placement)
        self._placement_status = QLabel()
        self._placement_status.setWordWrap(True)
        self._placement_inspection_output = QPlainTextEdit()
        self._placement_inspection_output.setReadOnly(True)
        self._placement_inspection_output.setMinimumHeight(180)
        layout.addWidget(self._load_placement_button)
        layout.addWidget(self._validate_placement_button)
        layout.addWidget(self._save_placement_button)
        layout.addWidget(self._placement_editor)
        layout.addWidget(self._inspect_topology_button)
        layout.addWidget(self._replay_placement_button)
        layout.addWidget(self._placement_status)
        layout.addWidget(self._placement_inspection_output)

        layout.addWidget(QLabel("Solver"))
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
        self._placement = None
        self._placement_path = None
        self._solver_result = None
        self._save_solution_button.setEnabled(False)
        self._solver_trace.clear()
        self._placement_inspection_output.clear()
        self._semantic_status.setText("Semantic level changed; validate it before solving.")
        self._placement_status.setText("Validate the semantic level before using this placement.")
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

    def _open_placement_dialog(self) -> None:
        """Load a portable placement for topology inspection or deterministic replay."""
        level = self._validate_level_editor()
        if level is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open rail placement",
            str(self._placement_path.parent if self._placement_path else Path.cwd()),
            "Placement JSON (*.json)",
        )
        if selected:
            self._load_placement_from_path(Path(selected), level)

    def _load_placement_from_path(self, path: Path, level: Level | None = None) -> None:
        """Read a placement into the in-window editor and validate it against the level."""
        active_level = level or self._validate_level_editor()
        if active_level is None:
            return
        try:
            raw_placement = path.read_text(encoding="utf-8")
            placement = load_placement(path)
            validate_placement(active_level, placement)
        except (OSError, ValueError) as error:
            self._show_error(f"Could not load rail placement: {error}")
            return
        self._placement_editor.setPlainText(raw_placement)
        self._set_validated_placement(placement, path)

    def _validate_placement_editor(self) -> RailPlacement | None:
        """Validate editable placement JSON against the current semantic level."""
        level = self._validate_level_editor()
        if level is None:
            return None
        try:
            placement = RailPlacement.model_validate_json(self._placement_editor.toPlainText())
            validate_placement(level, placement)
        except ValueError as error:
            self._show_error(f"Rail placement is invalid: {error}")
            return None
        self._set_validated_placement(placement, self._placement_path)
        return placement

    def _placement_editor_changed(self) -> None:
        """Prevent replaying an earlier placement after its JSON has been edited."""
        self._placement = None
        self._placement_status.setText(
            "Placement changed; validate it before inspection or replay."
        )
        self._placement_inspection_output.clear()
        self._update_semantic_controls()

    def _set_validated_placement(self, placement: RailPlacement, path: Path | None) -> None:
        """Adopt one placement that the current level accepts as legal."""
        self._placement = placement
        self._placement_path = path
        self._placement_status.setText(
            f"Validated placement for {placement.level_id!r} with {len(placement.rails)} rail(s)."
        )
        self._update_semantic_controls()

    def _save_placement_editor(self) -> None:
        """Persist the editable placement after it passes level-specific validation."""
        placement = self._validate_placement_editor()
        if placement is None:
            return
        suggested = self._placement_path or Path("placement.json")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save rail placement",
            str(suggested),
            "Placement JSON (*.json)",
        )
        if not selected:
            return
        output_path = Path(selected).with_suffix(".json")
        self._save_placement_to_path(placement, output_path)

    def _save_placement_to_path(self, placement: RailPlacement, path: Path) -> None:
        """Write a known-valid manual placement to an explicit destination."""
        try:
            save_placement(placement, path)
        except OSError as error:
            self._show_error(f"Could not save rail placement: {error}")
            return
        self._set_validated_placement(placement, path)
        self._placement_status.setText(f"Saved validated rail placement to {path}.")

    def _inspect_topology(self) -> None:
        """Compile the current placement and expose its deterministic port graph summary."""
        placement = self._validate_placement_editor()
        if placement is None or self._level is None:
            return
        try:
            board = compile_board(self._level, placement)
        except ValueError as error:
            self._show_error(f"Could not compile rail topology: {error}")
            return
        self._placement_inspection_output.setPlainText(_topology_summary(board))
        self._placement_status.setText("Compiled topology is shown below.")

    def _replay_placement(self) -> None:
        """Run the M6–M7 simulator for the visible manual placement."""
        placement = self._validate_placement_editor()
        if placement is None or self._level is None:
            return
        try:
            result = simulate(
                self._level,
                placement,
                SimulationOptions(max_ticks=self._max_ticks.value()),
            )
        except ValueError as error:
            self._show_error(f"Could not replay rail placement: {error}")
            return
        lines = [
            f"Status: {result.final_state.status.value}",
            f"Ticks: {result.final_state.tick}",
            "",
            "Trace:",
            *(_event_summary(event) for event in result.events),
        ]
        self._placement_inspection_output.setPlainText("\n".join(lines))
        outcome = "succeeded" if result.succeeded else "ended without success"
        self._placement_status.setText(
            f"Simulator {outcome} after {result.final_state.tick} tick(s)."
        )

    def _solve_level(self) -> None:
        """Search the in-window semantic level and render its verified replay."""
        level = self._validate_level_editor()
        if level is None:
            return
        max_rails = self._max_rails.value() if self._limit_rails.isChecked() else None
        self._solve_button.setEnabled(False)
        self._solver_status.setText("Searching with automatic safe pruning…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
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
        finally:
            QApplication.restoreOverrideCursor()
            self._solve_button.setEnabled(True)

        self._save_solution_button.setEnabled(self._solver_result.solved)
        if self._solver_result.placement is not None:
            self._placement_editor.setPlainText(
                self._solver_result.placement.model_dump_json(indent=2)
            )
            self._set_validated_placement(self._solver_result.placement, None)
        self._render_solver_result(self._solver_result)

    def _render_solver_result(self, result: SolverResult) -> None:
        """Render placement and trace from immutable simulator-verified output."""
        stats = result.statistics
        lines = [
            f"Status: {result.status.value}",
            "Search: "
            f"{stats.placements_considered} candidate(s), {stats.simulations} replay(s), "
            f"{stats.placements_pruned} pruned in {stats.elapsed_seconds:.3f} s",
            "Optimization: "
            f"{stats.connectivity_branches_pruned} partial branch(es), "
            f"{stats.transposition_hits} runtime cache hit(s)",
            f"Replay: {stats.runtime_states_visited} state(s), {stats.events_emitted} event(s)",
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
        suggested = self._solution_output_path or (
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
        self._solution_output_path = output_path
        self._solver_status.setText(f"Saved verified placement to {output_path}")

    def _update_semantic_controls(self) -> None:
        """Enable formalization and solver actions from in-memory workflow state."""
        annotations = self._cell_editor.annotations() if self._crops else ()
        complete = bool(annotations) and all(
            not annotation_errors(annotation, self._cell_types) for annotation in annotations
        )
        self._formalize_button.setEnabled(complete and bool(self._level_number.text().strip()))
        has_level_text = self._level is not None and bool(self._level_editor.toPlainText().strip())
        has_placement_text = bool(self._placement_editor.toPlainText().strip())
        self._solve_button.setEnabled(has_level_text)
        self._load_placement_button.setEnabled(has_level_text)
        self._validate_placement_button.setEnabled(has_level_text and has_placement_text)
        self._save_placement_button.setEnabled(has_level_text and has_placement_text)
        self._inspect_topology_button.setEnabled(has_level_text and has_placement_text)
        self._replay_placement_button.setEnabled(has_level_text and has_placement_text)
        self._update_instruction()

    def load_image(self, path: Path) -> None:
        image = cast(ColorImage | None, cv2.imread(str(path), cv2.IMREAD_COLOR))
        if image is None:
            self._show_error(f"Could not read image: {path}")
            return
        self._image = image
        self._image_path = path
        self._image_hash = image_sha256(path)
        self._level_number.setText(_infer_level_number(path) or "")
        self._restore_default_dimensions()
        self._reset_points()
        self._canvas.fit_in_view()
        self._log_activity(f"Loaded {path.name} ({image.shape[1]} × {image.shape[0]} px).")
        if self._automatic_detection:
            self._detect_grid()

    def _restore_default_dimensions(self) -> None:
        """Restore the manual fallback before analysing a newly loaded screenshot."""
        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(_DEFAULT_ROWS)
        self._cols.setValue(_DEFAULT_COLS)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)

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
                self._derive_dimensions_from_outer_intersection()
            else:
                self._update_grid_from_dimensions()
        else:
            self._add_reference_point(point)
        self._refresh_canvas()
        self._update_instruction()

    def _add_reference_point(self, point: PixelPoint) -> None:
        if self._grid is None:
            self._show_error("Finish the first four intersections before adding a reference.")
            return
        continuous = self._grid.pixel_to_grid(point.x, point.y)
        row = floor(continuous.row + 1.0)
        col = floor(continuous.col + 1.0)
        if not 0 <= row <= self._grid.rows or not 0 <= col <= self._grid.cols:
            self._show_error("Choose a visible grid intersection on the board.")
            return
        expected = self._grid.grid_to_pixel(row - 0.5, col - 0.5)
        axis_length = min(
            hypot(self._grid.col_axis_px.x, self._grid.col_axis_px.y),
            hypot(self._grid.row_axis_px.x, self._grid.row_axis_px.y),
        )
        distance = hypot(point.x - expected.x, point.y - expected.y)
        if distance > axis_length * _INTERSECTION_SELECTION_MAX_AXIS_FRACTION:
            self._show_error("Choose a point closer to a visible grid intersection.")
            return
        if any(
            reference.row == row and reference.col == col
            for reference in self._intersection_points()
        ):
            self._show_error(f"Grid intersection ({row}, {col}) is already a reference.")
            return
        self._reference_points.append(GridIntersectionPoint(row=row, col=col, pixel=point))
        self._detection_candidates.clear()
        self._update_grid_from_dimensions()

    def _detect_grid(self) -> None:
        """Detect and visibly score a grid, retaining manual calibration as fallback."""
        if self._image is None:
            self._show_error("Open a screenshot before detecting a grid.")
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
            self._log_activity(
                "Could not determine a regular grid or its dimensions automatically. "
                f"The current {self._rows.value()} rows × {self._cols.value()} columns are "
                f"unchanged (a new image starts with the {_DEFAULT_ROWS} × {_DEFAULT_COLS} "
                "defaults)."
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
            self._log_activity(
                f"The best automatic grid proposal has confidence {candidate.confidence:.2f}, "
                "so it was not applied below 0.70. The current dimensions are unchanged."
            )
            self._refresh_canvas()
            self._update_extraction_controls()
            self._update_instruction()
            return

        self._adopt_detection(candidate)
        self._log_activity(
            f"Accepted proposal 1 of {len(candidates)}: {candidate.grid.rows} rows × "
            f"{candidate.grid.cols} columns, confidence {candidate.confidence:.2f}, "
            f"RMS error {candidate.residual_rms_px:.2f} px."
        )
        self._refresh_canvas()
        self._update_extraction_controls()
        self._update_instruction()

    def _adopt_detection(self, candidate: CalibrationCandidate) -> None:
        """Make one accepted automatic proposal editable by the M1 controls."""
        self._grid = candidate.grid
        self._reference_points.clear()
        self._points = [
            candidate.grid.grid_to_pixel(-0.5, -0.5),
            candidate.grid.grid_to_pixel(-0.5, 0.5),
            candidate.grid.grid_to_pixel(0.5, -0.5),
            candidate.grid.grid_to_pixel(candidate.grid.rows - 0.5, candidate.grid.cols - 0.5),
        ]
        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(candidate.grid.rows)
        self._cols.setValue(candidate.grid.cols)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)

    def _derive_dimensions_from_outer_intersection(self) -> None:
        if self._image is None or len(self._points) != len(_POINT_LABELS):
            return
        self._clear_extraction()
        try:
            grid = grid_from_four_intersections(
                image_width=self._image.shape[1],
                image_height=self._image.shape[0],
                top_left=self._points[0],
                next_col=self._points[1],
                next_row=self._points[2],
                bottom_right=self._points[3],
                image_sha256=self._image_hash,
            )
        except ValueError as error:
            self._grid = None
            self._points.pop()
            self._log_activity(f"Could not use the outer intersection: {error}")
            self._update_extraction_controls()
            self._update_instruction()
            return

        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(grid.rows)
        self._cols.setValue(grid.cols)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)
        self._grid = grid
        self._update_extraction_controls()
        self._log_activity(
            f"Detected {grid.rows} rows × {grid.cols} columns from 4 intersections; "
            f"RMS error {grid.calibration.residual_rms_px:.2f} px."
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
                self._grid = grid_from_intersection_points(
                    rows=self._rows.value(),
                    cols=self._cols.value(),
                    image_width=self._image.shape[1],
                    image_height=self._image.shape[0],
                    intersections=self._intersection_points(),
                    image_sha256=self._image_hash,
                )
                self._log_activity(
                    f"Fitted {len(self._intersection_points())} grid intersections; "
                    f"RMS error {self._grid.calibration.residual_rms_px:.2f} px, "
                    f"max {self._grid.calibration.residual_max_px:.2f} px."
                )
            else:
                self._grid = grid_from_three_intersections(
                    rows=self._rows.value(),
                    cols=self._cols.value(),
                    image_width=self._image.shape[1],
                    image_height=self._image.shape[0],
                    top_left=self._points[0],
                    next_col=self._points[1],
                    next_row=self._points[2],
                    image_sha256=self._image_hash,
                )
        except ValueError as error:
            self._grid = None
            self._log_activity(f"Could not fit the grid: {error}")
        self._refresh_canvas()
        self._update_extraction_controls()
        self._update_instruction()

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
        has_selection = bool(self._points or self._reference_points)
        self._undo_calibration_button.setEnabled(has_selection)
        self._reset_calibration_button.setEnabled(
            bool(has_selection or self._grid or self._crops or self._detection_candidates)
        )
        if self._image is None:
            next_step = "Open a screenshot to begin."
        elif self._crops:
            incomplete = sum(
                bool(annotation_errors(annotation, self._cell_types))
                for annotation in self._cell_editor.annotations()
            )
            if incomplete:
                next_step = f"Review the {incomplete} highlighted cell(s) in the Cells workspace."
            elif self._level is None:
                next_step = "Formalize the fully reviewed cell annotations."
            else:
                next_step = "Review the formal Level view or continue to solving."
        elif len(self._points) < len(_POINT_LABELS):
            if not self._points:
                next_step = (
                    "Confirm the row and column counts above, then click the upper-left "
                    "grid intersection."
                )
            else:
                next_step = _POINT_LABELS[len(self._points)]
        elif self._grid is None:
            next_step = "Adjust the dimensions or undo the last intersection and try again."
        else:
            next_step = (
                "Review the fitted grid, optionally add another visible intersection, "
                "then extract the cells in memory."
            )
        self._instruction.setText(f"Next step: {next_step}")

    def _intersection_points(self) -> tuple[GridIntersectionPoint, ...]:
        if len(self._points) < 3:
            return ()
        points = [
            GridIntersectionPoint(row=0, col=0, pixel=self._points[0]),
            GridIntersectionPoint(row=0, col=1, pixel=self._points[1]),
            GridIntersectionPoint(row=1, col=0, pixel=self._points[2]),
        ]
        if len(self._points) == len(_POINT_LABELS):
            points.append(
                GridIntersectionPoint(
                    row=self._rows.value(),
                    col=self._cols.value(),
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
                f"{reference.row},{reference.col}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _REFERENCE_POINT_COLOR,
                2,
                cv2.LINE_AA,
            )
        self._canvas.set_image(_bgr_to_qimage(preview))

    @property
    def extracted_crops(self) -> tuple[CellCrop, ...]:
        """Return the current in-memory cells for future recognition stages."""
        return tuple(self._crops)

    def _update_extraction_controls(self) -> None:
        ready = (
            self._image is not None
            and self._grid is not None
            and len(self._points) == len(_POINT_LABELS)
            and bool(self._level_number.text().strip())
        )
        self._extract_button.setEnabled(ready)

    def _level_number_changed(self) -> None:
        if self._crops:
            self._clear_extraction()
        self._update_extraction_controls()
        self._update_semantic_controls()

    def _clear_extraction(self) -> None:
        had_extraction = bool(self._crops)
        self._crop_summary.setText(
            "Cell previews are out of date; extract them again." if had_extraction else ""
        )
        self._crop_summary.setVisible(had_extraction)
        self._crops = []
        self._crop_spec = None
        self._cell_editor.clear_cells()
        self._workspace_tabs.setTabEnabled(self._cells_tab_index, False)
        self._level_canvas.clear_image()
        self._workspace_tabs.setTabEnabled(self._level_tab_index, False)
        if self._workspace_tabs.currentIndex() in (self._cells_tab_index, self._level_tab_index):
            self._workspace_tabs.setCurrentIndex(0)
        self._level = None
        self._level_path = None
        self._rail_budget = None
        self._solver_result = None
        self._level_editor.clear()
        self._semantic_status.setText("Formalize the reviewed cells before solving.")
        self._save_solution_button.setEnabled(False)
        self._solver_trace.clear()
        self._update_semantic_controls()
        self._update_instruction()

    def _extract_cells(self) -> None:
        if self._image is None or self._grid is None:
            self._show_error("Finish a valid grid before extracting cells.")
            return
        spec = _INTERACTIVE_CROP_SPEC
        try:
            crops = CellExtractor().extract(self._image, self._grid, spec)
            annotations = self._automatic_annotations(crops)
        except (sqlite3.Error, ValueError) as error:
            self._show_error(f"Could not extract cells: {error}")
            return

        self._crops = crops
        self._crop_spec = spec
        self._cell_editor.set_cells(
            crops=crops,
            annotations=annotations,
            definitions=self._cell_types,
            columns=self._grid.cols,
        )
        self._workspace_tabs.setTabEnabled(self._cells_tab_index, True)
        self._workspace_tabs.setCurrentIndex(self._cells_tab_index)
        recognized = sum(annotation.source == "automatic" for annotation in annotations)
        self._crop_summary.setText(
            f"{recognized}/{len(crops)} recognized · {spec.output_size} × {spec.output_size} px"
        )
        self._crop_summary.show()
        self._log_activity(
            f"Extracted {len(crops)} cells; recognized {recognized} from confirmed samples."
        )
        self._update_extraction_controls()
        self._update_semantic_controls()
        self._update_instruction()

    def _automatic_annotations(
        self,
        crops: list[CellCrop],
        *,
        preserve_manual: bool = False,
    ) -> tuple[CellAnnotation, ...]:
        """Recognize cells from the local confirmed-sample library."""
        level_number = self._level_number.text().strip()
        existing = {
            (annotation.cell.row, annotation.cell.col): annotation
            for annotation in self._cell_editor.annotations()
            if preserve_manual and annotation.source == "manual"
        }
        samples = self._annotation_store.samples()
        recognizer: TemplateRecognizer | None = None
        samples_by_id = {sample.id: sample for sample in samples}
        if samples:
            catalog = TemplateCatalog(
                templates=tuple(
                    TemplateDefinition(
                        id=sample.id,
                        label=sample.type_id,
                        image=f"database/{sample.id}.png",
                    )
                    for sample in samples
                )
            )
            recognizer = TemplateRecognizer(
                catalog,
                tuple(
                    Template(definition=definition, image=sample.image)
                    for definition, sample in zip(catalog.templates, samples, strict=True)
                ),
            )

        annotations: list[CellAnnotation] = []
        for crop in crops:
            key = (crop.cell.row, crop.cell.col)
            if key in existing:
                annotations.append(existing[key])
                continue
            annotation_id = make_cell_annotation_id(level_number, crop.cell)
            if recognizer is None:
                annotations.append(CellAnnotation(id=annotation_id, cell=crop.cell))
                continue
            observation = recognizer.recognize(crop)
            if observation.status is not RecognitionStatus.ACCEPTED:
                annotations.append(CellAnnotation(id=annotation_id, cell=crop.cell))
                continue
            candidate = observation.candidates[0]
            definition = self._cell_types.get(candidate.label)
            sample = samples_by_id.get(candidate.template_id)
            if definition is None or sample is None:
                annotations.append(CellAnnotation(id=annotation_id, cell=crop.cell))
                continue
            properties = rotate_annotation_properties(
                sample.properties,
                definition,
                candidate.rotation_degrees,
            )
            annotations.append(
                CellAnnotation(
                    id=annotation_id,
                    cell=crop.cell,
                    type_id=definition.id,
                    properties=properties,
                    source="automatic",
                    confidence=observation.confidence,
                )
            )
        return tuple(annotations)

    def _cell_annotation_changed(self, annotation: CellAnnotation) -> None:
        """Persist complete manual evidence and refresh unresolved automatic matches."""
        try:
            self._annotation_store.delete_sample(annotation.id)
            if annotation.source == "manual" and not annotation_value_errors(
                annotation, self._cell_types
            ):
                crop = self._cell_editor.crop_for(annotation.cell)
                if crop is not None:
                    self._annotation_store.save_manual_sample(annotation, crop)
            self._refresh_automatic_annotations()
        except (OSError, sqlite3.Error, ValueError) as error:
            self._show_error(f"Could not update the confirmed cell sample: {error}")
            return
        self._update_annotation_summary()

    def _refresh_automatic_annotations(self) -> None:
        if not self._crops:
            return
        for annotation in self._automatic_annotations(self._crops, preserve_manual=True):
            self._cell_editor.set_annotation(annotation)

    def _update_annotation_summary(self) -> None:
        annotations = self._cell_editor.annotations()
        complete = sum(
            not annotation_errors(annotation, self._cell_types) for annotation in annotations
        )
        automatic = sum(annotation.source == "automatic" for annotation in annotations)
        self._crop_summary.setText(
            f"{complete}/{len(annotations)} ready · {automatic} recognized automatically"
        )
        self._update_semantic_controls()
        self._update_instruction()

    def _create_cell_type(self, cell: object) -> None:
        """Create a reusable type contextually and assign it to the requesting cell."""
        if not isinstance(cell, CellCoord):
            return
        display_name, accepted = QInputDialog.getText(
            self,
            "New cell type",
            "Type name",
        )
        display_name = display_name.strip()
        if not accepted or not display_name:
            return
        role_labels = {
            "Empty / buildable": "empty",
            "Obstacle / blocked": "obstacle",
            "Train / actor": "train",
            "Arrival goal": "goal",
            "New mechanic (not formalized yet)": "unmodeled",
        }
        role_label, accepted = QInputDialog.getItem(
            self,
            "Cell type role",
            "Graph-state role",
            tuple(role_labels),
            editable=False,
        )
        if not accepted:
            return
        base_id = f"custom.{_identifier_fragment(display_name)}"
        type_id = base_id
        suffix = 2
        while type_id in self._cell_types:
            type_id = f"{base_id}-{suffix}"
            suffix += 1
        definition = cell_type_for_role(
            type_id=type_id,
            display_name=display_name,
            role=cast(
                Literal["empty", "obstacle", "train", "goal", "unmodeled"],
                role_labels[role_label],
            ),
        )
        try:
            self._annotation_store.save_cell_type(definition)
        except (OSError, sqlite3.Error, ValueError) as error:
            self._show_error(f"Could not store the new cell type: {error}")
            return
        self._cell_types[definition.id] = definition
        self._cell_editor.set_definitions(self._cell_types)
        annotation = CellAnnotation(
            id=make_cell_annotation_id(self._level_number.text(), cell),
            cell=cell,
            type_id=definition.id,
            source="manual",
        )
        self._cell_editor.set_annotation(annotation)
        self._cell_annotation_changed(annotation)

    def _formalize_level(self) -> None:
        """Compile complete visual annotations and display their formal Level projection."""
        annotations = self._cell_editor.annotations()
        if not annotations or self._grid is None:
            self._show_error("Extract and review the cells before formalizing the level.")
            return
        invalid = [
            f"({item.cell.row}, {item.cell.col}): {', '.join(annotation_errors(item, self._cell_types))}"
            for item in annotations
            if annotation_errors(item, self._cell_types)
        ]
        if invalid:
            self._show_error("Complete every highlighted cell first. " + "; ".join(invalid))
            return
        rail_budget, accepted = QInputDialog.getInt(
            self,
            "Formalize level",
            "Available rails",
            self._rail_budget or 0,
            0,
            10_000,
        )
        if not accepted:
            return
        goal_assignments = self._request_goal_assignments(annotations)
        if goal_assignments is None:
            return
        try:
            level = formalize_annotations(
                level_number=self._level_number.text(),
                rows=self._grid.rows,
                cols=self._grid.cols,
                rail_budget=rail_budget,
                annotations=annotations,
                definitions=self._cell_types,
                goal_assignments=goal_assignments,
            )
        except ValueError as error:
            self._show_error(f"Could not formalize the level: {error}")
            return
        self._rail_budget = rail_budget
        self._level_path = None
        self._level_editor.setPlainText(level.model_dump_json(indent=2))
        self._set_validated_level(level, None)
        self._level_canvas.set_image(_bgr_to_qimage(render_level_schematic(level)))
        self._workspace_tabs.setTabEnabled(self._level_tab_index, True)
        self._workspace_tabs.setCurrentIndex(self._level_tab_index)
        self._level_canvas.fit_in_view()
        self._log_activity(f"Formalized {level.id!r} from {len(annotations)} reviewed cells.")
        self._update_semantic_controls()
        self._update_instruction()

    def _request_goal_assignments(
        self, annotations: tuple[CellAnnotation, ...]
    ) -> dict[str, CellCoord] | None:
        actors = [
            annotation
            for annotation in annotations
            if annotation.type_id is not None
            and any(
                isinstance(feature, ActorFeature)
                for feature in self._cell_types[annotation.type_id].features
            )
        ]
        goals = [
            annotation.cell
            for annotation in annotations
            if annotation.type_id is not None
            and any(
                isinstance(feature, GoalFeature)
                for feature in self._cell_types[annotation.type_id].features
            )
        ]
        if len(goals) <= 1:
            return {}
        choices = tuple(f"r{goal.row:03d} · c{goal.col:03d}" for goal in goals)
        assignments: dict[str, CellCoord] = {}
        for actor in actors:
            choice, accepted = QInputDialog.getItem(
                self,
                "Choose arrival goal",
                f"Goal for r{actor.cell.row:03d} · c{actor.cell.col:03d}",
                choices,
                editable=False,
            )
            if not accepted:
                return None
            assignments[actor.id] = goals[choices.index(choice)]
        return assignments

    def _show_error(self, message: str) -> None:
        self._log_activity(f"Error: {message}")
        QMessageBox.warning(self, "Railbound Solver", message)

    def _log_activity(self, message: str) -> None:
        """Append one concise event to the persistent sidebar activity log."""
        if self._activity_log.toPlainText():
            self._activity_log.appendPlainText("")
        self._activity_log.appendPlainText(message.strip())
        self._activity_log.ensureCursorVisible()


def run_calibration_gui(
    *,
    image_path: Path | None,
    automatic_detection: bool = True,
) -> int:
    """Open the primary native parse, review, and solve window."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
        window = CalibrationWindow(
            image_path=image_path,
            automatic_detection=automatic_detection,
        )
        window.show()
        return cast(int, application.exec())

    window = CalibrationWindow(
        image_path=image_path,
        automatic_detection=automatic_detection,
    )
    window.show()
    return 0


def _annotation_store_path() -> Path:
    """Return the platform-local database path used by the interactive library."""
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    base = Path(location) if location else Path.cwd() / ".railbound"
    return base / "cell-annotations.sqlite3"


def _infer_level_number(path: Path) -> str | None:
    """Extract a conventional world-level number from a screenshot filename."""
    match = re.search(r"(?<!\d)(\d+)[-_](\d+)(?!\d)", path.stem)
    return None if match is None else f"{match.group(1)}-{match.group(2)}"


def _identifier_fragment(value: str) -> str:
    """Convert a display name into a stable identifier fragment."""
    fragment = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip().lower()).strip("-.")
    return fragment or "cell-type"


def _event_summary(event: SimulationEvent) -> str:
    """Render one structured replay event as compact inspectable text."""
    actors = ", ".join(event.actor_ids) if event.actor_ids else "-"
    cell = event.target_cell or event.source_cell
    location = "" if cell is None else f" at ({cell.row}, {cell.col})"
    reason = "" if event.reason is None else f" [{event.reason.value}]"
    return (
        f"{event.sequence:04d}  tick {event.tick}  {event.kind.value}  {actors}{location}{reason}"
    )


def _topology_summary(board: CompiledBoard) -> str:
    """Render the M5 compiled graph without exposing mutable implementation details."""
    lines = [
        f"Level: {board.level_id}",
        f"Pieces: {len(board.pieces)}",
        f"Internal connections: {len(board.internal_connections)}",
        f"Adjacent connections: {len(board.adjacency_connections)}",
        f"Connected components: {len(board.connected_components)}",
        f"Boundary exits: {len(board.boundary_exits)}",
        f"Unconnected ports: {len(board.unconnected_ports)}",
    ]
    if board.pieces:
        lines.extend(("", "Pieces:"))
        for piece in board.pieces:
            if piece.source == "fixed":
                source = piece.id or "fixed"
            else:
                source = piece.piece_kind.value if piece.piece_kind is not None else "placement"
            orientation = "" if piece.orientation is None else f" {piece.orientation.value}"
            ports = ",".join(port.value for port in piece.ports)
            lines.append(
                f"  ({piece.cell.row}, {piece.cell.col}) {piece.source}: {source}{orientation} [{ports}]"
            )
    if board.unconnected_ports:
        lines.extend(("", "Unconnected ports:"))
        lines.extend(f"  {_port_node_summary(port)}" for port in board.unconnected_ports)
    if board.boundary_exits:
        lines.extend(("", "Boundary exits:"))
        lines.extend(f"  {_port_node_summary(port)}" for port in board.boundary_exits)
    return "\n".join(lines)


def _port_node_summary(port: PortNode) -> str:
    """Format one immutable graph vertex for human topology inspection."""
    return f"({port.cell.row}, {port.cell.col}) {port.port.value}"


def _bgr_to_qimage(image: ColorImage) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = width * 3
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
