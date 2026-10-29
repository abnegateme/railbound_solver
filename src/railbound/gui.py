"""Native Qt interface for parsing screenshots and solving semantic levels."""

from __future__ import annotations

import json
from math import floor, hypot
from pathlib import Path
from typing import Literal, cast

import cv2
from pydantic import TypeAdapter, ValidationError
from PySide6.QtCore import Qt, Signal
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
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
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
    GridIntersectionPoint,
    grid_from_four_intersections,
    grid_from_intersection_points,
    grid_from_three_intersections,
    image_sha256,
)
from railbound.crop_io import write_crop_set
from railbound.detection import (
    CalibrationCandidate,
    GridDetectionHints,
    GridDetector,
    render_detection_overlay,
)
from railbound.extraction import CellCrop, CellExtractor, CropSpec, render_crop_contact_sheet
from railbound.geometry import CellCoord, GridGeometry, PixelPoint
from railbound.grid_io import save_grid
from railbound.level import Level
from railbound.level_io import save_level
from railbound.placement_io import load_placement, save_placement
from railbound.recognition import (
    LevelDraft,
    RecognitionOverride,
    RecognitionReport,
    RecognitionSource,
    RecognitionStatus,
    RecognitionThresholds,
    Template,
    TemplateCatalog,
    TemplateDefinition,
    TemplateRecognizer,
    resolve,
)
from railbound.recognition_io import (
    load_template_catalog,
    load_template_images,
    save_level_draft,
    save_recognition_overrides,
    save_recognition_report,
    save_template_catalog,
)
from railbound.rendering import ColorImage, render_grid_overlay
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
_POINT_LABELS = (
    "Click the upper-left grid intersection.",
    "Click the adjacent grid intersection to its right.",
    "Click the adjacent grid intersection below it.",
    "Click the outer bottom-right grid intersection.",
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
        self._contact_sheet: ColorImage | None = None
        self._crop_preview: CropPreviewWindow | None = None
        self._grid_output = grid_output
        self._overlay_output = overlay_output
        self._automatic_detection = automatic_detection
        self._detection_candidates: list[CalibrationCandidate] = []
        self._recognition_report: RecognitionReport | None = None
        self._level_draft: LevelDraft | None = None
        self._actor_direction_selectors: dict[tuple[int, int], QComboBox] = {}
        self._actor_goal_selectors: dict[tuple[int, int], QComboBox] = {}
        self._catalog_templates: dict[str, Template] = {}
        self._catalog_recognizer_version = "template-ncc-v1"
        self._catalog_thresholds = RecognitionThresholds()
        self._template_catalog_path: Path | None = None
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
        layout = QVBoxLayout(calibration_controls)

        dimensions = QHBoxLayout()
        dimensions.addWidget(QLabel("Rows"))
        dimensions.addWidget(self._rows)
        dimensions.addSpacing(12)
        dimensions.addWidget(QLabel("Columns"))
        dimensions.addWidget(self._cols)
        dimensions.addStretch()
        layout.addLayout(dimensions)

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
        self._export_crops_button = QPushButton("Export crops and manifest…")
        self._export_crops_button.clicked.connect(self._export_crops)
        self._crop_summary = QLabel()
        self._crop_summary.setWordWrap(True)
        layout.addWidget(self._extract_button)
        layout.addWidget(self._show_crops_button)
        layout.addWidget(self._export_crops_button)
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
        layout.addStretch()

        activity = QWidget()
        activity_layout = QVBoxLayout(activity)
        activity_layout.setContentsMargins(9, 0, 9, 9)
        activity_layout.addWidget(QLabel("Activity and next step"))
        activity_layout.addWidget(self._activity_log, stretch=1)
        activity_layout.addWidget(self._instruction)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.addWidget(self._canvas, stretch=1)
        tabs = QTabWidget()
        tabs.addTab(calibration_controls, "Grid and crops")
        tabs.addTab(self._build_review_and_solver_tab(), "Review and solve")

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
        """Build the semantic review and solver controls in the main session."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel("Template catalog"))
        self._new_catalog_button = QPushButton("New template catalog")
        self._new_catalog_button.clicked.connect(self._new_template_catalog)
        self._open_catalog_button = QPushButton("Open template catalog…")
        self._open_catalog_button.clicked.connect(self._open_template_catalog_dialog)
        self._save_catalog_button = QPushButton("Save template catalog…")
        self._save_catalog_button.clicked.connect(self._save_template_catalog_dialog)
        self._catalog_status = QLabel("Create a catalog from extracted cells or open one.")
        self._catalog_status.setWordWrap(True)
        self._template_crop_selector = QComboBox()
        self._template_id_editor = QLineEdit()
        self._template_id_editor.setPlaceholderText(
            "optional; generated from label and source crop"
        )
        self._template_label_editor = QLineEdit()
        self._template_label_editor.setPlaceholderText(
            "semantic candidate label, e.g. terrain.empty"
        )
        label_hint = QLabel(
            "For automatic level construction use terrain.empty, terrain.occupied, actor.train, "
            "and goal.arrival. Other useful labels include track.straight, track.curve, "
            "track.crossing, device.switch, device.button, device.gate, and device.tunnel. "
            "Use one visual class per label."
        )
        label_hint.setWordWrap(True)
        self._template_id_editor.textChanged.connect(self._update_semantic_controls)
        self._template_label_editor.textChanged.connect(self._update_semantic_controls)
        self._add_template_button = QPushButton("Add/update selected crop as template")
        self._add_template_button.clicked.connect(self._add_template_from_crop)
        self._remove_template_button = QPushButton("Remove selected template")
        self._remove_template_button.clicked.connect(self._remove_selected_template)
        self._template_list = QListWidget()
        self._template_list.setMinimumHeight(110)
        self._template_list.currentRowChanged.connect(self._template_selection_changed)
        template_form = QFormLayout()
        template_form.addRow("Source crop", self._template_crop_selector)
        template_form.addRow("Template ID (optional)", self._template_id_editor)
        template_form.addRow("Label", self._template_label_editor)
        layout.addWidget(self._new_catalog_button)
        layout.addWidget(self._open_catalog_button)
        layout.addWidget(self._save_catalog_button)
        layout.addWidget(self._catalog_status)
        layout.addLayout(template_form)
        layout.addWidget(label_hint)
        layout.addWidget(self._add_template_button)
        layout.addWidget(self._remove_template_button)
        layout.addWidget(self._template_list)

        layout.addWidget(QLabel("Recognition review"))
        self._recognize_button = QPushButton("Recognize extracted cells with catalog")
        self._recognize_button.clicked.connect(self._recognize_crops)
        catalog_hint = QLabel(
            "Create a catalog from selected crops or open one with a top-level `templates` list; "
            "grid.json is calibration data, not a recognition catalog."
        )
        catalog_hint.setWordWrap(True)
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
        self._save_report_button = QPushButton("Save recognition report…")
        self._save_report_button.clicked.connect(self._save_recognition_report_dialog)
        self._save_overrides_button = QPushButton("Save recognition overrides…")
        self._save_overrides_button.clicked.connect(self._save_recognition_overrides_dialog)
        self._save_draft_button = QPushButton("Save level draft…")
        self._save_draft_button.clicked.connect(self._save_level_draft_dialog)
        layout.addWidget(self._recognize_button)
        layout.addWidget(catalog_hint)
        layout.addWidget(self._recognition_summary)
        layout.addWidget(self._recognition_view)
        layout.addWidget(QLabel("Recognition overrides (JSON)"))
        layout.addWidget(self._overrides_editor)
        layout.addWidget(self._apply_overrides_button)
        layout.addWidget(self._save_report_button)
        layout.addWidget(self._save_overrides_button)
        layout.addWidget(self._save_draft_button)

        layout.addWidget(QLabel("Build semantic level from reviewed labels"))
        self._semantic_mapping_summary = QLabel(
            "Resolve recognition cells to configure a semantic level from their labels."
        )
        self._semantic_mapping_summary.setWordWrap(True)
        self._semantic_mapping_form = QFormLayout()
        self._semantic_level_id = QLineEdit("reviewed-level")
        self._semantic_level_id.setPlaceholderText("level ID")
        self._semantic_inventory = QSpinBox()
        self._semantic_inventory.setRange(0, 10_000)
        self._semantic_inventory.setValue(0)
        self._semantic_settings_form = QFormLayout()
        self._semantic_settings_form.addRow("Level ID", self._semantic_level_id)
        self._semantic_settings_form.addRow("Available rails", self._semantic_inventory)
        self._build_level_from_labels_button = QPushButton(
            "Build validated level from reviewed labels"
        )
        self._build_level_from_labels_button.clicked.connect(self._build_level_from_labels)
        layout.addWidget(self._semantic_mapping_summary)
        layout.addLayout(self._semantic_settings_form)
        layout.addLayout(self._semantic_mapping_form)
        layout.addWidget(self._build_level_from_labels_button)

        layout.addWidget(QLabel("Authoritative semantic level"))
        self._create_level_from_draft_button = QPushButton(
            "Create editable level template from reviewed recognition"
        )
        self._create_level_from_draft_button.clicked.connect(self._create_level_from_draft)
        self._load_level_button = QPushButton("Open level.json or level draft…")
        self._load_level_button.clicked.connect(self._open_level_dialog)
        self._validate_level_button = QPushButton("Validate semantic level")
        self._validate_level_button.clicked.connect(self._validate_level_editor)
        self._save_level_button = QPushButton("Save level.json…")
        self._save_level_button.clicked.connect(self._save_level_editor)
        self._level_editor = QPlainTextEdit()
        self._level_editor.setPlaceholderText(
            "Create a template from reviewed recognition, or open/paste a complete authoritative "
            "level.json here. Confirm actors, rails, inventory, and goals before validation."
        )
        self._level_editor.setMinimumHeight(220)
        self._level_editor.textChanged.connect(self._level_editor_changed)
        self._semantic_status = QLabel()
        self._semantic_status.setWordWrap(True)
        layout.addWidget(self._create_level_from_draft_button)
        layout.addWidget(self._load_level_button)
        layout.addWidget(self._validate_level_button)
        layout.addWidget(self._save_level_button)
        layout.addWidget(self._level_editor)
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

    def _new_template_catalog(self) -> None:
        """Start an empty in-memory catalog without discarding the extracted crops."""
        self._catalog_templates.clear()
        self._catalog_recognizer_version = "template-ncc-v1"
        self._catalog_thresholds = RecognitionThresholds()
        self._template_catalog_path = None
        self._template_id_editor.clear()
        self._template_label_editor.clear()
        self._refresh_template_catalog_view()
        self._catalog_status.setText(
            "New empty catalog. Select extracted crops and give each visual class a label."
        )

    def _open_template_catalog_dialog(self) -> None:
        """Load labelled reference images and their portable catalog into this session."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open template catalog",
            str(
                self._template_catalog_path.parent
                if self._template_catalog_path is not None
                else self._image_path.parent
                if self._image_path is not None
                else Path.cwd()
            ),
            "Template catalog JSON (*.json)",
        )
        if selected:
            self._load_template_catalog_from_path(Path(selected))

    def _load_template_catalog_from_path(self, path: Path) -> None:
        """Adopt one portable catalog and decoded references for in-memory recognition."""
        try:
            catalog = load_template_catalog(path)
            templates = load_template_images(catalog, path)
        except ValidationError as error:
            first_error = error.errors()[0]["msg"]
            self._show_error(
                f"{path.name} is not a valid template catalog. Choose the labelled catalog JSON "
                "with a top-level `templates` list, not grid.json, a crop manifest, or a level "
                f"file. First problem: {first_error}"
            )
            return
        except (OSError, ValueError) as error:
            self._show_error(f"Could not load template catalog: {error}")
            return

        self._catalog_templates = {template.definition.id: template for template in templates}
        self._catalog_recognizer_version = catalog.recognizer_version
        self._catalog_thresholds = catalog.thresholds
        self._template_catalog_path = path
        self._refresh_template_catalog_view()
        self._catalog_status.setText(
            f"Loaded {len(templates)} template(s) from {path.name}. "
            "You can use them immediately or add/update references from this screenshot."
        )

    def _save_template_catalog_dialog(self) -> None:
        """Choose a portable catalog path for the in-memory reference set."""
        if not self._catalog_templates:
            self._show_error("Add or open at least one template before saving a catalog.")
            return
        suggested = self._template_catalog_path or Path("catalog.json")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save template catalog",
            str(suggested),
            "Template catalog JSON (*.json)",
        )
        if selected:
            self._save_template_catalog_to_path(Path(selected).with_suffix(".json"))

    def _save_template_catalog_to_path(self, path: Path) -> None:
        """Write catalog JSON and its labelled reference images below ``path``."""
        if not self._catalog_templates:
            self._show_error("Add or open at least one template before saving a catalog.")
            return
        saved_templates: dict[str, Template] = {}
        try:
            references_directory = path.parent / "references"
            references_directory.mkdir(parents=True, exist_ok=True)
            for template in self._ordered_catalog_templates():
                definition = TemplateDefinition(
                    id=template.definition.id,
                    label=template.definition.label,
                    image=f"references/{template.definition.id}.png",
                    rotations=template.definition.rotations,
                )
                image_path = path.parent / definition.image
                if not cv2.imwrite(str(image_path), template.image):
                    raise OSError(f"could not write template image: {image_path}")
                saved_templates[definition.id] = Template(
                    definition=definition,
                    image=template.image.copy(),
                )
            catalog = TemplateCatalog(
                recognizer_version=self._catalog_recognizer_version,
                thresholds=self._catalog_thresholds,
                templates=tuple(
                    saved_templates[template_id].definition
                    for template_id in sorted(saved_templates)
                ),
            )
            save_template_catalog(catalog, path)
        except (OSError, ValueError) as error:
            self._show_error(f"Could not save template catalog: {error}")
            return

        self._catalog_templates = saved_templates
        self._template_catalog_path = path
        self._refresh_template_catalog_view()
        self._catalog_status.setText(
            f"Saved {len(saved_templates)} template(s) and catalog to {path}."
        )

    def _add_template_from_crop(self) -> None:
        """Use one calibrated in-memory crop as a labelled visual reference."""
        crop = self._selected_template_crop()
        if crop is None:
            self._show_error("Extract cells and select a source crop before adding a template.")
            return
        template_id = self._template_id_editor.text().strip() or self._suggested_template_id(crop)
        existing = self._catalog_templates.get(template_id)
        try:
            if existing is None:
                definition = TemplateDefinition(
                    id=template_id,
                    label=self._template_label_editor.text().strip(),
                    image=f"references/{template_id}.png",
                )
            else:
                definition = TemplateDefinition(
                    id=template_id,
                    label=self._template_label_editor.text().strip(),
                    image=f"references/{template_id}.png",
                    rotations=existing.definition.rotations,
                )
        except ValueError as error:
            self._show_error(f"Template ID or label is invalid: {error}")
            return

        image = crop.image.copy()
        self._catalog_templates[definition.id] = Template(definition=definition, image=image)
        self._refresh_template_catalog_view(selected_id=definition.id)
        self._catalog_status.setText(
            f"Stored crop ({crop.cell.row}, {crop.cell.col}) as template {definition.id!r}."
        )

    def _remove_selected_template(self) -> None:
        """Remove the selected in-memory reference from the pending catalog."""
        row = self._template_list.currentRow()
        template_ids = self._catalog_template_ids()
        if row < 0 or row >= len(template_ids):
            self._show_error("Select a template to remove.")
            return
        removed_id = template_ids[row]
        del self._catalog_templates[removed_id]
        self._refresh_template_catalog_view()
        self._catalog_status.setText(f"Removed template {removed_id!r} from the in-memory catalog.")

    def _template_selection_changed(self, row: int) -> None:
        """Expose the selected template fields for deliberate editing or replacement."""
        template_ids = self._catalog_template_ids()
        if row < 0 or row >= len(template_ids):
            return
        template = self._catalog_templates[template_ids[row]]
        self._template_id_editor.setText(template.definition.id)
        self._template_label_editor.setText(template.definition.label)

    def _selected_template_crop(self) -> CellCrop | None:
        """Return the crop chosen by the catalog authoring controls."""
        index = self._template_crop_selector.currentIndex()
        if index < 0 or index >= len(self._crops):
            return None
        return self._crops[index]

    def _suggested_template_id(self, crop: CellCrop) -> str:
        """Generate a readable unique ID so catalog authoring only requires a label."""
        label = self._template_label_editor.text().strip().lower()
        characters = (
            character
            if character.isascii() and (character.isalnum() or character in ".-_:")
            else "-"
            for character in label
        )
        prefix = "".join(characters).strip("-.") or "reference"
        base = f"{prefix}-r{crop.cell.row:03d}-c{crop.cell.col:03d}"
        template_id = base
        suffix = 2
        while template_id in self._catalog_templates:
            template_id = f"{base}-{suffix}"
            suffix += 1
        return template_id

    def _ordered_catalog_templates(self) -> tuple[Template, ...]:
        """Return in-memory templates in the recognizer's stable ID order."""
        return tuple(
            self._catalog_templates[template_id] for template_id in self._catalog_template_ids()
        )

    def _catalog_template_ids(self) -> list[str]:
        """Return reference IDs in the same deterministic order used for recognition."""
        return sorted(self._catalog_templates)

    def _build_template_catalog(self) -> TemplateCatalog:
        """Create the validated catalog corresponding to the current in-memory images."""
        return TemplateCatalog(
            recognizer_version=self._catalog_recognizer_version,
            thresholds=self._catalog_thresholds,
            templates=tuple(template.definition for template in self._ordered_catalog_templates()),
        )

    def _refresh_template_crop_choices(self) -> None:
        """Present every row-major extracted crop as a possible labelled reference."""
        previous_index = self._template_crop_selector.currentIndex()
        self._template_crop_selector.blockSignals(True)
        self._template_crop_selector.clear()
        for index, crop in enumerate(self._crops):
            self._template_crop_selector.addItem(f"({crop.cell.row}, {crop.cell.col})", index)
        if self._crops:
            self._template_crop_selector.setCurrentIndex(
                min(max(previous_index, 0), len(self._crops) - 1)
            )
        self._template_crop_selector.blockSignals(False)

    def _refresh_template_catalog_view(self, *, selected_id: str | None = None) -> None:
        """Render in-memory references without writing catalog files prematurely."""
        self._template_list.blockSignals(True)
        self._template_list.clear()
        template_ids = self._catalog_template_ids()
        for template_id in template_ids:
            definition = self._catalog_templates[template_id].definition
            self._template_list.addItem(f"{definition.id} — {definition.label}")
        if selected_id is not None and selected_id in template_ids:
            self._template_list.setCurrentRow(template_ids.index(selected_id))
        self._template_list.blockSignals(False)
        self._update_semantic_controls()

    def _recognize_crops(self) -> None:
        """Recognize the current in-memory crops without writing intermediate files."""
        if self._image is None or self._grid is None or self._crop_spec is None or not self._crops:
            self._show_error("Extract calibrated cells before starting recognition.")
            return
        if not self._catalog_templates:
            self._show_error(
                "Create a template catalog from extracted crops or open an existing catalog first."
            )
            return
        try:
            catalog = self._build_template_catalog()
            recognizer = TemplateRecognizer(catalog, self._ordered_catalog_templates())
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

    def _recognition_overrides(self) -> tuple[RecognitionOverride, ...] | None:
        """Parse the visible manual decisions before applying or exporting them."""
        try:
            return _OVERRIDES_ADAPTER.validate_json(
                self._overrides_editor.toPlainText().strip() or "[]"
            )
        except ValueError as error:
            self._show_error(f"Recognition overrides are invalid: {error}")
            return None

    def _apply_recognition_overrides(self) -> None:
        """Apply visible manual recognition choices to the current report."""
        if self._recognition_report is None:
            self._show_error("Run recognition before applying overrides.")
            return
        overrides = self._recognition_overrides()
        if overrides is None:
            return
        try:
            self._level_draft = resolve(self._recognition_report, overrides)
        except ValueError as error:
            self._show_error(f"Could not apply recognition overrides: {error}")
            return

        self._refresh_semantic_mapping_form()
        self._render_recognition_review(self._recognition_report, self._level_draft)
        self._update_semantic_controls()

    def _save_recognition_report_dialog(self) -> None:
        """Persist the full matching evidence from the current interactive session."""
        if self._recognition_report is None:
            self._show_error("Run recognition before saving its report.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save recognition report",
            "recognition_report.json",
            "Recognition report JSON (*.json)",
        )
        if selected:
            self._save_recognition_report_to_path(Path(selected).with_suffix(".json"))

    def _save_recognition_report_to_path(self, path: Path) -> None:
        """Write the current immutable evidence report to one explicit path."""
        if self._recognition_report is None:
            self._show_error("Run recognition before saving its report.")
            return
        try:
            save_recognition_report(self._recognition_report, path)
        except OSError as error:
            self._show_error(f"Could not save recognition report: {error}")
            return
        self._recognition_summary.setText(f"Saved recognition report to {path}.")

    def _save_recognition_overrides_dialog(self) -> None:
        """Persist visible manual decisions independently from the recognition evidence."""
        overrides = self._recognition_overrides()
        if overrides is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save recognition overrides",
            "recognition_overrides.json",
            "Recognition overrides JSON (*.json)",
        )
        if selected:
            self._save_recognition_overrides_to_path(overrides, Path(selected).with_suffix(".json"))

    def _save_recognition_overrides_to_path(
        self, overrides: tuple[RecognitionOverride, ...], path: Path
    ) -> None:
        """Write explicit recognition choices in their portable JSON representation."""
        try:
            save_recognition_overrides(overrides, path)
        except OSError as error:
            self._show_error(f"Could not save recognition overrides: {error}")
            return
        self._recognition_summary.setText(f"Saved recognition overrides to {path}.")

    def _save_level_draft_dialog(self) -> None:
        """Persist the latest review state without turning it into a semantic level."""
        if self._level_draft is None:
            self._show_error("Apply recognition overrides before saving a level draft.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save level draft",
            "level_draft.json",
            "Level draft JSON (*.json)",
        )
        if selected:
            self._save_level_draft_to_path(Path(selected).with_suffix(".json"))

    def _save_level_draft_to_path(self, path: Path) -> None:
        """Write the current review-state document for semantic authoring later."""
        if self._level_draft is None:
            self._show_error("Apply recognition overrides before saving a level draft.")
            return
        try:
            save_level_draft(self._level_draft, path)
        except OSError as error:
            self._show_error(f"Could not save level draft: {error}")
            return
        self._recognition_summary.setText(f"Saved level draft to {path}.")

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
                "Every cell has a reviewed label. Build a semantic level from the standard "
                "terrain/train/goal labels below, or use the advanced JSON template for other rules."
            )
        self._recognition_view.setPlainText("\n".join(lines))

    def _draft_cells(self, label: str) -> tuple[CellCoord, ...]:
        """Return reviewed cells with one exact recognition label in row-major order."""
        if self._level_draft is None:
            return ()
        return tuple(
            resolved.cell
            for resolved in self._level_draft.resolved_cells
            if resolved.label == label
        )

    def _refresh_semantic_mapping_form(self) -> None:
        """Show only semantic facts that recognition labels do not encode."""
        while self._semantic_mapping_form.rowCount():
            self._semantic_mapping_form.removeRow(0)
        self._actor_direction_selectors.clear()
        self._actor_goal_selectors.clear()

        if self._level_draft is None:
            self._semantic_mapping_summary.setText(
                "Resolve recognition cells to configure a semantic level from their labels."
            )
            return

        actors = self._draft_cells("actor.train")
        goals = self._draft_cells("goal.arrival")
        empty_cells = self._draft_cells("terrain.empty")
        occupied_cells = self._draft_cells("terrain.occupied")
        if not self._level_draft.is_complete:
            self._semantic_mapping_summary.setText(
                "Resolve every recognition cell before building a semantic level."
            )
            return

        self._semantic_mapping_summary.setText(
            f"Recognized {len(actors)} train(s), {len(goals)} arrival goal(s), "
            f"{len(empty_cells)} empty cell(s), and {len(occupied_cells)} blocked cell(s). "
            "Choose train directions and rail budget; labels already supply positions and terrain. "
            "With one goal, every recognized train is assigned to it."
        )
        for index, cell in enumerate(actors, start=1):
            direction = QComboBox()
            direction.addItem("Choose direction…", None)
            for code, name in (("N", "North"), ("E", "East"), ("S", "South"), ("W", "West")):
                direction.addItem(f"{name} ({code})", code)
            key = (cell.row, cell.col)
            self._actor_direction_selectors[key] = direction
            self._semantic_mapping_form.addRow(
                f"Train {index} at ({cell.row}, {cell.col}) direction", direction
            )
            if len(goals) > 1:
                goal = QComboBox()
                for goal_index, goal_cell in enumerate(goals):
                    goal.addItem(
                        f"Goal at ({goal_cell.row}, {goal_cell.col})",
                        goal_index,
                    )
                self._actor_goal_selectors[key] = goal
                self._semantic_mapping_form.addRow(f"Train {index} arrival goal", goal)
        if not actors:
            self._semantic_mapping_summary.setText(
                "No actor.train label is present. Label each starting train before building a level."
            )
        elif not goals:
            self._semantic_mapping_summary.setText(
                "No goal.arrival label is present. Label each arrival cell before building a level."
            )

    def _build_level_from_labels(self) -> None:
        """Materialize the supported reviewed-label profile as a validated level."""
        if self._level_draft is None or not self._level_draft.is_complete:
            self._show_error("Resolve every recognition cell before building a semantic level.")
            return
        actors = self._draft_cells("actor.train")
        goals = self._draft_cells("goal.arrival")
        if not actors or not goals:
            self._show_error(
                "A level built from labels requires at least one actor.train and one goal.arrival."
            )
            return
        level_id = self._semantic_level_id.text().strip()
        if not level_id:
            self._show_error("Enter a level ID before building the semantic level.")
            return

        directions: list[str] = []
        for index, cell in enumerate(actors, start=1):
            direction = self._actor_direction_selectors[(cell.row, cell.col)].currentData()
            if not isinstance(direction, str):
                self._show_error(
                    f"Choose the direction for train {index} at ({cell.row}, {cell.col})."
                )
                return
            directions.append(direction)

        actor_documents: list[dict[str, object]] = []
        arrival_documents: list[dict[str, object]] = []
        for index, (cell, direction) in enumerate(zip(actors, directions, strict=True), start=1):
            actor_id = f"train-{index}"
            actor_documents.append(
                {
                    "id": actor_id,
                    "cell": _cell_document(cell),
                    "direction": direction,
                    "actor_type": "train",
                    "number": index,
                }
            )
            goal_index = 0
            selector = self._actor_goal_selectors.get((cell.row, cell.col))
            if selector is not None:
                selected_goal = selector.currentData()
                if not isinstance(selected_goal, int):
                    self._show_error(f"Choose an arrival goal for train {index}.")
                    return
                goal_index = selected_goal
            arrival_documents.append(
                {
                    "actor_id": actor_id,
                    "target_cell": _cell_document(goals[goal_index]),
                }
            )

        terrain_documents = [
            {"cell": _cell_document(cell), "kind": label}
            for label in ("terrain.empty", "terrain.occupied")
            for cell in self._draft_cells(label)
        ]
        buildable_cells = [
            _cell_document(cell)
            for label in ("terrain.empty", "actor.train", "goal.arrival")
            for cell in self._draft_cells(label)
        ]
        document: dict[str, object] = {
            "schema_version": 1,
            "ruleset_version": "railbound-observed-v1",
            "id": level_id,
            "metadata": {"source": "reviewed-recognition-labels"},
            "rows": self._level_draft.source.rows,
            "cols": self._level_draft.source.cols,
            "terrain": terrain_documents,
            "buildable_cells": buildable_cells,
            "fixed_tracks": [],
            "actors": actor_documents,
            "switches": [],
            "buttons": [],
            "gates": [],
            "tunnel_endpoints": [],
            "rail_inventory": {"total_budget": self._semantic_inventory.value()},
            "win_condition": {
                "required_arrivals": arrival_documents,
                "arrival_order": [],
                "required_couplings": [],
            },
        }
        try:
            level = Level.model_validate(document)
            self._validate_level_grid_dimensions(level)
        except ValueError as error:
            self._show_error(f"Could not build semantic level from labels: {error}")
            return
        self._level_path = None
        self._level_editor.setPlainText(level.model_dump_json(indent=2))
        self._set_validated_level(level, None)
        self._semantic_status.setText(
            "Built and validated level from reviewed labels. Actor and goal cells are included "
            "in the M9 buildable area so the solver can place their rail segments."
        )

    def _create_level_from_draft(self) -> None:
        """Create an editable M4 document without guessing unobserved game rules."""
        if self._level_draft is None:
            self._show_error(
                "Run recognition and apply its review decisions before creating a level."
            )
            return
        if not self._level_draft.is_complete:
            self._show_error(
                "Resolve every recognition cell before creating a semantic level template."
            )
            return
        template = _semantic_level_template(self._level_draft)
        self._level_path = None
        self._level_editor.setPlainText(json.dumps(template, indent=2) + "\n")
        self._semantic_status.setText(
            "Created an editable level template. Recognition supplied only terrain labels; add "
            "actors, buildable cells, fixed-track topology/devices, inventory, and win condition "
            "before validation and saving level.json."
        )

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
        except OSError as error:
            self._show_error(f"Could not load semantic level: {error}")
            return
        try:
            level = Level.model_validate_json(raw_level)
            self._validate_level_grid_dimensions(level)
        except ValueError as error:
            try:
                draft = LevelDraft.model_validate_json(raw_level)
            except ValueError:
                self._show_error(f"Could not load semantic level: {error}")
                return
            self._level_draft = draft
            self._recognition_summary.setText(
                f"Loaded reviewed recognition draft from {path.name}. Create an editable semantic "
                "level template below; this file is not itself a level.json."
            )
            self._recognition_view.setPlainText(
                "Loaded draft labels:\n"
                + "\n".join(
                    f"({cell.cell.row}, {cell.cell.col}) {cell.label}"
                    for cell in draft.resolved_cells
                )
            )
            self._refresh_semantic_mapping_form()
            self._update_semantic_controls()
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
        """Enable each in-window stage only when its in-memory input exists."""
        has_crops = bool(self._crops)
        has_catalog = bool(self._catalog_templates)
        has_report = self._recognition_report is not None
        has_draft = self._level_draft is not None
        has_level_text = bool(self._level_editor.toPlainText().strip())
        has_placement_text = bool(self._placement_editor.toPlainText().strip())
        has_template_fields = bool(self._template_label_editor.text().strip())
        self._template_crop_selector.setEnabled(has_crops)
        self._add_template_button.setEnabled(has_crops and has_template_fields)
        self._remove_template_button.setEnabled(self._template_list.currentRow() >= 0)
        self._save_catalog_button.setEnabled(has_catalog)
        self._recognize_button.setEnabled(has_crops and has_catalog)
        self._apply_overrides_button.setEnabled(has_report)
        self._save_report_button.setEnabled(has_report)
        self._save_overrides_button.setEnabled(has_report)
        self._save_draft_button.setEnabled(has_draft)
        self._build_level_from_labels_button.setEnabled(
            self._level_draft is not None
            and self._level_draft.is_complete
            and bool(self._draft_cells("actor.train"))
            and bool(self._draft_cells("goal.arrival"))
        )
        self._create_level_from_draft_button.setEnabled(
            self._level_draft is not None and self._level_draft.is_complete
        )
        self._validate_level_button.setEnabled(has_level_text)
        self._save_level_button.setEnabled(has_level_text)
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
        if self._image is None:
            next_step = "Open a screenshot to begin."
        elif self._crops:
            if not self._catalog_templates:
                next_step = "Open Review and solve, then create or load a template catalog."
            elif self._recognition_report is None:
                next_step = "Run recognition for the extracted cells with the loaded catalog."
            elif self._level_draft is not None and not self._level_draft.is_complete:
                next_step = "Review uncertain cells and apply recognition overrides."
            else:
                next_step = "Build and validate the semantic level from the reviewed labels."
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
        self._log_activity(f"Saved {grid_path} and {overlay_path}.")

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
        self._export_crops_button.setEnabled(bool(self._crops))

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
        self._refresh_template_crop_choices()
        self._refresh_semantic_mapping_form()
        if self._crop_preview is not None:
            self._crop_preview.hide()
        self._update_semantic_controls()
        self._update_instruction()

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
        self._refresh_template_crop_choices()
        self._crop_summary.setText(
            f"Extracted {len(crops)} cells in memory ({spec.mode}, {spec.output_size} × "
            f"{spec.output_size} px). No files were written."
        )
        self._log_activity("Grid calibrated and cells extracted in memory.")
        self._update_extraction_controls()
        self._update_semantic_controls()
        self._update_instruction()
        self._show_crop_preview()

    def _export_crops(self) -> None:
        """Write the optional M2 crop artifacts for sharing or batch recognition."""
        if self._grid is None or self._crop_spec is None or not self._crops:
            self._show_error("Extract cells before exporting crop artifacts.")
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose directory for crops, manifest, and contact sheet",
            str(self._image_path.parent if self._image_path is not None else Path.cwd()),
        )
        if selected:
            self._export_crops_to_directory(Path(selected))

    def _export_crops_to_directory(self, output_directory: Path) -> None:
        """Persist the current crop set using the same M2 portable artifact format as the CLI."""
        if self._grid is None or self._crop_spec is None or self._contact_sheet is None:
            self._show_error("Extract cells before exporting crop artifacts.")
            return
        try:
            manifest = write_crop_set(
                self._crops,
                grid=self._grid,
                spec=self._crop_spec,
                output_directory=output_directory,
            )
            contact_sheet_path = output_directory / "contact_sheet.png"
            if not cv2.imwrite(str(contact_sheet_path), self._contact_sheet):
                raise OSError(f"could not write contact sheet: {contact_sheet_path}")
        except (OSError, ValueError) as error:
            self._show_error(f"Could not export crop artifacts: {error}")
            return
        self._crop_summary.setText(
            f"Exported {len(manifest.crops)} crops, manifest, and contact sheet to {output_directory}."
        )

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
    grid_output: Path | None,
    overlay_output: Path | None,
    automatic_detection: bool = True,
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


def _cell_document(cell: CellCoord) -> dict[str, int]:
    """Return one portable coordinate object for a generated level document."""
    return {"row": cell.row, "col": cell.col}


def _semantic_level_template(draft: LevelDraft) -> dict[str, object]:
    """Build an editable M4 document while retaining only unambiguous terrain labels.

    Recognition cannot establish train direction, rail ports, placeability,
    inventory, or objectives.  Those fields remain deliberately empty for the
    reviewer instead of being guessed from image evidence.
    """
    terrain: list[dict[str, object]] = []
    for resolved in draft.resolved_cells:
        if not resolved.label.startswith("terrain."):
            continue
        terrain.append(
            {
                "cell": {"row": resolved.cell.row, "col": resolved.cell.col},
                "kind": resolved.label,
            }
        )
    return {
        "schema_version": 1,
        "ruleset_version": "railbound-observed-v1",
        "id": "reviewed-level",
        "metadata": {"source": "recognition-review"},
        "rows": draft.source.rows,
        "cols": draft.source.cols,
        "terrain": terrain,
        "buildable_cells": [],
        "fixed_tracks": [],
        "actors": [],
        "switches": [],
        "buttons": [],
        "gates": [],
        "tunnel_endpoints": [],
        "rail_inventory": {"total_budget": 0},
        "win_condition": {
            "required_arrivals": [],
            "arrival_order": [],
            "required_couplings": [],
        },
    }


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
