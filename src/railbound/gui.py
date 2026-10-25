"""Native Qt interface for assisted screenshot grid calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import cv2
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
    QPushButton,
    QSpinBox,
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
from railbound.rendering import ColorImage, render_grid_overlay

_POINT_COLORS = ((255, 0, 255), (0, 220, 0), (0, 0, 255), (0, 165, 255))
_REFERENCE_POINT_COLOR = (255, 255, 0)
_POINT_LABELS = (
    "1. Center of the upper-left cell (0, 0)",
    "2. Center of the cell immediately to its right (0, 1)",
    "3. Center of the cell immediately below it (1, 0)",
    "4. Center of the bottom-right cell",
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
    """Window for calibrating a grid and extracting its cells in memory."""

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

        self.setWindowTitle("Railbound Solver — Calibrate and extract")
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

        controls = QWidget()
        controls.setMinimumWidth(300)
        layout = QVBoxLayout(controls)
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
        content_layout.addWidget(controls)
        self.setCentralWidget(content)
        self._update_extraction_controls()

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
                "extract cells in memory, or save it."
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
        if self._crop_preview is not None:
            self._crop_preview.hide()

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
        QMessageBox.warning(self, "Railbound calibration", message)


def run_calibration_gui(
    *,
    image_path: Path | None,
    grid_output: Path | None,
    overlay_output: Path | None,
    automatic_detection: bool = False,
) -> int:
    """Open the native calibration window and return Qt's process exit code."""
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


def _bgr_to_qimage(image: ColorImage) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = width * 3
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
