"""Native Qt interface for assisted screenshot grid calibration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
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


class CalibrationWindow(QMainWindow):
    """Window for choosing four cell centers and saving a calibrated grid."""

    def __init__(
        self,
        *,
        image_path: Path | None = None,
        grid_output: Path | None = None,
        overlay_output: Path | None = None,
    ) -> None:
        super().__init__()
        self._image: ColorImage | None = None
        self._image_path: Path | None = None
        self._image_hash: str | None = None
        self._points: list[PixelPoint] = []
        self._reference_points: list[GridCalibrationPoint] = []
        self._grid: GridGeometry | None = None
        self._grid_output = grid_output
        self._overlay_output = overlay_output

        self.setWindowTitle("Railbound Solver — Calibrate grid")
        self.resize(1240, 820)
        self._canvas = CalibrationCanvas()
        self._canvas.point_selected.connect(self._select_point)
        self._rows = QSpinBox()
        self._cols = QSpinBox()
        for control in (self._rows, self._cols):
            control.setRange(2, 200)
            control.valueChanged.connect(self._update_grid_from_dimensions)
        self._rows.setValue(8)
        self._cols.setValue(10)
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
        self._update_grid_from_dimensions()

    def _derive_dimensions_from_last_cell(self) -> None:
        if self._image is None or len(self._points) != len(_POINT_LABELS):
            return
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
            return

        self._rows.blockSignals(True)
        self._cols.blockSignals(True)
        self._rows.setValue(grid.rows)
        self._cols.setValue(grid.cols)
        self._rows.blockSignals(False)
        self._cols.blockSignals(False)
        self._grid = grid
        self._status.setText(
            f"Detected {grid.rows} rows × {grid.cols} columns from 4 references; "
            f"RMS error {grid.calibration.residual_rms_px:.2f} px. "
            "Click any labeled cell center to add another reference."
        )

    def _update_grid_from_dimensions(self) -> None:
        if self._image is None or len(self._points) < 3:
            self._grid = None
            self._refresh_canvas()
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

    def _undo_last_point(self) -> None:
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
        self._grid = None
        self._refresh_canvas()
        self._update_instruction()

    def _update_instruction(self) -> None:
        if self._image is None:
            self._instruction.setText("Open a screenshot to begin.")
        elif len(self._points) < len(_POINT_LABELS):
            self._instruction.setText(_POINT_LABELS[len(self._points)])
        else:
            self._instruction.setText(
                "Grid is fitted. Click any labeled cell center to add a reference point, "
                "or save it."
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
) -> int:
    """Open the native calibration window and return Qt's process exit code."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
        window = CalibrationWindow(
            image_path=image_path,
            grid_output=grid_output,
            overlay_output=overlay_output,
        )
        window.show()
        return cast(int, application.exec())

    window = CalibrationWindow(
        image_path=image_path,
        grid_output=grid_output,
        overlay_output=overlay_output,
    )
    window.show()
    return 0


def _bgr_to_qimage(image: ColorImage) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = width * 3
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
