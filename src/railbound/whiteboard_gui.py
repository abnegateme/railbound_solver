"""Native sketch-style editor shell for composing Railbound levels."""

from __future__ import annotations

from math import sin
from typing import cast

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from railbound.geometry import CellCoord
from railbound.whiteboard import BoardDraft

_MAX_GRID_SIZE = 64
_BOARD_INSET = 48.0
_CHALK = QColor("#d9e4d9")
_CHALK_MUTED = QColor("#91a79b")
_ACCENT = QColor("#efb84a")
_WINDOW_STYLE = """
QMainWindow, QWidget#windowRoot {
    background: #111815;
    color: #edf2ed;
}
QFrame#header {
    background: #18231f;
    border-bottom: 1px solid #2c3b35;
}
QLabel#brand {
    color: #f5f2e8;
    font-size: 20px;
    font-weight: 700;
}
QLabel#subtitle, QLabel#fieldLabel, QLabel#boardMeta {
    color: #91a79b;
}
QLabel#fieldLabel {
    font-size: 11px;
    font-weight: 600;
}
QLabel#gridBadge {
    padding: 7px 12px;
    color: #dce8df;
    background: #23332d;
    border: 1px solid #34483f;
    border-radius: 14px;
    font-weight: 600;
}
QSpinBox {
    min-width: 72px;
    min-height: 34px;
    padding: 0 8px;
    color: #edf2ed;
    background: #101814;
    border: 1px solid #3b5148;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    selection-background-color: #bf8b2d;
}
QSpinBox:hover {
    border-color: #718a7d;
}
QSpinBox:focus {
    border: 2px solid #efb84a;
}
QPushButton#calculateButton {
    min-height: 38px;
    padding: 0 22px;
    color: #172019;
    background: #efb84a;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}
QPushButton#calculateButton:hover {
    background: #f4c665;
}
QPushButton#calculateButton:disabled {
    color: #708078;
    background: #2a3933;
}
QFrame#canvasFrame {
    background: #1d352d;
    border: 1px solid #344b42;
    border-radius: 14px;
}
QLabel#hint {
    padding: 7px 12px;
    color: #aabbb0;
    background: rgba(14, 28, 23, 178);
    border: 1px solid #40584d;
    border-radius: 10px;
}
QStatusBar {
    color: #91a79b;
    background: #111815;
    border-top: 1px solid #23332d;
}
"""


class SketchBoard(QWidget):
    """Responsive chalkboard canvas with deterministic hand-drawn grid lines."""

    hovered_cell_changed = Signal(object)

    def __init__(self, board: BoardDraft, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board = board
        self._hovered_cell: CellCoord | None = None
        self._grid_rect = QRectF()
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(900, 660)

    def board_resized(self) -> None:
        """Refresh the geometry after the draft dimensions change."""
        self._hovered_cell = None
        self.hovered_cell_changed.emit(None)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_board(painter)
        self._grid_rect = self._fitted_grid_rect()
        if self._grid_rect.isEmpty():
            return
        self._paint_hover(painter)
        self._paint_grid(painter)
        self._paint_labels(painter)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cell = self._cell_at(event.position())
        if cell == self._hovered_cell:
            return
        self._hovered_cell = cell
        self.hovered_cell_changed.emit(cell)
        self.update()

    def leaveEvent(self, event: QEvent) -> None:
        del event
        if self._hovered_cell is not None:
            self._hovered_cell = None
            self.hovered_cell_changed.emit(None)
            self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._grid_rect = self._fitted_grid_rect()

    def _paint_board(self, painter: QPainter) -> None:
        gradient = QLinearGradient(0.0, 0.0, float(width := self.width()), float(self.height()))
        gradient.setColorAt(0.0, QColor("#23483b"))
        gradient.setColorAt(0.52, QColor("#1c3c32"))
        gradient.setColorAt(1.0, QColor("#173128"))
        painter.fillRect(self.rect(), gradient)

        painter.setPen(QPen(QColor(225, 237, 226, 9), 1.0))
        height = self.height()
        for index in range(max(40, width // 13)):
            x = (index * 97 + 31) % max(width, 1)
            y = (index * 53 + 17) % max(height, 1)
            length = 5 + index % 19
            painter.drawLine(x, y, min(width, x + length), y + (index % 3) - 1)

    def _paint_grid(self, painter: QPainter) -> None:
        rect = self._grid_rect
        cell_width = rect.width() / self._board.cols
        cell_height = rect.height() / self._board.rows
        for col in range(self._board.cols + 1):
            x = rect.left() + col * cell_width
            self._draw_sketch_line(
                painter,
                QPointF(x, rect.top()),
                QPointF(x, rect.bottom()),
                seed=col * 19 + 7,
                boundary=col in (0, self._board.cols),
            )
        for row in range(self._board.rows + 1):
            y = rect.top() + row * cell_height
            self._draw_sketch_line(
                painter,
                QPointF(rect.left(), y),
                QPointF(rect.right(), y),
                seed=row * 23 + 101,
                boundary=row in (0, self._board.rows),
            )

    def _draw_sketch_line(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        *,
        seed: int,
        boundary: bool,
    ) -> None:
        color = QColor(_CHALK)
        color.setAlpha(180 if boundary else 92)
        painter.setPen(QPen(color, 1.65 if boundary else 1.05, Qt.PenStyle.SolidLine))
        painter.drawPath(self._wobbled_path(start, end, seed, 0.85))

        ghost = QColor(_CHALK)
        ghost.setAlpha(45 if boundary else 24)
        painter.setPen(QPen(ghost, 0.8))
        painter.drawPath(self._wobbled_path(start, end, seed + 43, 1.7))

    @staticmethod
    def _wobbled_path(start: QPointF, end: QPointF, seed: int, amplitude: float) -> QPainterPath:
        path = QPainterPath(start)
        horizontal = abs(end.x() - start.x()) >= abs(end.y() - start.y())
        for step in range(1, 13):
            progress = step / 12
            x = start.x() + (end.x() - start.x()) * progress
            y = start.y() + (end.y() - start.y()) * progress
            offset = sin(seed * 0.73 + step * 1.81) * amplitude
            if horizontal:
                y += offset
            else:
                x += offset
            path.lineTo(x, y)
        return path

    def _paint_hover(self, painter: QPainter) -> None:
        if self._hovered_cell is None:
            return
        cell_width = self._grid_rect.width() / self._board.cols
        cell_height = self._grid_rect.height() / self._board.rows
        highlight = QRectF(
            self._grid_rect.left() + self._hovered_cell.col * cell_width + 2.0,
            self._grid_rect.top() + self._hovered_cell.row * cell_height + 2.0,
            max(0.0, cell_width - 4.0),
            max(0.0, cell_height - 4.0),
        )
        color = QColor(_ACCENT)
        color.setAlpha(35)
        painter.fillRect(highlight, color)

    def _paint_labels(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(_CHALK_MUTED)
        cell_width = self._grid_rect.width() / self._board.cols
        cell_height = self._grid_rect.height() / self._board.rows
        if cell_width >= 28:
            for col in range(self._board.cols):
                area = QRectF(
                    self._grid_rect.left() + col * cell_width,
                    self._grid_rect.top() - 27,
                    cell_width,
                    18,
                )
                painter.drawText(area, Qt.AlignmentFlag.AlignCenter, str(col + 1))
        if cell_height >= 24:
            for row in range(self._board.rows):
                area = QRectF(
                    self._grid_rect.left() - 35,
                    self._grid_rect.top() + row * cell_height,
                    24,
                    cell_height,
                )
                painter.drawText(area, Qt.AlignmentFlag.AlignCenter, str(row + 1))

    def _fitted_grid_rect(self) -> QRectF:
        available_width = max(0.0, self.width() - _BOARD_INSET * 2)
        available_height = max(0.0, self.height() - _BOARD_INSET * 2)
        if available_width == 0.0 or available_height == 0.0:
            return QRectF()
        cell_size = min(available_width / self._board.cols, available_height / self._board.rows)
        width = cell_size * self._board.cols
        height = cell_size * self._board.rows
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _cell_at(self, point: QPointF) -> CellCoord | None:
        if not self._grid_rect.contains(point):
            return None
        col = min(
            int((point.x() - self._grid_rect.left()) * self._board.cols / self._grid_rect.width()),
            self._board.cols - 1,
        )
        row = min(
            int((point.y() - self._grid_rect.top()) * self._board.rows / self._grid_rect.height()),
            self._board.rows - 1,
        )
        return CellCoord(row=row, col=col)


class WhiteboardWindow(QMainWindow):
    """Top-level level editor window."""

    def __init__(self) -> None:
        super().__init__()
        self._board = BoardDraft()
        self.setWindowTitle("Railbound Whiteboard")
        self.resize(1280, 820)
        self.setMinimumSize(800, 600)
        self.setStyleSheet(_WINDOW_STYLE)

        root = QWidget()
        root.setObjectName("windowRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_workspace(), stretch=1)
        self.setCentralWidget(root)
        self._sync_controls()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(14)

        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        brand = QLabel("RAILBOUND  /  WHITEBOARD")
        brand.setObjectName("brand")
        subtitle = QLabel("Compose a level, then calculate its route")
        subtitle.setObjectName("subtitle")
        title_column.addWidget(brand)
        title_column.addWidget(subtitle)
        layout.addLayout(title_column)
        layout.addStretch()

        self._grid_badge = QLabel()
        self._grid_badge.setObjectName("gridBadge")
        layout.addWidget(self._grid_badge)

        calculate = QPushButton("Calculate")
        calculate.setObjectName("calculateButton")
        calculate.setEnabled(False)
        calculate.setToolTip("Place level primitives before calculating a solution")
        layout.addWidget(calculate)
        return header

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self._label("ROWS"))
        self._rows_input = self._dimension_input(self._board.rows)
        controls.addWidget(self._rows_input)
        controls.addSpacing(8)
        controls.addWidget(self._label("COLUMNS"))
        self._cols_input = self._dimension_input(self._board.cols)
        controls.addWidget(self._cols_input)
        controls.addStretch()
        self._board_meta = QLabel()
        self._board_meta.setObjectName("boardMeta")
        controls.addWidget(self._board_meta)
        layout.addLayout(controls)

        canvas_frame = QFrame()
        canvas_frame.setObjectName("canvasFrame")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(1, 1, 1, 1)
        self._canvas = SketchBoard(self._board)
        canvas_layout.addWidget(self._canvas)

        hint = QLabel("Primitive palette is the next step  ·  Hover a cell to inspect coordinates")
        hint.setObjectName("hint")
        hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hint.setParent(self._canvas)
        hint.adjustSize()
        hint.move(18, 18)
        self._hint = hint
        layout.addWidget(canvas_frame, stretch=1)

        self._rows_input.valueChanged.connect(self._resize_board)
        self._cols_input.valueChanged.connect(self._resize_board)
        self._canvas.hovered_cell_changed.connect(self._hover_changed)
        return workspace

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _dimension_input(value: int) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setRange(1, _MAX_GRID_SIZE)
        spin_box.setValue(value)
        spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return spin_box

    def _resize_board(self) -> None:
        try:
            self._board.resize(self._rows_input.value(), self._cols_input.value())
        except ValueError:
            self._sync_controls()
            return
        self._sync_controls()
        self._canvas.board_resized()

    def _sync_controls(self) -> None:
        self._rows_input.blockSignals(True)
        self._cols_input.blockSignals(True)
        self._rows_input.setMinimum(self._board.minimum_rows)
        self._cols_input.setMinimum(self._board.minimum_cols)
        self._rows_input.setValue(self._board.rows)
        self._cols_input.setValue(self._board.cols)
        self._rows_input.blockSignals(False)
        self._cols_input.blockSignals(False)
        self._grid_badge.setText(f"{self._board.rows} × {self._board.cols} GRID")
        self._board_meta.setText(
            f"{len(self._board.occupied_cells)} objects  ·  "
            f"{self._board.rows * self._board.cols} cells"
        )

    def _hover_changed(self, cell: CellCoord | None) -> None:
        if cell is None:
            self.statusBar().clearMessage()
            return
        self.statusBar().showMessage(f"Row {cell.row + 1}, column {cell.col + 1}")


def main() -> int:
    """Open the standalone whiteboard application."""
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication([])
    window = WhiteboardWindow()
    window.show()
    if not owns_application:
        return 0
    return cast(int, application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
