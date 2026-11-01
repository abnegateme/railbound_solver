"""Qt widgets for direct visual review of extracted cell annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from railbound.cell_annotation import (
    CellAnnotation,
    CellPropertyDefinition,
    CellTypeDefinition,
    PropertyValue,
    annotation_errors,
)
from railbound.extraction import BgrImage, CellCrop
from railbound.geometry import CellCoord

_NEW_TYPE = "__new_type__"
_CARD_STYLE = """
QFrame#cellCard {
    background-color: #ffffff;
    border: 2px solid #d9d6df;
    border-radius: 10px;
}
QFrame#cellCard[annotationStatus="automatic"] {
    border-color: #4f8f5b;
}
QFrame#cellCard[annotationStatus="manual"] {
    border-color: #6750a4;
}
QFrame#cellCard[annotationStatus="incomplete"] {
    border-color: #ba1a1a;
}
QLabel#cellIdentity {
    color: #6f7480;
    font-size: 10px;
}
QLabel#cellImage {
    background-color: #1f1f1f;
    border: none;
    border-radius: 6px;
}
QLabel#propertyLabel {
    color: #49454f;
    font-size: 10px;
}
QComboBox, QLineEdit, QSpinBox {
    min-height: 26px;
}
"""


class CellAnnotationCard(QFrame):
    """One crop with inline type and instance-property editors."""

    annotation_changed = Signal(object)
    new_type_requested = Signal(object)

    def __init__(
        self,
        *,
        crop: CellCrop,
        annotation: CellAnnotation,
        definitions: Mapping[str, CellTypeDefinition],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._crop = crop
        self._annotation = annotation
        self._definitions = dict(definitions)
        self._updating = False
        self.setObjectName("cellCard")
        self.setStyleSheet(_CARD_STYLE)
        self.setMinimumWidth(154)
        self.setMaximumWidth(190)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        identity = QLabel(f"r{crop.cell.row:03d} · c{crop.cell.col:03d}")
        identity.setObjectName("cellIdentity")
        layout.addWidget(identity)
        image = QLabel()
        image.setObjectName("cellImage")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setPixmap(
            QPixmap.fromImage(_bgr_to_qimage(crop.image)).scaled(
                128,
                128,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(image)

        self._type_selector = QComboBox()
        self._type_selector.currentIndexChanged.connect(self._type_changed)
        layout.addWidget(self._type_selector)
        self._properties = QWidget()
        self._properties_layout = QVBoxLayout(self._properties)
        self._properties_layout.setContentsMargins(0, 0, 0, 0)
        self._properties_layout.setSpacing(4)
        layout.addWidget(self._properties)
        layout.addStretch()
        self._refresh()

    @property
    def annotation(self) -> CellAnnotation:
        return self._annotation

    @property
    def crop(self) -> CellCrop:
        return self._crop

    def set_definitions(self, definitions: Mapping[str, CellTypeDefinition]) -> None:
        self._definitions = dict(definitions)
        self._refresh()

    def set_annotation(self, annotation: CellAnnotation) -> None:
        if annotation.cell != self._annotation.cell:
            raise ValueError("cannot assign another cell's annotation to this card")
        self._annotation = annotation
        self._refresh()

    def _refresh(self) -> None:
        self._updating = True
        self._type_selector.clear()
        self._type_selector.addItem("Choose type…", None)
        selected_index = 0
        for definition in sorted(
            self._definitions.values(), key=lambda item: (item.display_name.casefold(), item.id)
        ):
            self._type_selector.addItem(definition.display_name, definition.id)
            if definition.id == self._annotation.type_id:
                selected_index = self._type_selector.count() - 1
        self._type_selector.insertSeparator(self._type_selector.count())
        self._type_selector.addItem("New type…", _NEW_TYPE)
        self._type_selector.setCurrentIndex(selected_index)
        self._rebuild_properties()
        self._updating = False
        self._update_status()

    def _rebuild_properties(self) -> None:
        while self._properties_layout.count():
            item = self._properties_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self._annotation.type_id is None:
            return
        definition = self._definitions.get(self._annotation.type_id)
        if definition is None:
            return
        for property_definition in definition.properties:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            label = QLabel(property_definition.label)
            label.setObjectName("propertyLabel")
            editor = self._property_editor(property_definition)
            row_layout.addWidget(label)
            row_layout.addWidget(editor, stretch=1)
            self._properties_layout.addWidget(row)

    def _property_editor(self, definition: CellPropertyDefinition) -> QWidget:
        current = self._annotation.properties.get(definition.id)
        if definition.value_kind == "direction":
            editor = QComboBox()
            editor.addItem("—", None)
            for value, label in (
                ("N", "↑ N"),
                ("E", "→ E"),
                ("S", "↓ S"),
                ("W", "← W"),
            ):
                editor.addItem(label, value)
                if current == value:
                    editor.setCurrentIndex(editor.count() - 1)
            editor.currentIndexChanged.connect(
                lambda _index, key=definition.id, field=editor: self._property_changed(
                    key, field.currentData()
                )
            )
            return editor
        if definition.value_kind == "boolean":
            editor = QCheckBox()
            editor.setChecked(bool(current))
            editor.toggled.connect(
                lambda value, key=definition.id: self._property_changed(key, value)
            )
            return editor
        if definition.value_kind == "integer":
            editor = QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            if isinstance(current, int):
                editor.setValue(current)
            editor.valueChanged.connect(
                lambda value, key=definition.id: self._property_changed(key, value)
            )
            return editor
        if definition.choices:
            editor = QComboBox()
            editor.addItem("—", None)
            for choice in definition.choices:
                editor.addItem(choice, choice)
                if current == choice:
                    editor.setCurrentIndex(editor.count() - 1)
            editor.currentIndexChanged.connect(
                lambda _index, key=definition.id, field=editor: self._property_changed(
                    key, field.currentData()
                )
            )
            return editor
        editor = QLineEdit(str(current) if current is not None else "")
        editor.textChanged.connect(
            lambda value, key=definition.id: self._property_changed(key, value)
        )
        return editor

    def _type_changed(self, _index: int) -> None:
        if self._updating:
            return
        type_id = self._type_selector.currentData()
        if type_id == _NEW_TYPE:
            self.new_type_requested.emit(self._annotation.cell)
            self._refresh()
            return
        if not isinstance(type_id, str):
            self._annotation = self._annotation.model_copy(
                update={
                    "type_id": None,
                    "properties": {},
                    "source": "manual",
                    "confidence": None,
                }
            )
        else:
            definition = self._definitions[type_id]
            properties = {
                item.id: self._annotation.properties[item.id]
                for item in definition.properties
                if item.id in self._annotation.properties
            }
            self._annotation = self._annotation.model_copy(
                update={
                    "type_id": type_id,
                    "properties": properties,
                    "source": "manual",
                    "confidence": None,
                }
            )
        self._rebuild_properties()
        self._update_status()
        self.annotation_changed.emit(self._annotation)

    def _property_changed(self, key: str, value: PropertyValue | None) -> None:
        if self._updating:
            return
        properties = dict(self._annotation.properties)
        if value in (None, ""):
            properties.pop(key, None)
        else:
            properties[key] = value
        self._annotation = self._annotation.model_copy(
            update={"properties": properties, "source": "manual", "confidence": None}
        )
        self._update_status()
        self.annotation_changed.emit(self._annotation)

    def _update_status(self) -> None:
        errors = annotation_errors(self._annotation, self._definitions)
        status = "incomplete" if errors else self._annotation.source or "incomplete"
        self.setProperty("annotationStatus", status)
        self.style().unpolish(self)
        self.style().polish(self)
        if errors:
            self.setToolTip("\n".join(errors))
        elif self._annotation.source == "automatic":
            self.setToolTip(f"Recognized automatically · {self._annotation.confidence:.0%}")
        else:
            self.setToolTip("Reviewed manually")


class CellAnnotationGrid(QScrollArea):
    """Scrollable visual grid of crop-backed annotation cards."""

    annotation_changed = Signal(object)
    new_type_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = QWidget()
        self._layout = QGridLayout(self._content)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)
        self._cards: dict[tuple[int, int], CellAnnotationCard] = {}
        self.setWidgetResizable(True)
        self.setWidget(self._content)

    def set_cells(
        self,
        *,
        crops: Sequence[CellCrop],
        annotations: Sequence[CellAnnotation],
        definitions: Mapping[str, CellTypeDefinition],
        columns: int,
    ) -> None:
        """Replace the complete row-major editor contents."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()
        by_cell = {(item.cell.row, item.cell.col): item for item in annotations}
        for index, crop in enumerate(crops):
            key = (crop.cell.row, crop.cell.col)
            card = CellAnnotationCard(
                crop=crop,
                annotation=by_cell[key],
                definitions=definitions,
            )
            card.annotation_changed.connect(self.annotation_changed.emit)
            card.new_type_requested.connect(self.new_type_requested.emit)
            self._cards[key] = card
            row, col = divmod(index, columns)
            self._layout.addWidget(card, row, col)
        self._layout.setRowStretch((len(crops) + columns - 1) // columns, 1)

    def clear_cells(self) -> None:
        self.set_cells(crops=(), annotations=(), definitions={}, columns=1)

    def annotations(self) -> tuple[CellAnnotation, ...]:
        return tuple(card.annotation for _, card in sorted(self._cards.items()))

    def crop_for(self, cell: CellCoord) -> CellCrop | None:
        card = self._cards.get((cell.row, cell.col))
        return None if card is None else card.crop

    def set_definitions(self, definitions: Mapping[str, CellTypeDefinition]) -> None:
        for card in self._cards.values():
            card.set_definitions(definitions)

    def set_annotation(self, annotation: CellAnnotation) -> None:
        card = self._cards.get((annotation.cell.row, annotation.cell.col))
        if card is not None:
            card.set_annotation(annotation)


def _bgr_to_qimage(image: BgrImage) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
