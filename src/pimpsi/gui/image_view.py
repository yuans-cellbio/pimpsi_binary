"""pyqtgraph-based image viewer widgets."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg


COLORMAP_ALIASES = {
    "gray": "gray",
    "grey": "gray",
    "viridis": "viridis",
    "inferno": "inferno",
    "magma": "magma",
    "plasma": "plasma",
    "turbo": "turbo",
    "cividis": "cividis",
}

FALLBACK_COLORS = {
    "gray": [(0, 0, 0), (255, 255, 255)],
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "inferno": [(0, 0, 4), (87, 15, 109), (187, 55, 84), (249, 142, 8), (252, 255, 164)],
    "magma": [(0, 0, 4), (80, 18, 123), (182, 54, 121), (251, 136, 97), (252, 253, 191)],
    "plasma": [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
    "turbo": [(48, 18, 59), (50, 101, 255), (39, 216, 175), (245, 222, 58), (180, 4, 38)],
    "cividis": [(0, 32, 77), (40, 79, 110), (101, 120, 110), (170, 159, 101), (253, 234, 69)],
}


class ImageView(QtWidgets.QWidget):
    """A small wrapper around pyqtgraph.ImageView."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self._levels = None
        self._has_image = False
        self._image_shape = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def set_image(self, image, *, auto_levels: bool = True) -> None:
        if self._has_image:
            self._levels = self._view.getHistogramWidget().getLevels()
        should_auto_level = auto_levels and not self._has_image
        self._view.setImage(image, autoLevels=should_auto_level, autoRange=False, axes={"x": 1, "y": 0})
        if self._levels is not None:
            self._view.setLevels(*self._levels)
        self._view.view.setAspectLocked(True)
        self._image_shape = image.shape
        self._apply_image_bounds()
        self._has_image = True

    def reset_levels(self) -> None:
        self._levels = None
        self._has_image = False

    def set_colormap(self, name: str) -> None:
        mapped_name = COLORMAP_ALIASES.get(name, name)
        try:
            colormap = pg.colormap.getFromMatplotlib(mapped_name)
        except Exception:
            colors = FALLBACK_COLORS.get(mapped_name)
            if colors is None:
                colormap = pg.colormap.get(mapped_name)
            else:
                colormap = pg.ColorMap(np.linspace(0.0, 1.0, len(colors)), np.asarray(colors, dtype=np.ubyte))
        self._view.setColorMap(colormap)

    def map_scene_to_image(self, scene_pos: QtCore.QPointF) -> QtCore.QPointF:
        return self._view.imageItem.mapFromScene(scene_pos)

    def scene_contains_image(self, scene_pos: QtCore.QPointF) -> bool:
        if self._image_shape is None:
            return False
        height, width = self._image_shape
        point = self.map_scene_to_image(scene_pos)
        return 0 <= point.x() < width and 0 <= point.y() < height

    def _apply_image_bounds(self) -> None:
        if self._image_shape is None:
            return
        height, width = self._image_shape
        self._view.view.setLimits(
            xMin=0,
            xMax=width,
            yMin=0,
            yMax=height,
            minXRange=min(width, 1),
            minYRange=min(height, 1),
        )
        if not self._has_image:
            self._view.view.setRange(xRange=(0, width), yRange=(0, height), padding=0)
        self._view.view.setMouseEnabled(x=True, y=True)

    def add_item(self, item) -> None:
        self._view.view.addItem(item)

    def remove_item(self, item) -> None:
        self._view.view.removeItem(item)

    def clear(self) -> None:
        self._view.clear()
