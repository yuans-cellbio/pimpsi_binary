"""Region-of-interest geometry and mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


RoiShape = Literal["rectangle", "ellipse", "polygon", "freehand_polygon"]


@dataclass
class Roi:
    id: str
    label: str
    shape_type: RoiShape
    vertices_xy: list[tuple[float, float]]
    visible: bool = True
    locked: bool = False
    group: str | None = None
    notes: str | None = None
    color: str = "#00a6ff"

    def to_mask(self, image_shape: tuple[int, int]) -> NDArray[np.bool_]:
        """Return a 2D mask for ``image_shape=(height, width)``."""
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape dimensions must be positive")

        yy, xx = np.mgrid[:height, :width]
        x = xx + 0.5
        y = yy + 0.5

        if self.shape_type == "rectangle":
            return _rectangle_mask(x, y, self.vertices_xy)
        if self.shape_type == "ellipse":
            return _ellipse_mask(x, y, self.vertices_xy)
        if self.shape_type in {"polygon", "freehand_polygon"}:
            return _polygon_mask(x, y, self.vertices_xy)

        raise ValueError(f"Unsupported ROI shape_type: {self.shape_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "shape_type": self.shape_type,
            "vertices_xy": [[x, y] for x, y in self.vertices_xy],
            "visible": self.visible,
            "locked": self.locked,
            "group": self.group,
            "notes": self.notes,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Roi":
        return cls(
            id=data["id"],
            label=data["label"],
            shape_type=data["shape_type"],
            vertices_xy=[tuple(vertex) for vertex in data["vertices_xy"]],
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            group=data.get("group"),
            notes=data.get("notes"),
            color=data.get("color", "#00a6ff"),
        )


def _bounding_box(vertices_xy: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if len(vertices_xy) < 2:
        raise ValueError("rectangle and ellipse ROIs require at least two vertices")
    xs = [vertex[0] for vertex in vertices_xy]
    ys = [vertex[1] for vertex in vertices_xy]
    return min(xs), min(ys), max(xs), max(ys)


def _rectangle_mask(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices_xy: list[tuple[float, float]],
) -> NDArray[np.bool_]:
    left, top, right, bottom = _bounding_box(vertices_xy)
    return (x >= left) & (x < right) & (y >= top) & (y < bottom)


def _ellipse_mask(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices_xy: list[tuple[float, float]],
) -> NDArray[np.bool_]:
    left, top, right, bottom = _bounding_box(vertices_xy)
    radius_x = (right - left) / 2.0
    radius_y = (bottom - top) / 2.0
    if radius_x <= 0.0 or radius_y <= 0.0:
        return np.zeros_like(x, dtype=bool)

    center_x = left + radius_x
    center_y = top + radius_y
    return (((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2) <= 1.0


def _polygon_mask(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices_xy: list[tuple[float, float]],
) -> NDArray[np.bool_]:
    if len(vertices_xy) < 3:
        raise ValueError("polygon ROIs require at least three vertices")

    vertices = np.asarray(vertices_xy, dtype=np.float64)
    vertex_x = vertices[:, 0]
    vertex_y = vertices[:, 1]
    inside = np.zeros(x.shape, dtype=bool)

    previous = len(vertices) - 1
    for current in range(len(vertices)):
        yi = vertex_y[current]
        yj = vertex_y[previous]
        xi = vertex_x[current]
        xj = vertex_x[previous]
        crosses_y = (yi > y) != (yj > y)
        slope_x = ((xj - xi) * (y - yi) / ((yj - yi) + np.finfo(float).eps)) + xi
        inside ^= crosses_y & (x < slope_x)
        previous = current

    return inside
