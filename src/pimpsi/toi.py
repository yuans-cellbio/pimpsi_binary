"""Time-of-interest models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Toi:
    id: str
    label: str
    frame_start: int
    frame_end: int
    include_end: bool = False
    notes: str | None = None
    color: str = "#f2c94c"
    visible: bool = True

    def frame_indices(self) -> range:
        stop = self.frame_end + 1 if self.include_end else self.frame_end
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if stop <= self.frame_start:
            raise ValueError("TOI must include at least one frame")
        return range(self.frame_start, stop)

    def n_frames(self) -> int:
        return len(self.frame_indices())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "include_end": self.include_end,
            "notes": self.notes,
            "color": self.color,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Toi":
        return cls(
            id=data["id"],
            label=data["label"],
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            include_end=data.get("include_end", False),
            notes=data.get("notes"),
            color=data.get("color", "#f2c94c"),
            visible=data.get("visible", True),
        )
