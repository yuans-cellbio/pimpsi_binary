"""Small GUI-facing helpers that keep recording access lazy."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pimpsi.compute import calculate_contrast
from pimpsi.io import PimRecording


DisplayMode = Literal["intensity", "variance", "contrast", "perfusion"]


@dataclass(frozen=True)
class FrameRequest:
    mode: DisplayMode
    frame_index: int


class LazyFrameProvider:
    """Return only the requested display frame and keep a small recent-frame cache."""

    def __init__(self, recording: PimRecording, *, cache_size: int = 8, perfusion_clip_upper: float = 3000.0):
        self.recording = recording
        self.cache_size = cache_size
        self.perfusion_clip_upper = perfusion_clip_upper
        self._cache: OrderedDict[FrameRequest, NDArray[np.float64]] = OrderedDict()

    def frame(self, mode: DisplayMode, frame_index: int) -> NDArray[np.float64]:
        request = FrameRequest(mode, frame_index)
        if request in self._cache:
            self._cache.move_to_end(request)
            return self._cache[request]

        image = self._load_frame(mode, frame_index)
        self._cache[request] = image
        self._cache.move_to_end(request)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image

    def clear(self) -> None:
        self._cache.clear()

    def _load_frame(self, mode: DisplayMode, frame_index: int) -> NDArray[np.float64]:
        if mode == "intensity":
            return self.recording.get_intensity(frame_index)
        if mode == "variance":
            return self.recording.get_variance(frame_index)
        if mode == "contrast":
            return calculate_contrast(
                variance=self.recording.get_variance(frame_index),
                intensity=self.recording.get_intensity(frame_index),
                coherence_factor=self.recording.header.coherence_factor,
            )
        if mode == "perfusion":
            return self.recording.calculate_perfusion(
                frame_index,
                clip_upper=self.perfusion_clip_upper,
            )
        raise ValueError(f"Unsupported display mode: {mode!r}")

