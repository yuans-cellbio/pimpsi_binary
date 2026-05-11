"""ROI and TOI measurement routines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pimpsi.compute import calculate_perfusion_from_mean_intensity_variance
from pimpsi.io import PimRecording
from pimpsi.roi import Roi
from pimpsi.toi import Toi


DEFAULT_METRIC = "roi_toi_perfusion_from_mean_intensity_variance"
CSV_COLUMNS = [
    "source_file",
    "source_sha256",
    "file_version",
    "roi_id",
    "roi_label",
    "toi_id",
    "toi_label",
    "frame_start",
    "frame_end",
    "metric",
    "value",
    "n_pixels",
    "n_frames",
    "coherence_factor",
    "signal_gain",
    "perfusion_clip_upper",
    "negative_variance_policy",
]
IntensityMask = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MeasurementResult:
    source_file: str
    source_sha256: str
    file_version: int
    roi_id: str
    roi_label: str
    toi_id: str | None
    toi_label: str | None
    frame_start: int
    frame_end: int
    metric: str
    value: float
    n_pixels: int
    n_frames: int
    coherence_factor: float
    signal_gain: float
    perfusion_clip_upper: float
    negative_variance_policy: str = "signed_contrast"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "file_version": self.file_version,
            "roi_id": self.roi_id,
            "roi_label": self.roi_label,
            "toi_id": self.toi_id,
            "toi_label": self.toi_label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "metric": self.metric,
            "value": self.value,
            "n_pixels": self.n_pixels,
            "n_frames": self.n_frames,
            "coherence_factor": self.coherence_factor,
            "signal_gain": self.signal_gain,
            "perfusion_clip_upper": self.perfusion_clip_upper,
            "negative_variance_policy": self.negative_variance_policy,
        }


def measure_roi_toi(
    recording: PimRecording,
    roi: Roi,
    toi: Toi,
    metric: str = DEFAULT_METRIC,
    intensity_mask: IntensityMask | None = None,
    perfusion_clip_upper: float = 3000.0,
    negative_variance_policy: str = "signed_contrast",
) -> MeasurementResult:
    """Measure an ROI over a TOI.

    The default perfusion metric averages raw intensity and variance values first,
    then calculates perfusion from those means.
    """
    frame_indices = list(toi.frame_indices())
    _validate_frame_indices(recording, frame_indices)
    mask = roi.to_mask((recording.header.image_height, recording.header.image_width))

    stats = _collect_roi_toi_stats(recording, mask, frame_indices, intensity_mask)

    if metric == DEFAULT_METRIC:
        value = float(
            calculate_perfusion_from_mean_intensity_variance(
                intensity_values=stats["intensity_sum"] / stats["valid_count"],
                variance_values=stats["variance_sum"] / stats["valid_count"],
                coherence_factor=recording.header.coherence_factor,
                signal_gain=recording.header.signal_gain,
                clip_upper=perfusion_clip_upper,
            )
        )
    elif metric == "roi_toi_mean_intensity":
        value = stats["intensity_sum"] / stats["valid_count"]
    elif metric == "roi_toi_mean_variance":
        value = stats["variance_sum"] / stats["valid_count"]
    elif metric == "roi_area_pixels":
        value = float(mask.sum())
    elif metric == "roi_valid_pixel_count":
        value = float(stats["valid_count"])
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    return _result(
        recording=recording,
        roi=roi,
        toi=toi,
        frame_start=frame_indices[0],
        frame_end=frame_indices[-1],
        metric=metric,
        value=float(value),
        n_pixels=int(mask.sum()),
        n_frames=len(frame_indices),
        perfusion_clip_upper=perfusion_clip_upper,
        negative_variance_policy=negative_variance_policy,
    )


def measure_roi_per_frame(
    recording: PimRecording,
    roi: Roi,
    frames: range | list[int] | None = None,
    intensity_mask: IntensityMask | None = None,
    perfusion_clip_upper: float = 3000.0,
    negative_variance_policy: str = "signed_contrast",
) -> list[MeasurementResult]:
    """Calculate mean-first ROI perfusion independently for each frame."""
    if frames is None:
        frame_indices = list(range(recording.header.n_frames))
    else:
        frame_indices = list(frames)
    _validate_frame_indices(recording, frame_indices)

    mask = roi.to_mask((recording.header.image_height, recording.header.image_width))
    results = []
    for frame_index in frame_indices:
        stats = _collect_roi_toi_stats(recording, mask, [frame_index], intensity_mask)
        value = float(
            calculate_perfusion_from_mean_intensity_variance(
                intensity_values=stats["intensity_sum"] / stats["valid_count"],
                variance_values=stats["variance_sum"] / stats["valid_count"],
                coherence_factor=recording.header.coherence_factor,
                signal_gain=recording.header.signal_gain,
                clip_upper=perfusion_clip_upper,
            )
        )
        results.append(
            _result(
                recording=recording,
                roi=roi,
                toi=None,
                frame_start=frame_index,
                frame_end=frame_index,
                metric="roi_perfusion_per_frame",
                value=value,
                n_pixels=int(mask.sum()),
                n_frames=1,
                perfusion_clip_upper=perfusion_clip_upper,
                negative_variance_policy=negative_variance_policy,
            )
        )
    return results


def _collect_roi_toi_stats(
    recording: PimRecording,
    roi_mask: np.ndarray,
    frame_indices: list[int],
    intensity_mask: IntensityMask | None,
) -> dict[str, float]:
    intensity_sum = 0.0
    variance_sum = 0.0
    valid_count = 0

    for frame_index in frame_indices:
        intensity = recording.get_intensity(frame_index)
        variance = recording.get_variance(frame_index)
        valid_mask = roi_mask
        if intensity_mask is not None:
            mask_array = np.asarray(intensity_mask(intensity), dtype=bool)
            if mask_array.shape != intensity.shape:
                raise ValueError(
                    f"intensity_mask returned shape {mask_array.shape}, expected {intensity.shape}"
                )
            valid_mask = roi_mask & mask_array

        frame_count = int(valid_mask.sum())
        if frame_count == 0:
            continue
        intensity_sum += float(intensity[valid_mask].sum())
        variance_sum += float(variance[valid_mask].sum())
        valid_count += frame_count

    if valid_count == 0:
        raise ValueError("ROI/TOI selection contains no valid pixels")

    return {
        "intensity_sum": intensity_sum,
        "variance_sum": variance_sum,
        "valid_count": float(valid_count),
    }


def _validate_frame_indices(recording: PimRecording, frame_indices: list[int]) -> None:
    if not frame_indices:
        raise ValueError("at least one frame is required")
    for frame_index in frame_indices:
        if not 0 <= frame_index < recording.header.n_frames:
            raise IndexError(f"frame index {frame_index} is outside the recording")


def _result(
    recording: PimRecording,
    roi: Roi,
    toi: Toi | None,
    frame_start: int,
    frame_end: int,
    metric: str,
    value: float,
    n_pixels: int,
    n_frames: int,
    perfusion_clip_upper: float,
    negative_variance_policy: str = "signed_contrast",
) -> MeasurementResult:
    return MeasurementResult(
        source_file=str(recording.path),
        source_sha256=recording.header.sha256,
        file_version=recording.header.file_version,
        roi_id=roi.id,
        roi_label=roi.label,
        toi_id=toi.id if toi is not None else None,
        toi_label=toi.label if toi is not None else None,
        frame_start=frame_start,
        frame_end=frame_end,
        metric=metric,
        value=value,
        n_pixels=n_pixels,
        n_frames=n_frames,
        coherence_factor=recording.header.coherence_factor,
        signal_gain=recording.header.signal_gain,
        perfusion_clip_upper=perfusion_clip_upper,
        negative_variance_policy=negative_variance_policy,
    )
