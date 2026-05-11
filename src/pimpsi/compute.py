"""Numerical routines for PIMSoft PSI recordings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _signed_sqrt(values: ArrayLike) -> NDArray[np.float64]:
    values_array = np.asarray(values, dtype=np.float64)
    return np.sign(values_array) * np.sqrt(np.abs(values_array))


def calculate_contrast(
    variance: ArrayLike,
    intensity: ArrayLike,
    coherence_factor: float,
) -> NDArray[np.float64]:
    """Calculate sign-preserving speckle contrast from variance and intensity."""
    variance_sqrt = _signed_sqrt(variance)
    intensity_array = np.asarray(intensity, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        return (float(coherence_factor) * variance_sqrt) / intensity_array


def calculate_perfusion(
    variance: ArrayLike,
    intensity: ArrayLike,
    coherence_factor: float,
    signal_gain: float,
    clip_upper: float = 3000.0,
) -> NDArray[np.float64]:
    """Calculate perfusion while preserving the documented negative variance sign."""
    variance_sqrt = _signed_sqrt(variance)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    denominator = float(coherence_factor) * variance_sqrt

    with np.errstate(divide="ignore", invalid="ignore"):
        perfusion = float(signal_gain) * ((intensity_array / denominator) - 1.0)

    if clip_upper is not None:
        perfusion = np.minimum(perfusion, float(clip_upper))

    return perfusion


def calculate_perfusion_from_mean_intensity_variance(
    intensity_values: ArrayLike,
    variance_values: ArrayLike,
    coherence_factor: float,
    signal_gain: float,
    clip_upper: float = 3000.0,
) -> NDArray[np.float64]:
    """Average intensity and variance first, then calculate perfusion."""
    mean_intensity = np.mean(np.asarray(intensity_values, dtype=np.float64))
    mean_variance = np.mean(np.asarray(variance_values, dtype=np.float64))
    return calculate_perfusion(
        variance=mean_variance,
        intensity=mean_intensity,
        coherence_factor=coherence_factor,
        signal_gain=signal_gain,
        clip_upper=clip_upper,
    )


def block_average(image: ArrayLike, block_size: int) -> NDArray[np.float64]:
    """Average non-overlapping square blocks, dropping incomplete edge blocks."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError("image must be 2-dimensional")

    height, width = image_array.shape
    block_height = height // block_size
    block_width = width // block_size
    trimmed = image_array[: block_height * block_size, : block_width * block_size]

    if trimmed.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    return trimmed.reshape(block_height, block_size, block_width, block_size).mean(axis=(1, 3))
