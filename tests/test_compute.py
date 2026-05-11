import numpy as np
import pytest

from pimpsi.compute import (
    block_average,
    calculate_contrast,
    calculate_perfusion,
    calculate_perfusion_from_mean_intensity_variance,
)


def test_calculate_contrast_preserves_negative_variance_sign():
    variance = np.array([[4.0, -9.0]])
    intensity = np.array([[10.0, 10.0]])

    contrast = calculate_contrast(variance, intensity, coherence_factor=0.5)

    np.testing.assert_allclose(contrast, np.array([[0.1, -0.15]]))


def test_calculate_perfusion_uses_documented_formula_and_clips_upper_bound():
    variance = np.array([[4.0, 1.0]])
    intensity = np.array([[100.0, 1_000.0]])

    perfusion = calculate_perfusion(
        variance,
        intensity,
        coherence_factor=0.5,
        signal_gain=10.0,
        clip_upper=3000.0,
    )

    np.testing.assert_allclose(perfusion, np.array([[990.0, 3000.0]]))


def test_calculate_perfusion_allows_infinite_unclipped_values():
    perfusion = calculate_perfusion(
        variance=np.array([0.0]),
        intensity=np.array([100.0]),
        coherence_factor=0.5,
        signal_gain=10.0,
        clip_upper=None,
    )

    assert np.isposinf(perfusion[0])


def test_calculate_perfusion_keeps_negative_variance_behavior():
    perfusion = calculate_perfusion(
        variance=np.array([-4.0]),
        intensity=np.array([100.0]),
        coherence_factor=0.5,
        signal_gain=10.0,
        clip_upper=3000.0,
    )

    np.testing.assert_allclose(perfusion, np.array([-1010.0]))


def test_calculate_perfusion_from_mean_intensity_variance_averages_first():
    result = calculate_perfusion_from_mean_intensity_variance(
        intensity_values=np.array([10.0, 30.0]),
        variance_values=np.array([1.0, 9.0]),
        coherence_factor=1.0,
        signal_gain=1.0,
        clip_upper=3000.0,
    )

    assert result == pytest.approx((20.0 / np.sqrt(5.0)) - 1.0)


def test_block_average_drops_incomplete_edge_blocks():
    image = np.arange(30.0).reshape(5, 6)

    averaged = block_average(image, block_size=2)

    np.testing.assert_allclose(
        averaged,
        np.array(
            [
                [3.5, 5.5, 7.5],
                [15.5, 17.5, 19.5],
            ]
        ),
    )
