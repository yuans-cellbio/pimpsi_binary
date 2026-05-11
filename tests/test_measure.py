import numpy as np
import pytest

from pimpsi.compute import calculate_perfusion
from pimpsi.io import PimRecording
from pimpsi.measure import DEFAULT_METRIC, measure_roi_per_frame, measure_roi_toi
from pimpsi.roi import Roi
from pimpsi.toi import Toi
from pimsoft_fixtures import make_pimsoft_file


def test_roi_toi_measurement_defaults_to_mean_intensity_variance_then_perfusion(tmp_path):
    variance_frames = [
        np.array([[1.0, 100.0], [9.0, 25.0]]),
        np.array([[16.0, 36.0], [49.0, 64.0]]),
    ]
    intensity_frames = [
        np.array([[10.0, 100.0], [30.0, 50.0]]),
        np.array([[40.0, 60.0], [70.0, 80.0]]),
    ]
    path = tmp_path / "recording.dat"
    make_pimsoft_file(
        path,
        variance_frames=variance_frames,
        intensity_frames=intensity_frames,
        coherence_factor=1.0,
        signal_gain=1.0,
    )
    recording = PimRecording.open(path)
    roi = Roi(
        id="roi_001",
        label="all",
        shape_type="rectangle",
        vertices_xy=[(0.0, 0.0), (2.0, 2.0)],
    )
    toi = Toi(id="toi_001", label="both", frame_start=0, frame_end=2)

    result = measure_roi_toi(recording, roi, toi)

    all_intensity = np.concatenate([frame.ravel() for frame in intensity_frames])
    all_variance = np.concatenate([frame.ravel() for frame in variance_frames])
    expected_default = (all_intensity.mean() / np.sqrt(all_variance.mean())) - 1.0
    pixelwise_perfusion_mean = np.mean(
        [
            calculate_perfusion(
                variance=variance_frames[index],
                intensity=intensity_frames[index],
                coherence_factor=1.0,
                signal_gain=1.0,
                clip_upper=3000.0,
            ).mean()
            for index in range(2)
        ]
    )

    assert result.metric == DEFAULT_METRIC
    assert result.value == pytest.approx(expected_default)
    assert result.value != pytest.approx(pixelwise_perfusion_mean)
    assert result.n_pixels == 4
    assert result.n_frames == 2


def test_measurement_metrics_and_provenance(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(
        path,
        variance_frames=[np.array([[1.0, 4.0], [9.0, 16.0]])],
        intensity_frames=[np.array([[10.0, 20.0], [30.0, 40.0]])],
        file_version=2,
    )
    recording = PimRecording.open(path)
    roi = Roi(id="roi_001", label="top", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (2.0, 1.0)])
    toi = Toi(id="toi_001", label="frame0", frame_start=0, frame_end=0, include_end=True)

    result = measure_roi_toi(recording, roi, toi, metric="roi_toi_mean_intensity")

    assert result.value == pytest.approx(15.0)
    assert result.to_dict()["source_sha256"] == recording.header.sha256
    assert result.to_dict()["file_version"] == 2


def test_roi_per_frame_metric_uses_each_frame_independently(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(path)
    recording = PimRecording.open(path)
    roi = Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (3.0, 2.0)])

    results = measure_roi_per_frame(recording, roi, frames=range(2))

    assert [result.frame_start for result in results] == [0, 1]
    assert [result.metric for result in results] == ["roi_perfusion_per_frame", "roi_perfusion_per_frame"]


def test_measurement_can_exclude_pixels_with_intensity_mask(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(
        path,
        variance_frames=[np.array([[1.0, 4.0], [9.0, 16.0]])],
        intensity_frames=[np.array([[10.0, 20.0], [30.0, 40.0]])],
    )
    recording = PimRecording.open(path)
    roi = Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (2.0, 2.0)])
    toi = Toi(id="toi_001", label="frame0", frame_start=0, frame_end=1)

    result = measure_roi_toi(
        recording,
        roi,
        toi,
        metric="roi_valid_pixel_count",
        intensity_mask=lambda intensity: intensity > 20.0,
    )

    assert result.value == 2.0

