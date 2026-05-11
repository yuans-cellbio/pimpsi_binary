from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from pimpsi.export import export_image_set
from pimpsi.roi import Roi


@dataclass(frozen=True)
class FakeHeader:
    n_frames: int = 1
    image_height: int = 2
    image_width: int = 3
    coherence_factor: float = 0.5


class FakeRecording:
    def __init__(self, path: Path):
        self.path = path
        self.header = FakeHeader()

    def get_intensity(self, frame_index: int):
        assert frame_index == 0
        return np.asarray([[0.0, 128.0, 512.0], [1024.0, 2048.0, 4096.0]], dtype=np.float64)

    def get_variance(self, frame_index: int):
        assert frame_index == 0
        return np.ones((2, 3), dtype=np.float64)

    def calculate_perfusion(self, frame_index: int):
        assert frame_index == 0
        return np.full((2, 3), 42.0, dtype=np.float64)


class MultiFrameRecording:
    def __init__(self, path: Path):
        self.path = path
        self.header = FakeHeader(n_frames=2)

    def get_intensity(self, frame_index: int):
        return np.full((2, 3), frame_index + 1.0, dtype=np.float64)

    def get_variance(self, frame_index: int):
        return np.full((2, 3), frame_index + 10.0, dtype=np.float64)

    def calculate_perfusion(self, frame_index: int):
        return np.full((2, 3), frame_index + 100.0, dtype=np.float64)


def test_non_color_export_preserves_float_values_without_rescaling(tmp_path):
    recording = FakeRecording(tmp_path / "sample.dat")

    [path] = export_image_set(
        recording,
        frames=[0],
        channels=["intensity"],
        output_dir=tmp_path,
        single_frame=True,
        color=False,
    )

    image = tifffile.imread(path)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image, recording.get_intensity(0).astype(np.float32))


def test_color_export_uses_explicit_mapping_limits(tmp_path):
    recording = FakeRecording(tmp_path / "sample.dat")

    [path] = export_image_set(
        recording,
        frames=[0],
        channels=["intensity"],
        output_dir=tmp_path,
        single_frame=True,
        color=True,
        colormap="viridis",
        color_limits=(0.0, 4096.0),
    )

    image = tifffile.imread(path)
    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], np.asarray([68, 1, 84], dtype=np.uint8))
    np.testing.assert_array_equal(image[-1, -1], np.asarray([253, 231, 37], dtype=np.uint8))


def test_multi_channel_stack_is_written_as_time_channel_hyperstack(tmp_path):
    recording = MultiFrameRecording(tmp_path / "sample.dat")

    [path] = export_image_set(
        recording,
        frames=[0, 1],
        channels=["intensity", "perfusion"],
        output_dir=tmp_path,
        stacked=True,
        color=False,
    )

    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "TCYX"
        assert tif.series[0].shape == (2, 2, 2, 3)
    image = tifffile.imread(path)
    assert image[0, 0, 0, 0] == 1.0
    assert image[0, 1, 0, 0] == 100.0
    assert image[1, 0, 0, 0] == 2.0
    assert image[1, 1, 0, 0] == 101.0


def test_color_export_accepts_per_channel_mapping_settings(tmp_path):
    recording = MultiFrameRecording(tmp_path / "sample.dat")

    paths = export_image_set(
        recording,
        frames=[0],
        channels=["intensity", "perfusion"],
        output_dir=tmp_path,
        single_frame=True,
        color=True,
        colormap={"intensity": "viridis", "perfusion": "inferno"},
        color_limits={"intensity": (1.0, 2.0), "perfusion": (100.0, 101.0)},
    )

    images = {path.name: tifffile.imread(path) for path in paths}
    intensity = next(image for name, image in images.items() if "_intensity_" in name)
    perfusion = next(image for name, image in images.items() if "_perfusion_" in name)
    np.testing.assert_array_equal(intensity[0, 0], np.asarray([68, 1, 84], dtype=np.uint8))
    np.testing.assert_array_equal(perfusion[0, 0], np.asarray([0, 0, 4], dtype=np.uint8))


def test_color_roi_export_draws_masked_roi_outline_and_names_file(tmp_path):
    recording = FakeRecording(tmp_path / "sample.dat")
    roi = Roi(
        id="roi_001",
        label="all",
        shape_type="rectangle",
        vertices_xy=[(0.0, 0.0), (3.0, 2.0)],
        color="#ff0000",
    )

    [path] = export_image_set(
        recording,
        frames=[0],
        channels=["intensity"],
        output_dir=tmp_path,
        single_frame=True,
        color=True,
        colormap="gray",
        color_limits=(0.0, 4096.0),
        rois=[roi],
        intensity_mask=lambda intensity: (intensity >= 128.0) & (intensity <= 2048.0),
    )

    image = tifffile.imread(path)
    assert "_masked_" in path.name
    np.testing.assert_array_equal(image[0, 0], np.asarray([230, 230, 230], dtype=np.uint8))
    np.testing.assert_array_equal(image[0, 1], np.asarray([255, 0, 0], dtype=np.uint8))
