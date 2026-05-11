from hashlib import sha256
import struct

import numpy as np
import pytest

from pimpsi.io import PimRecording, trim_pim_binary
from pimsoft_fixtures import make_pimsoft_file


@pytest.mark.parametrize("file_version,expected_offset", [(1, 46), (2, 540), (3, 645)])
def test_recording_reads_versions_lazily_and_returns_row_major_frames(
    tmp_path,
    file_version,
    expected_offset,
):
    path = tmp_path / f"recording_v{file_version}.dat"
    variance_frames, intensity_frames = make_pimsoft_file(path, file_version=file_version)

    recording = PimRecording.open(path)

    assert isinstance(recording._data, np.memmap)
    assert recording.path == path
    assert recording.header.file_type == "PIMSOFT"
    assert recording.header.file_version == file_version
    assert recording.header.signal_gain == 10.0
    assert recording.header.coherence_factor == 0.5
    assert recording.header.number_of_images == 4
    assert recording.header.n_frames == 2
    assert recording.header.image_width == 3
    assert recording.header.image_height == 2
    assert recording.header.data_offset == expected_offset
    assert recording.header.variance_offset == expected_offset
    assert recording.header.intensity_offset == expected_offset + (2 * 3 * 2 * 8)
    assert recording.header.sha256 == sha256(path.read_bytes()).hexdigest()

    np.testing.assert_array_equal(recording.get_variance(1), variance_frames[1])
    np.testing.assert_array_equal(recording.get_intensity(1), intensity_frames[1])
    assert recording.get_intensity(1).shape == (2, 3)
    assert recording.get_intensity(1).flags.c_contiguous


def test_recording_validates_file_size(tmp_path):
    path = tmp_path / "truncated.dat"
    make_pimsoft_file(path, file_version=2)
    path.write_bytes(path.read_bytes()[:-8])

    with pytest.raises(ValueError, match="Invalid PIMSoft file size"):
        PimRecording.open(path)


def test_recording_rejects_unsupported_version(tmp_path):
    path = tmp_path / "unsupported.dat"
    make_pimsoft_file(path, file_version=1)
    data = bytearray(path.read_bytes())
    data[10:14] = struct.pack("<i", 99)
    path.write_bytes(data)

    with pytest.raises(ValueError, match="Unsupported PIMSoft file version"):
        PimRecording.open(path)


def test_recording_calculates_perfusion_for_requested_frame(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(
        path,
        file_version=1,
        variance_frames=[np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])],
        intensity_frames=[np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]])],
    )

    recording = PimRecording.open(path)
    perfusion = recording.calculate_perfusion(0, clip_upper=3000.0)

    expected = 10.0 * ((recording.get_intensity(0) / (0.5 * np.sqrt(recording.get_variance(0)))) - 1.0)
    np.testing.assert_allclose(perfusion, expected)


def test_frame_index_bounds_are_checked(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(path, file_version=1)
    recording = PimRecording.open(path)

    with pytest.raises(IndexError, match="frame_index"):
        recording.get_intensity(2)


@pytest.mark.parametrize("file_version", [1, 2, 3])
def test_trim_pim_binary_writes_selected_frames_in_original_format(tmp_path, file_version):
    path = tmp_path / f"recording_v{file_version}.dat"
    variance_frames = [
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        np.array([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]),
        np.array([[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]),
    ]
    intensity_frames = [
        np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]]),
        np.array([[200.0, 201.0, 202.0], [203.0, 204.0, 205.0]]),
        np.array([[300.0, 301.0, 302.0], [303.0, 304.0, 305.0]]),
    ]
    make_pimsoft_file(
        path,
        file_version=file_version,
        variance_frames=variance_frames,
        intensity_frames=intensity_frames,
    )
    recording = PimRecording.open(path)

    out_path = trim_pim_binary(recording, tmp_path / f"trimmed_v{file_version}.dat", [1, 2])
    trimmed = PimRecording.open(out_path)

    assert trimmed.header.file_version == file_version
    assert trimmed.header.number_of_images == 4
    assert trimmed.header.n_frames == 2
    np.testing.assert_array_equal(trimmed.get_variance(0), variance_frames[1])
    np.testing.assert_array_equal(trimmed.get_variance(1), variance_frames[2])
    np.testing.assert_array_equal(trimmed.get_intensity(0), intensity_frames[1])
    np.testing.assert_array_equal(trimmed.get_intensity(1), intensity_frames[2])
    if file_version == 3:
        assert trimmed.header.data_offset == 581 + (2 * 32)


def test_trim_pim_binary_rejects_source_overwrite(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(path)
    recording = PimRecording.open(path)

    with pytest.raises(ValueError, match="must not overwrite"):
        trim_pim_binary(recording, path, [0])
