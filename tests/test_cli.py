import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from pimpsi.io import PimRecording
from pimpsi.measure import DEFAULT_METRIC
from pimpsi.roi import Roi
from pimpsi.session import AnalysisSession
from pimpsi.toi import Toi
from pimsoft_fixtures import make_pimsoft_file


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "pimpsi", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_cli_inspect_prints_valid_metadata_json(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(path, file_version=3)

    completed = _run_cli("inspect", str(path), "--json")

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads(completed.stdout)
    assert metadata["file_version"] == 3
    assert metadata["image_width"] == 3
    assert metadata["image_height"] == 2
    assert metadata["n_frames"] == 2
    assert metadata["file_size_valid"] is True


def test_cli_inspect_prints_plain_metadata(tmp_path):
    path = tmp_path / "recording.dat"
    make_pimsoft_file(path, file_version=1)

    completed = _run_cli("inspect", str(path))

    assert completed.returncode == 0, completed.stderr
    assert "file_version: 1" in completed.stdout
    assert "file_size_valid: True" in completed.stdout


def test_cli_measure_writes_reproducible_csv_from_saved_session(tmp_path):
    variance_frames = [
        np.array([[1.0, 4.0], [9.0, 16.0]]),
        np.array([[25.0, 36.0], [49.0, 64.0]]),
    ]
    intensity_frames = [
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        np.array([[50.0, 60.0], [70.0, 80.0]]),
    ]
    recording_path = tmp_path / "recording.dat"
    make_pimsoft_file(
        recording_path,
        variance_frames=variance_frames,
        intensity_frames=intensity_frames,
        coherence_factor=1.0,
        signal_gain=1.0,
    )
    recording = PimRecording.open(recording_path)
    session = AnalysisSession.from_recording(recording)
    session.rois.append(
        Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (2.0, 2.0)])
    )
    session.tois.append(Toi(id="toi_001", label="both", frame_start=0, frame_end=2))
    session_path = tmp_path / "recording.pimpsi.json"
    session.save(session_path)
    out_path = tmp_path / "measurements.csv"

    first = _run_cli("measure", str(recording_path), "--session", str(session_path), "--out", str(out_path))
    first_csv = out_path.read_text()
    second = _run_cli("measure", str(recording_path), "--session", str(session_path), "--out", str(out_path))
    second_csv = out_path.read_text()

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_csv == second_csv
    rows = list(csv.DictReader(out_path.open()))
    assert len(rows) == 1
    assert rows[0]["metric"] == DEFAULT_METRIC
    expected = (np.concatenate([frame.ravel() for frame in intensity_frames]).mean() / np.sqrt(25.5)) - 1.0
    assert float(rows[0]["value"]) == pytest.approx(expected)
    assert rows[0]["roi_id"] == "roi_001"
    assert rows[0]["toi_id"] == "toi_001"


def test_cli_measure_applies_session_intensity_mask(tmp_path):
    variance_frames = [
        np.array([[1.0, 4.0], [9.0, 16.0]]),
        np.array([[25.0, 36.0], [49.0, 64.0]]),
    ]
    intensity_frames = [
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        np.array([[50.0, 60.0], [70.0, 80.0]]),
    ]
    recording_path = tmp_path / "recording.dat"
    make_pimsoft_file(
        recording_path,
        variance_frames=variance_frames,
        intensity_frames=intensity_frames,
        coherence_factor=1.0,
        signal_gain=1.0,
    )
    recording = PimRecording.open(recording_path)
    session = AnalysisSession.from_recording(recording)
    session.processing_profile.intensity_mask_enabled = True
    session.processing_profile.intensity_mask_lower = 30.0
    session.processing_profile.intensity_mask_upper = 70.0
    session.rois.append(
        Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (2.0, 2.0)])
    )
    session.tois.append(Toi(id="toi_001", label="both", frame_start=0, frame_end=2))
    session_path = tmp_path / "recording.pimpsi.json"
    session.save(session_path)
    out_path = tmp_path / "measurements.csv"

    completed = _run_cli("measure", str(recording_path), "--session", str(session_path), "--out", str(out_path))

    assert completed.returncode == 0, completed.stderr
    rows = list(csv.DictReader(out_path.open()))
    expected = (50.0 / np.sqrt(27.0)) - 1.0
    assert float(rows[0]["value"]) == pytest.approx(expected)
    assert rows[0]["n_pixels"] == "4"


def test_cli_measure_mask_flags_override_session(tmp_path):
    recording_path = tmp_path / "recording.dat"
    make_pimsoft_file(
        recording_path,
        variance_frames=[np.array([[1.0, 4.0], [9.0, 16.0]])],
        intensity_frames=[np.array([[10.0, 20.0], [30.0, 40.0]])],
        coherence_factor=1.0,
        signal_gain=1.0,
    )
    recording = PimRecording.open(recording_path)
    session = AnalysisSession.from_recording(recording)
    session.rois.append(
        Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (2.0, 2.0)])
    )
    session.tois.append(Toi(id="toi_001", label="frame0", frame_start=0, frame_end=1))
    session_path = tmp_path / "recording.pimpsi.json"
    session.save(session_path)
    out_path = tmp_path / "measurements.csv"

    completed = _run_cli(
        "measure",
        str(recording_path),
        "--session",
        str(session_path),
        "--out",
        str(out_path),
        "--intensity-mask-lower",
        "20",
        "--intensity-mask-upper",
        "30",
    )

    assert completed.returncode == 0, completed.stderr
    rows = list(csv.DictReader(out_path.open()))
    expected = (25.0 / np.sqrt(6.5)) - 1.0
    assert float(rows[0]["value"]) == pytest.approx(expected)


def test_cli_measure_can_write_multiple_metrics(tmp_path):
    recording_path = tmp_path / "recording.dat"
    make_pimsoft_file(recording_path)
    recording = PimRecording.open(recording_path)
    session = AnalysisSession.from_recording(recording)
    session.rois.append(
        Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (3.0, 2.0)])
    )
    session.tois.append(Toi(id="toi_001", label="both", frame_start=0, frame_end=2))
    session_path = tmp_path / "session.pimpsi.json"
    session.save(session_path)
    out_path = tmp_path / "measurements.csv"

    completed = _run_cli(
        "measure",
        str(recording_path),
        "--session",
        str(session_path),
        "--out",
        str(out_path),
        "--metric",
        DEFAULT_METRIC,
        "--metric",
        "roi_toi_mean_intensity",
    )

    assert completed.returncode == 0, completed.stderr
    rows = list(csv.DictReader(out_path.open()))
    assert [row["metric"] for row in rows] == [DEFAULT_METRIC, "roi_toi_mean_intensity"]


def test_cli_measure_rejects_source_mismatch(tmp_path):
    recording_path = tmp_path / "recording.dat"
    make_pimsoft_file(recording_path)
    recording = PimRecording.open(recording_path)
    session = AnalysisSession.from_recording(recording)
    session.source_sha256 = "not-the-recording"
    session.rois.append(
        Roi(id="roi_001", label="all", shape_type="rectangle", vertices_xy=[(0.0, 0.0), (3.0, 2.0)])
    )
    session.tois.append(Toi(id="toi_001", label="both", frame_start=0, frame_end=2))
    session_path = tmp_path / "session.pimpsi.json"
    session.save(session_path)

    completed = _run_cli(
        "measure",
        str(recording_path),
        "--session",
        str(session_path),
        "--out",
        str(tmp_path / "measurements.csv"),
    )

    assert completed.returncode == 1
    assert "source_sha256" in completed.stderr
