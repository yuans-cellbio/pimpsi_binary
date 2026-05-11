from pimpsi.roi import Roi
from pimpsi.session import AnalysisSession, ProcessingProfile
from pimpsi.toi import Toi


def test_session_roundtrip(tmp_path):
    session = AnalysisSession(
        source_file="mouse01.dat",
        source_sha256="abc123",
        source_file_size=12345,
        pimsoft_binary_version=3,
        processing_profile=ProcessingProfile(perfusion_clip_upper=2500.0, spatial_downscale=4),
        rois=[
            Roi(
                id="roi_001",
                label="cortex",
                shape_type="freehand_polygon",
                vertices_xy=[(1.25, 2.5), (3.0, 2.0), (2.75, 4.5)],
                notes="primary",
            )
        ],
        tois=[Toi(id="toi_001", label="baseline", frame_start=10, frame_end=20, include_end=True)],
        frame_annotations=[{"frame": 12, "label": "stimulus"}],
        labels=["baseline"],
        exports=[{"kind": "csv", "path": "measurements.csv"}],
    )
    path = tmp_path / "recording.pimpsi.json"

    session.save(path)
    loaded = AnalysisSession.load(path)

    assert loaded == session
    assert loaded.rois[0].vertices_xy == [(1.25, 2.5), (3.0, 2.0), (2.75, 4.5)]
    assert loaded.tois[0].frame_indices() == range(10, 21)

