"""Command-line interface for headless PIMSoft PSI workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from pimpsi.io import PimRecording
from pimpsi.export import write_measurement_csv
from pimpsi.measure import DEFAULT_METRIC, MeasurementResult, measure_roi_toi
from pimpsi.session import AnalysisSession


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as exc:
        print(f"pimpsi: error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pimpsi",
        description="Inspect and measure PIMSoft PSI binary recordings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a PIMSoft binary file")
    inspect_parser.add_argument("recording", type=Path, help="path to a PIMSoft binary file")
    inspect_parser.add_argument("--json", action="store_true", help="print metadata as JSON")
    inspect_parser.set_defaults(handler=inspect_command)

    measure_parser = subparsers.add_parser("measure", help="measure ROIs/TOIs from a saved session")
    measure_parser.add_argument("recording", type=Path, help="path to a PIMSoft binary file")
    measure_parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="path to a .pimpsi.json session file",
    )
    measure_parser.add_argument("--out", required=True, type=Path, help="output CSV path")
    measure_parser.add_argument(
        "--metric",
        action="append",
        choices=[
            DEFAULT_METRIC,
            "roi_toi_mean_intensity",
            "roi_toi_mean_variance",
            "roi_area_pixels",
            "roi_valid_pixel_count",
        ],
        help="metric to calculate; may be passed more than once",
    )
    measure_parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="measure even if session source metadata does not match the binary",
    )
    measure_parser.add_argument(
        "--intensity-mask-lower",
        type=float,
        help="override the session and exclude pixels below this intensity",
    )
    measure_parser.add_argument(
        "--intensity-mask-upper",
        type=float,
        help="override the session and exclude pixels above this intensity",
    )
    measure_parser.set_defaults(handler=measure_command)

    gui_parser = subparsers.add_parser("gui", help="start the PySide6 viewer")
    gui_parser.add_argument(
        "recording",
        nargs="?",
        type=Path,
        help="optional PIMSoft binary file to open",
    )
    gui_parser.set_defaults(handler=gui_command)

    return parser


def inspect_command(args: argparse.Namespace) -> None:
    recording = PimRecording.open(args.recording)
    metadata = _metadata(recording)
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    for key, value in metadata.items():
        print(f"{key}: {value}")


def measure_command(args: argparse.Namespace) -> None:
    recording = PimRecording.open(args.recording)
    session = AnalysisSession.load(args.session)
    if not args.allow_source_mismatch:
        _validate_session_source(recording, session)
    if not session.rois:
        raise ValueError("session contains no ROIs")
    if not session.tois:
        raise ValueError("session contains no TOIs")

    intensity_mask = _intensity_mask_from_args_or_session(args, session)
    metrics = args.metric or [DEFAULT_METRIC]
    results: list[MeasurementResult] = []
    for roi in session.rois:
        for toi in session.tois:
            for metric in metrics:
                results.append(
                    measure_roi_toi(
                        recording=recording,
                        roi=roi,
                        toi=toi,
                        metric=metric,
                        intensity_mask=intensity_mask,
                        perfusion_clip_upper=session.processing_profile.perfusion_clip_upper,
                        negative_variance_policy=session.processing_profile.negative_variance_policy,
                    )
                )

    _write_measurements(args.out, results)
    print(f"wrote {len(results)} measurements to {args.out}")


def gui_command(args: argparse.Namespace) -> None:
    from pimpsi.gui.main_window import run

    raise SystemExit(run(args.recording))


def _metadata(recording: PimRecording) -> dict[str, Any]:
    header = recording.header
    file_size = recording.path.stat().st_size
    expected_size = header.data_offset + (
        header.number_of_images * header.image_width * header.image_height * 8
    )
    return {
        "source_file": str(recording.path),
        "source_sha256": header.sha256,
        "source_file_size": file_size,
        "file_type": header.file_type,
        "file_version": header.file_version,
        "image_width": header.image_width,
        "image_height": header.image_height,
        "number_of_images": header.number_of_images,
        "n_frames": header.n_frames,
        "signal_gain": header.signal_gain,
        "coherence_factor": header.coherence_factor,
        "data_offset": header.data_offset,
        "variance_offset": header.variance_offset,
        "intensity_offset": header.intensity_offset,
        "expected_file_size": expected_size,
        "file_size_valid": file_size == expected_size,
    }


def _validate_session_source(recording: PimRecording, session: AnalysisSession) -> None:
    mismatches = []
    if session.source_sha256 and session.source_sha256 != recording.header.sha256:
        mismatches.append("source_sha256")
    if session.source_file_size is not None and session.source_file_size != recording.path.stat().st_size:
        mismatches.append("source_file_size")
    if (
        session.pimsoft_binary_version is not None
        and session.pimsoft_binary_version != recording.header.file_version
    ):
        mismatches.append("pimsoft_binary_version")
    if mismatches:
        raise ValueError(
            "session source metadata does not match recording: "
            + ", ".join(mismatches)
            + "; pass --allow-source-mismatch to override"
        )


def _intensity_mask_from_args_or_session(args: argparse.Namespace, session: AnalysisSession):
    lower = args.intensity_mask_lower
    upper = args.intensity_mask_upper
    if (lower is None) != (upper is None):
        raise ValueError("--intensity-mask-lower and --intensity-mask-upper must be passed together")
    enabled = session.processing_profile.intensity_mask_enabled
    if lower is None:
        lower = session.processing_profile.intensity_mask_lower
        upper = session.processing_profile.intensity_mask_upper
    else:
        enabled = True
    if not enabled:
        return None
    lower, upper = min(lower, upper), max(lower, upper)
    return lambda intensity: (np.asarray(intensity) >= lower) & (np.asarray(intensity) <= upper)


def _write_measurements(path: Path, results: list[MeasurementResult]) -> None:
    write_measurement_csv(path, results)


if __name__ == "__main__":
    raise SystemExit(main())
