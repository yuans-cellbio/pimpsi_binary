"""Sidecar JSON session state for PIMSoft PSI analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from pimpsi.roi import Roi
from pimpsi.toi import Toi


SCHEMA_VERSION = "0.1.0"


@dataclass
class ProcessingProfile:
    perfusion_clip_upper: float = 3000.0
    negative_variance_policy: str = "signed_contrast"
    roi_summary_policy: str = "mean_intensity_variance_then_perfusion"
    intensity_mask_policy: str = "exclude_pixels"
    spatial_downscale: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "perfusion_clip_upper": self.perfusion_clip_upper,
            "negative_variance_policy": self.negative_variance_policy,
            "roi_summary_policy": self.roi_summary_policy,
            "intensity_mask_policy": self.intensity_mask_policy,
            "spatial_downscale": self.spatial_downscale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProcessingProfile":
        if data is None:
            return cls()
        return cls(
            perfusion_clip_upper=data.get("perfusion_clip_upper", 3000.0),
            negative_variance_policy=data.get("negative_variance_policy", "signed_contrast"),
            roi_summary_policy=data.get(
                "roi_summary_policy",
                "mean_intensity_variance_then_perfusion",
            ),
            intensity_mask_policy=data.get("intensity_mask_policy", "exclude_pixels"),
            spatial_downscale=data.get("spatial_downscale"),
        )


@dataclass
class AnalysisSession:
    source_file: str
    source_sha256: str
    source_file_size: int | None = None
    pimsoft_binary_version: int | None = None
    schema_version: str = SCHEMA_VERSION
    processing_profile: ProcessingProfile = field(default_factory=ProcessingProfile)
    rois: list[Roi] = field(default_factory=list)
    tois: list[Toi] = field(default_factory=list)
    frame_annotations: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_file_size": self.source_file_size,
            "pimsoft_binary_version": self.pimsoft_binary_version,
            "processing_profile": self.processing_profile.to_dict(),
            "rois": [roi.to_dict() for roi in self.rois],
            "tois": [toi.to_dict() for toi in self.tois],
            "frame_annotations": self.frame_annotations,
            "labels": self.labels,
            "exports": self.exports,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisSession":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            source_file=data["source_file"],
            source_sha256=data["source_sha256"],
            source_file_size=data.get("source_file_size"),
            pimsoft_binary_version=data.get("pimsoft_binary_version"),
            processing_profile=ProcessingProfile.from_dict(data.get("processing_profile")),
            rois=[Roi.from_dict(roi) for roi in data.get("rois", [])],
            tois=[Toi.from_dict(toi) for toi in data.get("tois", [])],
            frame_annotations=data.get("frame_annotations", []),
            labels=data.get("labels", []),
            exports=data.get("exports", []),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisSession":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_recording(cls, recording: Any) -> "AnalysisSession":
        return cls(
            source_file=str(recording.path),
            source_sha256=recording.header.sha256,
            source_file_size=recording.path.stat().st_size,
            pimsoft_binary_version=recording.header.file_version,
        )

