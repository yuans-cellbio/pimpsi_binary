"""Headless export helpers for PIMSoft PSI recordings."""

from __future__ import annotations

from collections.abc import Callable
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from numpy.typing import NDArray

from pimpsi.compute import calculate_contrast
from pimpsi.io import PimRecording
from pimpsi.measure import CSV_COLUMNS, MeasurementResult
from pimpsi.roi import Roi


CHANNELS = ("intensity", "variance", "contrast", "perfusion")
_SAMPLE_AXIS = "S"
ColorMapSetting = str | dict[str, str]
ColorLimitSetting = tuple[float, float] | dict[str, tuple[float, float]] | None


def get_channel_frame(recording: PimRecording, channel: str, frame_index: int) -> NDArray[np.float64]:
    if channel == "intensity":
        return recording.get_intensity(frame_index)
    if channel == "variance":
        return recording.get_variance(frame_index)
    if channel == "contrast":
        return calculate_contrast(
            variance=recording.get_variance(frame_index),
            intensity=recording.get_intensity(frame_index),
            coherence_factor=recording.header.coherence_factor,
        )
    if channel == "perfusion":
        return recording.calculate_perfusion(frame_index)
    raise ValueError(f"Unsupported channel: {channel!r}")


def export_single_frame(recording: PimRecording, path: str | Path, channel: str, frame_index: int) -> None:
    image = get_channel_frame(recording, channel, frame_index)
    _write_image(Path(path), image)


def export_frame_stack(
    recording: PimRecording,
    path: str | Path,
    channels: list[str],
    frames: list[int],
) -> None:
    stack = np.asarray(
        [[get_channel_frame(recording, channel, frame_index) for frame_index in frames] for channel in channels],
        dtype=np.float64,
    )
    tifffile.imwrite(Path(path), stack, metadata={"axes": "CTYX"})


def export_average_image(
    recording: PimRecording,
    path: str | Path,
    channel: str,
    frames: list[int],
) -> None:
    if not frames:
        raise ValueError("at least one frame is required for averaged export")
    accumulator = None
    for frame_index in frames:
        image = get_channel_frame(recording, channel, frame_index)
        accumulator = image.copy() if accumulator is None else accumulator + image
    _write_image(Path(path), accumulator / len(frames))


def export_image_set(
    recording: PimRecording,
    *,
    frames: list[int],
    channels: list[str],
    output_dir: str | Path | None = None,
    single_frame: bool = False,
    stacked: bool = False,
    averaged: bool = False,
    color: bool = False,
    colormap: ColorMapSetting = "gray",
    color_limits: ColorLimitSetting = None,
    rois: list[Roi] | None = None,
    intensity_mask: Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None = None,
) -> list[Path]:
    """Export selected images next to the source ``.dat`` with deterministic names."""
    if not frames:
        raise ValueError("at least one frame is required")
    if not channels:
        raise ValueError("at least one channel is required")
    if not any([single_frame, stacked, averaged]):
        raise ValueError("at least one export option is required")

    out_dir = Path(output_dir) if output_dir is not None else recording.path.parent
    basename = recording.path.stem
    exported = []

    if single_frame:
        for channel in channels:
            for frame_index in frames:
                image = get_channel_frame(recording, channel, frame_index)
                path = out_dir / _image_export_name(
                    basename,
                    frames=[frame_index],
                    channel=channel,
                    color=color,
                    kind="single_frame",
                    masked=bool(rois and intensity_mask is not None),
                )
                _write_export_image(
                    path,
                    image,
                    recording=recording,
                    color=color,
                    colormap=_channel_colormap(colormap, channel),
                    color_limits=_channel_color_limits(color_limits, channel),
                    rois=rois,
                    masked_rois=_masked_rois(recording, frame_index, rois, intensity_mask),
                )
                exported.append(path)

    if stacked:
        if not color and len(channels) > 1:
            path = out_dir / _image_export_name(
                basename,
                frames=frames,
                channel="-".join(channels),
                color=color,
                kind="stacked",
                masked=False,
            )
            stack = np.asarray(
                [
                    [_image_export_plane(recording, get_channel_frame(recording, channel, frame_index)) for channel in channels]
                    for frame_index in frames
                ]
            )
            tifffile.imwrite(
                path,
                stack,
                imagej=True,
                metadata={"axes": "TCYX", "mode": "composite"},
            )
            exported.append(path)
        else:
            for channel in channels:
                images = [get_channel_frame(recording, channel, frame_index) for frame_index in frames]
                path = out_dir / _image_export_name(
                    basename,
                    frames=frames,
                    channel=channel,
                    color=color,
                    kind="stacked",
                    masked=bool(rois and intensity_mask is not None),
                )
                _write_export_stack(
                    path,
                    images,
                    recording=recording,
                    channel=channel,
                    color=color,
                    colormap=_channel_colormap(colormap, channel),
                    color_limits=_channel_color_limits(color_limits, channel),
                    rois=rois,
                    intensity_mask=intensity_mask,
                    frames=frames,
                )
                exported.append(path)

    if averaged:
        for channel in channels:
            accumulator = None
            for frame_index in frames:
                image = get_channel_frame(recording, channel, frame_index)
                accumulator = image.copy() if accumulator is None else accumulator + image
            path = out_dir / _image_export_name(
                basename,
                frames=frames,
                channel=channel,
                color=color,
                kind="averaged",
                masked=bool(rois and intensity_mask is not None),
            )
            _write_export_image(
                path,
                accumulator / len(frames),
                recording=recording,
                color=color,
                colormap=_channel_colormap(colormap, channel),
                color_limits=_channel_color_limits(color_limits, channel),
                rois=rois,
                masked_rois=_masked_rois_for_average(recording, frames, rois, intensity_mask),
            )
            exported.append(path)

    return exported


def export_roi_mask(
    recording: PimRecording,
    path: str | Path,
    rois: list[Roi],
) -> None:
    mask = np.zeros((recording.header.image_height, recording.header.image_width), dtype=np.uint16)
    for label, roi in enumerate(rois, start=1):
        mask[roi.to_mask(mask.shape)] = label
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(Path(path), mask)


def export_rgb_frame(
    recording: PimRecording,
    path: str | Path,
    frame_index: int,
    channel: str,
    rois: list[Roi] | None = None,
) -> None:
    image = get_channel_frame(recording, channel, frame_index)
    rgb = np.repeat(_normalize_uint8(image)[..., None], 3, axis=2)
    for roi in rois or []:
        boundary = _mask_boundary(roi.to_mask(image.shape))
        color = _hex_to_rgb(roi.color)
        rgb[boundary] = color
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, rgb)


def export_color_frame(
    recording: PimRecording,
    path: str | Path,
    frame_index: int,
    channel: str,
    colormap: str = "gray",
    color_limits: tuple[float, float] | None = None,
    rois: list[Roi] | None = None,
) -> None:
    image = get_channel_frame(recording, channel, frame_index)
    _write_export_image(
        Path(path),
        image,
        recording=recording,
        color=True,
        colormap=colormap,
        color_limits=color_limits,
        rois=rois,
    )


def write_measurement_csv(path: str | Path, results: list[MeasurementResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


def _write_image(path: Path, image: NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, image.astype(np.float32))
        return
    iio.imwrite(path, _normalize_uint8(image))


def _write_export_image(
    path: Path,
    image: NDArray[np.float64],
    *,
    recording: PimRecording,
    color: bool,
    colormap: str,
    color_limits: tuple[float, float] | None,
    rois: list[Roi] | None,
    masked_rois: list[tuple[Roi, NDArray[np.bool_]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color:
        tifffile.imwrite(
            path,
            _colorize(
                image,
                colormap=colormap,
                color_limits=color_limits,
                rois=rois,
                masked_rois=masked_rois,
            ),
        )
        return
    tifffile.imwrite(path, _image_export_plane(recording, image))


def _write_export_stack(
    path: Path,
    images: list[NDArray[np.float64]],
    *,
    recording: PimRecording,
    channel: str,
    color: bool,
    colormap: str,
    color_limits: tuple[float, float] | None,
    rois: list[Roi] | None,
    intensity_mask: Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None,
    frames: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "channel": channel,
        "channels": [channel],
        "frames": [frame + 1 for frame in frames],
    }
    if color:
        stack = np.asarray(
            [
                _colorize(
                    image,
                    colormap=colormap,
                    color_limits=color_limits,
                    rois=rois,
                    masked_rois=_masked_rois(recording, frame_index, rois, intensity_mask),
                )
                for image, frame_index in zip(images, frames, strict=True)
            ],
            dtype=np.uint8,
        )
        tifffile.imwrite(path, stack, photometric="rgb", metadata={**metadata, "axes": "TYX" + _SAMPLE_AXIS})
        return
    stack = np.asarray([_image_export_plane(recording, image) for image in images])
    tifffile.imwrite(path, stack, imagej=True, metadata={**metadata, "axes": "TYX"})


def _image_export_name(
    basename: str,
    *,
    frames: list[int],
    channel: str,
    color: bool,
    kind: str,
    masked: bool,
) -> str:
    frame_part = f"frame_{frames[0] + 1}" if len(frames) == 1 else f"frames_{frames[0] + 1}-{frames[-1] + 1}"
    image_part = "color_image" if color else "gray_image"
    mask_part = "_masked" if masked else ""
    return f"{basename}_{frame_part}_{channel}_{image_part}{mask_part}_{kind}.tif"


def _image_export_plane(recording: PimRecording, image: NDArray[np.float64]) -> NDArray[np.float32]:
    _ = recording
    return image.astype(np.float32)


def _channel_colormap(colormap: ColorMapSetting, channel: str) -> str:
    if isinstance(colormap, dict):
        return colormap.get(channel, "viridis")
    return colormap


def _channel_color_limits(color_limits: ColorLimitSetting, channel: str) -> tuple[float, float] | None:
    if isinstance(color_limits, dict):
        return color_limits.get(channel)
    return color_limits


def _colorize(
    image: NDArray[np.float64],
    *,
    colormap: str,
    color_limits: tuple[float, float] | None,
    rois: list[Roi] | None,
    masked_rois: list[tuple[Roi, NDArray[np.bool_]]] | None = None,
) -> NDArray[np.uint8]:
    gray = _normalize_uint8(image, limits=color_limits)
    rgb = _apply_colormap(gray, colormap)
    for roi in rois or []:
        boundary = _mask_boundary(roi.to_mask(image.shape))
        rgb[boundary] = (230, 230, 230) if masked_rois else _hex_to_rgb(roi.color)
    for roi, mask in masked_rois or []:
        boundary = _mask_boundary(mask)
        rgb[boundary] = _hex_to_rgb(roi.color)
    return rgb


def _masked_rois(
    recording: PimRecording,
    frame_index: int,
    rois: list[Roi] | None,
    intensity_mask: Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None,
) -> list[tuple[Roi, NDArray[np.bool_]]] | None:
    if intensity_mask is None or not rois:
        return None
    intensity = recording.get_intensity(frame_index)
    include = np.asarray(intensity_mask(intensity), dtype=bool)
    return [(roi, roi.to_mask(intensity.shape) & include) for roi in rois]


def _masked_rois_for_average(
    recording: PimRecording,
    frames: list[int],
    rois: list[Roi] | None,
    intensity_mask: Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None,
) -> list[tuple[Roi, NDArray[np.bool_]]] | None:
    if intensity_mask is None or not rois or not frames:
        return None
    accumulator = None
    for frame_index in frames:
        intensity = recording.get_intensity(frame_index)
        accumulator = intensity.copy() if accumulator is None else accumulator + intensity
    assert accumulator is not None
    include = np.asarray(intensity_mask(accumulator / len(frames)), dtype=bool)
    return [(roi, roi.to_mask(include.shape) & include) for roi in rois]


def _apply_colormap(gray: NDArray[np.uint8], colormap: str) -> NDArray[np.uint8]:
    name = colormap.lower()
    if name in {"gray", "grey"}:
        return np.repeat(gray[..., None], 3, axis=2)
    stops = _colormap_stops(name)
    x = gray.astype(np.float32) / 255.0
    positions = np.asarray([stop[0] for stop in stops], dtype=np.float32)
    colors = np.asarray([stop[1] for stop in stops], dtype=np.float32)
    channels = [np.interp(x, positions, colors[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _colormap_stops(name: str) -> list[tuple[float, tuple[int, int, int]]]:
    maps = {
        "viridis": [
            (0.0, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.5, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.0, (253, 231, 37)),
        ],
        "inferno": [
            (0.0, (0, 0, 4)),
            (0.25, (87, 15, 109)),
            (0.5, (187, 55, 84)),
            (0.75, (249, 142, 8)),
            (1.0, (252, 255, 164)),
        ],
        "magma": [
            (0.0, (0, 0, 4)),
            (0.25, (80, 18, 123)),
            (0.5, (182, 54, 121)),
            (0.75, (251, 136, 97)),
            (1.0, (252, 253, 191)),
        ],
        "plasma": [
            (0.0, (13, 8, 135)),
            (0.25, (126, 3, 168)),
            (0.5, (203, 71, 119)),
            (0.75, (248, 149, 64)),
            (1.0, (240, 249, 33)),
        ],
        "turbo": [
            (0.0, (48, 18, 59)),
            (0.2, (50, 111, 249)),
            (0.4, (34, 208, 188)),
            (0.6, (151, 251, 73)),
            (0.8, (245, 153, 39)),
            (1.0, (122, 4, 3)),
        ],
        "cividis": [
            (0.0, (0, 32, 76)),
            (0.25, (40, 78, 107)),
            (0.5, (87, 113, 111)),
            (0.75, (159, 152, 96)),
            (1.0, (255, 233, 69)),
        ],
    }
    return maps.get(name, maps["viridis"])


def _normalize_uint8(image: NDArray[np.float64], limits: tuple[float, float] | None = None) -> NDArray[np.uint8]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    if limits is None:
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))
    else:
        low, high = limits
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _mask_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    neighbors[1:, :] += mask[:-1, :]
    neighbors[:-1, :] += mask[1:, :]
    neighbors[:, 1:] += mask[:, :-1]
    neighbors[:, :-1] += mask[:, 1:]
    return mask & (neighbors < 4)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
