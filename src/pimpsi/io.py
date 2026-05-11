"""Lazy reader for PIMSoft PSI binary recordings."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pimpsi.compute import calculate_perfusion


HEADER_STRUCT = struct.Struct("<10sidddii")
DATA_OFFSETS = {1: 46, 2: 540}
VERSION_3_BASE_OFFSET = 581
VERSION_3_FRAME_HEADER_SIZE = 32
PIXEL_DTYPE = np.dtype("<f8")


@dataclass(frozen=True)
class PimHeader:
    file_type: str
    file_version: int
    signal_gain: float
    coherence_factor: float
    number_of_images: int
    n_frames: int
    image_width: int
    image_height: int
    data_offset: int
    variance_offset: int
    intensity_offset: int
    sha256: str


class PimRecording:
    """Lazy PIMSoft binary recording backed by ``numpy.memmap``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = self._read_header(self.path)
        self._data = np.memmap(
            self.path,
            dtype=PIXEL_DTYPE,
            mode="r",
            offset=self.header.data_offset,
            shape=(self.header.number_of_images, self.header.image_width, self.header.image_height),
            order="C",
        )

    @classmethod
    def open(cls, path: str | Path) -> Self:
        return cls(path)

    @staticmethod
    def _read_header(path: Path) -> PimHeader:
        path = Path(path)
        file_size = path.stat().st_size
        if file_size < HEADER_STRUCT.size:
            raise ValueError(f"{path} is too small to contain a PIMSoft header")

        with path.open("rb") as file:
            raw_header = file.read(HEADER_STRUCT.size)

        (
            raw_file_type,
            file_version,
            signal_gain,
            coherence_factor,
            raw_number_of_images,
            image_width,
            image_height,
        ) = HEADER_STRUCT.unpack(raw_header)

        file_type = raw_file_type.decode("ascii", errors="replace").strip("\x00 ")
        number_of_images = _parse_number_of_images(raw_number_of_images)
        _validate_header_values(file_version, number_of_images, image_width, image_height)

        n_frames = number_of_images // 2
        if file_version == 3:
            data_offset = VERSION_3_BASE_OFFSET + (n_frames * VERSION_3_FRAME_HEADER_SIZE)
        else:
            data_offset = DATA_OFFSETS[file_version]

        frame_pixels = image_width * image_height
        variance_offset = data_offset
        intensity_offset = data_offset + (n_frames * frame_pixels * PIXEL_DTYPE.itemsize)
        expected_size = data_offset + (number_of_images * frame_pixels * PIXEL_DTYPE.itemsize)
        if file_size != expected_size:
            raise ValueError(
                f"Invalid PIMSoft file size: expected {expected_size} bytes for "
                f"{number_of_images} images of {image_width}x{image_height}, got {file_size} bytes"
            )

        return PimHeader(
            file_type=file_type,
            file_version=file_version,
            signal_gain=signal_gain,
            coherence_factor=coherence_factor,
            number_of_images=number_of_images,
            n_frames=n_frames,
            image_width=image_width,
            image_height=image_height,
            data_offset=data_offset,
            variance_offset=variance_offset,
            intensity_offset=intensity_offset,
            sha256=_sha256_file(path),
        )

    def get_variance(self, frame_index: int) -> NDArray[np.float64]:
        self._validate_frame_index(frame_index)
        return np.array(self._data[frame_index].T, dtype=np.float64, order="C", copy=True)

    def get_intensity(self, frame_index: int) -> NDArray[np.float64]:
        self._validate_frame_index(frame_index)
        return np.array(
            self._data[self.header.n_frames + frame_index].T,
            dtype=np.float64,
            order="C",
            copy=True,
        )

    def calculate_perfusion(
        self,
        frame_index: int,
        clip_upper: float = 3000.0,
    ) -> NDArray[np.float64]:
        return calculate_perfusion(
            variance=self.get_variance(frame_index),
            intensity=self.get_intensity(frame_index),
            coherence_factor=self.header.coherence_factor,
            signal_gain=self.header.signal_gain,
            clip_upper=clip_upper,
        )

    def _validate_frame_index(self, frame_index: int) -> None:
        if not 0 <= frame_index < self.header.n_frames:
            raise IndexError(f"frame_index must be in [0, {self.header.n_frames})")


def trim_pim_binary(recording: PimRecording, output_path: str | Path, frames: list[int]) -> Path:
    """Write a frame-trimmed copy of ``recording`` in the same PIMSoft binary format."""
    if not frames:
        raise ValueError("at least one frame is required")
    output_path = Path(output_path)
    if output_path.resolve() == recording.path.resolve():
        raise ValueError("trimmed binary output must not overwrite the source recording")
    for frame_index in frames:
        recording._validate_frame_index(frame_index)

    header = recording.header
    frame_bytes = header.image_width * header.image_height * PIXEL_DTYPE.itemsize
    source_data_offset = header.data_offset
    output_frame_count = len(frames)
    output_image_count = output_frame_count * 2

    with recording.path.open("rb") as source:
        if header.file_version == 3:
            prefix = bytearray(source.read(VERSION_3_BASE_OFFSET))
            _write_number_of_images(prefix, output_image_count)
            source.seek(VERSION_3_BASE_OFFSET)
            frame_headers = source.read(header.n_frames * VERSION_3_FRAME_HEADER_SIZE)
            output_frame_headers = b"".join(
                frame_headers[frame_index * VERSION_3_FRAME_HEADER_SIZE : (frame_index + 1) * VERSION_3_FRAME_HEADER_SIZE]
                for frame_index in frames
            )
        else:
            prefix = bytearray(source.read(source_data_offset))
            _write_number_of_images(prefix, output_image_count)
            output_frame_headers = b""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            output.write(prefix)
            output.write(output_frame_headers)
            for plane_offset in (0, header.n_frames * frame_bytes):
                for frame_index in frames:
                    source.seek(source_data_offset + plane_offset + (frame_index * frame_bytes))
                    output.write(source.read(frame_bytes))

    return output_path


def _write_number_of_images(header_bytes: bytearray, number_of_images: int) -> None:
    values = list(HEADER_STRUCT.unpack(bytes(header_bytes[: HEADER_STRUCT.size])))
    values[4] = float(number_of_images)
    header_bytes[: HEADER_STRUCT.size] = HEADER_STRUCT.pack(*values)


def _parse_number_of_images(value: float) -> int:
    if not value.is_integer():
        raise ValueError(f"number_of_images must be an integer value, got {value!r}")
    return int(value)


def _validate_header_values(
    file_version: int,
    number_of_images: int,
    image_width: int,
    image_height: int,
) -> None:
    if file_version not in {1, 2, 3}:
        raise ValueError(f"Unsupported PIMSoft file version: {file_version}")
    if number_of_images <= 0 or number_of_images % 2 != 0:
        raise ValueError("number_of_images must be a positive even value")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
