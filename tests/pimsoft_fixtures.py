import struct

import numpy as np


HEADER_STRUCT = struct.Struct("<10sidddii")


def make_pimsoft_file(
    path,
    *,
    file_version=1,
    variance_frames=None,
    intensity_frames=None,
    signal_gain=10.0,
    coherence_factor=0.5,
):
    if variance_frames is None:
        variance_frames = [
            np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
            np.array([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]),
        ]
    if intensity_frames is None:
        intensity_frames = [
            np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]]),
            np.array([[200.0, 201.0, 202.0], [203.0, 204.0, 205.0]]),
        ]

    n_frames = len(variance_frames)
    height, width = variance_frames[0].shape
    data_offset = {1: 46, 2: 540}.get(file_version, 581 + (n_frames * 32))
    header = HEADER_STRUCT.pack(
        b"PIMSOFT\x00\x00\x00",
        file_version,
        signal_gain,
        coherence_factor,
        float(n_frames * 2),
        width,
        height,
    )
    padding = b"\x00" * (data_offset - len(header))
    payload = b"".join(
        frame.T.astype("<f8", copy=False).tobytes(order="C")
        for frame in [*variance_frames, *intensity_frames]
    )
    path.write_bytes(header + padding + payload)
    return variance_frames, intensity_frames

