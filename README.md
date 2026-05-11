# PIMPSI

PIMPSI is a small Python toolkit for reading, viewing, measuring, and exporting PIMSoft PSI binary recordings.

It provides both a PySide6 desktop GUI for interactive ROI/TOI analysis and a command-line interface for headless inspection and measurement workflows.

## Features

- Open PIMSoft binary recordings (`.dat`) lazily with `numpy.memmap`
- Inspect recording metadata and validate file size/layout
- View intensity, variance, contrast, and perfusion channels
- Create and edit ROIs and TOIs
- Save/load analysis sessions as `.pimpsi.json`
- Persist processing settings, including perfusion clipping and optional intensity masking
- Plot ROI traces with optional per-frame intensity masking
- Export:
  - data-preserving TIFF images/stacks
  - ImageJ-friendly multi-channel stacks
  - RGB representative images with per-channel color mapping and optional masked ROI outlines
  - ROI masks
  - measurement CSV files
- Trim PIMSoft binary recordings to selected frame ranges while preserving the original binary format

## Installation

From the repository root:

```bash
python -m pip install -e .
```

For the GUI and image export tools:

```bash
python -m pip install -e ".[gui]"
```

For development and tests:

```bash
python -m pip install -e ".[gui,test]"
```

## GUI

Start the viewer:

```bash
pimpsi gui
```

Open a recording directly:

```bash
pimpsi gui /path/to/recording.dat
```

The GUI supports interactive viewing, ROI/TOI editing, trace plotting, session save/load, image export, measurement export, ROI mask export, and binary trimming from the `Export` menu.

The main window has two synchronized viewers:

- The top viewer is locked to the intensity channel and is the canonical ROI editing surface.
- The bottom viewer can display any channel and shows read-only ROI outlines.
- New channel windows are also read-only for ROI geometry.

The trace panel includes optional intensity masking. When enabled, pixels outside the selected intensity range are excluded per frame from trace and measurement calculations. Mask settings are saved in the session file and restored when the session is loaded. Viewers show masked ROI outlines separately from the saved ROI geometry so the mask can be checked visually without changing the ROI itself.

## CLI

Inspect a recording:

```bash
pimpsi inspect /path/to/recording.dat
```

Print metadata as JSON:

```bash
pimpsi inspect /path/to/recording.dat --json
```

Measure ROIs and TOIs from a saved GUI session:

```bash
pimpsi measure /path/to/recording.dat \
  --session /path/to/session.pimpsi.json \
  --out measurements.csv
```

Choose metrics:

```bash
pimpsi measure /path/to/recording.dat \
  --session /path/to/session.pimpsi.json \
  --out measurements.csv \
  --metric roi_toi_mean_intensity \
  --metric roi_toi_mean_variance \
  --metric roi_toi_perfusion_from_mean_intensity_variance
```

Saved session intensity-mask settings are applied by `pimpsi measure`. You can also override them for a batch run:

```bash
pimpsi measure /path/to/recording.dat \
  --session /path/to/session.pimpsi.json \
  --out masked-measurements.csv \
  --intensity-mask-lower 100 \
  --intensity-mask-upper 5000
```

Both mask bounds must be supplied together. Masking is frame-local: each frame uses its own intensity image to decide which ROI pixels are included.

## Measurement Semantics

The default perfusion metric is:

```text
roi_toi_perfusion_from_mean_intensity_variance
```

It averages raw intensity and variance values over the selected ROI/TOI first, then calculates perfusion from those means. This matches the PIMSoft guidance that perfusion over multiple frames should be calculated from averaged intensity and variance, not by averaging per-pixel or per-frame perfusion values.

When intensity masking is enabled, the current behavior pools all valid ROI pixels across the selected frames. Frames with more included pixels therefore contribute more to the final ROI/TOI mean.

## Notes On Exports

Non-RGB TIFF exports preserve numeric image data as floating-point TIFFs. RGB exports are intended as representative images and use explicit color mapping limits. Multi-channel stacked TIFF exports are written as ImageJ-style hyperstacks.

ROI mask TIFF export writes static ROI geometry only. It does not apply intensity masking, because intensity masks are frame-dependent.

When RGB image export draws ROIs with intensity masking enabled, exported filenames include `_masked_` and the image includes masked ROI outlines.

## Development

Run the test suite:

```bash
pytest -q
```

Project layout:

- `src/pimpsi/io.py`: PIMSoft binary reading and trimming
- `src/pimpsi/measure.py`: ROI/TOI measurement routines
- `src/pimpsi/export.py`: image, mask, and CSV export helpers
- `src/pimpsi/gui/`: PySide6 GUI
- `tests/`: unit and GUI tests

## Status

This project is early-stage research/analysis software. Please validate exported measurements and images against your lab workflow before relying on them for production analysis.
