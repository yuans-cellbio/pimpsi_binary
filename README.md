# PIMPSI

PIMPSI is a small Python toolkit for reading, viewing, measuring, and exporting PIMSoft PSI binary recordings.

It provides both a PySide6 desktop GUI for interactive ROI/TOI analysis and a command-line interface for headless inspection and measurement workflows.

## Features

- Open PIMSoft binary recordings (`.dat`) lazily with `numpy.memmap`
- Inspect recording metadata and validate file size/layout
- View intensity, variance, contrast, and perfusion channels
- Create and edit ROIs and TOIs
- Save/load analysis sessions as `.pimpsi.json`
- Export:
  - data-preserving TIFF images/stacks
  - ImageJ-friendly multi-channel stacks
  - RGB representative images with per-channel color mapping
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

The GUI supports interactive viewing, ROI/TOI editing, trace plotting, session save/load, image export, measurement export, ROI mask export, and binary trimming from the `Extract` menu.

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

## Notes On Exports

Non-RGB TIFF exports preserve numeric image data as floating-point TIFFs. RGB exports are intended as representative images and use explicit color mapping limits. Multi-channel stacked TIFF exports are written as ImageJ-style hyperstacks.

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
