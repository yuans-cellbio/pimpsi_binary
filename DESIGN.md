PIMSoft PSI Binary Analyzer: Architecture Brief
Goal

Build an open-source, reproducible application for reading, visualizing, annotating, and analyzing binary recordings exported from PIMSoft / Perimed Pericam PSI laser speckle contrast imaging.

The app should replace reliance on proprietary PIMSoft for routine analysis while preserving correct numeric behavior, batch processing, transparent provenance, and reusable annotations.

The vendor documentation states that the binary file stores variance images followed by intensity images, and that No. of images is the total number of variance plus intensity images, meaning the actual recording frame count is number_of_images / 2. Version 2 and version 3 files also have larger headers than version 1, and version 3 includes per-frame image headers before image data.

Current old codebase

The old codebase contains:

archived/
  app.py
  app.ui
  LsciFile.py
  main.py
  mplcanvas.py

Treat these files as reference only, not as the foundation for the new app.

Major problems in the old code:

LsciFile.py assumes image data starts immediately after the 46-byte version 1 header. This is wrong for file version 2 and version 3.
It reads all variance, intensity, intensity masks, and perfusion frames into memory.
It calculates perfusion for the full stack eagerly during file loading.
It performs computational work on the GUI thread.
It uses Matplotlib as the main image viewer, which is too slow for frame-by-frame interactive image display.
It calculates negative variance using abs(variance) without preserving the sign behavior described by the vendor.
It applies intensity masks by multiplying intensity and variance images before perfusion calculation, which changes numeric values instead of treating pixels as excluded.
It sometimes averages already-computed perfusion images, although ROI/downscaled summaries should average intensity and variance first, then calculate perfusion.
It repeats 2D ROI masks into 3D arrays, causing unnecessary memory use.
ExportImage.getSelectedOptions() has a bug: "Mask" uses checkbox_perfusion instead of checkbox_roi.
roi_masks is not safely initialized in new_session().
Slider signal connections are repeatedly added during session reset.
It manipulates private Matplotlib selector fields, which is fragile.
Recommended strategy

Do not refactor the old GUI in place.

Build a new layered package:

pimpsi/
  pyproject.toml
  README.md
  DESIGN.md

  src/pimpsi/
    __init__.py
    io.py
    compute.py
    roi.py
    toi.py
    session.py
    measure.py
    export.py
    cli.py

    gui/
      __init__.py
      main_window.py
      image_view.py
      roi_panel.py
      toi_panel.py
      measurement_panel.py
      workers.py

  tests/
    test_io.py
    test_compute.py
    test_roi.py
    test_session.py
    test_measure.py
Technology choice

Use Python for now.

Recommended libraries:

numpy
scipy
pydantic
pandas
tifffile
imageio
scikit-image
PySide6
pyqtgraph
pytest
ruff
mypy

Optional later:

zarr
numba
napari

Use PySide6 + pyqtgraph for the dedicated GUI. Avoid Matplotlib for live image updates. Matplotlib can still be used for static export plots.

Core design rule

The GUI must not own the analysis logic.

The reusable library should support headless analysis:

from pimpsi.io import PimRecording
from pimpsi.session import AnalysisSession
from pimpsi.measure import measure_roi_toi

recording = PimRecording.open("recording.dat")
session = AnalysisSession.load("recording.pimpsi.json")

result = measure_roi_toi(
    recording=recording,
    roi=session.rois[0],
    toi=session.tois[0],
    metric="perfusion_from_mean_intensity_variance",
)

The GUI should only call this API.

File reading design

Implement PimRecording in io.py.

Responsibilities:

Parse binary header.
Detect file version.
Calculate correct image-data offsets.
Validate file size.
Expose lazy frame access using numpy.memmap.
Return images as normal row-major arrays with shape:
height x width

Internal binary layout appears column-major-like. The reader should transpose into normal image orientation.

Header behavior

Support:

Version 1:
  data_offset = 46

Version 2:
  data_offset = 540

Version 3:
  data_offset = 581 + n_frames * 32

Where:

n_frames = number_of_images / 2

Expose:

recording.header.file_type
recording.header.file_version
recording.header.signal_gain
recording.header.coherence_factor
recording.header.number_of_images
recording.header.n_frames
recording.header.image_width
recording.header.image_height
recording.header.data_offset
recording.header.variance_offset
recording.header.intensity_offset
recording.header.sha256
Computation design

Implement in compute.py.

Required functions:

calculate_contrast(
    variance,
    intensity,
    coherence_factor,
)

calculate_perfusion(
    variance,
    intensity,
    coherence_factor,
    signal_gain,
    clip_upper=3000.0,
)

block_average(
    image,
    block_size,
)

calculate_perfusion_from_mean_intensity_variance(
    intensity_values,
    variance_values,
    coherence_factor,
    signal_gain,
    clip_upper=3000.0,
)

Important: do not silently replace negative variance with abs(variance) in a way that loses the sign. The vendor formula treats negative variance as producing a negative contrast-like value before perfusion calculation.

ROI design

Implement in roi.py.

Support these ROI types:

rectangle
ellipse
polygon
freehand polygon

Each ROI should store geometry, not just a mask.

Example model:

class Roi:
    id: str
    label: str
    shape_type: str
    vertices_xy: list[tuple[float, float]]
    visible: bool = True
    locked: bool = False
    group: str | None = None
    notes: str | None = None

Generate masks on demand:

mask = roi.to_mask(image_shape=(height, width))

Do not store repeated 3D masks unless exporting masks explicitly.

TOI design

Implement in toi.py.

TOI means time/frame interval of interest.

class Toi:
    id: str
    label: str
    frame_start: int
    frame_end: int
    include_end: bool = False
    notes: str | None = None

Use zero-based frame indices internally.

Display one-based frame numbers in the GUI if desired, but make the internal representation explicit.

Session design

Implement in session.py.

Use a sidecar JSON file:

recording_name.pimpsi.json

It should store:

schema_version
source_file
source_sha256
source_file_size
pimsoft_binary_version
processing_profile
rois
tois
frame_annotations
labels
exports

Example:

{
  "schema_version": "0.1.0",
  "source_file": "mouse01.dat",
  "source_sha256": "abc123",
  "processing_profile": {
    "perfusion_clip_upper": 3000.0,
    "negative_variance_policy": "signed_contrast",
    "roi_summary_policy": "mean_intensity_variance_then_perfusion",
    "intensity_mask_policy": "exclude_pixels",
    "spatial_downscale": null
  },
  "rois": [],
  "tois": [],
  "frame_annotations": []
}
Measurement design

Implement in measure.py.

Primary supported metrics:

roi_perfusion_per_frame
roi_toi_perfusion_from_mean_intensity_variance
roi_toi_mean_intensity
roi_toi_mean_variance
roi_area_pixels
roi_valid_pixel_count

Default publication-quality metric:

roi_toi_perfusion_from_mean_intensity_variance

Do not default to averaging pixelwise perfusion values.

ROI measurement should:

Convert ROI geometry to a 2D mask.
Optionally combine with intensity-valid mask.
For each frame or TOI, collect intensity and variance values.
Average intensity and variance over selected pixels and frames.
Calculate perfusion from averaged intensity and averaged variance.
Export design

Implement in export.py.

Support:

CSV measurement table
JSON session
TIFF stack export
PNG snapshot export
ROI mask export
frame annotation table

CSV should include enough provenance:

source_file
source_sha256
file_version
roi_id
roi_label
toi_id
toi_label
frame_start
frame_end
metric
value
n_pixels
n_frames
coherence_factor
signal_gain
perfusion_clip_upper
negative_variance_policy
CLI design

Implement cli.py.

Commands:

pimpsi inspect recording.dat

pimpsi measure recording.dat \
  --session recording.pimpsi.json \
  --out measurements.csv

pimpsi export-tiff recording.dat \
  --session recording.pimpsi.json \
  --kind perfusion \
  --frames 0:100 \
  --out perfusion.tif

pimpsi export-mask recording.dat \
  --session recording.pimpsi.json \
  --roi roi_001 \
  --out roi_001_mask.tif

The CLI should work before the GUI is complete.

GUI design

Use PySide6 and pyqtgraph.

Main windows/panels:

File metadata panel
Image viewer
Frame slider
Display controls
ROI panel
TOI panel
Frame annotation panel
Measurement table
Trace plot panel
Export panel

Image viewer behavior:

Load only the current frame.
Compute displayed perfusion only for the current frame.
Cache recently viewed frames.
Never compute all frame perfusion images during file opening.
Run long measurement/export tasks in worker threads.
Disable controls during long tasks only when necessary.
Save session state frequently.
GUI display modes

Support:

Intensity
Variance
Contrast
Perfusion

Support scaling:

manual min/max
percentile scaling
auto per-frame scaling
fixed global scaling sampled from selected frames
Testing requirements

Create tests before building the GUI.

Minimum tests:

test_header_v1_offset
test_header_v2_offset
test_header_v3_offset
test_file_size_validation
test_number_of_images_even
test_column_order_transpose
test_negative_variance_perfusion
test_zero_variance_behavior
test_roi_polygon_mask
test_roi_toi_measurement
test_session_roundtrip

Synthetic binary files should be generated in tests. Do not rely only on real data.

Use of supplied real binary file

When I supply a real binary file, use it for integration testing only.

Do the following:

Read the header.
Report file version, image size, frame count, signal gain, coherence factor.
Confirm expected file size matches actual file size.
Load frame 0 intensity and variance.
Render frame 0 intensity.
Render frame 0 perfusion.
Export a small measurement table using a test ROI.
Confirm loading does not require reading the whole recording into RAM.
If possible, compare one frame or one ROI measurement against PIMSoft or vendor Matlab output.

Do not commit large binary files to the repository. Use:

tests/data/private/

and add it to .gitignore.

Implementation milestones
Milestone 1: Core binary reader

Deliver:

io.py
compute.py
unit tests
CLI inspect command

Acceptance:

pimpsi inspect example.dat

prints valid metadata and passes file-size validation.

Milestone 2: Correct computation

Deliver:

contrast calculation
perfusion calculation
block averaging
ROI/TOI summary calculation
unit tests with known values

Acceptance:

pytest tests/test_compute.py

passes all numeric tests.

Milestone 3: Session model

Deliver:

session.py
ROI schema
TOI schema
frame annotation schema
JSON save/load

Acceptance:

Session roundtrip does not alter ROI coordinates, TOI intervals, labels, or processing settings.
Milestone 4: CLI measurement

Deliver:

measure command
CSV export
JSON provenance

Acceptance:

pimpsi measure example.dat --session example.pimpsi.json --out measurements.csv

produces reproducible measurements without launching the GUI.

Milestone 5: Fast GUI viewer

Deliver:

PySide6 main window
pyqtgraph image viewer
frame slider
display mode switch
manual display limits

Acceptance:

Opening a file should parse metadata quickly.
Changing frames should not freeze the UI.
Perfusion should be calculated lazily for the displayed frame.
Milestone 6: ROI/TOI GUI

Deliver:

interactive ROI drawing
ROI table
TOI interval editor
trace plot
measurement table
session autosave

Acceptance:

User can draw ROI, define TOI, save session, reload session, and reproduce the same measurement.
Milestone 7: Export tools

Deliver:

CSV
TIFF
PNG snapshot
ROI masks
session JSON

Acceptance:

Exports include source metadata and processing profile.