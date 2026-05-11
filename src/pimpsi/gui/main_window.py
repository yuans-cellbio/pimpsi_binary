"""Main PySide6 analysis window for lazy PIMSoft PSI workflows."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from pimpsi.export import (
    export_image_set,
    export_roi_mask,
    write_measurement_csv,
)
from pimpsi.gui.image_view import ImageView
from pimpsi.gui.workers import DisplayMode, LazyFrameProvider
from pimpsi.io import PimRecording, trim_pim_binary
from pimpsi.measure import DEFAULT_METRIC, measure_roi_per_frame, measure_roi_toi
from pimpsi.roi import Roi
from pimpsi.session import AnalysisSession
from pimpsi.toi import Toi


CHANNELS = ["intensity", "variance", "contrast", "perfusion"]
COLORMAPS = ["viridis", "gray", "inferno", "magma", "plasma", "turbo", "cividis"]
EXPORT_COLORMAPS = ["viridis", "inferno", "magma", "plasma", "turbo", "cividis"]


class ChannelWindow(QtWidgets.QMainWindow):
    """Independent channel/frame viewer sharing the same lazy recording API."""

    def __init__(
        self,
        recording: PimRecording,
        channel: str = "intensity",
        frame_index: int = 0,
        rois: list[Roi] | None = None,
        colormap: str | None = None,
    ):
        super().__init__()
        self.recording = recording
        self.provider = LazyFrameProvider(recording)
        self.roi_items: dict[str, pg.ROI] = {}
        self.setWindowTitle(f"PIMSoft PSI - {channel}")
        self.resize(720, 520)

        self.image_view = ImageView()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(CHANNELS)
        self.mode_combo.setCurrentText(channel)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.setCurrentText(colormap or _default_colormap(channel))
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(1, recording.header.n_frames)
        self.frame_spin.setValue(frame_index + 1)

        toolbar = QtWidgets.QToolBar()
        toolbar.addWidget(QtWidgets.QLabel("Channel"))
        toolbar.addWidget(self.mode_combo)
        toolbar.addWidget(QtWidgets.QLabel("Map"))
        toolbar.addWidget(self.colormap_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Frame"))
        toolbar.addWidget(self.frame_spin)
        self.addToolBar(toolbar)
        self.setCentralWidget(self.image_view)

        self.mode_combo.currentTextChanged.connect(self._channel_changed)
        self.colormap_combo.currentTextChanged.connect(self._set_colormap)
        self.frame_spin.valueChanged.connect(self.refresh)
        self.set_rois(rois or [])
        self.image_view.set_colormap(self.colormap_combo.currentText())
        self.refresh()

    def refresh(self) -> None:
        channel = self.mode_combo.currentText()
        self.setWindowTitle(f"PIMSoft PSI - {channel}")
        self.image_view.set_image(self.provider.frame(channel, self.frame_spin.value() - 1))

    def _set_colormap(self, name: str) -> None:
        self.image_view.set_colormap(name)

    def _channel_changed(self) -> None:
        self.image_view.reset_levels()
        self.refresh()

    def set_frame(self, frame_index: int) -> None:
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(frame_index + 1)
        self.frame_spin.blockSignals(False)
        self.refresh()

    def set_rois(self, rois: list[Roi]) -> None:
        for item in list(self.roi_items.values()):
            self.image_view.remove_item(item)
        self.roi_items.clear()
        for roi in rois:
            item = _make_roi_item(roi, editable=False)
            self.roi_items[roi.id] = item
            self.image_view.add_item(item)


class MainWindow(QtWidgets.QMainWindow):
    """Session editor with lazy image viewing, ROI/TOI management, and export tools."""

    frame_changed = QtCore.Signal(int)

    def __init__(self, recording_path: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("PIMSoft PSI")
        self.resize(1280, 820)

        self.recording: PimRecording | None = None
        self.session: AnalysisSession | None = None
        self.session_path: Path | None = None
        self.frame_provider: LazyFrameProvider | None = None
        self.current_frame_index = 0
        self.roi_items: dict[str, pg.ROI] = {}
        self.secondary_roi_items: dict[str, pg.ROI] = {}
        self.channel_windows: list[ChannelWindow] = []
        self._updating_tables = False
        self._syncing_roi_items = False
        self._armed_roi_shape: str | None = None
        self._roi_tool_filter_installed = False
        self._trace_cache: dict[tuple[str, str, tuple[tuple[float, float], ...]], list[float]] = {}
        self._dirty_trace_rois: set[str] = set()
        self._trace_update_timer = QtCore.QTimer(self)
        self._trace_update_timer.setSingleShot(True)
        self._trace_update_timer.setInterval(500)
        self._trace_update_timer.timeout.connect(self.update_trace)

        self.image_view = ImageView()
        self.second_image_view = ImageView()
        self.second_provider: LazyFrameProvider | None = None
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(CHANNELS)
        self.secondary_mode_combo = QtWidgets.QComboBox()
        self.secondary_mode_combo.addItems(CHANNELS)
        self.secondary_mode_combo.setCurrentText("perfusion")
        self.top_colormap_combo = QtWidgets.QComboBox()
        self.top_colormap_combo.addItems(COLORMAPS)
        self.top_colormap_combo.setCurrentText("gray")
        self.bottom_colormap_combo = QtWidgets.QComboBox()
        self.bottom_colormap_combo.addItems(COLORMAPS)
        self.bottom_colormap_combo.setCurrentText("turbo")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setEnabled(False)
        self.status_label = QtWidgets.QLabel("No recording")

        self.roi_table = QtWidgets.QTableWidget(0, 5)
        self.roi_table.setHorizontalHeaderLabels(["Name", "Type", "Color", "Visible", "ID"])
        self.roi_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.toi_table = QtWidgets.QTableWidget(0, 7)
        self.toi_table.setHorizontalHeaderLabels(["Name", "Start", "End", "Include End", "Color", "Visible", "ID"])
        self.toi_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.toi_table.horizontalHeader().setStretchLastSection(True)

        self.trace_channel_combo = QtWidgets.QComboBox()
        self.trace_channel_combo.addItems(CHANNELS)
        self.trace_channel_combo.setCurrentText("intensity")
        self.auto_trace_checkbox = QtWidgets.QCheckBox("Auto")
        self.auto_trace_checkbox.setChecked(True)
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setLabel("bottom", "Frame")
        self.trace_plot.setLabel("left", "Mean value")
        self.trace_plot.addLegend(offset=(8, 8))

        self.metadata_table = QtWidgets.QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.verticalHeader().hide()
        self.metadata_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        self._build_actions()
        self._build_layout()
        self._connect_signals()
        self.image_view.set_colormap(self.top_colormap_combo.currentText())
        self.second_image_view.set_colormap(self.bottom_colormap_combo.currentText())
        self.image_view._view.scene.installEventFilter(self)
        self.second_image_view._view.scene.installEventFilter(self)

        if recording_path is not None:
            self.open_recording(recording_path)

    def _build_actions(self) -> None:
        self.action_open = QtGui.QAction("Open Binary", self)
        self.action_open.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.action_load_session = QtGui.QAction("Load Session", self)
        self.action_save_session = QtGui.QAction("Save Session", self)
        self.action_save_session.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.action_save_session_as = QtGui.QAction("Save Session As", self)
        self.action_export_images = QtGui.QAction("Images", self)
        self.action_export_roi_mask = QtGui.QAction("Save ROI(s) as TIFF", self)
        self.action_export_measurements = QtGui.QAction("Measurements CSV", self)
        self.action_trim_binary = QtGui.QAction("Trim Binary", self)
        self.action_new_channel = QtGui.QAction("New Channel Window", self)
        self.action_new_channel.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_load_session)
        file_menu.addAction(self.action_save_session)
        file_menu.addAction(self.action_save_session_as)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.action_new_channel)

        # extract_menu = self.menuBar().addMenu("Extract")

        export_menu = self.menuBar().addMenu("Export")
        export_menu.addAction(self.action_trim_binary)
        export_menu.addAction(self.action_export_images)
        export_menu.addAction(self.action_export_roi_mask)
        export_menu.addSeparator()
        export_menu.addAction(self.action_export_measurements)

    def _build_layout(self) -> None:
        toolbar = QtWidgets.QToolBar()
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet(
            """
            QToolBar#mainToolbar {
                spacing: 6px;
                padding: 6px;
                border-bottom: 1px solid #cfd7df;
                background: #f6f8fb;
            }
            QToolBar#mainToolbar QToolButton {
                padding: 5px 10px;
                border: 1px solid #b9c4d0;
                border-radius: 4px;
                background: #ffffff;
            }
            QToolBar#mainToolbar QToolButton:hover {
                border-color: #3b82f6;
                background: #eef5ff;
            }
            """
        )
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_save_session)
        toolbar.addAction(self.action_new_channel)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Top"))
        toolbar.addWidget(self.mode_combo)
        toolbar.addWidget(self.top_colormap_combo)
        toolbar.addWidget(QtWidgets.QLabel("Bottom"))
        toolbar.addWidget(self.secondary_mode_combo)
        toolbar.addWidget(self.bottom_colormap_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Frame"))
        toolbar.addWidget(self.frame_spin)
        self.addToolBar(toolbar)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.addWidget(self.frame_slider, 1)
        controls_layout.addWidget(self.status_label)

        image_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        image_splitter.addWidget(self.image_view)
        image_splitter.addWidget(self.second_image_view)
        image_splitter.setStretchFactor(0, 1)
        image_splitter.setStretchFactor(1, 1)

        viewer_panel = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(image_splitter, 1)
        viewer_layout.addWidget(controls)

        side_tabs = QtWidgets.QTabWidget()
        side_tabs.addTab(self._roi_panel(), "ROIs")
        side_tabs.addTab(self._toi_panel(), "TOIs")
        side_tabs.addTab(self.metadata_table, "Metadata")

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_panel.addWidget(side_tabs)
        right_panel.addWidget(self._trace_panel())
        right_panel.setStretchFactor(0, 3)
        right_panel.setStretchFactor(1, 2)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(viewer_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([860, 430])
        self.setCentralWidget(splitter)

    def _roi_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        buttons = QtWidgets.QHBoxLayout()
        self.add_rect_button = QtWidgets.QToolButton()
        self.add_rect_button.setText("Rect")
        self.add_ellipse_button = QtWidgets.QToolButton()
        self.add_ellipse_button.setText("Round")
        self.add_polygon_button = QtWidgets.QToolButton()
        self.add_polygon_button.setText("Poly")
        self.delete_roi_button = QtWidgets.QToolButton()
        self.delete_roi_button.setText("Remove")
        self.color_roi_button = QtWidgets.QToolButton()
        self.color_roi_button.setText("Color")
        for button in [
            self.add_rect_button,
            self.add_ellipse_button,
            self.add_polygon_button,
            self.color_roi_button,
            self.delete_roi_button,
        ]:
            if button in {self.add_rect_button, self.add_ellipse_button, self.add_polygon_button}:
                button.setCheckable(True)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        layout.addWidget(self.roi_table, 1)
        return panel

    def _toi_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        buttons = QtWidgets.QHBoxLayout()
        self.add_toi_button = QtWidgets.QToolButton()
        self.add_toi_button.setText("Add")
        self.delete_toi_button = QtWidgets.QToolButton()
        self.delete_toi_button.setText("Remove")
        self.color_toi_button = QtWidgets.QToolButton()
        self.color_toi_button.setText("Color")
        buttons.addWidget(self.add_toi_button)
        buttons.addWidget(self.color_toi_button)
        buttons.addWidget(self.delete_toi_button)
        layout.addLayout(buttons)
        layout.addWidget(self.toi_table, 1)
        return panel

    def _trace_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        self.update_trace_button = QtWidgets.QPushButton("Update")
        controls.addWidget(QtWidgets.QLabel("Channel"))
        controls.addWidget(self.trace_channel_combo)
        controls.addWidget(self.auto_trace_checkbox)
        controls.addWidget(self.update_trace_button)
        layout.addLayout(controls)
        layout.addWidget(self.trace_plot, 1)
        return panel

    def _connect_signals(self) -> None:
        self.action_open.triggered.connect(self._choose_recording)
        self.action_load_session.triggered.connect(self._choose_session)
        self.action_save_session.triggered.connect(self.save_session)
        self.action_save_session_as.triggered.connect(self.save_session_as)
        self.action_new_channel.triggered.connect(self.new_channel_window)
        self.action_export_images.triggered.connect(self.export_images)
        self.action_export_roi_mask.triggered.connect(self.export_roi_mask)
        self.action_export_measurements.triggered.connect(self.export_measurements)
        self.action_trim_binary.triggered.connect(self.trim_binary)

        self.mode_combo.currentTextChanged.connect(self._top_channel_changed)
        self.secondary_mode_combo.currentTextChanged.connect(self._bottom_channel_changed)
        self.top_colormap_combo.currentTextChanged.connect(self.image_view.set_colormap)
        self.bottom_colormap_combo.currentTextChanged.connect(self.second_image_view.set_colormap)
        self.frame_slider.valueChanged.connect(self._set_frame_from_slider)
        self.frame_spin.valueChanged.connect(self._set_frame_from_spin)
        self.add_rect_button.clicked.connect(lambda checked: self.arm_roi_tool("rectangle", checked))
        self.add_ellipse_button.clicked.connect(lambda checked: self.arm_roi_tool("ellipse", checked))
        self.add_polygon_button.clicked.connect(lambda checked: self.arm_roi_tool("polygon", checked))
        self.delete_roi_button.clicked.connect(self.delete_selected_rois)
        self.color_roi_button.clicked.connect(self.change_selected_roi_color)
        self.add_toi_button.clicked.connect(self.add_toi)
        self.delete_toi_button.clicked.connect(self.delete_selected_tois)
        self.color_toi_button.clicked.connect(self.change_selected_toi_color)
        self.roi_table.itemChanged.connect(self._roi_table_item_changed)
        self.toi_table.itemChanged.connect(self._toi_table_item_changed)
        self.roi_table.itemSelectionChanged.connect(self._maybe_update_trace)
        self.toi_table.itemSelectionChanged.connect(self._maybe_update_trace)
        self.trace_channel_combo.currentTextChanged.connect(self._maybe_update_trace)
        self.update_trace_button.clicked.connect(self.update_trace)

    def _choose_recording(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open PIMSoft Binary",
            "",
            "PIMSoft binary files (*.dat *.bin);;All files (*)",
        )
        if path:
            self.open_recording(path)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if self._armed_roi_shape is None:
            return super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel_roi_tool()
            return True
        if event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress:
            target_view = self._view_for_scene(watched)
            if target_view is not None and target_view.scene_contains_image(event.scenePos()):
                image_point = target_view.map_scene_to_image(event.scenePos())
                self.add_roi(self._armed_roi_shape, center_xy=(float(image_point.x()), float(image_point.y())))
                self.cancel_roi_tool()
            return True
        if event.type() in {
            QtCore.QEvent.Type.GraphicsSceneMouseMove,
            QtCore.QEvent.Type.GraphicsSceneMouseRelease,
            QtCore.QEvent.Type.GraphicsSceneWheel,
        }:
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and self._armed_roi_shape is not None:
            self.cancel_roi_tool()
            return
        super().keyPressEvent(event)

    def arm_roi_tool(self, shape_type: str, checked: bool = True) -> None:
        if not checked:
            self.cancel_roi_tool()
            return
        if self.recording is None or self.session is None:
            self._message("Open a recording before adding ROIs.")
            self._set_roi_tool_buttons(None)
            return
        self._armed_roi_shape = shape_type
        self._set_roi_tool_buttons(shape_type)
        if not self._roi_tool_filter_installed:
            QtWidgets.QApplication.instance().installEventFilter(self)
            self._roi_tool_filter_installed = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Click a channel view to place the ROI, or press Esc to cancel.")

    def cancel_roi_tool(self) -> None:
        self._armed_roi_shape = None
        self._set_roi_tool_buttons(None)
        if self._roi_tool_filter_installed:
            QtWidgets.QApplication.instance().removeEventFilter(self)
            self._roi_tool_filter_installed = False
        if QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().clearMessage()

    def _set_roi_tool_buttons(self, active_shape: str | None) -> None:
        mapping = {
            "rectangle": self.add_rect_button,
            "ellipse": self.add_ellipse_button,
            "polygon": self.add_polygon_button,
        }
        for shape, button in mapping.items():
            button.blockSignals(True)
            button.setChecked(shape == active_shape)
            button.blockSignals(False)

    def _view_for_scene(self, scene: QtCore.QObject) -> ImageView | None:
        if scene is self.image_view._view.scene:
            return self.image_view
        if scene is self.second_image_view._view.scene:
            return self.second_image_view
        return None

    def _top_channel_changed(self) -> None:
        self.image_view.reset_levels()
        self._refresh_current_frame()

    def _bottom_channel_changed(self) -> None:
        self.second_image_view.reset_levels()
        self._refresh_current_frame()

    def open_recording(self, path: str | Path) -> None:
        self.set_recording(PimRecording.open(path))

    def set_recording(self, recording: PimRecording) -> None:
        self.recording = recording
        self.session = AnalysisSession.from_recording(recording)
        self.session_path = None
        self.frame_provider = LazyFrameProvider(recording)
        self.second_provider = LazyFrameProvider(recording)
        self._trace_cache.clear()
        self._dirty_trace_rois.clear()
        self.current_frame_index = 0
        self._clear_roi_items()

        max_frame = max(0, recording.header.n_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(1, max_frame + 1)
        self.frame_spin.setRange(1, max_frame + 1)
        self.frame_slider.setValue(1)
        self.frame_spin.setValue(1)
        self.frame_slider.setEnabled(recording.header.n_frames > 1)
        self.frame_spin.setEnabled(True)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

        self._populate_metadata()
        self._sync_tables_from_session()
        self.image_view.reset_levels()
        self.second_image_view.reset_levels()
        self._refresh_current_frame()
        self._try_load_default_session()

    def _choose_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Session", "", "PIMPSI session (*.json)")
        if path:
            self.load_session(path)

    def load_session(self, path: str | Path) -> None:
        if self.recording is None:
            self._message("Open a recording before loading a session.")
            return
        self.session = AnalysisSession.load(path)
        self.session_path = Path(path)
        self._sync_tables_from_session()
        self._refresh_roi_items()
        self._sync_channel_window_rois()
        self._maybe_update_trace()

    def _try_load_default_session(self) -> None:
        if self.recording is None:
            return
        path = self._default_session_path()
        if path.exists():
            self.load_session(path)

    def _default_session_path(self) -> Path:
        if self.recording is None:
            return Path("recording.pimpsi.json")
        return self.recording.path.with_suffix(".pimpsi.json")

    def save_session(self) -> None:
        if self.session is None:
            self._message("There is no session to save.")
            return
        if self.session_path is None:
            self.session_path = self._default_session_path()
        self._sync_session_from_roi_items()
        self.session.save(self.session_path)
        self.statusBar().showMessage(f"Saved session to {self.session_path}", 5000)

    def save_session_as(self) -> None:
        if self.session is None:
            self._message("There is no session to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session",
            "recording.pimpsi.json",
            "PIMPSI session (*.json)",
        )
        if path:
            self.session_path = Path(path)
            self.save_session()

    def add_roi(self, shape_type: str, center_xy: tuple[float, float] | None = None) -> None:
        if self.recording is None or self.session is None:
            self._message("Open a recording before adding ROIs.")
            return
        width = self.recording.header.image_width
        height = self.recording.header.image_height
        color = _next_color(len(self.session.rois))
        center_x, center_y = center_xy or (width * 0.5, height * 0.5)
        roi_width = max(width * 0.12, 4.0)
        roi_height = max(height * 0.12, 4.0)
        left = _clamp(center_x - roi_width / 2.0, 0.0, max(width - roi_width, 0.0))
        top = _clamp(center_y - roi_height / 2.0, 0.0, max(height - roi_height, 0.0))
        right = min(left + roi_width, width)
        bottom = min(top + roi_height, height)
        if shape_type == "rectangle":
            vertices = [(left, top), (right, bottom)]
        elif shape_type == "ellipse":
            vertices = [(left, top), (right, bottom)]
        else:
            vertices = [
                (center_x, top),
                (right, bottom),
                (left, bottom),
            ]
        roi = Roi(
            id=f"roi_{uuid4().hex[:8]}",
            label=f"ROI {len(self.session.rois) + 1}",
            shape_type=shape_type,
            vertices_xy=vertices,
            color=color,
        )
        self.session.rois.append(roi)
        self._dirty_trace_rois.add(roi.id)
        self._add_roi_item(roi)
        self._sync_tables_from_session()
        self._sync_channel_window_rois()
        self._maybe_update_trace()

    def delete_selected_rois(self) -> None:
        if self.session is None:
            return
        selected_ids = self._selected_roi_ids()
        self.session.rois = [roi for roi in self.session.rois if roi.id not in selected_ids]
        for roi_id in selected_ids:
            self._dirty_trace_rois.discard(roi_id)
            self._drop_trace_cache_for_roi(roi_id)
            item = self.roi_items.pop(roi_id, None)
            if item is not None:
                self.image_view.remove_item(item)
            item = self.secondary_roi_items.pop(roi_id, None)
            if item is not None:
                self.second_image_view.remove_item(item)
        self._sync_tables_from_session()
        self._maybe_update_trace()
        self._sync_channel_window_rois()

    def change_selected_roi_color(self) -> None:
        if self.session is None:
            return
        selected = self._selected_roi_ids()
        if not selected:
            return
        color = QtWidgets.QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        for roi in self.session.rois:
            if roi.id in selected:
                roi.color = color.name()
        self._refresh_roi_items()
        self._sync_tables_from_session()
        self._sync_channel_window_rois()

    def add_toi(self) -> None:
        if self.recording is None or self.session is None:
            self._message("Open a recording before adding TOIs.")
            return
        start = self.current_frame_index
        end = min(self.recording.header.n_frames - 1, start + 10)
        self.session.tois.append(
            Toi(
                id=f"toi_{uuid4().hex[:8]}",
                label=f"TOI {len(self.session.tois) + 1}",
                frame_start=start,
                frame_end=end,
                include_end=True,
                color=_next_color(len(self.session.tois)),
            )
        )
        self._sync_tables_from_session()
        self._maybe_update_trace()

    def delete_selected_tois(self) -> None:
        if self.session is None:
            return
        selected_ids = self._selected_toi_ids()
        self.session.tois = [toi for toi in self.session.tois if toi.id not in selected_ids]
        self._sync_tables_from_session()
        self._maybe_update_trace()

    def change_selected_toi_color(self) -> None:
        if self.session is None:
            return
        selected = self._selected_toi_ids()
        if not selected:
            return
        color = QtWidgets.QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        for toi in self.session.tois:
            if toi.id in selected:
                toi.color = color.name()
        self._sync_tables_from_session()

    def new_channel_window(self) -> None:
        if self.recording is None:
            self._message("Open a recording first.")
            return
        window = ChannelWindow(
            self.recording,
            channel=self.mode_combo.currentText(),
            frame_index=self.current_frame_index,
            colormap=self.top_colormap_combo.currentText(),
            rois=self.session.rois if self.session is not None else [],
        )
        self.channel_windows.append(window)
        self.frame_changed.connect(window.set_frame)
        window.destroyed.connect(lambda: self.channel_windows.remove(window) if window in self.channel_windows else None)
        window.show()

    def export_current_image(self) -> None:
        self.export_images()

    def export_stack(self) -> None:
        self.export_images()

    def export_average(self) -> None:
        self.export_images()

    def export_images(self) -> None:
        if self.recording is None:
            return
        options = self._image_export_options()
        if options is None:
            return
        directory = self._choose_export_directory("Export Images")
        if directory is None:
            return
        exported = export_image_set(
            self.recording,
            frames=[frame - 1 for frame in options["frames"]],
            channels=options["channels"],
            output_dir=directory,
            single_frame=options["single_frame"],
            stacked=options["stacked"],
            averaged=options["averaged"],
            color=options["color"],
            colormap=options["colormap"],
            color_limits=options["color_limits"],
            rois=options["rois"] if options["draw_rois"] else [],
        )
        self.statusBar().showMessage(f"Exported {len(exported)} image file(s) to {directory}", 5000)

    def export_roi_mask(self) -> None:
        if self.recording is None or self.session is None:
            return
        rois = self._roi_mask_export_options()
        if rois is None:
            return
        if not rois:
            self._message("Add or select at least one ROI before exporting a mask.")
            return
        directory = self._choose_export_directory("Save ROI(s) as TIFF")
        if directory is None:
            return
        path = directory / f"{self.recording.path.stem}_roi_masks.tif"
        export_roi_mask(self.recording, path, rois)
        self.statusBar().showMessage(f"Exported ROI TIFF to {path}", 5000)

    def export_measurements(self) -> None:
        if self.recording is None or self.session is None:
            return
        options = self._measurement_export_options()
        if options is None:
            return
        directory = self._choose_export_directory("Export Measurements")
        if directory is None:
            return
        path = directory / f"{self.recording.path.stem}_measurements.csv"
        self._sync_session_from_roi_items()
        results = []
        for roi in options["rois"]:
            for toi in options["tois"]:
                for metric in options["metrics"]:
                    results.append(
                        measure_roi_toi(
                            self.recording,
                            roi,
                            toi,
                            metric=metric,
                            perfusion_clip_upper=self.session.processing_profile.perfusion_clip_upper,
                        )
                    )
            if options["frame_by_frame"]:
                results.extend(
                    measure_roi_per_frame(
                        self.recording,
                        roi,
                        frames=options["frames"],
                        perfusion_clip_upper=self.session.processing_profile.perfusion_clip_upper,
                    )
                )
        write_measurement_csv(path, results)
        self.statusBar().showMessage(f"Exported {len(results)} measurements to {path}", 5000)

    def trim_binary(self) -> None:
        if self.recording is None:
            return
        frames = self._trim_binary_options()
        if frames is None:
            return
        start = frames[0]
        end = frames[-1]
        default_name = f"{self.recording.path.stem}_frames_{start + 1}-{end + 1}{self.recording.path.suffix}"
        path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Trimmed Binary",
            str(self.recording.path.with_name(default_name)),
            "PIMSoft binary (*.dat);;All files (*)",
        )
        if not path_text:
            return
        try:
            path = trim_pim_binary(self.recording, path_text, frames)
        except Exception as exc:
            self._message(f"Could not trim binary: {exc}")
            return
        self.statusBar().showMessage(f"Saved trimmed binary to {path}", 5000)

    def _choose_export_directory(self, title: str) -> Path | None:
        if self.recording is None:
            return None
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            title,
            str(self.recording.path.parent),
        )
        return Path(directory) if directory else None

    def update_trace(self) -> None:
        if self.recording is None or self.session is None:
            return
        rois = [roi for roi in self.session.rois if roi.visible]
        if not rois:
            self.trace_plot.clear()
            return
        trace_frames = list(range(self.recording.header.n_frames))
        if not trace_frames:
            return

        channel = self.trace_channel_combo.currentText()
        provider = self.frame_provider or LazyFrameProvider(self.recording)
        self.trace_plot.clear()
        self._add_toi_trace_regions()
        self.statusBar().showMessage(f"Updating {channel} trace over {len(trace_frames)} frame(s)...")
        QtWidgets.QApplication.processEvents()
        for roi in rois:
            values = self._trace_values_for_roi(roi, channel, trace_frames, provider)
            self.trace_plot.plot(
                trace_frames,
                values,
                pen=pg.mkPen(roi.color, width=2),
                name=roi.label,
            )
        self.trace_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.trace_plot.autoRange()
        self.statusBar().showMessage("Trace updated", 3000)

    def _add_toi_trace_regions(self) -> None:
        if self.session is None:
            return
        for toi in self.session.tois:
            if not toi.visible:
                continue
            stop = toi.frame_end + 1 if toi.include_end else toi.frame_end
            fill = QtGui.QColor(toi.color)
            fill.setAlpha(52)
            region = pg.LinearRegionItem(
                values=(toi.frame_start, stop),
                orientation="vertical",
                brush=pg.mkBrush(fill),
                pen=pg.mkPen(QtGui.QColor(toi.color), width=1),
                movable=False,
            )
            region.setZValue(-10)
            self.trace_plot.addItem(region)

    def _set_frame_from_slider(self, frame_index: int) -> None:
        if self.frame_spin.value() != frame_index:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame_index)
            self.frame_spin.blockSignals(False)
        self._set_frame(frame_index - 1)

    def _set_frame_from_spin(self, frame_index: int) -> None:
        if self.frame_slider.value() != frame_index:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)
        self._set_frame(frame_index - 1)

    def _set_frame(self, frame_index: int) -> None:
        self.current_frame_index = frame_index
        self._refresh_current_frame()

    def _refresh_current_frame(self) -> None:
        if self.recording is None or self.frame_provider is None or self.second_provider is None:
            return
        top_mode = self.mode_combo.currentText()
        bottom_mode = self.secondary_mode_combo.currentText()
        top_image = self.frame_provider.frame(top_mode, self.current_frame_index)
        bottom_image = self.second_provider.frame(bottom_mode, self.current_frame_index)
        self.image_view.set_image(top_image)
        self.second_image_view.set_image(bottom_image)
        self.status_label.setText(
            f"{self.current_frame_index + 1}/{self.recording.header.n_frames} | "
            f"top {top_mode} | bottom {bottom_mode}"
        )
        self.frame_changed.emit(self.current_frame_index)

    def _add_roi_item(self, roi: Roi) -> None:
        item = _make_roi_item(roi, editable=True)
        self.roi_items[roi.id] = item
        self.image_view.add_item(item)
        item.sigRegionChanged.connect(lambda _=None, roi_id=roi.id: self._roi_item_changed(roi_id, "top"))
        secondary_item = _make_roi_item(roi, editable=True)
        self.secondary_roi_items[roi.id] = secondary_item
        self.second_image_view.add_item(secondary_item)
        secondary_item.sigRegionChanged.connect(lambda _=None, roi_id=roi.id: self._roi_item_changed(roi_id, "bottom"))

    def _roi_item_changed(self, roi_id: str, source: str) -> None:
        if self.session is None:
            return
        item = self.roi_items.get(roi_id) if source == "top" else self.secondary_roi_items.get(roi_id)
        roi = self._roi_by_id(roi_id)
        if item is None or roi is None or self._syncing_roi_items:
            return
        roi.vertices_xy = _vertices_from_item(item, roi.shape_type)
        self._dirty_trace_rois.add(roi.id)
        self._drop_trace_cache_for_roi(roi.id)
        self._sync_roi_item_geometry(roi, source=source)
        self._update_roi_row(roi)
        self._maybe_update_trace()
        self._sync_channel_window_rois()

    def _refresh_roi_items(self) -> None:
        if self.session is None:
            self._clear_roi_items()
            return
        self._clear_roi_items()
        for roi in self.session.rois:
            self._add_roi_item(roi)
        self._sync_channel_window_rois()

    def _clear_roi_items(self) -> None:
        for item in list(self.roi_items.values()):
            try:
                self.image_view.remove_item(item)
            except Exception:
                pass
        self.roi_items.clear()
        for item in list(self.secondary_roi_items.values()):
            try:
                self.second_image_view.remove_item(item)
            except Exception:
                pass
        self.secondary_roi_items.clear()

    def _sync_roi_item_geometry(self, roi: Roi, source: str | None = None) -> None:
        targets = []
        if source != "top" and roi.id in self.roi_items:
            targets.append(self.roi_items[roi.id])
        if source != "bottom" and roi.id in self.secondary_roi_items:
            targets.append(self.secondary_roi_items[roi.id])
        if not targets:
            return
        self._syncing_roi_items = True
        try:
            for item in targets:
                _set_roi_item_geometry(item, roi)
        finally:
            self._syncing_roi_items = False

    def _sync_channel_window_rois(self) -> None:
        rois = self.session.rois if self.session is not None else []
        for window in list(self.channel_windows):
            window.set_rois(rois)

    def _sync_tables_from_session(self) -> None:
        self._updating_tables = True
        try:
            self.roi_table.setRowCount(0)
            self.toi_table.setRowCount(0)
            if self.session is None:
                return
            for roi in self.session.rois:
                row = self.roi_table.rowCount()
                self.roi_table.insertRow(row)
                self._set_item(self.roi_table, row, 0, roi.label)
                self._set_item(self.roi_table, row, 1, roi.shape_type)
                color_item = self._set_item(self.roi_table, row, 2, roi.color)
                color_item.setBackground(QtGui.QColor(roi.color))
                visible_item = self._set_item(self.roi_table, row, 3, "")
                visible_item.setCheckState(QtCore.Qt.CheckState.Checked if roi.visible else QtCore.Qt.CheckState.Unchecked)
                self._set_item(self.roi_table, row, 4, roi.id)
            for toi in self.session.tois:
                row = self.toi_table.rowCount()
                self.toi_table.insertRow(row)
                self._set_item(self.toi_table, row, 0, toi.label)
                self._set_item(self.toi_table, row, 1, str(toi.frame_start + 1))
                self._set_item(self.toi_table, row, 2, str(toi.frame_end + 1))
                include_item = self._set_item(self.toi_table, row, 3, "")
                include_item.setCheckState(
                    QtCore.Qt.CheckState.Checked if toi.include_end else QtCore.Qt.CheckState.Unchecked
                )
                color_item = self._set_item(self.toi_table, row, 4, toi.color)
                color_item.setBackground(QtGui.QColor(toi.color))
                visible_item = self._set_item(self.toi_table, row, 5, "")
                visible_item.setCheckState(QtCore.Qt.CheckState.Checked if toi.visible else QtCore.Qt.CheckState.Unchecked)
                self._set_item(self.toi_table, row, 6, toi.id)
        finally:
            self._updating_tables = False

    def _sync_session_from_roi_items(self) -> None:
        if self.session is None:
            return
        for roi in self.session.rois:
            item = self.roi_items.get(roi.id)
            if item is not None:
                roi.vertices_xy = _vertices_from_item(item, roi.shape_type)

    def _set_item(self, table: QtWidgets.QTableWidget, row: int, column: int, text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        table.setItem(row, column, item)
        return item

    def _roi_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_tables or self.session is None:
            return
        roi_id = self.roi_table.item(item.row(), 4).text()
        roi = self._roi_by_id(roi_id)
        if roi is None:
            return
        roi.label = self.roi_table.item(item.row(), 0).text()
        roi.color = self.roi_table.item(item.row(), 2).text()
        visible_item = self.roi_table.item(item.row(), 3)
        roi.visible = visible_item.checkState() == QtCore.Qt.CheckState.Checked
        roi_item = self.roi_items.get(roi.id)
        if roi_item is not None:
            roi_item.setVisible(roi.visible)
            roi_item.setPen(pg.mkPen(roi.color, width=2))
        secondary_item = self.secondary_roi_items.get(roi.id)
        if secondary_item is not None:
            secondary_item.setVisible(roi.visible)
            secondary_item.setPen(pg.mkPen(roi.color, width=2))
        self._sync_channel_window_rois()
        self._maybe_update_trace()

    def _trace_values_for_roi(
        self,
        roi: Roi,
        channel: str,
        trace_frames: list[int],
        provider: LazyFrameProvider,
    ) -> list[float]:
        cache_key = (roi.id, channel, _roi_geometry_signature(roi))
        if roi.id not in self._dirty_trace_rois and cache_key in self._trace_cache:
            return self._trace_cache[cache_key]

        if self.recording is None:
            return []
        mask = roi.to_mask((self.recording.header.image_height, self.recording.header.image_width))
        values = []
        for frame_index in trace_frames:
            image = provider.frame(channel, frame_index)
            values.append(float(image[mask].mean()) if mask.any() else np.nan)
        self._drop_trace_cache_for_roi(roi.id, channel=channel)
        self._trace_cache[cache_key] = values
        self._dirty_trace_rois.discard(roi.id)
        return values

    def _drop_trace_cache_for_roi(self, roi_id: str, channel: str | None = None) -> None:
        stale_keys = [
            key
            for key in self._trace_cache
            if key[0] == roi_id and (channel is None or key[1] == channel)
        ]
        for key in stale_keys:
            del self._trace_cache[key]

    def _toi_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_tables or self.session is None:
            return
        toi_id = self.toi_table.item(item.row(), 6).text()
        toi = self._toi_by_id(toi_id)
        if toi is None:
            return
        try:
            toi.label = self.toi_table.item(item.row(), 0).text()
            max_frame = self.recording.header.n_frames if self.recording is not None else 1
            start_display = _clamp_int(int(self.toi_table.item(item.row(), 1).text()), 1, max_frame)
            end_display = _clamp_int(int(self.toi_table.item(item.row(), 2).text()), 1, max_frame)
            if end_display < start_display:
                end_display = start_display
            toi.frame_start = start_display - 1
            toi.frame_end = end_display - 1
            toi.include_end = self.toi_table.item(item.row(), 3).checkState() == QtCore.Qt.CheckState.Checked
            toi.color = self.toi_table.item(item.row(), 4).text()
            toi.visible = self.toi_table.item(item.row(), 5).checkState() == QtCore.Qt.CheckState.Checked
        except ValueError:
            self._message("TOI start and end must be integer frame numbers.")
            self._sync_tables_from_session()
            return
        self._sync_tables_from_session()
        self._maybe_update_trace()

    def _update_roi_row(self, roi: Roi) -> None:
        for row in range(self.roi_table.rowCount()):
            if self.roi_table.item(row, 4).text() == roi.id:
                self.roi_table.item(row, 0).setText(roi.label)
                return

    def _populate_metadata(self) -> None:
        if self.recording is None:
            self.metadata_table.setRowCount(0)
            return
        header = self.recording.header
        rows = [
            ("File", str(self.recording.path)),
            ("SHA-256", header.sha256),
            ("Type", header.file_type),
            ("Version", header.file_version),
            ("Width", header.image_width),
            ("Height", header.image_height),
            ("Frames", header.n_frames),
            ("Images", header.number_of_images),
            ("Signal gain", header.signal_gain),
            ("Coherence factor", header.coherence_factor),
            ("Data offset", header.data_offset),
        ]
        self.metadata_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.metadata_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.metadata_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
        self.metadata_table.resizeColumnsToContents()

    def _selected_roi_ids(self) -> set[str]:
        return {
            self.roi_table.item(index.row(), 4).text()
            for index in self.roi_table.selectionModel().selectedRows()
        }

    def _selected_toi_ids(self) -> set[str]:
        return {
            self.toi_table.item(index.row(), 6).text()
            for index in self.toi_table.selectionModel().selectedRows()
        }

    def _selected_rois_or_all(self) -> list[Roi]:
        if self.session is None:
            return []
        selected = self._selected_roi_ids()
        return [roi for roi in self.session.rois if not selected or roi.id in selected]

    def _selected_tois_or_all(self) -> list[Toi]:
        if self.session is None:
            return []
        selected = self._selected_toi_ids()
        return [toi for toi in self.session.tois if not selected or toi.id in selected]

    def _trace_frames(self) -> list[int]:
        if self.recording is None:
            return []
        tois = self._selected_tois_or_all()
        if not tois:
            return list(range(self.recording.header.n_frames))
        frames = []
        for toi in tois:
            frames.extend(list(toi.frame_indices()))
        return sorted(set(frame for frame in frames if 0 <= frame < self.recording.header.n_frames))

    def _selected_frames_for_export(self) -> list[int]:
        frames = self._trace_frames()
        return frames or [self.current_frame_index]

    def _choose_channels(self) -> list[str]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Choose Channels")
        layout = QtWidgets.QVBoxLayout(dialog)
        checks = []
        for channel in CHANNELS:
            check = QtWidgets.QCheckBox(channel)
            check.setChecked(channel in {self.mode_combo.currentText(), self.secondary_mode_combo.currentText()})
            layout.addWidget(check)
            checks.append(check)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return []
        return [check.text() for check in checks if check.isChecked()]

    def _set_fixed_dialog_width(self, dialog: QtWidgets.QDialog, title: str, minimum: int = 520) -> None:
        title_width = dialog.fontMetrics().horizontalAdvance(title) + 160
        dialog.setFixedWidth(max(minimum, title_width))

    def _add_frame_range_controls(
        self,
        frame_layout: QtWidgets.QFormLayout,
        *,
        start_value: int,
        end_value: int,
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QSpinBox, QtWidgets.QSlider, QtWidgets.QSpinBox]:
        frame_count = self.recording.header.n_frames
        start_slider, start_spin = self._make_frame_scroll_text_pair(frame_count, start_value)
        end_slider, end_spin = self._make_frame_scroll_text_pair(frame_count, end_value)
        frame_layout.addRow("Start", self._row_widget(start_slider, start_spin))
        frame_layout.addRow("End", self._row_widget(end_slider, end_spin))
        return start_slider, start_spin, end_slider, end_spin

    def _make_frame_scroll_text_pair(
        self, frame_count: int, value: int
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QSpinBox]:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        spin = QtWidgets.QSpinBox()
        slider.setRange(1, frame_count)
        spin.setRange(1, frame_count)
        slider.setValue(value)
        spin.setValue(value)
        spin.setFixedWidth(76)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return slider, spin

    def _row_widget(self, *widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return row

    def _add_toi_frame_controls(
        self,
        frame_layout: QtWidgets.QFormLayout,
        *,
        start_slider: QtWidgets.QSlider,
        start_spin: QtWidgets.QSpinBox,
        end_slider: QtWidgets.QSlider,
        end_spin: QtWidgets.QSpinBox,
    ) -> tuple[QtWidgets.QCheckBox, list[tuple[QtWidgets.QRadioButton, Toi]]]:
        use_toi_check = QtWidgets.QCheckBox("Use TOI")
        frame_layout.addRow(use_toi_check)
        toi_group = QtWidgets.QGroupBox("TOIs")
        toi_layout = QtWidgets.QVBoxLayout(toi_group)
        toi_buttons = []
        for toi in (self.session.tois if self.session is not None else []):
            start, end = self._toi_export_display_range(toi)
            radio = QtWidgets.QRadioButton(f"{toi.label} ({start}-{end})")
            toi_layout.addWidget(radio)
            toi_buttons.append((radio, toi))
        if not toi_buttons:
            toi_layout.addWidget(QtWidgets.QLabel("No TOIs available"))
        toi_group.setEnabled(False)
        frame_layout.addRow(toi_group)

        def apply_selected_toi() -> None:
            selected_toi = next((toi for radio, toi in toi_buttons if radio.isChecked()), None)
            if selected_toi is None:
                return
            start, end = self._toi_export_display_range(selected_toi)
            start_spin.setValue(start)
            end_spin.setValue(end)

        def update_toi_state(checked: bool) -> None:
            toi_group.setEnabled(checked and bool(toi_buttons))
            start_slider.setEnabled(not checked)
            start_spin.setEnabled(not checked)
            end_slider.setEnabled(not checked)
            end_spin.setEnabled(not checked)
            if checked and toi_buttons and not any(radio.isChecked() for radio, _ in toi_buttons):
                toi_buttons[0][0].setChecked(True)
            if checked:
                apply_selected_toi()

        use_toi_check.toggled.connect(update_toi_state)
        for radio, _ in toi_buttons:
            radio.toggled.connect(lambda checked: apply_selected_toi() if checked else None)
        return use_toi_check, toi_buttons

    def _toi_export_display_range(self, toi: Toi) -> tuple[int, int]:
        try:
            frames = list(toi.frame_indices())
        except ValueError:
            frames = [toi.frame_start]
        if not frames:
            frames = [toi.frame_start]
        frame_count = self.recording.header.n_frames if self.recording is not None else 1
        start = _clamp_int(frames[0] + 1, 1, frame_count)
        end = _clamp_int(frames[-1] + 1, 1, frame_count)
        return min(start, end), max(start, end)

    def _channel_scale_defaults(self, channel: str) -> tuple[float, float]:
        if self.recording is None:
            return 0.0, 1.0
        image = self.frame_provider.frame(channel, self.current_frame_index) if self.frame_provider is not None else None
        if image is None:
            return 0.0, 1.0
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(finite.min())
        high = float(finite.max())
        if high <= low:
            high = low + 1.0
        return low, high

    def _add_color_scale_controls(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
        channel_combo: QtWidgets.QComboBox,
        colormap_combo: QtWidgets.QComboBox,
    ):
        scale_box = QtWidgets.QGroupBox("Color Mapping Scale")
        scale_layout = QtWidgets.QFormLayout(scale_box)
        low_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        high_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        low_spin = QtWidgets.QDoubleSpinBox()
        high_spin = QtWidgets.QDoubleSpinBox()
        for slider in (low_slider, high_slider):
            slider.setRange(0, 1000)
        for spin in (low_spin, high_spin):
            spin.setDecimals(6)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setFixedWidth(116)
        low_slider.setValue(0)
        high_slider.setValue(1000)
        scale_layout.addRow("Low", self._row_widget(low_slider, low_spin))
        scale_layout.addRow("High", self._row_widget(high_slider, high_spin))
        parent_layout.addWidget(scale_box)

        state = {"low": 0.0, "high": 1.0, "syncing": False, "channel": channel_combo.currentText()}
        settings: dict[str, dict[str, object]] = {}

        def export_default_colormap(channel: str) -> str:
            name = _default_colormap(channel)
            return name if name in EXPORT_COLORMAPS else "viridis"

        def ensure_settings(channel: str) -> dict[str, object]:
            if channel not in settings:
                low, high = self._channel_scale_defaults(channel)
                settings[channel] = {
                    "colormap": export_default_colormap(channel),
                    "limits": (low, high),
                    "range": (low, high),
                }
            return settings[channel]

        def value_from_slider(slider_value: int) -> float:
            return state["low"] + ((state["high"] - state["low"]) * slider_value / 1000.0)

        def slider_from_value(value: float) -> int:
            if state["high"] <= state["low"]:
                return 0
            return int(round((value - state["low"]) / (state["high"] - state["low"]) * 1000.0))

        def save_current_settings() -> None:
            channel = state["channel"]
            if not channel:
                return
            ensure_settings(channel)
            settings[channel]["colormap"] = colormap_combo.currentText()
            settings[channel]["limits"] = (low_spin.value(), high_spin.value())

        def load_channel_settings(channel: str) -> None:
            if not channel:
                return
            channel_settings = ensure_settings(channel)
            low, high = channel_settings["range"]
            value_low, value_high = channel_settings["limits"]
            state["low"] = low
            state["high"] = high
            state["syncing"] = True
            colormap_combo.setCurrentText(str(channel_settings["colormap"]))
            for spin in (low_spin, high_spin):
                spin.setRange(low, high)
                spin.setSingleStep((high - low) / 100.0)
            low_spin.setValue(float(value_low))
            high_spin.setValue(float(value_high))
            low_slider.setValue(_clamp_int(slider_from_value(float(value_low)), 0, 1000))
            high_slider.setValue(_clamp_int(slider_from_value(float(value_high)), 0, 1000))
            state["syncing"] = False

        def low_slider_changed(value: int) -> None:
            if not state["syncing"]:
                low_spin.setValue(min(value_from_slider(value), high_spin.value()))

        def high_slider_changed(value: int) -> None:
            if not state["syncing"]:
                high_spin.setValue(max(value_from_slider(value), low_spin.value()))

        def spin_changed() -> None:
            if state["syncing"]:
                return
            state["syncing"] = True
            low_slider.setValue(_clamp_int(slider_from_value(low_spin.value()), 0, 1000))
            high_slider.setValue(_clamp_int(slider_from_value(high_spin.value()), 0, 1000))
            state["syncing"] = False
            save_current_settings()

        def channel_changed(channel: str) -> None:
            if state["syncing"]:
                return
            save_current_settings()
            state["channel"] = channel
            load_channel_settings(channel)

        def colormap_changed() -> None:
            if not state["syncing"]:
                save_current_settings()

        low_slider.valueChanged.connect(low_slider_changed)
        high_slider.valueChanged.connect(high_slider_changed)
        low_spin.valueChanged.connect(spin_changed)
        high_spin.valueChanged.connect(spin_changed)
        channel_combo.currentTextChanged.connect(channel_changed)
        colormap_combo.currentTextChanged.connect(colormap_changed)
        ensure_settings(channel_combo.currentText())
        load_channel_settings(channel_combo.currentText())
        return scale_box, low_spin, high_spin, settings, save_current_settings

    def _image_export_options(self) -> dict | None:
        if self.recording is None:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Image Export Options")
        layout = QtWidgets.QVBoxLayout(dialog)

        frame_box = QtWidgets.QGroupBox("Frames")
        frame_layout = QtWidgets.QFormLayout(frame_box)
        start_slider, start_spin, end_slider, end_spin = self._add_frame_range_controls(
            frame_layout,
            start_value=self.current_frame_index + 1,
            end_value=self.recording.header.n_frames,
        )
        self._add_toi_frame_controls(
            frame_layout,
            start_slider=start_slider,
            start_spin=start_spin,
            end_slider=end_slider,
            end_spin=end_spin,
        )
        layout.addWidget(frame_box)

        channel_checks = []
        channel_combo = QtWidgets.QComboBox()
        channel_box = QtWidgets.QGroupBox("Channels")
        channel_layout = QtWidgets.QVBoxLayout(channel_box)
        for channel in CHANNELS:
            check = QtWidgets.QCheckBox(channel)
            check.setChecked(channel in {self.mode_combo.currentText(), self.secondary_mode_combo.currentText()})
            channel_layout.addWidget(check)
            channel_checks.append(check)
            channel_combo.addItem(channel)
        channel_combo.setCurrentText(self.mode_combo.currentText())
        layout.addWidget(channel_box)

        option_box = QtWidgets.QGroupBox("Options")
        option_layout = QtWidgets.QVBoxLayout(option_box)
        stacked_check = QtWidgets.QCheckBox("Stacked images")
        single_check = QtWidgets.QCheckBox("Single frame images")
        averaged_check = QtWidgets.QCheckBox("Average image")
        single_check.setChecked(True)
        color_check = QtWidgets.QCheckBox("RGB representative image")
        color_note = QtWidgets.QLabel("RGB exports are representative images only; data TIFF exports preserve original values.")
        color_note.setWordWrap(True)
        colormap_combo = QtWidgets.QComboBox()
        colormap_combo.addItems(EXPORT_COLORMAPS)
        default_export_map = _default_colormap(self.mode_combo.currentText())
        colormap_combo.setCurrentText(default_export_map if default_export_map in EXPORT_COLORMAPS else "viridis")
        roi_check = QtWidgets.QCheckBox("Draw selected ROIs")
        option_layout.addWidget(stacked_check)
        image_variant_box = QtWidgets.QGroupBox("Single and Average Images")
        image_variant_layout = QtWidgets.QVBoxLayout(image_variant_box)
        image_variant_layout.addWidget(single_check)
        image_variant_layout.addWidget(averaged_check)
        image_variant_layout.addWidget(color_check)
        image_variant_layout.addWidget(color_note)
        option_layout.addWidget(image_variant_box)
        layout.addWidget(option_box)
        scale_box, low_spin, high_spin, color_settings, save_color_settings = self._add_color_scale_controls(
            layout,
            channel_combo,
            colormap_combo,
        )
        scale_layout = scale_box.layout()
        if isinstance(scale_layout, QtWidgets.QFormLayout):
            scale_layout.insertRow(0, "Channel", channel_combo)
            scale_layout.insertRow(1, "Map", colormap_combo)
        scale_layout.addRow(roi_check)

        def refresh_rgb_channel_combo() -> None:
            save_color_settings()
            checked_channels = [check.text() for check in channel_checks if check.isChecked()]
            current = channel_combo.currentText()
            target = current if current in checked_channels else (checked_channels[0] if checked_channels else "")
            channel_combo.blockSignals(True)
            channel_combo.clear()
            channel_combo.addItems(checked_channels)
            channel_combo.blockSignals(False)
            if target:
                channel_combo.setCurrentText(target)
                channel_combo.currentTextChanged.emit(target)

        for check in channel_checks:
            check.toggled.connect(refresh_rgb_channel_combo)
        refresh_rgb_channel_combo()

        roi_checks = []
        roi_box = QtWidgets.QGroupBox("ROIs")
        roi_layout = QtWidgets.QVBoxLayout(roi_box)
        selected = self._selected_roi_ids()
        if self.session is not None and self.session.rois:
            for roi in self.session.rois:
                check = QtWidgets.QCheckBox(roi.label)
                check.setChecked(not selected or roi.id in selected)
                roi_layout.addWidget(check)
                roi_checks.append((check, roi))
        else:
            roi_layout.addWidget(QtWidgets.QLabel("No ROIs available"))
        layout.addWidget(roi_box)

        def update_color_options() -> None:
            has_rgb_source = single_check.isChecked() or averaged_check.isChecked()
            color_check.setEnabled(has_rgb_source)
            if not has_rgb_source and color_check.isChecked():
                color_check.setChecked(False)
            enabled = has_rgb_source and color_check.isChecked()
            color_note.setEnabled(enabled)
            channel_combo.setEnabled(enabled)
            colormap_combo.setEnabled(enabled)
            scale_box.setEnabled(enabled)
            roi_check.setEnabled(enabled)
            roi_box.setEnabled(enabled and roi_check.isChecked())

        single_check.toggled.connect(update_color_options)
        stacked_check.toggled.connect(update_color_options)
        averaged_check.toggled.connect(update_color_options)
        color_check.toggled.connect(update_color_options)
        roi_check.toggled.connect(update_color_options)
        update_color_options()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._set_fixed_dialog_width(dialog, "Image Export Options")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        start = min(start_spin.value(), end_spin.value())
        end = max(start_spin.value(), end_spin.value())
        channels = [check.text() for check in channel_checks if check.isChecked()]
        if not channels:
            self._message("Select at least one channel.")
            return None
        single = single_check.isChecked()
        stacked = stacked_check.isChecked()
        averaged = averaged_check.isChecked()
        color = color_check.isChecked()
        if color and not (single or averaged):
            self._message("RGB export is available for single frame or average image exports.")
            return None
        if not any([single, stacked, averaged]):
            self._message("Select at least one image export option.")
            return None
        if color and high_spin.value() <= low_spin.value():
            self._message("Set the color mapping high value above the low value.")
            return None
        save_color_settings()
        for channel in channels:
            if channel not in color_settings:
                low, high = self._channel_scale_defaults(channel)
                default_map = _default_colormap(channel)
                color_settings[channel] = {
                    "colormap": default_map if default_map in EXPORT_COLORMAPS else "viridis",
                    "limits": (low, high),
                    "range": (low, high),
                }
        colormaps = {
            channel: str(color_settings[channel]["colormap"])
            for channel in channels
            if channel in color_settings
        }
        color_limits = {
            channel: color_settings[channel]["limits"]
            for channel in channels
            if channel in color_settings
        }
        draw_rois = color and roi_check.isChecked()
        rois = [roi for check, roi in roi_checks if check.isChecked()]
        return {
            "frames": list(range(start, end + 1)),
            "channels": channels,
            "single_frame": single,
            "stacked": stacked,
            "averaged": averaged,
            "color": color,
            "colormap": colormaps if color else colormap_combo.currentText(),
            "color_limits": color_limits if color else None,
            "draw_rois": draw_rois,
            "rois": rois,
        }

    def _roi_mask_export_options(self) -> list[Roi] | None:
        if self.session is None or not self.session.rois:
            self._message("Add at least one ROI before exporting masks.")
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save ROI(s) as TIFF")
        layout = QtWidgets.QVBoxLayout(dialog)
        roi_box = QtWidgets.QGroupBox("ROIs")
        roi_layout = QtWidgets.QVBoxLayout(roi_box)
        selected = self._selected_roi_ids()
        roi_checks = []
        for roi in self.session.rois:
            check = QtWidgets.QCheckBox(roi.label)
            check.setChecked(not selected or roi.id in selected)
            roi_layout.addWidget(check)
            roi_checks.append((check, roi))
        layout.addWidget(roi_box)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._set_fixed_dialog_width(dialog, "Save ROI(s) as TIFF")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        rois = [roi for check, roi in roi_checks if check.isChecked()]
        if not rois:
            self._message("Select at least one ROI.")
            return None
        return rois

    def _trim_binary_options(self) -> list[int] | None:
        if self.recording is None:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Trim Binary")
        layout = QtWidgets.QVBoxLayout(dialog)

        frame_box = QtWidgets.QGroupBox("Frames")
        frame_layout = QtWidgets.QFormLayout(frame_box)
        start_slider, start_spin, end_slider, end_spin = self._add_frame_range_controls(
            frame_layout,
            start_value=self.current_frame_index + 1,
            end_value=self.recording.header.n_frames,
        )
        self._add_toi_frame_controls(
            frame_layout,
            start_slider=start_slider,
            start_spin=start_spin,
            end_slider=end_slider,
            end_spin=end_spin,
        )
        layout.addWidget(frame_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._set_fixed_dialog_width(dialog, "Trim Binary")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        start = min(start_spin.value(), end_spin.value()) - 1
        end = max(start_spin.value(), end_spin.value()) - 1
        return list(range(start, end + 1))

    def _measurement_export_options(self) -> dict | None:
        if self.recording is None or self.session is None or not self.session.rois:
            self._message("Add at least one ROI before exporting measurements.")
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Measurement Export Options")
        layout = QtWidgets.QVBoxLayout(dialog)
        frame_box = QtWidgets.QGroupBox("Frames")
        frame_layout = QtWidgets.QFormLayout(frame_box)
        start_slider, start_spin, end_slider, end_spin = self._add_frame_range_controls(
            frame_layout,
            start_value=1,
            end_value=self.recording.header.n_frames,
        )
        self._add_toi_frame_controls(
            frame_layout,
            start_slider=start_slider,
            start_spin=start_spin,
            end_slider=end_slider,
            end_spin=end_spin,
        )
        layout.addWidget(frame_box)
        roi_checks = []
        metric_checks = []
        roi_box = QtWidgets.QGroupBox("ROIs")
        roi_layout = QtWidgets.QVBoxLayout(roi_box)
        selected = self._selected_roi_ids()
        for roi in self.session.rois:
            check = QtWidgets.QCheckBox(roi.label)
            check.setChecked(not selected or roi.id in selected)
            roi_layout.addWidget(check)
            roi_checks.append((check, roi))
        layout.addWidget(roi_box)
        channel_box = QtWidgets.QGroupBox("Channels")
        channel_layout = QtWidgets.QVBoxLayout(channel_box)
        for label, metric in [
            ("Intensity", "roi_toi_mean_intensity"),
            ("Variance", "roi_toi_mean_variance"),
            ("Perfusion", DEFAULT_METRIC),
        ]:
            check = QtWidgets.QCheckBox(label)
            check.setChecked(label == "Perfusion")
            channel_layout.addWidget(check)
            metric_checks.append((check, metric))
        layout.addWidget(channel_box)
        table_box = QtWidgets.QGroupBox("Tables")
        table_layout = QtWidgets.QVBoxLayout(table_box)
        averaged_check = QtWidgets.QCheckBox("Averaged measurement table")
        averaged_check.setChecked(True)
        frame_by_frame_check = QtWidgets.QCheckBox("Frame-by-frame measurement table")
        note = QtWidgets.QLabel(
            "Frame-by-frame measurements are for plotting. Do not average multiple frames' perfusion "
            "measurements; the manual states perfusion over multiple frames should be calculated from "
            "averaged intensity and variance."
        )
        note.setWordWrap(True)
        table_layout.addWidget(averaged_check)
        table_layout.addWidget(frame_by_frame_check)
        table_layout.addWidget(note)
        layout.addWidget(table_box)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._set_fixed_dialog_width(dialog, "Measurement Export Options")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        rois = [roi for check, roi in roi_checks if check.isChecked()]
        if not rois:
            self._message("Select at least one ROI.")
            return None
        metrics = [metric for check, metric in metric_checks if check.isChecked()]
        if averaged_check.isChecked() and not metrics:
            self._message("Select at least one measurement channel.")
            return None
        if not averaged_check.isChecked() and not frame_by_frame_check.isChecked():
            self._message("Select at least one measurement table.")
            return None
        start = min(start_spin.value(), end_spin.value()) - 1
        end = max(start_spin.value(), end_spin.value()) - 1
        toi = Toi(id="export_frames", label="Export frames", frame_start=start, frame_end=end, include_end=True)
        return {
            "rois": rois,
            "tois": [toi] if averaged_check.isChecked() else [],
            "metrics": metrics,
            "frames": list(range(start, end + 1)),
            "frame_by_frame": frame_by_frame_check.isChecked(),
        }

    def _roi_by_id(self, roi_id: str) -> Roi | None:
        if self.session is None:
            return None
        return next((roi for roi in self.session.rois if roi.id == roi_id), None)

    def _toi_by_id(self, toi_id: str) -> Toi | None:
        if self.session is None:
            return None
        return next((toi for toi in self.session.tois if toi.id == toi_id), None)

    def _maybe_update_trace(self) -> None:
        if self.auto_trace_checkbox.isChecked():
            self._trace_update_timer.start()

    def _message(self, text: str) -> None:
        QtWidgets.QMessageBox.information(self, "PIMSoft PSI", text)


def _bounds(vertices_xy: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in vertices_xy]
    ys = [y for _, y in vertices_xy]
    return min(xs), min(ys), max(xs), max(ys)


def _default_colormap(channel: str) -> str:
    if channel == "perfusion":
        return "turbo"
    return "gray"


def _roi_geometry_signature(roi: Roi) -> tuple[tuple[float, float], ...]:
    return tuple((round(x, 3), round(y, 3)) for x, y in roi.vertices_xy)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _clamp_int(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _make_roi_item(roi: Roi, *, editable: bool) -> pg.ROI:
    pen = pg.mkPen(roi.color, width=2)
    if roi.shape_type == "rectangle":
        left, top, right, bottom = _bounds(roi.vertices_xy)
        item = pg.RectROI([left, top], [right - left, bottom - top], pen=pen, movable=editable)
    elif roi.shape_type == "ellipse":
        left, top, right, bottom = _bounds(roi.vertices_xy)
        item = pg.EllipseROI([left, top], [right - left, bottom - top], pen=pen, movable=editable)
    else:
        item = pg.PolyLineROI(roi.vertices_xy, closed=True, pen=pen, movable=editable)
    item.setVisible(roi.visible)
    if not editable:
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        for handle in item.getHandles():
            handle.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    return item


def _set_roi_item_geometry(item: pg.ROI, roi: Roi) -> None:
    if roi.shape_type in {"rectangle", "ellipse"}:
        left, top, right, bottom = _bounds(roi.vertices_xy)
        item.setPos([left, top], update=False)
        item.setSize([right - left, bottom - top], update=True)
        return
    if hasattr(item, "setPoints"):
        item.setPoints(roi.vertices_xy, closed=True)


def _vertices_from_item(item: pg.ROI, shape_type: str) -> list[tuple[float, float]]:
    if shape_type in {"rectangle", "ellipse"}:
        pos = item.pos()
        size = item.size()
        return [(float(pos.x()), float(pos.y())), (float(pos.x() + size.x()), float(pos.y() + size.y()))]
    vertices = []
    for handle in item.getHandles():
        pos = item.mapToParent(handle.pos())
        vertices.append((float(pos.x()), float(pos.y())))
    return vertices


def _next_color(index: int) -> str:
    colors = ["#00a6ff", "#ff6b35", "#2ec4b6", "#e71d36", "#7b2cbf", "#4f772d"]
    return colors[index % len(colors)]


def run(recording_path: str | Path | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(recording_path)
    window.show()
    return app.exec()
