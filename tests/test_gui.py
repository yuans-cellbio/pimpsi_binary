import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from pimpsi.gui.image_view import ImageView
from pimpsi.gui.main_window import MainWindow, _vertices_from_item
from pimpsi.gui.workers import LazyFrameProvider
from pimpsi.roi import Roi


@dataclass(frozen=True)
class FakeHeader:
    n_frames: int = 4
    image_height: int = 2
    image_width: int = 3
    coherence_factor: float = 0.5
    file_type: str = "PIMSOFT"
    file_version: int = 1
    number_of_images: int = 8
    signal_gain: float = 10.0
    data_offset: int = 46
    variance_offset: int = 46
    intensity_offset: int = 238
    sha256: str = "abc123"


class FakeRecording:
    def __init__(self):
        self.header = FakeHeader()
        self.path = Path(__file__)
        self.intensity_calls = []
        self.variance_calls = []
        self.perfusion_calls = []

    def get_intensity(self, frame_index):
        self.intensity_calls.append(frame_index)
        return np.full((2, 3), frame_index + 100.0)

    def get_variance(self, frame_index):
        self.variance_calls.append(frame_index)
        return np.full((2, 3), frame_index + 10.0)

    def calculate_perfusion(self, frame_index, clip_upper=3000.0):
        self.perfusion_calls.append((frame_index, clip_upper))
        return np.full((2, 3), frame_index + 1000.0)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_lazy_frame_provider_loads_only_requested_frames():
    recording = FakeRecording()
    provider = LazyFrameProvider(recording, cache_size=2)

    first = provider.frame("intensity", 2)
    second = provider.frame("intensity", 2)
    perfusion = provider.frame("perfusion", 3)

    np.testing.assert_array_equal(first, second)
    assert recording.intensity_calls == [2]
    assert recording.variance_calls == []
    assert recording.perfusion_calls == [(3, 3000.0)]
    assert perfusion[0, 0] == 1003.0


def test_lazy_frame_provider_contrast_loads_current_intensity_and_variance_only():
    recording = FakeRecording()
    provider = LazyFrameProvider(recording)

    provider.frame("contrast", 1)

    assert recording.intensity_calls == [1]
    assert recording.variance_calls == [1]
    assert recording.perfusion_calls == []


def test_main_window_set_recording_displays_current_frame_lazily(qapp):
    recording = FakeRecording()
    window = MainWindow()

    window.set_recording(recording)
    window.frame_slider.setValue(3)
    window.mode_combo.setCurrentText("perfusion")

    assert recording.intensity_calls == [0, 2]
    assert recording.variance_calls == []
    assert recording.perfusion_calls == [(0, 3000.0), (2, 3000.0), (2, 3000.0)]
    assert window.frame_slider.maximum() == 4
    assert "3/4" in window.status_label.text()
    window.close()


def test_main_window_default_colormaps_are_gray_and_turbo(qapp):
    window = MainWindow()

    assert window.top_colormap_combo.currentText() == "gray"
    assert window.bottom_colormap_combo.currentText() == "turbo"
    window.close()


def test_image_view_accepts_friendly_colormap_names(qapp):
    image_view = ImageView()

    image_view.set_colormap("gray")
    image_view.set_colormap("turbo")

    image_view.close()


def test_main_window_adds_roi_to_table_and_scene(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)

    window.add_roi("rectangle")

    assert len(window.session.rois) == 1
    assert window.roi_table.rowCount() == 1
    assert window.roi_table.item(0, 0).text() == "ROI 1"
    assert window.session.rois[0].id in window.roi_items
    window.close()


def test_roi_tool_button_arms_without_creating_roi(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)

    window.add_rect_button.click()

    assert window._armed_roi_shape == "rectangle"
    assert window.add_rect_button.isChecked()
    assert len(window.session.rois) == 0
    window.cancel_roi_tool()
    window.close()


def test_armed_roi_tool_places_roi_on_channel_scene_click(qapp):
    class SceneClick:
        def __init__(self, scene_pos):
            self._scene_pos = scene_pos

        def type(self):
            return QtCore.QEvent.Type.GraphicsSceneMousePress

        def scenePos(self):
            return self._scene_pos

    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.arm_roi_tool("rectangle")
    scene_pos = window.image_view._view.imageItem.mapToScene(QtCore.QPointF(1.5, 1.0))

    handled = window.eventFilter(window.image_view._view.scene, SceneClick(scene_pos))

    assert handled is True
    assert window._armed_roi_shape is None
    assert not window.add_rect_button.isChecked()
    assert len(window.session.rois) == 1
    window.close()


def test_click_placed_roi_is_smaller_than_legacy_default(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)

    window.add_roi("rectangle", center_xy=(1.5, 1.0))

    left, top = window.session.rois[0].vertices_xy[0]
    right, bottom = window.session.rois[0].vertices_xy[1]
    assert right - left <= recording.header.image_width
    assert bottom - top <= recording.header.image_height
    window.close()


def test_main_window_trace_uses_selected_roi_and_channel_lazily(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    recording.intensity_calls.clear()
    recording.variance_calls.clear()
    recording.perfusion_calls.clear()

    window.trace_channel_combo.setCurrentText("variance")
    window.update_trace()

    assert recording.variance_calls == [0, 1, 2, 3]
    assert recording.intensity_calls == []
    assert recording.perfusion_calls == []
    window.close()


def test_auto_trace_is_debounced_during_roi_edits(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    recording.intensity_calls.clear()
    window.frame_provider.clear()

    window._maybe_update_trace()

    assert window._trace_update_timer.isActive()
    assert recording.intensity_calls == []
    window._trace_update_timer.stop()
    window.close()


def test_main_window_trace_draws_one_line_per_visible_roi(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    window.add_roi("ellipse")

    window.update_trace()

    assert len(window.trace_plot.listDataItems()) == 2
    window.close()


def test_trace_recalculates_only_edited_roi(qapp, monkeypatch):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    window.add_roi("ellipse")
    call_counts = {}
    original_to_mask = Roi.to_mask

    def counted_to_mask(self, image_shape):
        call_counts[self.id] = call_counts.get(self.id, 0) + 1
        return original_to_mask(self, image_shape)

    monkeypatch.setattr(Roi, "to_mask", counted_to_mask)
    window.update_trace()
    call_counts.clear()
    edited_roi_id = window.session.rois[0].id

    window.roi_items[edited_roi_id].setPos([1.0, 1.0])
    QtWidgets.QApplication.processEvents()
    window._trace_update_timer.stop()
    window.update_trace()

    assert call_counts == {edited_roi_id: 1}
    window.close()


def test_main_window_preserves_roi_items_when_frame_changes(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    roi_id = window.session.rois[0].id
    primary_item = window.roi_items[roi_id]
    secondary_item = window.secondary_roi_items[roi_id]

    window.frame_slider.setValue(2)

    assert window.roi_items[roi_id] is primary_item
    assert window.secondary_roi_items[roi_id] is secondary_item
    window.close()


def test_channel_window_syncs_with_main_frame_bar(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.new_channel_window()
    child = window.channel_windows[0]

    window.frame_slider.setValue(4)

    assert child.frame_spin.value() == 4
    window.close()
    child.close()


def test_toi_table_clamps_one_based_frame_numbers(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_toi()

    window.toi_table.item(0, 1).setText("-10")
    window.toi_table.item(0, 2).setText("99")

    assert window.session.tois[0].frame_start == 0
    assert window.session.tois[0].frame_end == 3
    assert window.toi_table.item(0, 1).text() == "1"
    assert window.toi_table.item(0, 2).text() == "4"
    window.close()


def test_polygon_roi_vertices_read_from_pyqtgraph_handles(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("polygon")
    roi_id = window.session.rois[0].id

    vertices = _vertices_from_item(window.roi_items[roi_id], "polygon")

    assert len(vertices) == 3
    assert all(isinstance(x, float) and isinstance(y, float) for x, y in vertices)
    window.close()


def test_bottom_roi_edit_syncs_to_top_roi(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    roi_id = window.session.rois[0].id
    bottom_item = window.secondary_roi_items[roi_id]

    bottom_item.setPos([1.0, 1.0])
    QtWidgets.QApplication.processEvents()

    assert window.session.rois[0].vertices_xy[0] == (1.0, 1.0)
    assert window.roi_items[roi_id].pos().x() == 1.0
    assert window.roi_items[roi_id].pos().y() == 1.0
    window.close()


def test_trace_always_uses_all_frames_even_with_selected_toi(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    window.add_toi()
    window.session.tois[0].frame_start = 1
    window.session.tois[0].frame_end = 2
    recording.intensity_calls.clear()
    window.frame_provider.clear()

    window.update_trace()

    assert recording.intensity_calls == [0, 1, 2, 3]
    window.close()


def test_invisible_toi_is_not_drawn_as_trace_region(qapp):
    recording = FakeRecording()
    window = MainWindow()
    window.set_recording(recording)
    window.add_roi("rectangle")
    window.add_toi()
    window.session.tois[0].visible = False

    window.update_trace()

    assert not any(isinstance(item, pg.LinearRegionItem) for item in window.trace_plot.items())
    window.close()
