import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import sys

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import numpy as np

from PyQt6 import uic
from PyQt6 import QtCore, QtGui, QtWidgets
from qtrangeslider import QRangeSlider
from LsciFile import PimsoftBinary

from mplcanvas import MplCanvas

from skimage.draw import polygon

import tifffile

class Selector:
    def __init__(self, name, selector):
        self.name = name
        self.selector = selector

    def calculate_center(self):
        if isinstance(self.selector, PolygonSelector):
            if hasattr(self.selector, "verts"):
                center = np.array(self.selector.verts)
                return(np.mean(center, axis = 0))
        # Add other selector types here if needed
        return (0, 0)
    
class RenameDialog(QtWidgets.QDialog):
    def __init__(self, default_text = ""):
        super().__init__()
        
        self.setWindowTitle("Rename ROI")
        self.setModal(True)
        
        self.layout = QtWidgets.QVBoxLayout()

        # Add label
        self.label = QtWidgets.QLabel("Rename:")
        self.layout.addWidget(self.label)
        
        # Add line edit widget
        self.line_edit = QtWidgets.QLineEdit(self)
        self.line_edit.setText(default_text)
        self.layout.addWidget(self.line_edit)

        # Add buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        
        self.confirm_button = QtWidgets.QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.confirm_button)
        
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(self.button_layout)
        
        self.setLayout(self.layout)

    def get_line_edit_value(self):
        return self.line_edit.text()

class ExportImage(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Export Image")
        self.setModal(True)

        main_layout = QtWidgets.QHBoxLayout()

        # Checkbox group and layout
        checkbox_group = QtWidgets.QGroupBox("Image Options")
        checkbox_layout = QtWidgets.QVBoxLayout()
        
        # Checkbox with default values
        self.checkbox_intensity = QtWidgets.QCheckBox("Intensity Image(s)")
        self.checkbox_perfusion = QtWidgets.QCheckBox("Perfusion Image(s)")
        self.checkbox_roi = QtWidgets.QCheckBox("ROI mask(s)")

        self.checkbox_intensity.setChecked(True)
        self.checkbox_perfusion.setChecked(True)
        self.checkbox_roi.setChecked(True)

        # Add checkboxes to checkbox layout
        checkbox_layout.addWidget(self.checkbox_intensity)
        checkbox_layout.addWidget(self.checkbox_perfusion)
        checkbox_layout.addWidget(self.checkbox_roi)
        checkbox_layout.addStretch()  # Add stretch to push checkboxes to the top

        checkbox_group.setLayout(checkbox_layout)  # Set the layout for the group box

        # listWidget layout
        listwidget_layout = QtWidgets.QVBoxLayout()

        # Create and add a list widget for multiple selection
        self.label_frames = QtWidgets.QLabel("Frames:")
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.listWidget.addItems(["All frames", "Time of interest", "Average all frames", "Average time of interest"])
        self.listWidget.setCurrentRow(0)
        
        listwidget_layout.addWidget(self.label_frames)
        listwidget_layout.addWidget(self.listWidget)

        self.listWidget.setFixedHeight(100)

        # Add group box and list widget to main layout
        main_layout.addWidget(checkbox_group)
        main_layout.addLayout(listwidget_layout)
        
        # Create and add Export and Cancel buttons
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        # Create a vertical layout to hold the main layout and button box
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addWidget(self.buttonBox)
        
        self.setLayout(layout)

    def accept(self):
        if not self.listWidget.selectedItems():
            QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select at least one frame.")
        else:
            super().accept()

    def getSelectedOptions(self):
        selected_options = {
            "Intensity": self.checkbox_intensity.isChecked(),
            "Perfusion": self.checkbox_perfusion.isChecked(),
            "Mask": self.checkbox_perfusion.isChecked(),
            "Frames": [item.text() for item in self.listWidget.selectedItems()]
        }
        return selected_options


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Load the UI
        uic.loadUi('app.ui', self)

        self.horizontalSlider_TimeOfInterest = QRangeSlider()
        self.horizontalSlider_TimeOfInterest.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider_TimeOfInterest.setObjectName("horizontalSlider_TimeOfInterest")
        self.horizontalLayout_TimeOfInterest.addWidget(self.horizontalSlider_TimeOfInterest)
        self.horizontalSlider_TimeOfInterest.setMinimum(1)
        self.horizontalSlider_TimeOfInterest.setMaximum(100)
        self.horizontalSlider_TimeOfInterest.setValue([1, 100])
        self.horizontalSlider_TimeOfInterest.setEnabled(False)

        self.new_session()
        # Menu > File
        # Open binary file
        self.actionOpen_Binary_File.triggered.connect(self.open_binary_file_dialog)

        # Connect ROI buttons to the method
        self.toolButton_Add_ROI_Polygon.clicked.connect(self.activate_polygon_selector)
        self.pushButton_Delete_ROI.clicked.connect(self.remove_selected_roi)
        self.pushButton_Rename_ROI.clicked.connect(self.rename_roi)

        # Connect Image Export function
        self.actionImageExport.triggered.connect(self.image_export)

        # Connect Measurement Export function
        self.actionMeasurementExport.triggered.connect(self.measurement_export)

    
    @property
    def current_frame_index(self):
        return self._current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, value):
        self._current_frame_index = value
        self.on_current_frame_index_changed()

    def on_current_frame_index_changed(self):
        if hasattr(self, "psi_file"):
            if self.current_frame_index != self.horizontalSlider_Frame.value() - 1:
                self.horizontalSlider_Frame.setValue(self.current_frame_index + 1)
            if self.current_frame_index != self.spinBox_Frame.value() - 1:
                self.spinBox_Frame.setValue(self.current_frame_index + 1)

            self.update_canvas()

            if len(self.selectors) > 0:
                self.update_roi_perfusion_plot_vline()

    def new_session(self):

        # Initate session information
        self.current_frame_index = 0
        self.intensity_thresholding_method = "otsu"
        self.intensity_limits = [0, 15000]
        self.perfusion_limits = [0, 3000]

        self.frame_averaging_window = 1

        # Initiate ROI table
        self.tableWidget_ROI.setColumnCount(2)
        self.tableWidget_ROI.setHorizontalHeaderLabels(["ROI", "Perfusion"])
        self.tableWidget_ROI.setRowCount(0)

        # Set column resize modes
        self.tableWidget_ROI.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tableWidget_ROI.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch) 

        if hasattr(self, "psi_file"):
            self.intensity_limits = [0, np.max(self.psi_file.intensity_images)]
            self.perfusion_limits = [0, np.max(self.psi_file.perfusion_images)]
            self.apparent_frame_number = self.psi_file.perfusion_images.shape[0] // self.frame_averaging_window
            # self.apparent_frame_number = self.psi_file.apparent_frame_number

        # TODO: Initiate canvas        
        
        if hasattr(self, "psi_file"):
            # Initiate frame slider
            self.horizontalSlider_Frame.setEnabled(True)
            self.horizontalSlider_Frame.setMinimum(1)
            self.horizontalSlider_Frame.setMaximum(self.apparent_frame_number)
            self.horizontalSlider_Frame.setValue(self.current_frame_index)
            self.horizontalSlider_Frame.valueChanged.connect(self.frame_slider_value_changed)

            # Initiate frame spin box
            self.spinBox_Frame.setEnabled(True)
            self.spinBox_Frame.setMinimum(1)
            self.spinBox_Frame.setMaximum(self.apparent_frame_number)
            self.spinBox_Frame.setValue(self.current_frame_index)
            self.spinBox_Frame.valueChanged.connect(self.frame_spinbox_value_changed)

            # Inititate selector
            if self.selectors:
                for _ in range(len(self.selectors)):
                    selector = self.selectors.pop(0).selector
                    selector._xs, selector._ys = [0], [0]
                    selector._selection_completed = True
                    selector.set_visible(False)
                    selector.disconnect_events()

                    self.update_table()
                    self.update_canvas()

        else:
            self.horizontalSlider_Frame.setEnabled(False)
            self.spinBox_Frame.setEnabled(False)
            self.horizontalSlider_TimeOfInterest.setEnabled(False)
            self.selectors = []

    def frame_slider_value_changed(self, value):
        self.current_frame_index = value - 1

    def frame_spinbox_value_changed(self, value):
        self.current_frame_index = value - 1

    def open_binary_file_dialog(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Binary File", "", "Binary Files (*.dat)")
        if fileName:
            try:
                self.psi_file = PimsoftBinary(fileName)
                self.statusbar.showMessage(self.psi_file.file_path.split("/")[-1] + " Loaded")

                self.new_session()

                self.update_canvas()
            except Exception as e:
                print("ERROR: ", e)

    def update_canvas(self):
        psi_file = self.psi_file
        index = self.current_frame_index

        intensity_image = psi_file.get_frame(index, "intensity")
        perfusion_image = psi_file.get_frame(index, "perfusion")

        # Draw image in canvas
        self.show_image(self.widget_Canvas_Intensity, intensity_image, cmap = "Greys_r", vmin = self.intensity_limits[0], vmax = self.intensity_limits[1])
        self.show_image(self.widget_Canvas_Perfusion, perfusion_image, cmap = "hot", vmin = self.perfusion_limits[0], vmax = self.perfusion_limits[1])

        # Update status bar
        self.statusbar.showMessage(self.psi_file.file_path.split("/")[-1] + ": Frame " + str(self.current_frame_index + 1))

    def show_image(self, canvas, image, cmap='RdYlBu_r', vmin=None, vmax=None):
        # `canvas` is an instance of MplCanvas
        # `image` is an numpy.ndarray with 1 frame of image

        if vmin is None: 
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        if not hasattr(canvas, 'image'):
            # First time setup: create the image plot and colorbar
            canvas.image = canvas.axes.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            canvas.colorbar = canvas.figure.colorbar(canvas.image, ax=canvas.axes, shrink = 0.8)

            # Set font size for the colorbar labels and ticks
            # TODO: The fontsize is hard coded, which may appear different with different screen resolutions. 
            canvas.colorbar.ax.tick_params(labelsize=10)
            colorbar_label = canvas.colorbar.ax.yaxis.label
            colorbar_label.set_size(10)
        else:
            # Update the existing image plot
            canvas.image.set_data(image)
            canvas.image.set_clim(vmin, vmax)  # Update the color limits if necessary
            canvas.colorbar.update_normal(canvas.image)

        if hasattr(canvas, 'text'):
            for text in canvas.text:
                text.remove()
        
        canvas.text = []

        if self.selectors:
            for _, selector in enumerate(self.selectors):
                if len(selector.selector.verts) > 2:
                    center = selector.calculate_center()
                    canvas.text.append(canvas.axes.text(center[0], center[1], selector.name, color='white', ha='right', va='center', fontsize = 10))
        canvas.draw()

    def get_new_selector_name(self, base_name="New ROI"):
        existing_names = [selector.name for selector in self.selectors]
        if base_name not in existing_names:
            return base_name

        counter = 1
        new_name = f"{base_name} {counter}"
        while new_name in existing_names:
            counter += 1
            new_name = f"{base_name} {counter}"
        
        return new_name

    def activate_polygon_selector(self):
        line_props=dict(color='c', linestyle='-', linewidth=0.5, alpha=1)
        handle_props = dict(marker='o', markersize=1, color='c', alpha=1)
        polygon_selector = PolygonSelector(self.widget_Canvas_Intensity.axes, self.onselect, useblit=True, props=line_props, handle_props=handle_props)
        selector_name = self.get_new_selector_name()
        self.selectors.append(Selector(name=selector_name, selector=polygon_selector))

        self.update_table()

    def remove_selected_roi(self):
        selected_items = self.tableWidget_ROI.selectedItems()
        if self.tableWidget_ROI.rowCount() > 0:
            if selected_items:
                selected_index = selected_items[0].row()

                selector = self.selectors.pop(selected_index).selector
                selector._xs, selector._ys = [0], [0]
                selector._selection_completed = True
                selector.set_visible(False)
                selector.disconnect_events()

                self.update_table()
                self.update_canvas()

    def rename_selector(self, index, new_name):
        if 0 <= index < len(self.selectors):
            self.selectors[index].name = new_name

    def onselect(self, _):
        rois = []
        for _, selector in enumerate(self.selectors):
            if len(selector.selector.verts) > 2:
                label = selector.name
                verts = selector.selector.verts
                mask = self.vertice_to_mask(verts)
                rois.append([label, mask])
        self.roi_masks = rois
        self.plot_roi_perfusion()
        self.update_roi_toi_perfusion()
        self.update_canvas()

    def update_table(self):
        roi_num = len(self.selectors)
        if roi_num >= 1:
            self.tableWidget_ROI.setRowCount(roi_num)
            for i, selector in enumerate(self.selectors):
                self.tableWidget_ROI.setItem(i, 0, QtWidgets.QTableWidgetItem(selector.name))

            if not self.horizontalSlider_TimeOfInterest.isEnabled():
                # Initiate frame slider
                self.horizontalSlider_TimeOfInterest.setEnabled(True)
                self.horizontalSlider_TimeOfInterest.setMinimum(1)
                self.horizontalSlider_TimeOfInterest.setMaximum(self.apparent_frame_number)
                self.horizontalSlider_TimeOfInterest.setValue((1, self.apparent_frame_number))
                self.horizontalSlider_TimeOfInterest.valueChanged.connect(self.range_slider_value_changed)
        else:
            self.tableWidget_ROI.setRowCount(0)
            self.horizontalSlider_TimeOfInterest.setEnabled(False)
            self.horizontalSlider_TimeOfInterest.setMinimum(1)
            self.horizontalSlider_TimeOfInterest.setMaximum(100)
            self.horizontalSlider_TimeOfInterest.setValue((1, 100))


    def rename_roi(self):
        items = self.tableWidget_ROI.selectedItems()
        self.tableWidget_ROI.clearSelection()
        if items:
            item = items[0]
            index = item.row()
            label = self.tableWidget_ROI.item(index, 0).text()
            dialog = RenameDialog(label)
            result = dialog.exec()

            if result == QtWidgets.QDialog.DialogCode.Accepted:
                new_label = dialog.get_line_edit_value()
                if new_label != label:
                    self.tableWidget_ROI.setItem(index, 0, QtWidgets.QTableWidgetItem(new_label))
                    self.selectors[index].name = new_label
                    self.roi_masks[index][0] = new_label
                    if len(self.selectors[index].selector.verts) > 2:
                        self.update_canvas()
                        self.plot_roi_perfusion()

    def vertice_to_mask(self, verts):
        verts = np.array(verts)
        # Separate the vertices into x and y coordinates
        r = verts[:, 1]  # y-coordinates
        c = verts[:, 0]  # x-coordinates

        _, height, width = self.psi_file.perfusion_images.shape

        # Create the mask
        rr, cc = polygon(r, c, (height, width))
        mask = np.zeros((height, width), dtype=bool)
        mask[rr, cc] = True

        return(mask)
    
    def plot_roi_perfusion(self):
        perfusion_images = self.psi_file.perfusion_images
        canvas = self.widget_Canvas_Measurement_Plot
        ax = canvas.axes
        ax.clear()
        ax.axis('on')
        rois = self.roi_masks
        color_map = plt.get_cmap('tab10')
        for i, roi in enumerate(rois):
            label = roi[0]
            mask = roi[1]
            masks = np.array([mask for _ in range(perfusion_images.shape[0])])
            roi_perfusion = self.psi_file.roi_perfusion_per_frame(masks)
            ax.plot(roi_perfusion, marker='o', markersize=2, linestyle='-', linewidth=0.5, color=color_map(i), label=label)

        ax.set_xlim(0, perfusion_images.shape[0] - 1)
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Perfusion Value")
        ax.set_position([0.125, 0.2, 0.6, 0.6])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside the plot

        # Draw vline to indicate the current frame
        canvas.vline_current_frame = ax.axvline(self.current_frame_index, color='r', linewidth=0.5, linestyle="dotted")

        # Draw span for TOI
        if len(self.selectors) > 0:
            toi = self.horizontalSlider_TimeOfInterest.value()
            toi = [x - 1 for x in toi]
            canvas.toi_span = ax.axvspan(xmin=toi[0], xmax=toi[1], color='blue', alpha=0.3)

        canvas.draw()

    def update_roi_toi_perfusion(self):
        frames_limits = self.horizontalSlider_TimeOfInterest.value()
        frames = [x for x in range(frames_limits[0]-1, frames_limits[1])]
        rois = self.roi_masks
        for i, roi in enumerate(rois):
            mask = roi[1]
            masks = np.array([mask for _ in frames])
            roi_perfusion = self.psi_file.roi_perfusion_by_toi(masks, frames)
            self.tableWidget_ROI.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{roi_perfusion:.3f}"))


    def range_slider_value_changed(self):
        toi = self.horizontalSlider_TimeOfInterest.value()
        toi = [x - 1 for x in toi]
        canvas = self.widget_Canvas_Measurement_Plot
        ax = canvas.axes
        
        if hasattr(canvas, 'toi_span'):
            canvas.toi_span.remove()
            del canvas.toi_span
        canvas.toi_span = ax.axvspan(xmin=toi[0], xmax=toi[1], color='blue', alpha=0.3)

        if self.tableWidget_ROI.rowCount() > 0:
            self.update_roi_toi_perfusion()

        canvas.draw()
        

    def update_roi_perfusion_plot_vline(self):
        canvas = self.widget_Canvas_Measurement_Plot
        ax = canvas.axes

        # Draw vline to indicate the current frame
        if hasattr(canvas, 'vline_current_frame'):
            canvas.vline_current_frame.remove()
            del canvas.vline_current_frame
        canvas.vline_current_frame = ax.axvline(self.current_frame_index, color='r', linewidth=0.5, linestyle="dotted")

        canvas.draw()

    def image_export(self):
        dialog = ExportImage()
        if dialog.exec():
            selected_options = dialog.getSelectedOptions()
            saving_intensity = selected_options['Intensity']
            saving_perfusion = selected_options['Perfusion']
            saving_mask = selected_options['Mask']
            frame_options = selected_options['Frames']

            output_dir = self.psi_file.file_path
            root, ext = os.path.splitext(output_dir)
            output_dir = root
            os.makedirs(output_dir, exist_ok=True)

            if 'All frames' in frame_options:
                if saving_intensity:
                    intensity_images = self.psi_file.intensity_images
                    tifffile.imwrite(f'{output_dir}/intensity_images_all_frames.tif', intensity_images)
                if saving_perfusion:
                    perfusion_images = self.psi_file.perfusion_images
                    tifffile.imwrite(f'{output_dir}/perfusion_images_all_frame.tif', perfusion_images)
                if saving_mask:
                    for _, roi in enumerate(self.roi_masks):
                        roi_label = roi[0]
                        roi_mask = roi[1]
                        roi_masks = np.array([roi_mask for _ in range(self.psi_file.perfusion_images.shape[0])])
                        roi_masks = np.logical_and(roi_masks, self.psi_file.intensity_masks)
                        tifffile.imwrite(f'{output_dir}/roi_masks_{roi_label}_all_frames.tif', roi_masks)
            if 'Time of interest' in frame_options:
                frames_limits = self.horizontalSlider_TimeOfInterest.value()
                frames = [x for x in range(frames_limits[0]-1, frames_limits[1])]
                if saving_intensity:
                    intensity_images = self.psi_file.intensity_images
                    intensity_images = intensity_images[frames, :, :]
                    tifffile.imwrite(f'{output_dir}/intensity_images_TOI.tif', intensity_images)
                if saving_perfusion:
                    perfusion_images = self.psi_file.perfusion_images
                    perfusion_images = perfusion_images[frames, :, :]
                    tifffile.imwrite(f'{output_dir}/perfusion_images_TOI.tif', perfusion_images)
                if saving_mask:
                    intensity_masks = self.psi_file.intensity_masks
                    intensity_masks = intensity_masks[frames, :, :]
                    for _, roi in enumerate(self.roi_masks):
                        roi_label = roi[0]
                        roi_mask = roi[1]
                        roi_masks = np.array([roi_mask for _ in frames])
                        roi_masks = np.logical_and(roi_masks, intensity_masks)
                        tifffile.imwrite(f'{output_dir}/roi_masks_{roi_label}_TOI.tif', roi_masks)
            if 'Average all frames' in frame_options:
                if saving_intensity:
                    intensity_images = self.psi_file.intensity_images
                    intensity_images = intensity_images.mean(axis = 0)
                    tifffile.imwrite(f'{output_dir}/intensity_images_average_all_frames.tif', intensity_images)
                if saving_perfusion:
                    perfusion_images = self.psi_file.perfusion_images
                    perfusion_images = perfusion_images.mean(axis = 0)
                    tifffile.imwrite(f'{output_dir}/perfusion_images_average_all_frame.tif', perfusion_images)
                if saving_mask:
                    for _, roi in enumerate(self.roi_masks):
                        roi_label = roi[0]
                        roi_mask = roi[1]
                        roi_masks = np.array([roi_mask for _ in range(self.psi_file.perfusion_images.shape[0])])
                        roi_masks = np.logical_and(roi_masks, self.psi_file.intensity_masks)
                        roi_masks = roi_masks.mean(axis = 0)
                        tifffile.imwrite(f'{output_dir}/roi_masks_{roi_label}_average_all_frames.tif', roi_masks)
            if 'Average time of interest' in frame_options:
                frames_limits = self.horizontalSlider_TimeOfInterest.value()
                frames = [x for x in range(frames_limits[0]-1, frames_limits[1])]
                if saving_intensity:
                    intensity_images = self.psi_file.intensity_images
                    intensity_images = intensity_images[frames, :, :].mean(axis = 0)
                    tifffile.imwrite(f'{output_dir}/intensity_images_average_TOI.tif', intensity_images)
                if saving_perfusion:
                    perfusion_images = self.psi_file.perfusion_images
                    perfusion_images = perfusion_images[frames, :, :].mean(axis = 0)
                    tifffile.imwrite(f'{output_dir}/perfusion_images_average_TOI.tif', perfusion_images)
                if saving_mask:
                    intensity_masks = self.psi_file.intensity_masks
                    intensity_masks = intensity_masks[frames, :, :].mean(axis = 0)
                    for _, roi in enumerate(self.roi_masks):
                        roi_label = roi[0]
                        roi_mask = roi[1]
                        roi_masks = np.array([roi_mask for _ in frames])
                        roi_masks = np.logical_and(roi_masks, intensity_masks)
                        roi_masks = roi_masks.mean(axis = 0)
                        tifffile.imwrite(f'{output_dir}/roi_masks_{roi_label}_average_TOI.tif', roi_masks)

            self.statusbar.showMessage("Images saved!")

    def measurement_export(self):
        output_dir = self.psi_file.file_path
        root, ext = os.path.splitext(output_dir)
        output_dir = root
        os.makedirs(output_dir, exist_ok=True)
        file_path = f'{output_dir}/Perfusion_Measurement.txt'

        # Open the file in write mode
        with open(file_path, 'w') as file:
            row_count = self.tableWidget_ROI.rowCount()
            column_count = self.tableWidget_ROI.columnCount()

            # Write column headers
            headers = []
            for column in range(column_count):
                header_item = self.tableWidget_ROI.horizontalHeaderItem(column)
                headers.append(header_item.text())
            file.write('\t'.join(headers) + '\n')

            # Iterate through each cell and write data to the file
            for row in range(row_count):
                row_data = []
                for column in range(column_count):
                    item = self.tableWidget_ROI.item(row, column)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')  # Handle empty cells

                file.write('\t'.join(row_data) + '\n')

        self.statusbar.showMessage("Result table saved!")
            




            
        
##################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())