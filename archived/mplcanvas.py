import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.colorbar = None  # Handle colorbar separately
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)  # Ensure it has a Qt parent if required

        # Initially hide the axes
        self.axes.axis('off')  # Turns off the axis lines and labels

        # You can also make the background transparent if that's desired
        self.axes.set_facecolor('none')  # Set the axes background color to transparent
        self.figure.patch.set_facecolor('none')  # Set the figure background to transparent

        # To remove the white frame (in case it's not already done by the above):
        self.figure.set_edgecolor('none')
        self.figure.set_frameon(False)