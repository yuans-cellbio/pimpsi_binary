import numpy as np
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu
import struct

class PimsoftBinary:
    '''
    This class deals with the binary files exported by Perimed PSI NR imager using the Pimsoft software. The binary file has a header recording basic information about the recording, including the coherence factor and the signal gain used to calculate perfusion data. 

    The class is designed to self-contained. It calculates and measures perfusion using the methods recommended by the manufacturer. It also implements methods to save and retrieve the session data.  
    '''
    def __init__(self, file_path):
        # The file path is used to match session information. 
        self.file_path = file_path
        
        # The following attributes are retrieved from the binary header. 
        self.file_type = ''
        self.file_version = 0
        self.signal_gain = 0.0
        self.coherence_factor = 0.0
        self.number_of_images = 0
        self.image_width = 0
        self.image_height = 0

        # Attributes for spatial and temporal averaging
        self.spatial_window = 1
        self.temporal_window = 1

        # Attributes used to record intensity, variance, and perfusion data. 
        self.variance_images = None
        self.intensity_images = None
        self.perfusion_images = None

        # Attributes used to set intensity filter
        # `intensity_thresholds` is either a single threshold value or a list of threshold values with the same length of intensity images. 
        self.intensity_thresholds = 0
        self.intensity_masks = None

        self.read_pimsoft_binary()

    def spatial_averaging(images, spatial_window):
        kernel = np.ones((spatial_window, spatial_window)) / spatial_window**2
        averaged_images = np.array([convolve(image, kernel) for image in images])

        return averaged_images

    def temporal_averaging(images, temporal_window):
        frame, height, width = images.shape
        num_chunks = frame // temporal_window
        images = images[:num_chunks * temporal_window].reshape((num_chunks, temporal_window, height, width))
        averaged_images = images.mean(axis=1)

        return averaged_images
    
    def set_intensity_thresholds(self, thresholds):
        if thresholds == 'otsu':
            self.thresholds = [threshold_otsu(frame) for frame in self.intensity_images]
        else:
            self.thresholds = thresholds
    
    def update_intensity_masks(self):
        intensity_masks = np.zeros_like(self.intensity_images, dtype=bool)
        for i, frame in enumerate(self.intensity_images):
            intensity_masks[i] = frame > self.thresholds[i]

        self.intensity_masks = intensity_masks
    
    def update_perfusion_images(self):
        intensity_images = self.intensity_images
        variance_images = self.variance_images
        if self.spatial_window > 1:
            intensity_images = self.spatial_averaging(intensity_images, self.spatial_window)
            variance_images = self.spatial_averaging(variance_images, self.spatial_window)
        if self.temporal_window > 1:
            intensity_images = self.temporal_averaging(intensity_images, self.temporal_window)
            variance_images = self.temporal_averaging(variance_images, self.temporal_window)

        # Use intensity masks to limit the area to show perfusion images. 
        intensity_images = np.multiply(intensity_images, self.intensity_masks)
        variance_images = np.multiply(variance_images, self.intensity_masks)

        # Deal with negative variance values
        # TODO: Ask Perimed why there are negative variance values. 
        variance_images = np.abs(variance_images)

        # Assign beta and gain
        coherence_factor = self.coherence_factor
        signal_gain = self.signal_gain

        variance_sqrt = np.sqrt(variance_images)
        variance_sqrt = np.where(variance_sqrt == 0, 1e-10, variance_sqrt)

        perfusion_images = signal_gain * (intensity_images / (coherence_factor * variance_sqrt) - 1)

        # perfusion_images = np.clip(perfusion_images, 0, 30000)

        self.perfusion_images = perfusion_images

    def roi_perfusion_per_frame(self, roi_masks):
        intensity_images = self.intensity_images
        variance_images = self.variance_images

        # Only use the interaction between roi mask and intensity mask
        roi_masks = np.logical_and(roi_masks, self.intensity_masks)
        # intensity_images = np.multiply(intensity_images, roi_masks)
        # variance_images = np.multiply(variance_images, roi_masks)

        # Calculate the average of intensity and variance
        avg_intensity = np.ma.array(intensity_images, mask=~roi_masks).mean(axis=(1, 2))
        avg_variance = np.ma.array(variance_images, mask=~roi_masks).mean(axis=(1, 2))

        # Deal with negative variance values
        # TODO: Ask Perimed why there are negative variance values.
        avg_variance = np.abs(avg_variance)

        # Assign beta and gain
        coherence_factor = self.coherence_factor
        signal_gain = self.signal_gain

        avg_variance_sqrt = np.sqrt(avg_variance)
        avg_variance_sqrt = np.where(avg_variance_sqrt == 0, 1e-10, avg_variance_sqrt)

        avg_perfusion = signal_gain * (avg_intensity / (coherence_factor * avg_variance_sqrt) - 1)

        return(avg_perfusion)
    
    def roi_perfusion_by_toi(self, roi_masks, frames):
        intensity_images = self.intensity_images[frames, :, :]
        variance_images = self.variance_images[frames, :, :]
        intensity_masks = self.intensity_masks[frames, :, :]

        # Only use the interaction between roi mask and intensity mask
        roi_masks = np.logical_and(roi_masks, intensity_masks)

        # Calculate the average of intensity and variance
        avg_intensity = np.mean(intensity_images, where=roi_masks)
        avg_variance = np.mean(variance_images, where=roi_masks)

        # Deal with negative variance values
        # TODO: Ask Perimed why there are negative variance values.
        avg_variance = np.abs(avg_variance)

        # Assign beta and gain
        coherence_factor = self.coherence_factor
        signal_gain = self.signal_gain

        avg_variance_sqrt = np.sqrt(avg_variance)
        avg_variance_sqrt = np.where(avg_variance_sqrt == 0, 1e-10, avg_variance_sqrt)

        avg_perfusion = signal_gain * (avg_intensity / (coherence_factor * avg_variance_sqrt) - 1)

        return(avg_perfusion)



    def read_pimsoft_binary(self):
        with open(self.file_path, 'rb') as file:
            # Read the header
            self.file_type = file.read(10).decode().strip()
            self.file_version, = struct.unpack('i', file.read(4))
            self.signal_gain, = struct.unpack('d', file.read(8))
            self.coherence_factor, = struct.unpack('d', file.read(8))
            self.number_of_images, = struct.unpack('d', file.read(8))
            self.image_width, = struct.unpack('i', file.read(4))
            self.image_height, = struct.unpack('i', file.read(4))

            # Calculate total number of pixels per image
            total_pixels = self.image_width * self.image_height
            number_of_frames = int(self.number_of_images / 2)

            # Read all frames at once
            total_variance_data = np.fromfile(file, dtype=np.float64, count=total_pixels * number_of_frames)
            total_intensity_data = np.fromfile(file, dtype=np.float64, count=total_pixels * number_of_frames)

            # Reshape the data to (number_of_frames, image_height, image_width)
            self.variance_images = total_variance_data.reshape((number_of_frames, self.image_width, self.image_height)).transpose(0, 2, 1)
            self.intensity_images = total_intensity_data.reshape((number_of_frames, self.image_width, self.image_height)).transpose(0, 2, 1)
            
            # Get intensity masks
            self.set_intensity_thresholds('otsu')
            self.update_intensity_masks()

            # Calculate perfusion images
            self.update_perfusion_images()

    def get_frame(self, index, image_type=None):
            if image_type == "intensity":
                image = self.intensity_images[index]
            else:
                image = self.perfusion_images[index]
            return(image)
