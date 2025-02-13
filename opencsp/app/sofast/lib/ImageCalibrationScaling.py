from numpy import ndarray
import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement


class ImageCalibrationScaling(ImageCalibrationAbstract):
    @staticmethod
    def get_calibration_name() -> str:
        """The name of this calibration class type"""
        return "ImageCalibrationScaling"

    def apply_to_images(self, measurement: Measurement) -> ndarray:
        """
        Performs camera-projector brightness values calibration in two steps.

        1) Using captured mask images, pixel values are scaled to match their
           expected min/max values given the calibration response curve.
        2) The camera to display digital number (DN) response interpolation
           is applied to all pixel values.

        Parameters
        ----------
        measurement : Measurement
            Measurement object to apply calibration to.

        Returns
        -------
        ndarray
            Calibrated fringe images, float.

        """
        # Convert camera images to observed display values
        im_dark_disp = self.response_function(measurement.mask_images[..., 0:1])  # M x N x 1
        im_light_disp = self.response_function(measurement.mask_images[..., 1:2])  # M x N x 1
        fringe_images_disp = self.response_function(measurement.fringe_images)  # M x N x n

        # Calculate delta image
        im_delta_disp = im_light_disp - im_dark_disp  # M x N x 1

        # Remove zeros from delta mask (will not be active pixels)
        im_delta_disp[im_delta_disp <= 0] = np.nan

        # Calculate min/max camera values for nominal pixels
        max_disp = self.display_values[-1].astype(float)

        # Create scale image from observed display value to projected display value
        im_scale = max_disp / im_delta_disp  # M x N x 1

        # Subtract display value offset from fringe images
        fringe_images_disp -= im_dark_disp

        # Scale fringe images from observed display value to projected display value
        fringe_images_disp *= im_scale

        # Remove nans
        fringe_images_disp[np.isnan(fringe_images_disp)] = 0

        # Return scaled fringe images
        return fringe_images_disp
