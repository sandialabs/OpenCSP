from numpy import ndarray

from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement


class ImageCalibrationGlobal(ImageCalibrationAbstract):
    @staticmethod
    def get_calibration_name() -> str:
        """The name of this calibration class type"""
        return "ImageCalibrationGlobal"

    def apply_to_images(self, measurement: Measurement) -> ndarray:
        """
        Performs camera-projector brightness values calibration by applying the
        camera-display response curve to all pixels equally.

        Parameters
        ----------
        fringe_images : ndarray
            Measurement fringe images.

        Returns
        -------
        ndarray
            Calibrated fringe images, float.

        """
        return self.response_function(measurement.fringe_images)
