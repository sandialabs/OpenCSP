"""Unit test suite to test ImageCalibrationGlobal class
"""

import datetime as dt
import unittest

import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
import opencsp.app.sofast.lib.OpticScreenDistance as osd
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestImageCalibrationGlobal:
    @classmethod
    def setup_class(cls):
        # Create data
        cls.camera_values = np.concatenate(([0.0, 0.0], np.linspace(1, 255, 8))).astype('uint8')
        cls.display_values = np.linspace(0, 255, 10).astype('uint8')

        # Create frames
        frames = np.ones((100, 200, 10)).astype('uint8')
        frames *= cls.camera_values.reshape((1, 1, -1))

        # Create calibration object
        cls.calibration = ImageCalibrationGlobal.from_data(frames, cls.display_values)

    def test_min_display_camera_values(self):
        disp_min, cam_min = self.calibration.calculate_min_display_camera_values()
        np.testing.assert_equal(disp_min, 56.0)
        np.testing.assert_equal(cam_min, 1.0)

    def test_apply_to_images(self):
        # Create mask images
        mask_images = np.zeros((100, 200, 2))
        mask_images[..., 0] = 0
        mask_images[..., 1] = 255
        mask_images = mask_images.astype('uint8')

        # Create fringe images
        fringe_images = np.ones((100, 200, 8)) * self.camera_values[2:].reshape((1, 1, -1))
        fringe_images = fringe_images.astype('uint8')

        # Expected fringe images are same as display values
        fringe_images_calibrated_exp = np.ones((100, 200, 8)) * self.display_values[2:].astype(float).reshape(
            (1, 1, -1)
        )

        # Create measurement object
        optic_screen_dist_measure = osd.OpticScreenDistance(Vxyz((0, 0, 0)), 10)
        measurement = Measurement(
            mask_images,
            fringe_images,
            np.array([0.0]),
            np.array([0.0]),
            optic_screen_dist_measure,
            dt.datetime.now(),
            'Test',
        )

        # Calibrate
        fringe_images_calibrated = self.calibration.apply_to_images(measurement)

        # Test
        np.testing.assert_allclose(fringe_images_calibrated_exp, fringe_images_calibrated)


if __name__ == '__main__':
    unittest.main()
