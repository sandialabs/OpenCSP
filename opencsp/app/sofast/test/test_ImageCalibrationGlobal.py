"""Unit test suite to test ImageCalibrationGlobal class"""

import datetime as dt
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.file_tools as ft


class TestImageCalibrationGlobal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Get data directories
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "ImageCalibrationGlobal")
        cls.out_dir = os.path.join(path, "data", "output", "ImageCalibrationGlobal")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

        # Create data
        cls.camera_values = np.concatenate(([0.0, 0.0], np.linspace(1, 255, 8))).astype("uint8")
        cls.display_values = np.linspace(0, 255, 10).astype("uint8")

        # Create frames
        frames = np.ones((100, 200, 10)).astype("uint8")
        frames *= cls.camera_values.reshape((1, 1, -1))

        # Create calibration object
        cls.calibration = ImageCalibrationGlobal.from_data(frames, cls.display_values)

    def tearDown(self):
        # Make sure we release all matplotlib resources.
        plt.close("all")

    def test_min_display_camera_values(self):
        disp_min, cam_min = self.calibration.calculate_min_display_camera_values()
        np.testing.assert_equal(disp_min, 56.0)
        np.testing.assert_equal(cam_min, 1.0)

    def test_apply_to_images(self):
        # Create mask images
        mask_images = np.zeros((100, 200, 2))
        mask_images[..., 0] = 0
        mask_images[..., 1] = 255
        mask_images = mask_images.astype("uint8")

        # Create fringe images
        fringe_images = np.ones((100, 200, 8)) * self.camera_values[2:].reshape((1, 1, -1))
        fringe_images = fringe_images.astype("uint8")

        # Expected fringe images are same as display values
        fringe_images_calibrated_exp = np.ones((100, 200, 8)) * self.display_values[2:].astype(float).reshape(
            (1, 1, -1)
        )

        # Create measurement object
        dist_optic_screen_measure = osd.DistanceOpticScreen(Vxyz((0, 0, 0)), 10)
        measurement = Measurement(
            mask_images,
            fringe_images,
            np.array([0.0]),
            np.array([0.0]),
            dist_optic_screen_measure,
            dt.datetime.now(),
            "Test",
        )

        # Calibrate
        fringe_images_calibrated = self.calibration.apply_to_images(measurement)

        # Test
        np.testing.assert_allclose(fringe_images_calibrated_exp, fringe_images_calibrated)

    def test_to_from_hdf(self):
        # load our testing data from a known file
        file_cal_global = os.path.join(self.data_dir, "cal_global.h5")
        cal = ImageCalibrationGlobal.load_from_hdf(file_cal_global)
        np.testing.assert_almost_equal(cal.camera_values, [5, 10, 15, 20, 25, 30, 35, 40, 45], decimal=0.1)
        np.testing.assert_almost_equal(
            cal.display_values, [0, 28.33, 56.66, 85, 113.33, 141.66, 170, 198.33, 226.66], decimal=0.1
        )

        # make sure our destination hdf file doesn't exist yet
        cal_path_name_ext = os.path.join(self.out_dir, "test_to_from_hdf.h5")
        ft.delete_file(cal_path_name_ext, error_on_not_exists=False)
        self.assertFalse(ft.file_exists(cal_path_name_ext))

        # save to an hdf file
        cal.save_to_hdf(cal_path_name_ext)
        self.assertTrue(ft.file_exists(cal_path_name_ext))

        # load and compare
        cal2 = ImageCalibrationGlobal.load_from_hdf(cal_path_name_ext)
        np.testing.assert_almost_equal(cal.camera_values, cal2.camera_values, decimal=1e-6)
        np.testing.assert_almost_equal(cal.display_values, cal2.display_values, decimal=1e-6)

    def test_plot_gray_levels_cal_no_calibration(self):
        # load the calibration
        file_cal_global = os.path.join(self.data_dir, "cal_global.h5")
        cal = ImageCalibrationGlobal.load_from_hdf(file_cal_global)

        # plot (the calibration class should take care of closing the plot)
        cal.plot_gray_levels()


if __name__ == "__main__":
    unittest.main()
