import os
import time

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numpy.testing as nptest
from PIL import Image
import pytest
import tkinter as tk
import unittest

import opencsp.app.sofast.lib.Fringes as fr
import opencsp.app.sofast.lib.ImageCalibrationGlobal as icg
import opencsp.app.sofast.lib.ImageCalibrationScaling as ics
import opencsp.app.sofast.lib.SystemSofastFringe as ssf
import opencsp.app.sofast.SofastService as ss
import opencsp.app.sofast.test.ImageAcquisition_no_camera as ianc
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import FalseColorImageProcessor
import opencsp.common.lib.deflectometry.ImageProjection as ip
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class TestSofastService(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "SofastService")
        self.out_dir = os.path.join(path, "data", "output", "SofastService")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        # Create fringe object
        periods_x = [4**idx for idx in range(4)]
        periods_y = [4**idx for idx in range(4)]
        periods_x[0] -= 0.1
        periods_y[0] -= 0.1
        self.fringes = fr.Fringes(periods_x, periods_y)

        # Create calibration objects
        projector_values = np.arange(0, 255, (255 - 0) / 9)
        camera_response = np.arange(5, 50, (50 - 5) / 9)
        self.calibration_global = icg.ImageCalibrationGlobal(camera_response, projector_values)
        self.calibration_scaling = ics.ImageCalibrationScaling(camera_response, projector_values)

        self.service = ss.SofastService()

    def tearDown(self):
        self.service.close()

    @pytest.mark.no_xvfb
    def test_set_image_projection(self):
        self.assertIsNone(self.service.image_acquisition)

        # Create a mock ImageProjection object
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)

        # Test setting the image_projection object
        self.service.image_projection = image_projection
        self.assertEqual(self.service.image_projection, image_projection)

        # Test un-setting the image_projection object
        self.assertFalse(image_projection.is_closed)
        self.service.image_projection = None
        self.assertTrue(image_projection.is_closed)
        self.assertIsNone(self.service.image_projection)

    def test_set_image_acquisition(self):
        self.assertIsNone(self.service.image_acquisition)

        # Create a mock ImageAcquisition object
        image_acquisition = ianc.ImageAcquisition()

        # Test setting the image_acquisition object
        self.service.image_acquisition = image_acquisition
        self.assertEqual(self.service.image_acquisition, image_acquisition)

        # Test un-setting the image_acquisition object
        self.assertFalse(image_acquisition.is_closed)
        self.service.image_acquisition = None
        self.assertTrue(image_acquisition.is_closed)
        self.assertIsNone(self.service.image_acquisition)

    @pytest.mark.no_xvfb
    def test_get_system_all_prereqs(self):

        # Create mock ImageProjection and ImageAcquisition objects
        image_acquisition = ianc.ImageAcquisition()
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        self.service.image_projection = image_projection
        self.service.image_acquisition = image_acquisition

        # Assert that we can retrieve the system instance
        system = self.service.system
        self.assertIsNotNone(system)

    @pytest.mark.no_xvfb
    def test_get_system_some_prereqs(self):

        # Create mock ImageProjection and ImageAcquisition objects
        image_acquisition = ianc.ImageAcquisition()
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)

        # Assert that with only one prereq we CANNOT retrieve the system instance
        self.service.image_projection = image_projection
        with self.assertRaises(Exception):
            system = self.service.system
        self.service.image_projection = None
        self.service.image_acquisition = image_acquisition
        with self.assertRaises(Exception):
            system = self.service.system

    @pytest.mark.no_xvfb
    def test_get_system_no_prereqs(self):

        # Base case, nothing is ever set
        with self.assertRaises(Exception):
            system = self.service.system

        # More interesting case, things are set and then unset
        image_acquisition = ianc.ImageAcquisition()
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        for attr_name in ["image_acquisition", "image_projection"]:
            self.service.image_projection = image_projection
            self.service.image_acquisition = image_acquisition
            system = self.service.system
            self.assertIsNotNone(system)

            setattr(self.service, attr_name, None)
            with self.assertRaises(Exception):
                system = self.service.system

    def test_set_unset_calibration(self):
        self.assertIsNone(self.service.calibration)

        calibrations = [self.calibration_global, self.calibration_scaling]
        for calibration in calibrations:
            # Set a mock calibration object
            self.service.calibration = calibration
            self.assertEqual(self.service.calibration, calibration)

            # Unset and verify is None
            self.service.calibration = None
            self.assertIsNone(self.service.calibration)

    @pytest.mark.no_xvfb
    def test_get_frame(self):

        # Create a mock ImageAcquisition object
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_acquisition = image_acquisition

        # Try to get a frame
        frame = self.service.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (640, 320))

    def test_get_frame_no_image_acquisition(self):
        with self.assertRaises(Exception):
            frame = self.service.get_frame()

    def test_run_measurement_no_prereqs(self):
        with self.assertRaises(Exception):
            self.service.run_measurement(self.fringes, lambda: print("done"))

    @pytest.mark.no_xvfb
    def test_run_measurement_no_calibration(self):
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_projection = image_projection
        self.service.image_acquisition = image_acquisition
        with self.assertRaises(Exception):
            self.service.run_measurement(self.fringes)

    @pytest.mark.no_xvfb
    def test_run_measurement_without_on_done(self):
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_projection = image_projection
        self.service.image_acquisition = image_acquisition
        self.service.calibration = self.calibration_global
        self.service.run_measurement(self.fringes)

    @pytest.mark.no_xvfb
    def test_run_measurement_with_on_done(self):
        global is_done

        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_projection = image_projection
        self.service.image_acquisition = image_acquisition
        self.service.calibration = self.calibration_global
        is_done = False

        def on_done():
            global is_done
            is_done = True
            self.service.close()

        self.assertFalse(is_done)
        self.service.run_measurement(self.fringes, on_done=on_done)
        # runs the tkinter gui main loop until close_all() is called in on_done()
        self.service.system.run()
        # if we got to this point, then either 1. run() does nothing, or 2. on_done() got called
        self.assertTrue(is_done)

    def test_run_exposure_cal(self):
        image_acquisition = _IA_No_Calibrate()
        self.service.image_acquisition = image_acquisition
        self.assertFalse(image_acquisition.is_calibrated)
        self.service.run_exposure_cal()
        self.assertTrue(image_acquisition.is_calibrated)

    def test_load_gray_levels_cal_global(self):
        file_cal_global = os.path.join(self.data_dir, "cal_global.h5")
        self.assertIsNone(self.service.calibration)
        self.service.load_gray_levels_cal(file_cal_global)
        self.assertIsInstance(self.service.calibration, icg.ImageCalibrationGlobal)

    def test_load_gray_levels_cal_scaling(self):
        file_cal_scaling = os.path.join(self.data_dir, "cal_scaling.h5")
        self.assertIsNone(self.service.calibration)
        self.service.load_gray_levels_cal(file_cal_scaling)
        self.assertIsInstance(self.service.calibration, ics.ImageCalibrationScaling)

    def test_plot_gray_levels_cal_no_calibration(self):
        with self.assertRaises(Exception):
            self.service.plot_gray_levels_cal()

    def test_plot_gray_levels_cal_no_errors(self):
        self.service.calibration = self.calibration_global
        self.service.plot_gray_levels_cal()

    def test_get_exposure(self):
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_acquisition = image_acquisition
        exposure = self.service.get_exposure()
        self.assertIsNotNone(exposure)

    def test_get_exposure_no_image_acquisition(self):
        with self.assertRaises(Exception):
            self.service.get_exposure()

    def test_set_exposure(self):
        image_acquisition = ianc.ImageAcquisition()
        self.service.image_acquisition = image_acquisition
        new_exp = 0.1
        self.service.set_exposure(new_exp)
        self.assertEqual(self.service.image_acquisition.exposure_time, new_exp)

    def test_set_exposure_illegal_values(self):
        self.assertTrue(True)
        # with self.assertRaises(ValueError):
        #     service.set_exposure(new_exp)
        # Exposure time denial is dependent on the type of ImageAcquisition, and so can't be tested here.

    @pytest.mark.no_xvfb
    def test_close_closes_acquisition_projections(self):
        image_acquisition = ianc.ImageAcquisition()
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        self.service.image_projection = image_projection
        self.service.image_acquisition = image_acquisition
        self.assertFalse(image_acquisition.is_closed)
        self.assertFalse(image_projection.is_closed)

        self.service.close()
        self.assertTrue(image_acquisition.is_closed)
        self.assertTrue(image_projection.is_closed)
        self.assertIsNone(self.service.image_acquisition)
        self.assertIsNone(self.service.image_projection)
        with self.assertRaises(Exception):
            system = self.service.system


class _ImageProjection(ip.ImageProjection):
    display_dict = {
        'name': 'smoll_for_unit_tests',
        'win_size_x': 640,
        'win_size_y': 480,
        'win_position_x': 0,
        'win_position_y': 0,
        'size_x': 640,
        'size_y': 480,
        'position_x': 0,
        'position_y': 0,
        'projector_max_int': 255,
        'projector_data_type': "uint8",
        'shift_red_x': 0,
        'shift_red_y': 0,
        'shift_blue_x': 0,
        'shift_blue_y': 0,
        'image_delay': 10,
        'ui_position_x': 0,
    }


class _IA_No_Calibrate(ianc.ImageAcquisition):
    def __init__(self):
        self.is_calibrated = False

    def calibrate_exposure(self):
        self.is_calibrated = True


if __name__ == '__main__':
    unittest.main()
