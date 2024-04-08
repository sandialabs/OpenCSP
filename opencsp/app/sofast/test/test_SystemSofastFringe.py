"""Unit test suite to test the System class
"""

import numpy as np
import os
import unittest

import pytest

import opencsp.app.sofast.lib.ImageCalibrationGlobal as icg
import opencsp.app.sofast.lib.ImageCalibrationScaling as ics
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.app.sofast.test.ImageAcquisition_no_camera import ImageAcquisition
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.deflectometry.test.test_ImageProjection as test_ip
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft


@pytest.mark.no_xvfb
class TestSystemSofastFringe(unittest.TestCase):
    def setUp(self):
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "SystemSofastFringe")
        self.out_dir = os.path.join(path, "data", "output", "SystemSofastFringe")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        # Create fringe object
        self.fringes = Fringes.from_num_periods()

        # Create calibration objects
        projector_values = np.arange(0, 255, (255 - 0) / 9)
        camera_response = np.arange(5, 50, (50 - 5) / 9)
        self.calibration_global = icg.ImageCalibrationGlobal(camera_response, projector_values)
        self.calibration_scaling = ics.ImageCalibrationScaling(camera_response, projector_values)

    def tearDown(self):
        with et.ignored(Exception):
            ImageAcquisition.instance().close()
        with et.ignored(Exception):
            ImageProjection.instance().close()

    def test_SystemSofastFringe(self):
        # Get test data location
        file_im_proj = os.path.join(opencsp_code_dir(), 'test/data/sofast_common/image_projection_test.h5')

        # Create fringe object
        periods_x = [0.9, 3.9]
        periods_y = [15.9, 63.9]
        fringes = Fringes(periods_x, periods_y)

        # Instantiate image projection class
        im_proj = ImageProjection.load_from_hdf_and_display(file_im_proj)

        # Instantiate image acquisition class
        im_aq = ImageAcquisition()

        # Set camera settings
        im_aq.frame_size = (100, 80)
        im_aq.frame_rate = 7
        im_aq.exposure_time = 300000
        im_aq.gain = 230

        # Create system class
        system = SystemSofastFringe(im_aq)

        # Load fringes
        system.set_fringes(fringes, 0)

        # Define functions to put in system queue
        def f1():
            system.capture_mask_and_fringe_images(system.run_next_in_queue)

        def f2():
            system.close_all()

        # Load function in queue
        system.set_queue([f1, f2])

        # Run
        system.run()

    def test_system_all_prereqs(self):
        # Create mock ImageProjection and ImageAcquisition objects
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        ia = ImageAcquisition()

        # Create the system instance
        sys = SystemSofastFringe(ia)

    def test_system_some_prereqs(self):
        # With just a projector
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        with self.assertRaises(RuntimeError):
            sys = SystemSofastFringe()
        ip.close()

        # With just a camera
        ia = ImageAcquisition()
        with self.assertRaises(RuntimeError):
            sys = SystemSofastFringe(ia)
        ia.close()

    def test_system_no_prereqs(self):
        # Base case, nothing is ever set
        with self.assertRaises(RuntimeError):
            sys = SystemSofastFringe()

        # More interesting case, things are set and then unset
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        ia = ImageAcquisition()
        ip.close()
        ia.close()
        with self.assertRaises(RuntimeError):
            sys = SystemSofastFringe()

    def test_run_measurement_no_calibration(self):
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        ia = ImageAcquisition()
        sys = SystemSofastFringe(ia)
        sys.set_fringes(Fringes.from_num_periods(), 0)
        with self.assertRaises(RuntimeError):
            sys.run_measurement()

    def test_run_measurement_without_on_done(self):
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        ia = ImageAcquisition()
        sys = SystemSofastFringe(ia)
        sys.calibration = self.calibration_global
        sys.set_fringes(self.fringes)

        # Shouldn't raise any errors, everything set
        # Note that this doesn't actually evaluate anything because we don't have a tkinter loop running
        sys.run_measurement()

    def test_run_measurement_with_on_done(self):
        global is_done
        global sys
        is_done = False

        def on_done():
            global is_done
            is_done = True
            sys.close_all()

        # create the prerequisites and the system
        ip = test_ip._ImageProjection.in_new_window(test_ip._ImageProjection.display_dict)
        ia = ImageAcquisition()
        sys = SystemSofastFringe(ia)
        sys.calibration = self.calibration_global
        sys.set_fringes(self.fringes)

        # capture images
        self.assertFalse(is_done)
        sys.run_measurement(on_done)
        # runs the tkinter gui main loop until close_all() is called in on_done()
        sys.run()
        # if we got to this point, then either 1. run() does nothing, or 2. on_done() got called
        self.assertTrue(is_done)

        # Verify we have the right number of images
        nfringes = 4 * 4 * 2  # self.fringes.num_images
        self.assertEqual(nfringes, len(sys.fringe_images_captured[0]))

    def test_close_all_closes_acquisition_projections(self):
        # build system, including multiple image_acquisitions
        d = test_ip._ImageProjection.display_dict
        d['win_position_x'] = -640
        ip = test_ip._ImageProjection.in_new_window(d)
        ia1 = ImageAcquisition()
        ia2 = ImageAcquisition()
        sys = SystemSofastFringe([ia1, ia2])

        # verify that the image acquisitions are not closed yet
        self.assertFalse(ia1.is_closed)
        self.assertFalse(ia2.is_closed)

        # close the system
        sys.close_all()

        # verify that the image acquisitions are closed
        self.assertTrue(ia1.is_closed)
        self.assertTrue(ia2.is_closed)


if __name__ == '__main__':
    unittest.main()
