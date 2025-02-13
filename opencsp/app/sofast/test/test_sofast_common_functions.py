import os
import unittest

import numpy as np
import pytest

import opencsp.app.sofast.lib.Fringes as fr
import opencsp.app.sofast.lib.ImageCalibrationGlobal as icg
import opencsp.app.sofast.lib.ImageCalibrationScaling as ics
import opencsp.app.sofast.lib.sofast_common_functions as scf
import opencsp.app.sofast.lib.SystemSofastFringe as ssf
import opencsp.app.sofast.test.ImageAcquisition_no_camera as ianc
import opencsp.common.lib.deflectometry.ImageProjection as ip
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft


class test_sofast_common_functions(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "sofast_common_functions")
        self.cal_dir = os.path.join(path, "data", "input", "ImageCalibrationGlobal")
        self.out_dir = os.path.join(path, "data", "output", "sofast_common_functions")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        # Load ImageProjectionData
        self.file_image_projection_input = os.path.join(
            opencsp_code_dir(), "test/data/sofast_common/image_projection_test.h5"
        )
        self.image_projection_data = ip.ImageProjectionData.load_from_hdf(self.file_image_projection_input)

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

    def tearDown(self):
        with et.ignored(Exception):
            ip.ImageProjection.instance().close()
        with et.ignored(Exception):
            ianc.ImageAcquisition.instance().close()

    @pytest.mark.no_xvfb
    def test_check_projector_loaded(self):
        # No instance loaded yet, should throw an error
        with self.assertRaises(RuntimeError):
            scf.check_projector_loaded("test_check_projector_loaded")

        # Create a mock ImageProjection object
        image_projection = ip.ImageProjection.in_new_window(self.image_projection_data)

        # No more error!
        self.assertTrue(scf.check_projector_loaded("test_check_projector_loaded"))

    def test_check_acquisition_loaded(self):
        # No instance loaded yet, should throw an error
        with self.assertRaises(RuntimeError):
            scf.check_camera_loaded("test_check_acquisition_loaded")

        # Create a mock ImageAcquisition object
        image_acquisition = ianc.ImageAcquisition()

        # No more error!
        self.assertTrue(scf.check_camera_loaded("test_check_acquisition_loaded"))

    @pytest.mark.no_xvfb
    def test_check_system_loaded(self):
        # No instance loaded yet, should throw an error
        with self.assertRaises(RuntimeError):
            scf.check_system_fringe_loaded(None, "test_check_projector_loaded")

        # Create the prerequisites
        im_proj = ip.ImageProjection.in_new_window(self.image_projection_data)
        ia = ianc.ImageAcquisition()

        # Still no instance loaded, should throw an error
        with self.assertRaises(RuntimeError):
            scf.check_system_fringe_loaded(None, "test_check_projector_loaded")

        # Create a system instance
        sys = ssf.SystemSofastFringe()

        # No more error!
        self.assertTrue(scf.check_system_fringe_loaded(sys, "test_check_projector_loaded"))

        # Release the prerequisites, should throw an error again
        im_proj.close()
        ia.close()
        with self.assertRaises(RuntimeError):
            scf.check_system_fringe_loaded(None, "test_check_projector_loaded")

    @pytest.mark.no_xvfb
    def test_check_calibration_loaded(self):
        # No instance loaded yet, should throw an error
        with self.assertRaises(RuntimeError):
            scf.check_calibration_loaded(None, "test_check_calibration_loaded")

        # Create the prerequisites and system instance
        im_proj = ip.ImageProjection.in_new_window(self.image_projection_data)
        ia = ianc.ImageAcquisition()
        sys = ssf.SystemSofastFringe()

        # Create the calibration instance
        cal_path_name_ext = os.path.join(self.cal_dir, "cal_global.h5")
        cal = icg.ImageCalibrationGlobal.load_from_hdf(cal_path_name_ext)
        sys._calibration = cal

        # No more error!
        self.assertTrue(scf.check_calibration_loaded(sys, "test_check_calibration_loaded"))

    def test_run_exposure_cal(self):
        global sys

        # Create the prerequisites and system instance
        im_proj = ip.ImageProjection.in_new_window(self.image_projection_data)
        ia = ianc.IA_No_Calibrate()
        sys = ssf.SystemSofastFringe()

        # Finish the test after the on_done method
        def on_done():
            global sys
            sys.root.destroy()

        # Run the calibration
        # on_done will exit the main loop
        self.assertFalse(ia.is_calibrated)
        scf.run_exposure_cal(on_done=on_done)
        sys.root.mainloop()

        # Check that calibration was finished
        self.assertTrue(ia.is_calibrated)

    def test_get_exposure_no_image_acquisition(self):
        with self.assertRaises(RuntimeError):
            scf.get_exposure()


if __name__ == "__main__":
    unittest.main()
