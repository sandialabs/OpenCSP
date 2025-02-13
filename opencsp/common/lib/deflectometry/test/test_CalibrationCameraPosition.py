"""Tests the calculation of the Sofast camera position from known points
"""

from os.path import join
import unittest

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.deflectometry.CalibrationCameraPosition import CalibrationCameraPosition
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.file_tools as ft


class TestCalibrationCameraPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Tests the CalibrationCameraPosition process"""
        # Define inputs
        file_camera_sofast = join(opencsp_code_dir(), "test/data/sofast_common/camera_sofast.h5")
        file_point_locations = join(opencsp_code_dir(), "test/data/sofast_common/aruco_corner_locations.csv")
        file_cal_image = join(
            opencsp_code_dir(), "test/data/camera_position_calibration/data_measurement/image_sofast_camera.png"
        )
        file_exp = join(opencsp_code_dir(), "test/data/camera_position_calibration/data_expected/camera_rvec_tvec.csv")

        cls.save_dir = join(opencsp_code_dir(), "common/lib/deflectometry/test/data/output/camera_pose_calibration/")
        ft.create_directories_if_necessary(cls.save_dir)

        # Load input data
        camera = Camera.load_from_hdf(file_camera_sofast)
        pts_marker_data = np.loadtxt(file_point_locations, delimiter=",", dtype=float, skiprows=1)
        pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
        corner_ids = pts_marker_data[:, 1]
        image = load_image_grayscale(file_cal_image)

        # Perform camera position calibraiton
        cal_camera_position = CalibrationCameraPosition(camera, pts_xyz_marker, corner_ids, image)
        cal_camera_position.run_calibration()

        # Get calculated vectors
        rvec, tvec = cal_camera_position.get_data()

        # Test data
        cls.data_exp = np.loadtxt(file_exp, delimiter=",")
        cls.data_meas = np.vstack((rvec, tvec))
        cls.cal = cal_camera_position

    def tearDown(self) -> None:
        # Make sure we release all matplotlib resources.
        plt.close("all")

    def test_camera_rvec_tvec(self):
        """Tests the camera position vectors"""
        np.testing.assert_allclose(self.data_exp, self.data_meas, rtol=0, atol=1e-6)

    def test_to_csv(self):
        """Tests saving to csv"""
        self.cal.save_data_as_csv(join(self.save_dir, "data.csv"))


if __name__ == "__main__":
    save_dir = join(opencsp_code_dir(), "common/lib/deflectometry/test/data/output/camera_pose_calibration/")
    ft.create_directories_if_necessary(save_dir)
    lt.logger(join(save_dir, "log.txt"), lt.log.ERROR)

    unittest.main()
