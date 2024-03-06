"""Tests the calculation of the Sofast camera position from known points
"""
import os
from os.path import join
import unittest

import numpy as np

from opencsp.common.lib.deflectometry.CalibrationCameraPosition import (
    CalibrationCameraPosition,
)
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale


class TestCalibrationCameraPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls, dir_input: str = None, dir_output: str = None):
        """Tests the CalibrationCameraPosition process. If directories are None,
        uses default test data directory. All input files listed below must be
        on the dir_input path.

        Input Files (in dir_input):
        ---------------------------
            - camera_sofast.h5
            - point_locations.csv
            - image_sofast_camera.png

        Expected Files (in dir_output):
        ------------------------------
            - camera_rvec_tvec.csv

        Parameters
        ----------
        dir_input : str
            Input/measurment file directory, by default, None
        dir_output : str
            Expected output file directory, by default None

        """
        if (dir_input is None) or (dir_output is None):
            # Define default data directories
            base_dir = os.path.dirname(__file__)
            dir_input = join(base_dir, 'data', 'data_measurement')
            dir_output = join(base_dir, 'data', 'data_expected')

        verbose = 1  # 0=no output, 1=only print outputs, 2=print outputs and show plots, 3=plots only with no printing

        # Define inputs
        file_camera_sofast = join(dir_input, 'camera_sofast.h5')
        file_point_locations = join(dir_input, 'point_locations.csv')
        file_cal_image = join(dir_input, 'image_sofast_camera.png')

        # Load input data
        camera = Camera.load_from_hdf(file_camera_sofast)
        pts_marker_data = np.loadtxt(
            file_point_locations, delimiter=',', dtype=float, skiprows=1
        )
        pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
        corner_ids = pts_marker_data[:, 1]
        image = load_image_grayscale(file_cal_image)

        # Perform camera position calibraiton
        cal_camera_position = CalibrationCameraPosition(
            camera, pts_xyz_marker, corner_ids, image
        )
        cal_camera_position.verbose = verbose
        cal_camera_position.run_calibration()

        # Get calculated vectors
        rvec, tvec = cal_camera_position.get_data()

        # Test data
        cls.data_exp = np.loadtxt(
            join(dir_output, 'camera_rvec_tvec.csv'), delimiter=','
        )
        cls.data_meas = np.vstack((rvec, tvec))

    def test_camera_rvec_tvec(self):
        """Tests the camera position vectors"""
        np.testing.assert_allclose(
            self.data_exp, self.data_meas, rtol=0, atol=1e-6)
        print('rvec/tvec tested successfully.')


if __name__ == '__main__':
    test = TestCalibrationCameraPosition()
    test.setUpClass()

    test.test_camera_rvec_tvec()
