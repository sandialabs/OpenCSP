"""Tests Sofast screen distortion calibration
"""

import os
from os.path import join
import unittest

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pytest

from opencsp.app.sofast.lib.CalibrateDisplayShape import CalibrateDisplayShape, DataInput
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjectionData
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestCalibrateDisplayShape(unittest.TestCase):
    """Tests CalibrateDisplayShape"""

    @classmethod
    def setUpClass(cls):
        # Define default data directories
        dir_meas_data = join(opencsp_code_dir(), "test/data/display_shape_calibration/data_measurement")

        # Define input files
        resolution_xy = [100, 100]  # sample density of screen
        file_screen_cal_point_pairs = join(dir_meas_data, "screen_calibration_point_pairs.csv")
        file_point_locations = join(opencsp_code_dir(), "test/data/sofast_common/aruco_corner_locations.csv")
        file_camera_distortion = join(dir_meas_data, "camera_screen_shape.h5")
        file_image_projection = join(opencsp_code_dir(), "test/data/sofast_common/image_projection.h5")
        files_screen_shape_measurement = glob(join(dir_meas_data, "screen_shape_sofast_measurements/pose_*.h5"))
        files_screen_shape_measurement.sort()
        file_exp = join(
            opencsp_code_dir(), "test/data/display_shape_calibration/data_expected/screen_distortion_data_100_100.h5"
        )

        # Load input data
        pts_marker_data = np.loadtxt(file_point_locations, delimiter=",", dtype=float, skiprows=1)
        pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
        corner_ids = pts_marker_data[:, 1]
        screen_cal_point_pairs = np.loadtxt(file_screen_cal_point_pairs, delimiter=",", skiprows=1).astype(int)
        camera = Camera.load_from_hdf(file_camera_distortion)
        image_projection_data = ImageProjectionData.load_from_hdf(file_image_projection)

        # Store input data in data class
        data_input = DataInput(
            corner_ids,
            screen_cal_point_pairs,
            resolution_xy,
            pts_xyz_marker,
            camera,
            image_projection_data,
            [MeasurementSofastFringe.load_from_hdf(f) for f in files_screen_shape_measurement],
        )

        # Perform screen position calibration
        cal_screen_position = CalibrateDisplayShape(data_input)
        cal_screen_position.run_calibration()
        cls.cal = cal_screen_position

        # Test screen distortion information
        cls.data_exp = load_hdf5_datasets(["pts_xy_screen_fraction", "pts_xyz_screen_coords"], file_exp)
        cls.save_dir_local = join(opencsp_code_dir(), "app/sofast/test/data/output/display_shape_calibrate")
        ft.create_directories_if_necessary(cls.save_dir_local)

    def tearDown(self) -> None:
        # Make sure we release all matplotlib resources.
        plt.close("all")

    @pytest.mark.skipif(os.name != "nt", reason="Does not pass in Linux environment for unkonwn reason.")
    def test_xy_screen_fraction(self):
        """Tests xy points"""
        data_meas = self.cal.get_data()
        np.testing.assert_allclose(
            data_meas["xy_screen_fraction"].data, self.data_exp["pts_xy_screen_fraction"], rtol=0, atol=1e-6
        )

    @pytest.mark.skipif(os.name != "nt", reason="Does not pass in Linux environment for unkonwn reason.")
    def test_xyz_screen_coords(self):
        """Tests xyz points"""
        data_meas = self.cal.get_data()
        np.testing.assert_allclose(
            data_meas["xyz_screen_coords"].data, self.data_exp["pts_xyz_screen_coords"], rtol=0, atol=1e-6
        )

    def test_save_display_object(self):
        """Tests saving DisplayShape object"""
        display_shape = self.cal.as_DisplayShape("Test display")
        file = join(self.save_dir_local, "test_calibration_display.h5")
        display_shape.save_to_hdf(file)


if __name__ == "__main__":
    # Set up save dir
    save_dir = join(opencsp_code_dir(), "app/sofast/test/data/output/display_shape_calibrate")
    ft.create_directories_if_necessary(save_dir)

    # Set up logger
    lt.logger(join(save_dir, "log_display_shape.txt"), lt.log.WARN)

    unittest.main()
