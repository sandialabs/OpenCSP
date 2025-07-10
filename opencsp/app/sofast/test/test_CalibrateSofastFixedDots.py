"""Example script that performs dot location calibration using photogrammetry."""

from os.path import join
import unittest

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.CalibrateSofastFixedDots import CalibrateSofastFixedDots
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestFixedPatternSetupCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Tests dot-location calibration"""
        cls.dir_save = join(opencsp_code_dir(), "app/sofast/test/data/output/dot_location_calibration")
        ft.create_directories_if_necessary(cls.dir_save)

        # Set up logger
        lt.logger(log_dir_body_ext=join(cls.dir_save, "log.txt"), level=lt.log.ERROR)

        # Define dot location images and origins
        dir_meas = join(opencsp_code_dir(), "test/data/dot_location_calibration/data_measurement")
        dir_exp = join(opencsp_code_dir(), "test/data/dot_location_calibration/data_expected")
        files = [
            join(dir_meas, "images/DSC03992.JPG"),
            join(dir_meas, "images/DSC03993.JPG"),
            join(dir_meas, "images/DSC03994.JPG"),
            join(dir_meas, "images/DSC03995.JPG"),
            join(dir_meas, "images/DSC03996.JPG"),
        ]
        origins = np.array(([1201, 1120, 1135, 964, 918], [828, 857, 852, 851, 862]))
        origins = Vxy(origins)

        # Define other files
        file_camera_calibration = join(dir_meas, "camera_calibration.h5")
        file_xyz_points = join(dir_meas, "aruco_corner_locations.csv")
        cls.file_fpd_dot_locs_exp = join(dir_exp, "fixed_pattern_dot_locations.h5")

        # Load marker corner locations
        data = np.loadtxt(file_xyz_points, delimiter=",")
        pts_xyz_corners = Vxyz(data[:, 2:5].T)
        ids_corners = data[:, 1]

        # Load cameras
        camera_marker = Camera.load_from_hdf(file_camera_calibration)

        # Perform dot location calibration
        cal_dot_locs = CalibrateSofastFixedDots(
            files, origins, camera_marker, pts_xyz_corners, ids_corners, -32, 31, -31, 32
        )
        cal_dot_locs.plot = True
        cal_dot_locs.blob_search_threshold = 3.0
        cal_dot_locs.blob_detector.minArea = 3.0
        cal_dot_locs.blob_detector.maxArea = 30.0

        cal_dot_locs.run()

        cls.cal = cal_dot_locs

    def test_save_dot_location_hdf(self):
        """Tests saving dot location data"""
        dot_locs = self.cal.get_dot_location_object()
        dot_locs.save_to_hdf(join(self.dir_save, "fixed_pattern_dot_locations.h5"))

    def test_save_figures(self):
        """Tests saving figures"""
        self.cal.save_figures(self.dir_save)

    def test_dot_xyz_locations(self):
        """Tests dot locations"""
        # Load expected dot locations
        dot_locs_exp = DotLocationsFixedPattern.load_from_hdf(self.file_fpd_dot_locs_exp)

        dot_locs = self.cal.get_dot_location_object()

        # Test
        np.testing.assert_allclose(dot_locs.xyz_dot_loc, dot_locs_exp.xyz_dot_loc, atol=1e-6, rtol=0)
        np.testing.assert_allclose(dot_locs.x_dot_index, dot_locs_exp.x_dot_index, atol=1e-6, rtol=0)
        np.testing.assert_allclose(dot_locs.y_dot_index, dot_locs_exp.y_dot_index, atol=1e-6, rtol=0)

    def tearDown(self) -> None:
        # Make sure we release all matplotlib resources.
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
