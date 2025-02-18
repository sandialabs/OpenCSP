"""Unit test suite to test DisplayShape class
"""

import unittest
from os.path import dirname, join

import numpy as np

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.file_tools as ft


class TestDisplayShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define screen X and Y extent
        LX = 5.0  # meters
        LY = 5.0  # meters
        LZ = 3.0  # meters

        # Define test points
        cls.test_Vxy_pts = Vxy(([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]))

        # Define rectangular input data
        cls.grid_data_rect2D = {"screen_x": LX, "screen_y": LY, "screen_model": "rectangular2D"}

        # Define 2D input data
        cls.grid_data_2D = {
            "xy_screen_fraction": Vxy(([0, 1, 0, 1], [0, 0, 1, 1])),
            "xy_screen_coords": Vxy(([-LX / 2, LX / 2, -LX / 2, LX / 2], [-LY / 2, -LY / 2, LY / 2, LY / 2])),
            "screen_model": "distorted2D",
        }

        # Define 3D input data
        cls.grid_data_3D = {
            "xy_screen_fraction": Vxy(([0, 1, 0, 1], [0, 0, 1, 1])),
            "xyz_screen_coords": Vxyz(
                ([-LX / 2, LX / 2, -LX / 2, LX / 2], [-LY / 2, -LY / 2, LY / 2, LY / 2], [LZ, LZ, 0, 0])
            ),
            "screen_model": "distorted3D",
        }

        # Define expected 2D points
        cls.exp_Vxy_disp_pts = Vxyz(
            (
                [-LX / 2, 0.0, LX / 2, -LX / 2, 0.0, LX / 2, -LX / 2, 0.0, LX / 2],
                [-LY / 2, -LY / 2, -LY / 2, 0.0, 0.0, 0.0, LY / 2, LY / 2, LY / 2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )

        # Define expected 3D points
        cls.exp_Vxyz_disp_pts = Vxyz(
            (
                [-LX / 2, 0.0, LX / 2, -LX / 2, 0.0, LX / 2, -LX / 2, 0.0, LX / 2],
                [-LY / 2, -LY / 2, -LY / 2, 0.0, 0.0, 0.0, LY / 2, LY / 2, LY / 2],
                [LZ, LZ, LZ, LZ / 2, LZ / 2, LZ / 2, 0.0, 0.0, 0.0],
            )
        )

        # Set up save path
        cls.save_dir = join(dirname(__file__), "data/output/display_shape")
        ft.create_directories_if_necessary(cls.save_dir)

    def test_rectangular2D(self):
        # Instantiate display object
        name = "Test DisplayShape"
        disp = DisplayShape(self.grid_data_rect2D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(calc.data, self.exp_Vxy_disp_pts.data, rtol=0, atol=0)

    def test_distorted2D(self):
        # Instantiate display object
        name = "Test DisplayShape"
        disp = DisplayShape(self.grid_data_2D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(calc.data, self.exp_Vxy_disp_pts.data, rtol=0, atol=1e-7)

    def test_distorted3D(self):
        # Instantiate display object
        name = "Test DisplayShape"
        disp = DisplayShape(self.grid_data_3D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(calc.data, self.exp_Vxyz_disp_pts.data, rtol=0, atol=1e-7)

    def test_save_load_hdf_dist_3d(self):
        # Instantiate display object
        name = "Test DisplayShape 3D"
        disp = DisplayShape(self.grid_data_3D, name)
        file = join(self.save_dir, "test_display_shape_dist_3d.h5")

        # Save
        disp.save_to_hdf(file)

        # Load
        disp_load = DisplayShape.load_from_hdf(file)

        # Compare
        self.assertEqual(disp.grid_data["screen_model"], disp_load.grid_data["screen_model"])
        self.assertEqual(disp.name, disp_load.name)
        np.testing.assert_equal(
            disp.grid_data["xy_screen_fraction"].data, disp_load.grid_data["xy_screen_fraction"].data
        )
        np.testing.assert_equal(disp.grid_data["xyz_screen_coords"].data, disp_load.grid_data["xyz_screen_coords"].data)

    def test_save_load_hdf_dist_2d(self):
        # Instantiate display object
        name = "Test DisplayShape 2D"
        disp = DisplayShape(self.grid_data_2D, name)
        file = join(self.save_dir, "test_display_shape_dist_2d.h5")

        # Save
        disp.save_to_hdf(file)

        # Load
        disp_load = DisplayShape.load_from_hdf(file)

        # Compare
        self.assertEqual(disp.grid_data["screen_model"], disp_load.grid_data["screen_model"])
        self.assertEqual(disp.name, disp_load.name)
        np.testing.assert_equal(
            disp.grid_data["xy_screen_fraction"].data, disp_load.grid_data["xy_screen_fraction"].data
        )
        np.testing.assert_equal(disp.grid_data["xy_screen_coords"].data, disp_load.grid_data["xy_screen_coords"].data)

    def test_save_load_hdf_rectangular(self):
        # Instantiate display object
        name = "Test DisplayShape Rectangular"
        disp = DisplayShape(self.grid_data_rect2D, name)
        file = join(self.save_dir, "test_display_shape_rect.h5")

        # Save
        disp.save_to_hdf(file)

        # Load
        disp_load = DisplayShape.load_from_hdf(file)

        # Compare
        self.assertEqual(disp.grid_data["screen_model"], disp_load.grid_data["screen_model"])
        self.assertEqual(disp.name, disp_load.name)
        self.assertEqual(disp.grid_data["screen_x"], disp_load.grid_data["screen_x"])
        self.assertEqual(disp.grid_data["screen_y"], disp_load.grid_data["screen_y"])


if __name__ == "__main__":
    unittest.main()
