"""Unit test suite to test Surface2D type classes
"""

from os.path import join
import unittest

import numpy as np

from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.deflectometry.Surface2DPlano import Surface2DPlano
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft


class Test2DSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test data
        cls.data_test = [generate_2DParabolic(), generate_2DPlano()]
        # Save location
        cls.dir_save = join(opencsp_code_dir(), "common/lib/deflectometry/test/data/output/Surface2D")
        ft.create_directories_if_necessary(cls.dir_save)

    def test_intersect(self):
        """Tests the intersection of rays with fit surface."""
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: Vxyz = test[1]

                np.testing.assert_allclose(surface.v_surf_int_pts_optic.data, data_exp.data)

    def test_calculate_slopes(self):
        """Tests slope calculations."""
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: np.ndarray = test[2]

                np.testing.assert_allclose(surface.slopes, data_exp)

    def test_fit_slopes(self):
        """Tests slope fit"""
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: np.ndarray = test[3]

                np.testing.assert_allclose(surface.slope_coefs, data_exp)

    def test_fit_surf(self):
        """Tests surface fit"""
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: np.ndarray = test[4]

                np.testing.assert_allclose(surface.surf_coefs, data_exp)

    def test_design_normal(self):
        """
        Test normal of design surface at align point

        """
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: Uxyz = test[5]

                # Calculate normals of design surface
                n_fit = Uxyz(surface.normal_fit_at_align_point().data)

                # Test
                np.testing.assert_allclose(n_fit.data, data_exp.data)

    def test_fit_normal(self):
        """
        Test normal of fit surface at align point

        """
        for i, test in enumerate(self.data_test):
            with self.subTest(i=i):
                # Get data
                surface: Surface2DAbstract = test[0]
                data_exp: Uxyz = test[6]

                # Calculate normals of fit surface
                n_design = Uxyz(surface.normal_design_at_align_point().data)

                # Test
                np.testing.assert_allclose(n_design.data, data_exp.data)

    def test_io(self):
        """Test saving to HDF5"""
        prefix = "test_folder/"
        for idx, surf in enumerate(self.data_test):
            surf_cur: Surface2DAbstract = surf[0]
            file = join(self.dir_save, f"test_surface_{idx:d}.h5")
            # Test saving
            surf_cur.save_to_hdf(file, prefix)
            # Test loading
            surf_cur.load_from_hdf(file, prefix)


def generate_2DParabolic() -> tuple[Surface2DParabolic, Vxyz, np.ndarray, np.ndarray, np.ndarray, Uxyz, Uxyz]:
    """
    Generates data for 2DParabolic case
    """
    # Generate surface: z = 1/4*x^2 + 1/4*y^2
    initial_focal_lengths_xy = (1.0, 1.0)
    robust_least_squares = False
    downsample = 1
    surface = Surface2DParabolic(initial_focal_lengths_xy, robust_least_squares, downsample)

    # Define reflection geometry
    x_int = 2 * (np.sqrt(2) - 1)
    z_int = 0.25 * x_int**2

    u_active_pixel_pointing_optic = Uxyz(([-1, 1, 0, 0, 0], [0, 0, -1, 1, 0], [-1, -1, -1, -1, -1]))
    v_screen_points_optic = Vxyz(([-x_int, x_int, 0, 0, 0], [0, 0, -x_int, x_int, 0], [1, 1, 1, 1, 1]))
    v_optic_cam_optic = Vxyz((0, 0, 1))
    u_measure_pixel_pointing_optic = Uxyz((0, 0, -1))
    v_align_point_optic = Vxyz((0, 0, 0))
    v_optic_screen_optic = Vxyz((0, 0, 1))

    # Set spatial data
    surface.set_spatial_data(
        u_active_pixel_pointing_optic,
        v_screen_points_optic,
        v_optic_cam_optic,
        u_measure_pixel_pointing_optic,
        v_align_point_optic,
        v_optic_screen_optic,
    )

    # Define expected data
    v_surf_int_pts_exp = Vxyz(([-x_int, x_int, 0, 0, 0], [0, 0, -x_int, x_int, 0], [z_int, z_int, z_int, z_int, 0]))
    slopes_exp = np.array(([-x_int / 2, x_int / 2, 0, 0, 0], [0, 0, -x_int / 2, x_int / 2, 0]))
    slope_coefs_exp = np.array(([0, 0.5, 0], [0, 0, 0.5]))
    surf_coefs_exp = np.array([0, 0, 0.25, 0, 0, 0.25])
    u_design_exp = Uxyz((0.0, 0.0, 1.0))
    u_fit_exp = Uxyz((0.0, 0.0, 1.0))

    # Calculate surface intersection with initial polynomial shape
    surface.calculate_surface_intersect_points()

    # Calculate slopes
    surface.calculate_slopes()

    # Calculate surface fit coefficients
    surface.fit_slopes()

    # Pack data
    return (surface, v_surf_int_pts_exp, slopes_exp, slope_coefs_exp, surf_coefs_exp, u_design_exp, u_fit_exp)


def generate_2DPlano() -> tuple[Surface2DPlano, Vxyz, np.ndarray, np.ndarray, np.ndarray, Uxyz, Uxyz]:
    """
    Generates data for 2DPlano case
    """
    # Generate surface: z = 0*x + 0*y
    robust_least_squares = False
    downsample = 1
    surface = Surface2DPlano(robust_least_squares, downsample)

    # Define reflection geometry
    x_int = 1
    z_int = 0

    u_active_pixel_pointing_optic = Uxyz(([-1, 1, 0, 0, 0], [0, 0, -1, 1, 0], [-1, -1, -1, -1, -1]))
    v_screen_points_optic = Vxyz(([-2 * x_int, 2 * x_int, 0, 0, 0], [0, 0, -2 * x_int, 2 * x_int, 0], [1, 1, 1, 1, 1]))
    v_optic_cam_optic = Vxyz((0, 0, 1))
    u_measure_pixel_pointing_optic = Uxyz((0, 0, -1))
    v_align_point_optic = Vxyz((0, 0, 0))
    v_optic_screen_optic = Vxyz((0, 0, 1))

    # Set spatial data
    surface.set_spatial_data(
        u_active_pixel_pointing_optic,
        v_screen_points_optic,
        v_optic_cam_optic,
        u_measure_pixel_pointing_optic,
        v_align_point_optic,
        v_optic_screen_optic,
    )

    # Define expected data
    v_surf_int_pts_exp = Vxyz(([-x_int, x_int, 0, 0, 0], [0, 0, -x_int, x_int, 0], [z_int, z_int, z_int, z_int, 0]))
    slopes_exp = np.array(([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=float)
    slope_coefs_exp = np.array(([0, 0]), dtype=float)
    surf_coefs_exp = np.array([0, 0, 0], dtype=float)
    u_design_exp = Uxyz((0.0, 0.0, 1.0))
    u_fit_exp = Uxyz((0.0, 0.0, 1.0))

    # Calculate surface intersection with initial polynomial shape
    surface.calculate_surface_intersect_points()

    # Calculate slopes
    surface.calculate_slopes()

    # Calculate surface fit coefficients
    surface.fit_slopes()

    # Pack data
    return (surface, v_surf_int_pts_exp, slopes_exp, slope_coefs_exp, surf_coefs_exp, u_design_exp, u_fit_exp)


if __name__ == "__main__":
    unittest.main()
