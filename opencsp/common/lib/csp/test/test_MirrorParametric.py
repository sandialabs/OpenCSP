"""Unit test to test MirrorParametric class"""

import numpy as np

from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestMirrorParametric:
    """Test class for MirrorParametric"""

    def get_region_test_mirror(self) -> RegionXY:
        """Returns a test mirror region"""
        return RegionXY.from_vertices(Vxy(([0.5, -0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5])))

    def get_test_flat_mirror(self, height: float) -> MirrorParametric:
        """Returns a flat mirror with defined height"""

        def surf_func_flat(x, y):
            return height

        region = self.get_region_test_mirror()
        return MirrorParametric(surf_func_flat, region)

    def test_mirror_z_flat(self):
        """Tests z height of flat mirror"""
        # Define mirror
        height = 0.5
        mirror = self.get_test_flat_mirror(height)
        # Calculate height
        x_vec = y_vec = np.arange(-0.5, 0.5, 0.1)
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)
        v_pts = Vxy((x_mat, y_mat))
        height_array = mirror.surface_displacement_at(v_pts)
        # Test height
        np.testing.assert_array_equal(height_array, np.ones(len(v_pts)) * height)

    def test_mirror_normals_flat(self):
        """Tsets mirror normal vectors of flat mirror"""
        # Define mirror
        height = 0.5
        mirror = self.get_test_flat_mirror(height)
        # Calculate height
        x_vec = y_vec = np.arange(-0.5, 0.5, 0.1)
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)
        v_pts = Vxy((x_mat, y_mat))
        v_normals = mirror.surface_norm_at(v_pts)
        # Test height
        v_normals_exp = np.zeros((3, len(v_pts)))
        v_normals_exp[2] = 1
        v_normals_exp = Vxyz(v_normals_exp)
        np.testing.assert_array_equal(v_normals.data, v_normals_exp.data)

    def test_surface_displacement_at_parabolic(self):
        """Tests z height of parabolic mirror"""
        focal_length = 100.0
        region = self.get_region_test_mirror()
        mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, region)
        x = np.arange(-0.5, 0.5, 0.1)
        y = np.zeros(x.shape)
        z_exp = x**2 / (4 * focal_length)
        # Calculate z
        p_samp = Vxy((x, y))
        z_calc = mirror.surface_displacement_at(p_samp)
        # Test
        np.testing.assert_array_almost_equal(z_exp, z_calc)

    def test_surface_norm_at_parabolic(self):
        """Tests normal vectors of parabolic mirror"""
        focal_length = 100.0
        region = self.get_region_test_mirror()
        mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, region)
        x = np.arange(-0.5, 0.5, 0.1)
        y = np.arange(-0.5, 0.5, 0.1)
        dfdx = x / (2 * focal_length)
        dfdy = y / (2 * focal_length)
        u_exp = Uxyz((-dfdx, -dfdy, np.ones(x.shape)))
        # Calculate z
        p_samp = Vxy((x, y))
        u_calc = mirror.surface_norm_at(p_samp)
        # Test
        np.testing.assert_array_almost_equal(u_exp.data, u_calc.data)
