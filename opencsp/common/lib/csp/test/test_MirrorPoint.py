"""Unit test to test the MirrorPoint class"""

import numpy as np

from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestMirrorPoint:
    """Test class for testing MirrorPoint class"""

    def get_region_test_mirror(self) -> RegionXY:
        """Returns test mirror region"""
        return RegionXY.from_vertices(Vxy(([0.5, -0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5])))

    def get_test_mirror_flat(self, height: float, interpolation_type: str) -> MirrorPoint:
        """Returns a test instance of a MirrorPoint object"""
        # Calculate surface xyz points
        xv = yv = np.arange(-0.5, 0.5, 0.1)
        X, Y = np.meshgrid(xv, yv)
        Z = np.ones(X.shape) * height
        surface_points = Pxyz((X, Y, Z))
        # Calculate normal vectors
        nvecs = np.zeros((3, len(surface_points)))
        nvecs[2] = 1
        normal_vectors = Uxyz(nvecs)
        # Get mirror shape
        shape = self.get_region_test_mirror()
        # Create mirror object
        mirror = MirrorPoint(surface_points, normal_vectors, shape, interpolation_type)
        return mirror

    def test_mirror_z_flat_bilinear(self):
        """Tests z height output for flat mirror"""
        # Define mirror
        height = 0.1
        mirror = self.get_test_mirror_flat(height, "bilinear")
        # Define sample points
        xv = yv = np.arange(-0.45, 0.45, 0.1)
        X, Y = np.meshgrid(xv, yv)
        pts_samp = Vxy((X, Y))
        z_exp = np.ones(X.size) * height
        # Sample mirror
        z_calc = mirror.surface_displacement_at(pts_samp)
        # Test
        np.testing.assert_array_almost_equal(z_exp, z_calc)

    def test_mirror_normal_flat_bilinear(self):
        """Tests normal vector output for flat mirror"""
        # Define mirror
        height = 0.25
        mirror = self.get_test_mirror_flat(height, "bilinear")
        # Define sample points
        xv = yv = np.arange(-0.45, 0.45, 0.1)
        X, Y = np.meshgrid(xv, yv)
        pts_samp = Vxy((X, Y))
        # Expected normals
        norms_exp = np.zeros((3, len(pts_samp)))
        norms_exp[2] = 1
        norms_exp = Vxyz(norms_exp)
        # Sample mirror
        norms_calc = mirror.surface_norm_at(pts_samp)
        # Test
        np.testing.assert_array_almost_equal(norms_exp.data, norms_calc.data)


if __name__ == "__main__":
    Test = TestMirrorPoint()
    Test.test_mirror_z_array_flat_biinear()
    Test.test_mirror_z_flat_bilinear()
    Test.test_mirror_normal_array_flat_bilinear()
    Test.test_mirror_normal_flat_bilinear()
