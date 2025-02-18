import os
import unittest

import numpy as np

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.PatternSofastFixed import PatternSofastFixed
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


class TestDotLocationsFixedPattern(unittest.TestCase):
    def test_DotLocationsFixedPattern(self):
        # Generate test data
        xv = np.array([-2, -1, 0, 1, 2])
        yv = np.array([-3, -2, -1, 0, 1, 2])
        x = np.array(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            dtype=float,
        )
        y = np.array(
            [
                [11, 11, 11, 11, 11],
                [12, 12, 12, 12, 12],
                [13, 13, 13, 13, 13],
                [14, 14, 14, 14, 14],
                [15, 15, 15, 15, 15],
                [16, 16, 16, 16, 16],
            ],
            dtype=float,
        )
        z = np.zeros((6, 5)) + 0.1
        xyz = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)

        # Instantiate object
        fp = DotLocationsFixedPattern(xv, yv, xyz)

        # Test
        np.testing.assert_equal(fp.nx, 5)
        np.testing.assert_equal(fp.ny, 6)
        np.testing.assert_equal(fp.x_min, -2)
        np.testing.assert_equal(fp.x_max, 2)
        np.testing.assert_equal(fp.y_min, -3)
        np.testing.assert_equal(fp.y_max, 2)
        np.testing.assert_equal(fp.x_offset, 2)
        np.testing.assert_equal(fp.y_offset, 3)

        # Calculate
        idxs = Vxy(([-2, -1, 0, 1, 2], [-3, -2, -1, 0, 2]), dtype=int)
        pts_xyz = fp.xy_indices_to_screen_coordinates(idxs)
        np.testing.assert_equal(pts_xyz.x, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_equal(pts_xyz.y, np.array([11, 12, 13, 14, 16]))
        np.testing.assert_equal(pts_xyz.z, np.array([0.1, 0.1, 0.1, 0.1, 0.1]))

    def test_from_Display(self):
        # Load display
        file_disp = os.path.join(opencsp_code_dir(), "test/data/sofast_common/display_distorted_3d.h5")
        display = Display.load_from_hdf(file_disp)
        fp_proj = PatternSofastFixed(30, 30, 5, 5)

        fp = DotLocationsFixedPattern.from_projection_and_display(fp_proj, display)

        # Test
        x_exp = np.array(
            [
                [-3.20926192, -0.29103971, 2.61672562],
                [-3.2057312, -0.2908884, 2.61473907],
                [-3.20352939, -0.29055366, 2.61233896],
            ]
        )
        y_exp = np.array(
            [
                [-1.28036655, -1.27780583, -1.27613219],
                [-0.11836353, -0.11801777, -0.11784412],
                [1.04289668, 1.04089699, 1.03854907],
            ]
        )
        z_exp = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        xyz_exp = np.concatenate((x_exp[..., None], y_exp[..., None], z_exp[..., None]), axis=2)
        np.testing.assert_allclose(fp.xyz_dot_loc, xyz_exp, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
