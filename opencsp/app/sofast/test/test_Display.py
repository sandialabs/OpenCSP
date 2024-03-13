"""Unit test suite to test Display class
"""
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.Display import Display
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestDisplay:
    @classmethod
    def setup_class(cls):
        # Define screen X and Y extent
        LX = 5.0  # meters
        LY = 5.0  # meters
        LZ = 3.0  # meters

        # Define test points
        cls.test_Vxy_pts = Vxy(
            ([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
        )

        # Define rectangular input data
        cls.grid_data_rect2D = {
            'screen_x': LX,
            'screen_y': LY,
            'screen_model': 'rectangular2D',
        }

        # Define 2D input data
        cls.grid_data_2D = {
            'Pxy_screen_fraction': Vxy(([0, 1, 0, 1], [0, 0, 1, 1])),
            'Pxy_screen_coords': Vxy(
                ([-LX / 2, LX / 2, -LX / 2, LX / 2], [-LY / 2, -LY / 2, LY / 2, LY / 2])
            ),
            'screen_model': 'distorted2D',
        }

        # Define 3D input data
        cls.grid_data_3D = {
            'Pxy_screen_fraction': Vxy(([0, 1, 0, 1], [0, 0, 1, 1])),
            'Pxyz_screen_coords': Vxyz(
                (
                    [-LX / 2, LX / 2, -LX / 2, LX / 2],
                    [-LY / 2, -LY / 2, LY / 2, LY / 2],
                    [LZ, LZ, 0, 0],
                )
            ),
            'screen_model': 'distorted3D',
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

    def test_rectangular2D(self):
        # Instantiate display object
        V_cam_screen_screen = Vxyz((0, 0, 1))
        R_screen_cam = Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))
        name = 'Test Display'
        disp = Display(V_cam_screen_screen, R_screen_cam, self.grid_data_rect2D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(
            calc.data, self.exp_Vxy_disp_pts.data, rtol=0, atol=0
        )

    def test_distorted2D(self):
        # Instantiate display object
        V_cam_screen_screen = Vxyz((0, 0, 1))
        R_screen_cam = Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))
        name = 'Test Display'
        disp = Display(V_cam_screen_screen, R_screen_cam, self.grid_data_2D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(
            calc.data, self.exp_Vxy_disp_pts.data, rtol=0, atol=1e-7
        )

    def test_distorted3D(self):
        # Instantiate display object
        V_cam_screen_screen = Vxyz((0, 0, 1))
        R_screen_cam = Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))
        name = 'Test Display'
        disp = Display(V_cam_screen_screen, R_screen_cam, self.grid_data_3D, name)

        # Perform calculation
        calc = disp.interp_func(self.test_Vxy_pts)

        # Test
        np.testing.assert_allclose(
            calc.data, self.exp_Vxyz_disp_pts.data, rtol=0, atol=1e-7
        )
