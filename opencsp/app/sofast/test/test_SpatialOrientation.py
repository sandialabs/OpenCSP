"""Unit test suite to test the SpatialOrientation class
"""
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestSpatialOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        base_dir = os.path.join(
            opencsp_code_dir(), 'test/data/sofast_measurements'
        )

        # Define test data files for single facet processing
        data_file_facet = os.path.join(base_dir, 'calculations_facet/data.h5')

        # Create spatial orientation objects
        datasets = [
            'DataSofastCalculation/geometry/general/r_optic_cam_refine_1',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_2',
        ]
        # Load data
        data = load_hdf5_datasets(datasets, data_file_facet)
        display = Display.load_from_hdf(data_file_facet)

        r_cam_optic = Rotation.from_rotvec(data['r_optic_cam_refine_1']).inv()
        v_cam_optic_cam = Vxyz(data['v_cam_optic_cam_refine_2'])
        ori = SpatialOrientation(display.r_cam_screen, display.v_cam_screen_cam)
        ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        # Save spatial orientation
        cls.so = ori

    def test_translation_ring_1(self):
        v_exp = np.zeros(3)
        v_calc = (
            self.so.v_cam_optic_cam
            + self.so.v_optic_screen_optic.rotate(self.so.r_optic_cam)
            + self.so.v_screen_cam_cam
        )
        np.testing.assert_allclose(v_exp, v_calc.data.squeeze(), rtol=0, atol=1e-10)

    def test_translation_ring_2(self):
        v_exp = np.zeros(3)
        v_calc = (
            self.so.v_cam_screen_cam
            + self.so.v_screen_optic_optic.rotate(self.so.r_optic_cam)
            + self.so.v_optic_cam_cam
        )
        np.testing.assert_allclose(v_exp, v_calc.data.squeeze(), rtol=0, atol=1e-10)

    def test_rotation_ring_1(self):
        I_exp = np.eye(3)
        I_calc = self.so.r_optic_screen * self.so.r_cam_optic * self.so.r_screen_cam
        np.testing.assert_allclose(I_exp, I_calc.as_matrix(), atol=1e-9, rtol=0)

    def test_rotation_ring_2(self):
        I_exp = np.eye(3)
        I_calc = self.so.r_cam_screen * self.so.r_optic_cam * self.so.r_screen_optic
        np.testing.assert_allclose(I_exp, I_calc.as_matrix(), atol=1e-9, rtol=0)
