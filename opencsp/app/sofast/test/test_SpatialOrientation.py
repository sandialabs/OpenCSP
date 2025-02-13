"""Unit test suite to test the SpatialOrientation class"""

from os.path import join
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
import opencsp.common.lib.tool.file_tools as ft


class TestSpatialOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        base_dir = join(opencsp_code_dir(), "test/data/sofast_fringe")

        # Define test data files for single facet processing
        data_file_facet = join(base_dir, "data_expected_facet/data.h5")

        # Load data
        datasets = [
            "DataSofastCalculation/general/CalculationDataGeometryGeneral/r_optic_cam_refine_1",
            "DataSofastCalculation/general/CalculationDataGeometryGeneral/v_cam_optic_cam_refine_2",
        ]

        ori = SpatialOrientation.load_from_hdf(data_file_facet)

        data = load_hdf5_datasets(datasets, data_file_facet)
        r_cam_optic = Rotation.from_rotvec(data["r_optic_cam_refine_1"]).inv()
        v_cam_optic_cam = Vxyz(data["v_cam_optic_cam_refine_2"])

        ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        # Save spatial orientation
        cls.so = ori

        # Set up save path
        cls.save_dir = join(opencsp_code_dir(), "app/sofast/test/data/output/spatial_orientation")
        ft.create_directories_if_necessary(cls.save_dir)

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

    def test_io_oriented_optic(self):
        file = join(self.save_dir, "test_spatial_orientation_oriented_optic.h5")
        # Save
        self.so.save_to_hdf(file)
        # Load
        ori = SpatialOrientation.load_from_hdf(file)
        # Check optic is oriented
        self.assertEqual(ori.optic_oriented, True)

    def test_io_unoriented_optic(self):
        file = join(self.save_dir, "test_spatial_orientation_unoriented_optic.h5")
        # Save
        r_cam_screen = self.so.r_cam_screen
        v_cam_screen_cam = self.so.v_cam_screen_cam
        ori_1 = SpatialOrientation(r_cam_screen, v_cam_screen_cam)
        ori_1.save_to_hdf(file)
        # Load
        ori_2 = SpatialOrientation.load_from_hdf(file)
        # Check optic not oriented
        self.assertEqual(ori_1.optic_oriented, False)
        self.assertEqual(ori_2.optic_oriented, False)


if __name__ == "__main__":
    unittest.main()
