"""Unit test suite to test SlopeSolver class"""

from os.path import join
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.file_tools as ft


class TestSlopeSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        base_dir = join(opencsp_code_dir(), "test/data/sofast_fringe")
        # Save location
        cls.dir_save = join(opencsp_code_dir(), "common/lib/deflectometry/test/data/output/SlopeSolver")
        ft.create_directories_if_necessary(cls.dir_save)

        # Define test data files for single facet processing
        cls.data_file_facet = join(base_dir, "data_expected_facet/data.h5")
        data_file_measurement = join(base_dir, "data_measurement/measurement_facet.h5")

        # Create spatial orientation objects
        datasets = [
            "DataSofastCalculation/general/CalculationDataGeometryGeneral/r_optic_cam_refine_1",
            "DataSofastCalculation/general/CalculationDataGeometryGeneral/v_cam_optic_cam_refine_2",
            "DataSofastCalculation/facet/facet_000/CalculationDataGeometryFacet/u_pixel_pointing_facet",
            "DataSofastCalculation/facet/facet_000/CalculationDataGeometryFacet/u_cam_measure_point_facet",
            "DataSofastCalculation/facet/facet_000/CalculationDataGeometryFacet/v_screen_points_facet",
        ]
        # Load data
        data = h5.load_hdf5_datasets(datasets, cls.data_file_facet)
        measurement = MeasurementSofastFringe.load_from_hdf(data_file_measurement)
        ori = SpatialOrientation.load_from_hdf(cls.data_file_facet)

        # Create spatial orientation object
        r_cam_optic = Rotation.from_rotvec(data["r_optic_cam_refine_1"]).inv()
        v_cam_optic_cam = Vxyz(data["v_cam_optic_cam_refine_2"])
        ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        # Perform calculations
        surface = Surface2DParabolic.load_from_hdf(cls.data_file_facet, "DataSofastInput/optic_definition/facet_000/")
        kwargs = {
            "v_optic_cam_optic": ori.v_optic_cam_optic,
            "u_active_pixel_pointing_optic": Uxyz(data["u_pixel_pointing_facet"]),
            "u_measure_pixel_pointing_optic": Uxyz(data["u_cam_measure_point_facet"]),
            "v_screen_points_facet": Vxyz(data["v_screen_points_facet"]),
            "v_optic_screen_optic": ori.v_optic_screen_optic,
            "v_align_point_optic": measurement.v_measure_point_facet,
            "dist_optic_screen": measurement.dist_optic_screen,
            "surface": surface,
        }

        # Solve slopes
        ss = SlopeSolver(**kwargs)
        ss.fit_surface()
        ss.solve_slopes()
        cls.data_slope = ss.get_data()

    def test_transform_alignment(self):
        data = h5.load_hdf5_datasets(
            ["DataSofastCalculation/facet/facet_000/SlopeSolverData/trans_alignment"], self.data_file_facet
        )

        np.testing.assert_allclose(data["trans_alignment"], self.data_slope.trans_alignment.matrix, atol=1e-8, rtol=0)

    def test_int_pts(self):
        data = h5.load_hdf5_datasets(
            ["DataSofastCalculation/facet/facet_000/SlopeSolverData/v_surf_points_facet"], self.data_file_facet
        )
        np.testing.assert_allclose(
            data["v_surf_points_facet"], self.data_slope.v_surf_points_facet.data, atol=1e-8, rtol=0
        )

    def test_slopes(self):
        data = h5.load_hdf5_datasets(
            ["DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy"], self.data_file_facet
        )
        np.testing.assert_allclose(data["slopes_facet_xy"], self.data_slope.slopes_facet_xy, atol=1e-8, rtol=0)

    def test_save_hdf(self):
        self.data_slope.save_to_hdf(join(self.dir_save, "slope_solver_data.h5"))


if __name__ == "__main__":
    unittest.main()
