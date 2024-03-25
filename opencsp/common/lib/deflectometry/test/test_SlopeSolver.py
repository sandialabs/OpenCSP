"""Unit test suite to test SlopeSolver class
"""

import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestSlopeSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        base_dir = os.path.join(opencsp_code_dir(), 'test/data/measurements_sofast_fringe')

        # Define test data files for single facet processing
        cls.data_file_facet = os.path.join(base_dir, 'calculations_facet/data.h5')
        data_file_measurement = os.path.join(base_dir, 'measurement_facet.h5')

        # Create spatial orientation objects
        datasets = [
            'DataSofastCalculation/geometry/general/r_optic_cam_refine_1',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_2',
            'DataSofastCalculation/geometry/facet_000/u_pixel_pointing_facet',
            'DataSofastCalculation/geometry/facet_000/u_cam_measure_point_facet',
            'DataSofastCalculation/geometry/facet_000/v_screen_points_facet',
            'DataSofastInput/surface_params/facet_000/initial_focal_lengths_xy',
            'DataSofastInput/surface_params/facet_000/downsample',
            'DataSofastInput/surface_params/facet_000/robust_least_squares',
            'DataSofastInput/surface_params/facet_000/surface_type',
        ]
        # Load data
        data = load_hdf5_datasets(datasets, cls.data_file_facet)
        display = Display.load_from_hdf(cls.data_file_facet)
        measurement = Measurement.load_from_hdf(data_file_measurement)

        # Create spatial orientation object
        r_cam_optic = Rotation.from_rotvec(data['r_optic_cam_refine_1']).inv()
        v_cam_optic_cam = Vxyz(data['v_cam_optic_cam_refine_2'])
        ori = SpatialOrientation(display.r_cam_screen, display.v_cam_screen_cam)
        ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        # Perform calculations
        surface = Surface2DParabolic(
            initial_focal_lengths_xy=data['initial_focal_lengths_xy'],
            robust_least_squares=bool(data['robust_least_squares']),
            downsample=data['downsample'],
        )
        kwargs = {
            'v_optic_cam_optic': ori.v_optic_cam_optic,
            'u_active_pixel_pointing_optic': Uxyz(data['u_pixel_pointing_facet']),
            'u_measure_pixel_pointing_optic': Uxyz(data['u_cam_measure_point_facet']),
            'v_screen_points_facet': Vxyz(data['v_screen_points_facet']),
            'v_optic_screen_optic': ori.v_optic_screen_optic,
            'v_align_point_optic': measurement.measure_point,
            'dist_optic_screen': measurement.optic_screen_dist,
            'surface': surface,
        }

        # Solve slopes
        ss = SlopeSolver(**kwargs)
        ss.fit_surface()
        ss.solve_slopes()
        cls.data_slope = ss.get_data()

    def test_transform_alignment(self):
        data = load_hdf5_datasets(['DataSofastCalculation/facet/facet_000/trans_alignment'], self.data_file_facet)

        np.testing.assert_allclose(data['trans_alignment'], self.data_slope.trans_alignment.matrix, atol=1e-8, rtol=0)

    def test_int_pts(self):
        data = load_hdf5_datasets(['DataSofastCalculation/facet/facet_000/v_surf_points_facet'], self.data_file_facet)
        np.testing.assert_allclose(
            data['v_surf_points_facet'], self.data_slope.v_surf_points_facet.data, atol=1e-8, rtol=0
        )

    def test_slopes(self):
        data = load_hdf5_datasets(['DataSofastCalculation/facet/facet_000/slopes_facet_xy'], self.data_file_facet)
        np.testing.assert_allclose(data['slopes_facet_xy'], self.data_slope.slopes_facet_xy, atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
