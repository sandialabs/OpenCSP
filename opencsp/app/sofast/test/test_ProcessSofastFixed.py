"""Example script that runs fixed pattern deflectometry analysis on saved data
"""

from os.path import join
import unittest

import numpy as np

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.hdf5_tools as ht
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class test_ProcessSofastFixed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads data and runs ProcessSofastFixed"""
        dir_base = join(opencsp_code_dir(), 'test/data/measurements_sofast_fixed')

        # Definitions
        file_camera = join(dir_base, "camera.h5")
        file_facet = join(dir_base, "Facet_NSTTF.json")
        file_ori = join(dir_base, 'spatial_orientation.h5')
        file_dot_locs = join(dir_base, 'sofast_fixed_dot_locations.h5')
        file_meas = join(dir_base, 'sofast_fixed_measurement.h5')
        file_exp = join(dir_base, 'data_calculation_sofast_fixed.h5')

        cls.save_dir = join(opencsp_code_dir(), 'app/sofast/test/data/output')
        ft.create_directories_if_necessary(cls.save_dir)

        lt.logger(join(cls.save_dir, 'sofast_fixed_process_log.txt'), lt.log.ERROR)

        surface = Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1)

        # Load data
        camera = Camera.load_from_hdf(file_camera)
        facet_data = DefinitionFacet.load_from_json(file_facet)
        dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
        orientation = SpatialOrientation.load_from_hdf(file_ori)
        measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

        # Load expected data
        datasets = ['CalculationsFixedPattern/Facet_000/SlopeSolverData/slopes_facet_xy']
        data = ht.load_hdf5_datasets(datasets, file_exp)
        cls.exp_slopes_xy = data['slopes_facet_xy']

        # Instantiate class
        cls.process_sofast_fixed = ProcessSofastFixed(orientation, camera, dot_locs, facet_data)
        cls.process_sofast_fixed.load_measurement_data(measurement)

        # Process
        cls.process_sofast_fixed.process_single_facet_optic(surface)

    def test_save_as_hdf(self):
        """Tests saving to HDF file"""
        self.process_sofast_fixed.save_to_hdf(join(self.save_dir, 'data_calculation_sofast_fixed.h5'))

    def test_save_figure(self):
        """Tests generating and saving a figure"""
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()

        fig_mng = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror = self.process_sofast_fixed.get_mirror('bilinear')
        mirror.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_mng.axis)
        fig_mng.save(self.save_dir, 'sofast_fixed_orthorectified_slope_magnitude', 'png')

    def test_slopes_xy(self):
        """Tests slope data"""
        np.testing.assert_allclose(
            self.process_sofast_fixed.data_slope_solver.slopes_facet_xy, self.exp_slopes_xy, rtol=0, atol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
