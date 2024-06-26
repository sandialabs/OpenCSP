from os.path import join, dirname
import unittest

import matplotlib.pyplot as plt
import matplotlib.testing.compare as mplt

import opencsp.app.sofast.lib.load_saved_data as lsd
from opencsp.common.lib.geometry.Uxyz import Uxyz
import opencsp.common.lib.tool.file_tools as ft
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.log_tools as lt


class TestStandardPlotOutput(unittest.TestCase):
    """Tests creating a standard plot suite for a single facet.

    NOTE: To update the unit test data, run the test and copy the .PNG files from
    the data/output folder to the data/input folder. Run the test again to confirm passing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Set up directories
        cls.dir_output = join(dirname(__file__), 'data/output/StandardPlotOutput')
        ft.create_directories_if_necessary(cls.dir_output)
        cls.dir_input = join(dirname(__file__), 'data/input/StandardPlotOutput')
        ft.create_directories_if_necessary(cls.dir_input)

    def test_facet(self):
        """Generates figures for single facet"""
        # General setup
        dir_out = join(self.dir_output, 'facet')
        ft.create_directories_if_necessary(dir_out)
        dir_in = join(self.dir_input, 'facet')
        ft.create_directories_if_necessary(dir_in)

        lt.logger(join(dir_out, 'log.txt'), level=lt.log.INFO)

        # Define data file
        file_data = join(opencsp_code_dir(), 'test/data/sofast_fringe/data_expected_facet/data.h5')

        # Load Sofast measurement data
        optic_meas = lsd.load_mirror_from_hdf(file_data)
        optic_ref = lsd.load_ideal_mirror_from_hdf(file_data, 100.0)

        # Define viewing/illumination geometry
        v_target_center = Vxyz((0, 0, 100))
        v_target_normal = Vxyz((0, 0, -1))
        source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

        # Create standard output plots
        output = StandardPlotOutput()
        output.optic_measured = optic_meas
        output.optic_reference = optic_ref

        # Update slope visualization parameters
        output.options_slope_vis.clim = 7
        output.options_slope_vis.deviation_clim = 1.5

        # Update curvature visualization parameters
        output.options_curvature_vis.processing = ['smooth']
        output.options_curvature_vis.smooth_kernel_width = 5

        # Update file options
        output.options_file_output.to_save = True
        output.options_file_output.output_dir = dir_out
        output.options_file_output.number_in_name = False

        # Update raytrace options
        output.options_ray_trace_vis.ensquared_energy_max_semi_width = 1

        # Define ray trace parameters
        output.params_ray_trace.source = source
        output.params_ray_trace.v_target_center = v_target_center
        output.params_ray_trace.v_target_normal = v_target_normal

        # Create standard output plots
        output.plot()

        # Get list of created plots
        files = [
            "Slope_Magnitude_measured_xy.png",
            "Slope_X_measured_xy.png",
            "Slope_Y_measured_xy.png",
            "Curvature_Combined_measured_xy.png",
            "Curvature_X_measured_xy.png",
            "Curvature_Y_measured_xy.png",
            "Slope_Magnitude_reference_xy.png",
            "Slope_X_reference_xy.png",
            "Slope_Y_reference_xy.png",
            "Curvature_Combined_reference_xy.png",
            "Curvature_X_reference_xy.png",
            "Curvature_Y_reference_xy.png",
            "Slope_Deviation_Magnitude_xy.png",
            "Slope_Deviation_X_xy.png",
            "Slope_Deviation_Y_xy.png",
            "Ray_Trace_Image_measured_xy.png",
            "Ray_Trace_Image_reference_xy.png",
            "Ensquared_Energy_xy.png",
        ]
        for file in files:
            file_in = join(dir_in, file)
            file_out = join(dir_out, file)
            self._compare_actual_expected_images(file_out, file_in)

    def _compare_actual_expected_images(self, actual_location: str, expected_location: str, tolerance=0.2) -> bool:
        """Tests if image files match."""
        self.assertIsNone(mplt.compare_images(expected_location, actual_location, tolerance))

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
