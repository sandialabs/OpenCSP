from os.path import join, dirname
import unittest

import matplotlib.pyplot as plt
import matplotlib.testing.compare as mplt

import opencsp.app.sofast.lib.load_sofast_hdf_data as lsd
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestStandardPlotOutput(unittest.TestCase):
    """Tests creating a standard plot suite for a single facet.

    NOTE: To update the unit test data, run the test and copy the .PNG files from
    the data/output folder to the data/input folder. Run the test again to confirm passing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Create output directory
        cls.dir_output = join(dirname(__file__), "data/output/StandardPlotOutput")
        ft.create_directories_if_necessary(cls.dir_output)

        # Create and clear output facet directory
        cls.dir_output_facet = join(cls.dir_output, "facet")
        ft.create_directories_if_necessary(cls.dir_output_facet)
        ft.delete_files_in_directory(cls.dir_output_facet, "*.*")

        # Define input directory
        cls.dir_input = join(dirname(__file__), "data/input/StandardPlotOutput")

        # Define input facet directory
        cls.dir_input_facet = join(cls.dir_input, "facet")
        ft.create_directories_if_necessary(cls.dir_input_facet)

        lt.logger(join(cls.dir_output, "log.txt"), level=lt.log.WARN)

    def test_facet(self):
        """Generates figures for single facet"""
        # General setup
        dir_in = self.dir_input_facet
        dir_out = self.dir_output_facet

        # Define data file
        file_data = join(opencsp_code_dir(), "test/data/sofast_fringe/data_expected_facet/data.h5")

        # Load Sofast measurement data
        optic_meas = lsd.load_mirror(file_data)
        optic_ref = lsd.load_mirror_ideal(file_data, 100.0)

        # Define viewing/illumination geometry
        v_target_center = Vxyz((0, 0, 100))
        v_target_normal = Vxyz((0, 0, -1))
        source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)

        # Create standard output plots
        output = StandardPlotOutput()

        # Update slope visualization parameters
        output.options_slope_vis.resolution = 0.05
        output.options_slope_vis.clim = 7
        output.options_slope_vis.quiver_color = ["white", "green", "blue"]
        output.options_slope_vis.quiver_density = 0.2
        output.options_slope_vis.quiver_scale = 55

        # Update slope deviation visualization parameters
        output.options_slope_deviation_vis.resolution = 0.05
        output.options_slope_deviation_vis.clim = 1.5
        output.options_slope_vis.quiver_color = ["magenta", "blue", "red"]
        output.options_slope_vis.quiver_density = [0.3, 0.2, 0.1]
        output.options_slope_vis.quiver_scale = [10, 10, 10]

        # Update curvature visualization parameters
        output.options_curvature_vis.resolution = 0.05
        output.options_curvature_vis.processing = ["smooth"]
        output.options_curvature_vis.smooth_kernel_width = 5
        output.options_curvature_vis.clim = 30

        # Update file options
        output.options_file_output.to_save = True
        output.options_file_output.output_dir = dir_out
        output.options_file_output.number_in_name = False
        output.options_file_output.close_after_save = True

        # Update raytrace options
        output.options_ray_trace_vis.enclosed_energy_max_semi_width = 1
        output.options_ray_trace_vis.ray_trace_optic_res = 0.2

        # Define ray trace parameters
        output.params_ray_trace.source = source
        output.params_ray_trace.v_target_center = v_target_center
        output.params_ray_trace.v_target_normal = v_target_normal

        # Test no plots are made with no optics loaded but with plotting turned on
        output.plot()
        files = ft.files_in_directory_by_extension(self.dir_output_facet, [".png"])
        if len(files[".png"]) != 0:
            raise AssertionError(f'There should be no files, but the following exist: {files[".png"]}')

        # Test no plots are made when all plotting is turned off but optics loaded
        output.optic_measured = optic_meas
        output.optic_reference = optic_ref

        output.options_curvature_vis.to_plot = False
        output.options_ray_trace_vis.to_plot = False
        output.options_slope_vis.to_plot = False
        output.options_slope_deviation_vis.to_plot = False

        output.plot()
        files = ft.files_in_directory_by_extension(self.dir_output_facet, [".png"])
        if len(files[".png"]) != 0:
            raise AssertionError(f'There should be no files, but the following exist: {files[".png"]}')

        # Create standard output plots with plotting turned on and optics loaded
        output.options_curvature_vis.to_plot = True
        output.options_ray_trace_vis.to_plot = True
        output.options_slope_vis.to_plot = True
        output.options_slope_deviation_vis.to_plot = True
        output.plot()

        # Test all plots match
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
        for idx, file in enumerate(files):
            with self.subTest(i=idx):
                file_in = join(dir_in, file)
                file_out = join(dir_out, file)
                self._compare_actual_expected_images(file_out, file_in)

    def _compare_actual_expected_images(self, actual_location: str, expected_location: str, tolerance=0.2) -> bool:
        # Tests if image files match
        res = mplt.compare_images(expected_location, actual_location, tolerance)
        if res is not None:
            raise AssertionError(res)

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
