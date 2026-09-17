import configparser
from os.path import join, dirname
import os
import unittest

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.testing.compare as mplt
import numpy as np
import pandas as pd

from opencsp.common.lib.csp.StandardPlotOutput import (
    StandardPlotOutput,
    _OptionsSave,
    _OptionsSliceOutput,
    coords_to_indices,
)
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
        import opencsp.app.sofast.lib.load_sofast_hdf_data as lsd
        from opencsp.common.lib.csp.LightSourceSun import LightSourceSun

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
            "Meas_Slope_XY.png",
            "Meas_Slope_X.png",
            "Meas_Slope_Y.png",
            "Meas_Curvature_XY.png",
            "Meas_Curvature_X.png",
            "Meas_Curvature_Y.png",
            "Ref_Slope_XY.png",
            "Ref_Slope_X.png",
            "Ref_Slope_Y.png",
            "Ref_Curvature_XY.png",
            "Ref_Curvature_X.png",
            "Ref_Curvature_Y.png",
            "Slope_Deviation_XY.png",
            "Slope_Deviation_X.png",
            "Slope_Deviation_Y.png",
            "Meas_Ray_Trace_Image.png",
            "Ref_Ray_Trace_Image.png",
            "Ensquared_Energy.png",
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


class TestOptionsSave(unittest.TestCase):
    """Tests for the _OptionsSave dataclass defaults and mutation."""

    def test_default_values(self):
        """All flags should default to True."""
        opts = _OptionsSave()
        self.assertTrue(opts.curvature_deviation)
        self.assertTrue(opts.slope_deviation)
        self.assertTrue(opts.enclosed_energy)
        self.assertTrue(opts.curvature)
        self.assertTrue(opts.slope)

    def test_field_mutation(self):
        """Each field can be independently set to False."""
        opts = _OptionsSave()
        opts.curvature_deviation = False
        opts.slope_deviation = False
        opts.enclosed_energy = False
        opts.curvature = False
        opts.slope = False
        self.assertFalse(opts.curvature_deviation)
        self.assertFalse(opts.slope_deviation)
        self.assertFalse(opts.enclosed_energy)
        self.assertFalse(opts.curvature)
        self.assertFalse(opts.slope)

    def test_instances_are_independent(self):
        """Two separate instances should not share state."""
        a = _OptionsSave()
        b = _OptionsSave()
        a.slope = False
        self.assertTrue(b.slope)

    def test_standard_plot_output_exposes_options_save(self):
        """StandardPlotOutput should expose options_save as _OptionsSave."""
        spo = StandardPlotOutput()
        self.assertIsInstance(spo.options_save, _OptionsSave)


class TestOptionsSliceOutput(unittest.TestCase):
    """Tests for the _OptionsSliceOutput dataclass defaults and mutation."""

    def test_default_values(self):
        """Check all defaults match the documented values."""
        opts = _OptionsSliceOutput()
        self.assertTrue(opts.to_plot)
        self.assertEqual(opts.slice_x_loc, [0])
        self.assertEqual(opts.slice_y_loc, [0])
        self.assertEqual(opts.plot_x_slope_offset, 0)
        self.assertEqual(opts.plot_y_slope_offset, 0)
        self.assertEqual(opts.plot_x_slope_deviation_offset, 0)
        self.assertEqual(opts.plot_y_slope_deviation_offset, 0)
        self.assertEqual(opts.plot_x_curvature_offset, 0)
        self.assertEqual(opts.plot_y_curvature_offset, 0)
        self.assertEqual(opts.plot_x_curvature_deviation_offset, 0)
        self.assertEqual(opts.plot_y_curvature_deviation_offset, 0)
        self.assertFalse(opts.uncertainty_to_plot)
        self.assertIsNone(opts.xlim)
        self.assertIsNone(opts.ylim)

    def test_slice_loc_lists_are_independent(self):
        """Each instance should get its own default list (not a shared reference)."""
        a = _OptionsSliceOutput()
        b = _OptionsSliceOutput()
        a.slice_x_loc.append(1.0)
        self.assertEqual(b.slice_x_loc, [0], "slice_x_loc lists should not be shared between instances")

    def test_xlim_ylim_assignment(self):
        """xlim and ylim should accept two-element tuples."""
        opts = _OptionsSliceOutput()
        opts.xlim = (-0.5, 0.5)
        opts.ylim = (-1.0, 1.0)
        self.assertEqual(opts.xlim, (-0.5, 0.5))
        self.assertEqual(opts.ylim, (-1.0, 1.0))

    def test_offset_fields(self):
        """Offset fields should accept float values."""
        opts = _OptionsSliceOutput()
        opts.plot_x_slope_offset = 2.5
        opts.plot_y_curvature_deviation_offset = -1.0
        self.assertEqual(opts.plot_x_slope_offset, 2.5)
        self.assertEqual(opts.plot_y_curvature_deviation_offset, -1.0)

    def test_standard_plot_output_exposes_options_slice_output(self):
        """StandardPlotOutput should expose options_slice_output as _OptionsSliceOutput."""
        spo = StandardPlotOutput()
        self.assertIsInstance(spo.options_slice_output, _OptionsSliceOutput)


class TestCoordsToIndices(unittest.TestCase):
    """Tests for the module-level coords_to_indices helper."""

    def setUp(self):
        self.vec = np.linspace(-1.0, 1.0, 21)  # [-1.0, -0.9, ..., 0.9, 1.0]

    def test_exact_match(self):
        """A coordinate that exactly matches a vector element returns that index."""
        indices = coords_to_indices([0.0], self.vec)
        self.assertEqual(len(indices), 1)
        self.assertAlmostEqual(self.vec[indices[0]], 0.0, places=10)

    def test_nearest_match(self):
        """A coordinate between two elements returns the index of the nearest one."""
        indices = coords_to_indices([0.05], self.vec)
        # vec has 0.0 at index 10 and 0.1 at index 11; 0.05 is equidistant but argmin picks first
        self.assertIn(indices[0], [10, 11])

    def test_multiple_coords(self):
        """Multiple coordinates produce a list of the same length."""
        coords = [-1.0, 0.0, 1.0]
        indices = coords_to_indices(coords, self.vec)
        self.assertEqual(len(indices), 3)

    def test_scalar_input(self):
        """A scalar (non-list) coordinate is accepted and returns a single-element list."""
        indices = coords_to_indices(0.0, self.vec)
        self.assertEqual(len(indices), 1)

    def test_clipped_to_max_index(self):
        """Coordinates beyond the vector range are clipped to max_index (len-2)."""
        indices = coords_to_indices([999.0], self.vec)
        self.assertEqual(indices[0], len(self.vec) - 2)

    def test_negative_clipped_to_zero(self):
        """Coordinates far below the vector minimum map to index 0 (nearest)."""
        indices = coords_to_indices([-999.0], self.vec)
        self.assertEqual(indices[0], 0)


class TestSetPlotControlFromSettings(unittest.TestCase):
    """Tests that set_plot_control_from_settings correctly reads _OptionsSave
    and _OptionsSliceOutput fields from a ConfigParser object."""

    def _make_settings(self, pairs: dict) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["Default"] = pairs
        return cfg

    # --- _OptionsSave fields ---

    def test_options_save_all_false(self):
        """All _OptionsSave flags should be set to False when provided."""
        settings = self._make_settings(
            {
                "plots.options_save.curvature_deviation": "False",
                "plots.options_save.slope_deviation": "False",
                "plots.options_save.enclosed_energy": "False",
                "plots.options_save.curvature": "False",
                "plots.options_save.slope": "False",
            }
        )
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertFalse(spo.options_save.curvature_deviation)
        self.assertFalse(spo.options_save.slope_deviation)
        self.assertFalse(spo.options_save.enclosed_energy)
        self.assertFalse(spo.options_save.curvature)
        self.assertFalse(spo.options_save.slope)

    def test_options_save_partial_override(self):
        """Only the provided keys should change; others keep their defaults."""
        settings = self._make_settings({"plots.options_save.slope": "False"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertFalse(spo.options_save.slope)
        # Other flags remain True (their defaults)
        self.assertTrue(spo.options_save.curvature_deviation)
        self.assertTrue(spo.options_save.enclosed_energy)

    def test_options_save_empty_settings(self):
        """Empty settings should leave all _OptionsSave flags at their defaults (True)."""
        settings = configparser.ConfigParser()
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertTrue(spo.options_save.curvature_deviation)
        self.assertTrue(spo.options_save.slope_deviation)
        self.assertTrue(spo.options_save.enclosed_energy)
        self.assertTrue(spo.options_save.curvature)
        self.assertTrue(spo.options_save.slope)

    # --- _OptionsSliceOutput fields ---

    def test_options_slice_output_to_plot_false(self):
        """to_plot should be set to False when provided."""
        settings = self._make_settings({"plots.options_slice_output.to_plot": "False"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertFalse(spo.options_slice_output.to_plot)

    def test_options_slice_output_slice_x_loc(self):
        """slice_x_loc should be parsed from a comma-separated string."""
        settings = self._make_settings({"plots.options_slice_output.slice_x_loc": "-0.5, 0.0, 0.5"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertEqual(spo.options_slice_output.slice_x_loc, [-0.5, 0.0, 0.5])

    def test_options_slice_output_slice_y_loc(self):
        """slice_y_loc should be parsed from a comma-separated string."""
        settings = self._make_settings({"plots.options_slice_output.slice_y_loc": "0.1, 0.2"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertEqual(spo.options_slice_output.slice_y_loc, [0.1, 0.2])

    def test_options_slice_output_offsets(self):
        """Float offset fields should be read correctly."""
        settings = self._make_settings(
            {
                "plots.options_slice_output.plot_x_slope_offset": "1.5",
                "plots.options_slice_output.plot_y_slope_offset": "2.5",
                "plots.options_slice_output.plot_x_curvature_offset": "0.5",
                "plots.options_slice_output.plot_y_curvature_offset": "-0.5",
                "plots.options_slice_output.plot_x_slope_deviation_offset": "3.0",
                "plots.options_slice_output.plot_y_slope_deviation_offset": "-3.0",
                "plots.options_slice_output.plot_x_curvature_deviation_offset": "0.25",
                "plots.options_slice_output.plot_y_curvature_deviation_offset": "-0.25",
            }
        )
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertAlmostEqual(spo.options_slice_output.plot_x_slope_offset, 1.5)
        self.assertAlmostEqual(spo.options_slice_output.plot_y_slope_offset, 2.5)
        self.assertAlmostEqual(spo.options_slice_output.plot_x_curvature_offset, 0.5)
        self.assertAlmostEqual(spo.options_slice_output.plot_y_curvature_offset, -0.5)
        self.assertAlmostEqual(spo.options_slice_output.plot_x_slope_deviation_offset, 3.0)
        self.assertAlmostEqual(spo.options_slice_output.plot_y_slope_deviation_offset, -3.0)
        self.assertAlmostEqual(spo.options_slice_output.plot_x_curvature_deviation_offset, 0.25)
        self.assertAlmostEqual(spo.options_slice_output.plot_y_curvature_deviation_offset, -0.25)

    def test_options_slice_output_xlim(self):
        """xlim should be parsed into a two-element tuple of floats."""
        settings = self._make_settings({"plots.options_slice_output.xlim": "-0.5, 0.5"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertEqual(spo.options_slice_output.xlim, (-0.5, 0.5))

    def test_options_slice_output_ylim(self):
        """ylim should be parsed into a two-element tuple of floats."""
        settings = self._make_settings({"plots.options_slice_output.ylim": "-1.0, 1.0"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertEqual(spo.options_slice_output.ylim, (-1.0, 1.0))

    def test_options_slice_output_uncertainty_to_plot(self):
        """uncertainty_to_plot should be set to True when provided."""
        settings = self._make_settings({"plots.options_slice_output.uncertainty_to_plot": "True"})
        spo = StandardPlotOutput()
        spo.set_plot_control_from_settings(settings)
        self.assertTrue(spo.options_slice_output.uncertainty_to_plot)


class TestPlotCombinedMeasuredReferenceSlopes(unittest.TestCase):
    """Tests related to plot_combined_measured_reference_slopes.

    This method is currently commented out in StandardPlotOutput (intentionally
    disabled via triple-quote block). Tests verify that the method is not
    callable on the class in its current state, and that the surrounding
    public interface (plot/options) behaves correctly without it.
    """

    def test_method_not_present_on_class(self):
        """plot_combined_measured_reference_slopes is intentionally commented out
        and should not be an attribute of StandardPlotOutput."""
        spo = StandardPlotOutput()
        self.assertFalse(
            hasattr(spo, 'plot_combined_measured_reference_slopes'),
            "Method is intentionally disabled; it should not exist on the class.",
        )

    def test_plot_does_not_call_combined_when_slope_vis_off(self):
        """With slope vis disabled, plot() should complete without error even
        though plot_combined_measured_reference_slopes is absent."""
        spo = StandardPlotOutput()
        spo.options_slope_vis.to_plot = False
        spo.options_curvature_vis.to_plot = False
        spo.options_slope_deviation_vis.to_plot = False
        spo.options_ray_trace_vis.to_plot = False
        # Should not raise regardless of combined method being absent
        spo.plot()

    def test_options_slice_output_to_plot_false_skips_combined(self):
        """When options_slice_output.to_plot is False, the combined plot path
        is not reached, so the absent method causes no error."""
        spo = StandardPlotOutput()
        spo.options_slope_vis.to_plot = False
        spo.options_curvature_vis.to_plot = False
        spo.options_slope_deviation_vis.to_plot = False
        spo.options_ray_trace_vis.to_plot = False
        spo.options_slice_output.to_plot = False
        spo.plot()  # should not raise


if __name__ == "__main__":
    unittest.main()
