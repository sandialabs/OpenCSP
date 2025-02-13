import json
from os.path import join, dirname
import unittest

import matplotlib.pyplot as plt
import matplotlib.testing.compare as mplt

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SofastConfiguration import SofastConfiguration
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestSofastConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup paths
        cls.dir_input = join(dirname(__file__), "data/input/SofastConfiguration")
        ft.create_directories_if_necessary(cls.dir_input)
        cls.dir_output = join(dirname(__file__), "data/output/SofastConfiguration")
        ft.create_directories_if_necessary(cls.dir_output)

        # Set up logger
        lt.logger(join(cls.dir_output, "log.txt"), lt.log.INFO)

        # Create sofast fringe instance
        cls.process_sofast_fringe = cls._get_process_sofast_fringe()

        # Create sofast fixed instance
        cls.process_sofast_fixed = cls._get_process_sofast_fixed()

    @staticmethod
    def _get_process_sofast_fixed():
        # Define sample data directory
        dir_data_sofast = join(opencsp_code_dir(), "test/data/sofast_fixed")
        dir_data_common = join(opencsp_code_dir(), "test/data/sofast_common")

        # Directory setup
        file_meas = join(dir_data_sofast, "data_measurement/measurement_facet.h5")
        file_camera = join(dir_data_common, "camera_sofast.h5")
        file_dot_locs = join(dir_data_sofast, "data_measurement/fixed_pattern_dot_locations.h5")
        file_ori = join(dir_data_common, "spatial_orientation.h5")
        file_facet = join(dir_data_common, "Facet_NSTTF.json")

        # Load saved single facet SofastFixed collection data
        camera = Camera.load_from_hdf(file_camera)
        facet_data = DefinitionFacet.load_from_json(file_facet)
        fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
        orientation = SpatialOrientation.load_from_hdf(file_ori)
        measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

        # Instantiate SofastFixed class and load measurement data
        sofast = ProcessSofastFixed(orientation, camera, fixed_pattern_dot_locs)
        sofast.load_measurement_data(measurement)

        # Process
        # - Since we are using an NSTTF mirror with about a 150m focal length,
        #     we will seed the initial focal lengths with 150 in x and y
        # - Since we are testing SofastFixed (which has sparse data compared
        #     to SofastFringe), we will not downsample the data (set downsample to 1)
        surface = Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1)
        pt_known = measurement.origin
        xy_known = (0, 0)
        sofast.process_single_facet_optic(facet_data, surface, pt_known, xy_known)

        return sofast

    @staticmethod
    def _get_process_sofast_fringe():
        # Define sample data directory
        dir_data_sofast = join(opencsp_code_dir(), "test/data/sofast_fringe")
        dir_data_common = join(opencsp_code_dir(), "test/data/sofast_common")

        # Directory Setup
        file_measurement = join(dir_data_sofast, "data_measurement/measurement_facet.h5")
        file_camera = join(dir_data_common, "camera_sofast_downsampled.h5")
        file_display = join(dir_data_common, "display_distorted_2d.h5")
        file_orientation = join(dir_data_common, "spatial_orientation.h5")
        file_calibration = join(dir_data_sofast, "data_measurement/image_calibration.h5")
        file_facet = join(dir_data_common, "Facet_NSTTF.json")

        # Load saved single facet Sofast collection data
        camera = Camera.load_from_hdf(file_camera)
        display = Display.load_from_hdf(file_display)
        orientation = SpatialOrientation.load_from_hdf(file_orientation)
        measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
        calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
        facet_data = DefinitionFacet.load_from_json(file_facet)

        # Define surface definition (parabolic surface)
        # - Since we are using an NSTTF mirror with about a 300m focal length,
        #     we will seed the initial focal lengths with 300 in x and y
        # - Since we are testing SofastFringe (which has dense data sampling),
        #     we will downsample the data by a factor of 10
        surface = Surface2DParabolic(initial_focal_lengths_xy=(300.0, 300.0), robust_least_squares=True, downsample=10)

        # Calibrate fringes
        measurement.calibrate_fringe_images(calibration)

        # Instantiate sofast object
        sofast = ProcessSofastFringe(measurement, orientation, camera, display)

        # Process
        sofast.process_optic_singlefacet(facet_data, surface)

        return sofast

    def test_visualize_setup_fringe(self):
        # Create configuration object
        config = SofastConfiguration()
        config.load_sofast_object(self.process_sofast_fringe)

        # Create figures
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        config.visualize_setup(
            ax=ax,
            v_screen_object_screen=self.process_sofast_fringe.data_geometry_facet[
                0
            ].spatial_orientation.v_screen_optic_screen,
            r_object_screen=self.process_sofast_fringe.data_geometry_facet[0].spatial_orientation.r_optic_screen,
        )
        file_out = join(self.dir_output, "fringe_setup_visualize.png")
        fig.savefig(file_out)

        # Compare
        file_in = join(self.dir_input, "fringe_setup_visualize.png")
        self.compare_actual_expected_images(file_out, file_in)

    def test_measurement_stats_fringe(self):
        # Create configuration object
        config = SofastConfiguration()
        config.load_sofast_object(self.process_sofast_fringe)

        # Get measured stats
        stats = config.get_measurement_stats()

        # Get expected stats
        file_stats_in = join(self.dir_input, "stats_fringe.json")
        with open(file_stats_in, "r", encoding="utf-8") as f:
            stats_in = json.load(f)

        # Compare
        self.assertAlmostEqual(stats[0]["delta_x_sample_points_average"], stats_in[0]["delta_x_sample_points_average"])
        self.assertAlmostEqual(stats[0]["delta_y_sample_points_average"], stats_in[0]["delta_y_sample_points_average"])
        self.assertAlmostEqual(stats[0]["number_samples"], stats_in[0]["number_samples"])
        self.assertAlmostEqual(
            stats[0]["focal_lengths_parabolic_xy"][0], stats_in[0]["focal_lengths_parabolic_xy"][0], 3
        )
        self.assertAlmostEqual(
            stats[0]["focal_lengths_parabolic_xy"][1], stats_in[0]["focal_lengths_parabolic_xy"][1], 3
        )

    def test_visualize_setup_fixed(self):
        # Create configuration object
        config = SofastConfiguration()
        config.load_sofast_object(self.process_sofast_fixed)

        # Create figures
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        config.visualize_setup(
            ax=ax,
            v_screen_object_screen=self.process_sofast_fixed.data_geometry_facet[
                0
            ].spatial_orientation.v_screen_optic_screen,
            r_object_screen=self.process_sofast_fixed.data_geometry_facet[0].spatial_orientation.r_optic_screen,
        )
        file_out = join(self.dir_output, "fixed_setup_visualize.png")
        fig.savefig(file_out)

        # Compare
        file_in = join(self.dir_input, "fixed_setup_visualize.png")
        self.compare_actual_expected_images(file_out, file_in)

    def test_measurement_stats_fixed(self):
        # Create configuration object
        config = SofastConfiguration()
        config.load_sofast_object(self.process_sofast_fixed)

        # Get measured stats
        stats = config.get_measurement_stats()

        # Get expected stats
        file_stats_in = join(self.dir_input, "stats_fixed.json")
        with open(file_stats_in, "r", encoding="utf-8") as f:
            stats_in = json.load(f)

        # Compare
        self.assertAlmostEqual(stats[0]["delta_x_sample_points_average"], stats_in[0]["delta_x_sample_points_average"])
        self.assertAlmostEqual(stats[0]["delta_y_sample_points_average"], stats_in[0]["delta_y_sample_points_average"])
        self.assertAlmostEqual(stats[0]["number_samples"], stats_in[0]["number_samples"])
        self.assertAlmostEqual(
            stats[0]["focal_lengths_parabolic_xy"][0], stats_in[0]["focal_lengths_parabolic_xy"][0], 3
        )
        self.assertAlmostEqual(
            stats[0]["focal_lengths_parabolic_xy"][1], stats_in[0]["focal_lengths_parabolic_xy"][1], 3
        )

    def compare_actual_expected_images(self, actual_location: str, expected_location: str, tolerance=0.2):
        output = mplt.compare_images(expected_location, actual_location, tolerance)
        if output is not None:
            raise AssertionError(output)

    def tearDown(self) -> None:
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
