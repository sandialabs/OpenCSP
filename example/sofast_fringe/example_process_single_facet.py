"""Module for processing and analyzing SOFAST data for a single facet mirror.

This script performs the following steps:
1. Load saved single facet SOFAST collection data from an HDF5 file.
2. Save projected sinusoidal fringe images to PNG format.
3. Save captured sinusoidal fringe images and mask images to PNG format.
4. Process data with SOFAST and save processed data to HDF5.
5. Generate a suite of plots and save image files.

Examples
--------
To run the script, simply execute it as a standalone program:

>>> python example_process_single_facet.py

This will perform the processing steps and save the results to the data/output/single_facet directory
with the following subfolders:
1_images_fringes_projected - The patterns sent to the display during the SOFAST measurement of the optic.
2_images_captured - The captured images of the displayed patterns as seen by the SOFAST camera
3_processed_data - The processed data from SOFAST.
4_processed_output_figures - The output figure suite from a SOFAST characterization.

Notes
-----
- The script assumes that the input data files are located in the specified directories.
- Chat GPT 40 assisted with the generation of some docstrings in this file.
"""

import json
from os.path import join, basename, dirname, splitext

import argparse
import configparser

import imageio.v3 as imageio

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.SofastConfiguration import SofastConfiguration
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st


def process_single_facet(
    verbose: bool,
    file_camera: str,
    file_display: str,
    file_orientation: str,
    file_facet: str,
    file_calibration: str,
    file_measurement: str,
    dir_save: str,
    measurement_id: str,
    post_process_id: str,
    plots: StandardPlotOutput,
):
    """Performs processing of previously collected SOFAST data of single facet mirror.

    1. Load saved single facet SOFAST collection data from HDF5 file
    2. Save projected sinusoidal fringe images to PNG format
    3. Save captured sinusoidal fringe images and mask images to PNG format
    4. Processes data with SOFAST and save processed data to HDF5
    5. Generate plot suite and save image files
    """
    # General setup
    # =============

    # Set up save dir
    ft.create_directories_if_necessary(dir_save)

    # Construct output file prefix
    output_file_prefix = measurement_id + "_" + post_process_id + "_"

    # 1. Load saved single facet Sofast collection data
    # =================================================
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    orientation = SpatialOrientation.load_from_hdf(file_orientation)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    facet_data = DefinitionFacet.load_from_json(file_facet)

    # 2. Save projected sinusoidal fringe images to PNG format
    # ========================================================
    fringes = Fringes(measurement.fringe_periods_x, measurement.fringe_periods_y)
    images = fringes.get_frames(640, 320, "uint8", [0, 255])  # writes images we projected from sofast projector to disk
    dir_save_cur = join(dir_save, "B1_projected_fringes")
    ft.create_directories_if_necessary(dir_save_cur)
    # Save y images
    for idx_image in range(measurement.num_y_ims):
        image = images[..., idx_image]
        imageio.imwrite(join(dir_save_cur, output_file_prefix + f"y_{idx_image:02d}.png"), image)
    # Save x images
    for idx_image in range(measurement.num_x_ims):
        image = images[..., measurement.num_y_ims + idx_image]
        imageio.imwrite(join(dir_save_cur, output_file_prefix + f"x_{idx_image:02d}.png"), image)

    # 3. Save captured sinusoidal fringe images and mask images to PNG format
    # =======================================================================
    dir_save_cur = join(dir_save, "B2_captured_fringes")
    ft.create_directories_if_necessary(dir_save_cur)

    # Save mask (like a pixel mask value (all 0s, all 255s)) images
    for idx_image in [0, 1]:
        image = measurement.mask_images[..., idx_image]
        imageio.imwrite(join(dir_save_cur, output_file_prefix + f"mask_{idx_image:02d}.png"), image)
    # Save y images (when lines were vertical, e.g.)
    for idx_image in range(measurement.num_y_ims):
        image = measurement.fringe_images_y[..., idx_image]
        imageio.imwrite(join(dir_save_cur, output_file_prefix + f"y_{idx_image:02d}.png"), image)
    # Save x images (when lines were horizontal, e.g.)
    for idx_image in range(measurement.num_x_ims):
        image = measurement.fringe_images_x[..., idx_image]
        imageio.imwrite(join(dir_save_cur, output_file_prefix + f"x_{idx_image:02d}.png"), image)

    # 4. Processes data with Sofast and save processed data to HDF5
    # =============================================================
    dir_save_cur = join(dir_save, "B3_output_analysis")
    ft.create_directories_if_necessary(dir_save_cur)

    # Define surface definition (parabolic surface), this is the mirror
    surface = Surface2DParabolic(initial_focal_lengths_xy=(300.0, 300.0), robust_least_squares=True, downsample=10)

    # Calibrate fringes - (aka sinosoidal image)
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, orientation, camera, display)

    # Process
    sofast.process_optic_singlefacet(facet_data, surface)

    # Get measurement statistics
    config = SofastConfiguration()
    config.load_sofast_object(sofast)
    measurement_stats = config.get_measurement_stats()

    # Save processed data to HDF5 format
    sofast.save_to_hdf(join(dir_save_cur, output_file_prefix + "data_sofast_processed.h5"))

    # Save measurement stats as JSON
    with open(join(dir_save_cur, output_file_prefix + "measurement_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(measurement_stats, f, indent=3)

    # 5. Generate plot suite and save images files
    # ============================================
    dir_save_cur = join(dir_save, "B4_output_figures")
    ft.create_directories_if_necessary(dir_save_cur)

    # Get measured and reference optics
    mirror_measured = sofast.get_optic().mirror.no_parent_copy()
    mirror_reference = MirrorParametric.generate_symmetric_paraboloid(100, mirror_measured.region)

    # Save optic objects and output destination
    plots.optic_measured = mirror_measured
    plots.optic_reference = mirror_reference
    plots.options_file_output.output_dir = dir_save_cur
    plots.options_file_output.file_prefix = output_file_prefix

    # Create standard output plots
    plots.plot()


def example_process_single_facet_driver(arg_settings_dir_body_ext: str = None, verbose_param=None):
    """
    Sets up and runs the example_process_single_facet() routine.

    Parameters
    ----------

    arg_settings_dir_body_ext : str
        Full path and filename for settings file, containing inputand output directories, plot control settings, etc.
         Optional.  If not provided, internal defaults are used.
        See code for options sought within file.

    verbose : bool
        If true, output detailed progress and calculation output.
    """
    # Setup plot control, whcih might have some values set from settings, if provided.
    plots = StandardPlotOutput()

    # Get settings
    if arg_settings_dir_body_ext is None:
        print("Using default control settings.")
        # Verbose control
        if verbose_param is None:
            verbose = False
        else:
            verbose = verbose_param
        # Define sample data directories
        dir_data_sofast = join(opencsp_code_dir(), "test/data/sofast_fringe")
        dir_data_common = join(opencsp_code_dir(), "test/data/sofast_common")
        # Input files
        file_camera = join(dir_data_common, "camera_sofast_downsampled.h5")
        file_display = join(dir_data_common, "display_distorted_2d.h5")
        file_orientation = join(dir_data_common, "spatial_orientation.h5")
        file_facet = join(dir_data_common, "Facet_NSTTF.json")
        file_calibration = join(dir_data_sofast, "data_measurement/image_calibration.h5")
        file_measurement = join(dir_data_sofast, "data_measurement/measurement_facet.h5")
        # Define save dir
        dir_save = join(dirname(__file__), "data/output/single_facet")
        # Strings denoting computation.
        measurement_id = "Time_Mirror_InstrumentMode"
        post_process_id = "PostSpec"
        # Set plot control parameters to the values we want for the default example.
        plots.options_slope_vis.clim = 7
        plots.options_slope_vis.resolution = 0.001
        plots.options_slope_deviation_vis.clim = 1.5
        plots.options_slope_deviation_vis.resolution = 0.001
        plots.options_curvature_vis.resolution = 0.001
        plots.options_ray_trace_vis.enclosed_energy_max_semi_width = 1
        plots.options_file_output.to_save = True
        plots.options_file_output.number_in_name = False

        # Define viewing/illumination geometry
        v_target_center = Vxyz((0, 0, 100))
        v_target_normal = Vxyz((0, 0, -1))
        source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

        # Define ray trace parameters
        plots.params_ray_trace.source = source
        plots.params_ray_trace.v_target_center = v_target_center
        plots.params_ray_trace.v_target_normal = v_target_normal

    else:
        print("Loading control from settings file:", arg_settings_dir_body_ext)
        if not ft.file_exists(arg_settings_dir_body_ext):
            print("ERROR: In " + basename(__file__) + ", settings file does not exist. Settings file:")
            print("   ", arg_settings_dir_body_ext)
            assert False
        settings = configparser.ConfigParser()
        settings.read(arg_settings_dir_body_ext)
        # Verbose control
        verbose_setting = settings["Default"]["verbose"]
        if verbose_param is None:
            verbose = verbose_setting
        else:
            verbose = verbose_param
        # Input files
        file_camera = settings["Default"]["file_camera"]
        file_display = settings["Default"]["file_display"]
        file_orientation = settings["Default"]["file_orientation"]
        file_facet = settings["Default"]["file_facet"]
        file_calibration = settings["Default"]["file_calibration"]
        file_measurement = settings["Default"]["file_measurement"]
        # Define save dir
        dir_save = settings["Default"]["dir_save"]
        # Strings denoting computation
        measurement_id = st.verify_contiguous(settings["Default"]["measurement_id"])
        post_process_id = st.verify_contiguous(settings["Default"]["post_process_id"])
        # Set plot control parameters
        plots.set_plot_control_from_settings(settings)

    # Ensure output directory is ready
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    logfile_dir_body_ext = join(
        dir_save, measurement_id + "_" + post_process_id + "_" + splitext(basename(__file__))[0] + '_log.txt'
    )
    print("logfile_dir_body_ext = ", logfile_dir_body_ext)
    lt.logger(logfile_dir_body_ext, lt.log.INFO)
    # Output standard lines
    if verbose:
        lt.info_strings_from_file(join(dirname(__file__), splitext(basename(__file__))[0] + '_README.md'))
        lt.info('Starting program ' + __file__)
        lt.info('verbose = ' + str(verbose))
        lt.info('file_camera = ' + str(file_camera))
        lt.info('file_display = ' + str(file_display))
        lt.info('file_orientation = ' + str(file_orientation))
        lt.info('file_facet = ' + str(file_facet))
        lt.info('file_calibration = ' + str(file_calibration))
        lt.info('file_measurement = ' + str(file_measurement))
        lt.info('dir_save = ' + str(dir_save))
        lt.info('measurement_id = ' + str(measurement_id))
        lt.info('post_process_id = ' + str(post_process_id))
    if verbose:
        lt.info('Calling routine example_process_single_facet(...)...')

    # Process and output
    process_single_facet(
        verbose,
        file_camera,
        file_display,
        file_orientation,
        file_facet,
        file_calibration,
        file_measurement,
        dir_save,
        measurement_id,
        post_process_id,
        plots,
    )


if __name__ == "__main__":
    # Parse command-line arguments, if any.
    # Execute "python <this_file>.py --help" to see usage tips.
    #
    # Source - https://stackoverflow.com/a
    # Posted by Martijn Pieters, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-04, License - CC BY-SA 4.0
    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"),
        description='Analyze SOFAST measurement of a single facet, image processing, fitting, and producing analysis plots.  See "example_process_single_facet_README.md" for details.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--settings_dir_body_ext",
        required=False,
        dest="settings_dir_body_ext",
        default=None,
        help="Settings file defining run parameters (input/output directories, etc).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Output detailed information reporting run progress and calculations.",
    )
    args = parser.parse_args()
    arg_settings_dir_body_ext_main: str = args.settings_dir_body_ext
    verbose_main: bool = args.verbose

    # Manual override for use when debugging.  Comment this line for normal runs.
    # verbose: bool = True

    # Call driver.
    example_process_single_facet_driver(arg_settings_dir_body_ext_main, verbose_main)
