"""
Processes SOFAST data with debug mode enabled.

This module provides a function to process optical measurements using the SOFAST
framework while enabling debug mode. The function performs the following tasks:
loading necessary data files, calibrating fringe images, and processing the
measurements. If an error occurs during processing, debug figures are saved for
further analysis.

Usage:
    To execute the processing with debug mode, run the module directly. The output
    figures and log files will be saved in a specified directory structure.
"""

# ChatGPT 4o-mini assisted with docstring creation

from os.path import join, dirname

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_process_in_debug_mode():
    """
    Processes SOFAST data with debug mode enabled. This example intentionally runs SOFAST
    on a single facet with erroneous facet corner location information. When SOFAST is in
    debug mode, the cause of the most common errors can easily be diagnosed by looking at
    the produced figures from debug mode. Run this script as-is, the look through the output
    figures produced (location is printed in terminal) to see the cause of the error (the
    erroneous facet corner definition file).

    This function performs the following steps:

        1. Sets up the necessary directories and logging for output.
        2. Loads required data files, including camera, display, spatial orientation,
           measurement, calibration, and facet definition data.
        3. Calibrates the fringe images from the measurement data.
        4. Instantiates a SOFAST processing object and enables debug mode.
        5. Processes the optical data for a single facet. If an error occurs during
           processing, all debug figures are saved to the specified output directory.

    Notes
    -----
    The function needs the following files to run:

        - Measurement data (HDF5 format)
        - Camera calibration data (HDF5 format)
        - Display shape data (HDF5 format)
        - Spatial orientation data (HDF5 format)
        - Image calibration data (HDF5 format)
        - Facet definition data (JSON format)

    Example
    -------
    To process the SOFAST data with debug mode enabled, simply call the function:

    >>> example_process_in_debug_mode()

    The resulting debug figures will be saved in the output directory if an error
    occurs during processing.
    """
    # 1. General setup
    # ================

    # Define save dir
    dir_save = join(dirname(__file__), "data/output/sofast_with_debug_mode_on")
    ft.create_directories_if_necessary(dir_save)
    dir_save_debug = join(dir_save, "debug")
    ft.create_directories_if_necessary(dir_save_debug)

    # Set up logger
    lt.logger(join(dir_save, "log.txt"), lt.log.INFO)

    # Define sample data directory
    dir_data_sofast = join(opencsp_code_dir(), "test/data/sofast_fringe")
    dir_data_common = join(opencsp_code_dir(), "test/data/sofast_common")

    # Directory Setup
    file_measurement = join(dir_data_sofast, "data_measurement/measurement_facet.h5")
    file_camera = join(dir_data_common, "camera_sofast_downsampled.h5")
    file_display = join(dir_data_common, "display_distorted_2d.h5")
    file_orientation = join(dir_data_common, "spatial_orientation.h5")
    file_calibration = join(dir_data_sofast, "data_measurement/image_calibration.h5")
    file_facet = join(dirname(__file__), "data/input/incorrect_facet_definition.json")

    # 2. Load saved single facet Sofast collection data
    # =================================================
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    orientation = SpatialOrientation.load_from_hdf(file_orientation)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    facet_data = DefinitionFacet.load_from_json(file_facet)

    # 3. Processes data with SOFAST and handle errors
    # ===============================================
    # Define surface definition (parabolic surface), this is the mirror
    surface = Surface2DParabolic(initial_focal_lengths_xy=(300.0, 300.0), robust_least_squares=True, downsample=10)

    # Calibrate fringes - (aka sinosoidal image)
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, orientation, camera, display)

    # Turn on debug mode
    sofast.params.debug_geometry.debug_active = True
    sofast.params.debug_geometry.debug_active = True
    sofast.params.debug_geometry.save_dir = dir_save_debug
    sofast.params.debug_geometry.figure_idx = 0
    sofast.params.debug_slope_solver.debug_active = True
    sofast.params.debug_slope_solver.save_dir = dir_save_debug
    sofast.params.debug_slope_solver.figure_idx = 100

    # Process
    try:
        sofast.process_optic_singlefacet(facet_data, surface)
    except ValueError:
        # Save all debug figures
        lt.info(f'An error occured when processing SOFAST data. Saving all debug figures to {dir_save}.')
        for idx, fig in enumerate(sofast.params.debug_geometry.figures):
            fig.savefig(join(dir_save, f'{idx:02d}.png'))


if __name__ == "__main__":
    example_process_in_debug_mode()
