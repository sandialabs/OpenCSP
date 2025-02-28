"""Module for processing and analyzing SOFAST data for a single cosmetic mirror.

This script performs the following steps:
1. Load saved single facet SOFAST collection data from an HDF5 file.
2. Save projected sinusoidal fringe images to PNG format.
3. Save captured sinusoidal fringe images and mask images to PNG format.
4. Process data with SOFAST and save processed data to HDF5.
5. Generate a suite of plots and save image files.

Examples
--------
To run the script, simply execute it as a standalone program:

>>> python example_laptop.py

This will perform the processing steps and save the results to the data/output/laptop directory
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
from os.path import join, dirname

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
import numpy as np


def example_laptop():
    """Performs processing of previously collected SOFAST data using a laptop webcam and a cosmetic mirror.

    1. Load saved single facet SOFAST collection data from HDF5 file
    2. Save projected sinusoidal fringe images to PNG format
    3. Save captured sinusoidal fringe images and mask images to PNG format
    4. Processes data with SOFAST and save processed data to HDF5
    5. Generate plot suite and save images files
    """
    # General setup
    # =============

    # Define save dir
    dir_save = join(dirname(__file__), 'data/output/laptop')
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    lt.logger(join(dir_save, 'example_laptop_log.txt'), lt.log.WARN)

    # Phase 1: Setup a sofast laptop experiment by loading the calibration data
    # (note that this data has already been collected for this opencsp example)
    # =========================================================================

    # Load the calibration data from physically setting up the laptop and cosmetic mirror in your room.
    dir_calibration_data = join(dirname(__file__), 'data/calibration')

    # This calibration data was prepared by doing the following:
    #  1) Get resolution of laptop screen and use this to adjust the display size to the desired location
    #       NOTE: if a screen scale is applied, this may need to be applied to input dimensions as well.
    #       NOTE: If using two screens with difference scales it is recommended to set the scale to be the same for both monitors.
    # file_calibration = join(dir_measurement_data, '20240515_104737_measurement_fringe.h5')
    #  2) We assume that the laptop screen has little to no distortion and use the 2d rectangular definition.
    #       NOTE: for setting up your own laptop sofast, open display_rectangular.h5 via HDFView, set screen x/y in meters, and save it.
    file_display = join(dir_calibration_data, 'screen_laptop_full/display_rectangular.h5')
    #  3) NOTE: We measured the distance from the center of the crosshairs on the cosmetic mirror to the center of the laptop webcam lens
    #       NOTE: for setting up your own laptop sofast, open spatial_orientation.h5 via HDFView, set optic_oriented to 0, set r_cam_screen to (0, 0, 0), set v_cam_screen_cam to x=?, y=?, z=0, and save it
    file_orientation = join(dir_calibration_data, 'screen_laptop_full/spatial_orientation.h5')
    #  4) NOTE: camera.h5 was measured by following 2.4.1 of the sofast user guide. See https://sandia-csp.app.box.com/file/1636393938364.
    file_laptop_camera = join(dir_calibration_data, 'camera_calibration/camera.h5')

    # This data was collected by running the sofast command line client with the above calibration data on the cosmetic mirror, and laptop webcam in our room
    dir_measurement_data = join(dirname(__file__), 'data/large_cosmetic_mirror')
    file_measurement = join(dir_measurement_data, 'measurement.h5')
    file_calibration = join(dir_measurement_data, 'calibration.h5')

    ex_calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)  # pixel intensity calibration 1-255

    ex_display = Display.load_from_hdf(
        file_display
    )  # this is a grid representation of the screen (aruco marker calibration algorithm output)
    ex_orientation = SpatialOrientation.load_from_hdf(file_orientation)  # camera to display
    ex_laptop_camera = Camera.load_from_hdf(file_laptop_camera)

    # Phase 2: Load the sofast laptop measurement data
    # (note that this data has already been collected for this opencsp example)
    # =========================================================================

    ex_measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)  # the sofast experimental data

    # 2. Save projected sinusoidal fringe images to PNG format
    # ========================================================
    fringes = Fringes(ex_measurement.fringe_periods_x, ex_measurement.fringe_periods_y)
    images = fringes.get_frames(
        640, 320, 'uint8', [0, 255]
    )  # writes images we projected from laptop screen "projector" to disk
    dir_save_cur = join(dir_save, '1_images_fringes_projected')
    ft.create_directories_if_necessary(dir_save_cur)
    # Save y images
    for idx_image in range(ex_measurement.num_y_ims):
        image = images[..., idx_image]
        imageio.imwrite(join(dir_save_cur, f'y_{idx_image:02d}.png'), image)
    # Save x images
    for idx_image in range(ex_measurement.num_x_ims):
        image = images[..., ex_measurement.num_y_ims + idx_image]
        imageio.imwrite(join(dir_save_cur, f'x_{idx_image:02d}.png'), image)

    # 3. Save captured sinusoidal fringe images and mask images to PNG format
    # =======================================================================
    dir_save_cur = join(dir_save, '2_images_captured')
    ft.create_directories_if_necessary(dir_save_cur)

    # Save mask (like a pixel mask value (all 0s, all 255s)) images
    for idx_image in [0, 1]:
        image = ex_measurement.mask_images[..., idx_image]
        img_normalized = (image - image.min()) / (image.max() - image.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        imageio.imwrite(join(dir_save_cur, f'mask_{idx_image:02d}.png'), img_normalized)
    # Save y images (when lines were vertical, e.g.)
    for idx_image in range(ex_measurement.num_y_ims):
        image = ex_measurement.fringe_images_y[..., idx_image]
        img_normalized = (image - image.min()) / (image.max() - image.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        imageio.imwrite(join(dir_save_cur, f'y_{idx_image:02d}.png'), img_normalized)
    # Save x images (when lines were horizontal, e.g.)
    for idx_image in range(ex_measurement.num_x_ims):
        image = ex_measurement.fringe_images_x[..., idx_image]
        img_normalized = (image - image.min()) / (image.max() - image.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        imageio.imwrite(join(dir_save_cur, f'x_{idx_image:02d}.png'), img_normalized)

    # 4. Processes data with Sofast and save processed data to HDF5
    # =============================================================
    dir_save_cur = join(dir_save, '3_processed_data')
    ft.create_directories_if_necessary(dir_save_cur)

    # Define surface definition (parabolic surface), this is the mirror
    surface = Surface2DParabolic(initial_focal_lengths_xy=(2.0, 2.0), robust_least_squares=False, downsample=5)

    # Calibrate fringes - (aka sinosoidal image)
    ex_measurement.calibrate_fringe_images(ex_calibration)

    # Instantiate sofast object
    sofast = Sofast(ex_measurement, ex_orientation, ex_laptop_camera, ex_display)
    sofast.params.mask.keep_largest_area = True

    # Process
    sofast.process_optic_undefined(surface)

    # Save processed data to HDF5 format
    sofast.save_to_hdf(join(dir_save_cur, 'data_sofast_cosmetic_mirror_fringe.h5'))

    # Save measurement statistics to JSON
    config = SofastConfiguration()
    config.load_sofast_object(sofast)
    measurement_stats = config.get_measurement_stats()

    # Save measurement stats as JSON
    with open(join(dir_save_cur, 'cosmetic_mirror_measurement_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(measurement_stats, f, indent=3)

    # 5. Generate plot suite and save images files
    # ============================================
    dir_save_cur = join(dir_save, '4_processed_output_figures')
    ft.create_directories_if_necessary(dir_save_cur)

    # Get measured and reference optics
    mirror_measured = sofast.get_optic('bilinear').mirror.no_parent_copy()
    mirror_reference = MirrorParametric.generate_symmetric_paraboloid(0.55, mirror_measured.region)

    # Define viewing/illumination geometry
    # x and y is ground plane
    # z is pointing up to the sky

    # target is 100 meters up in air
    v_target_center = Vxyz(
        (0, 0, 0.55)
    )  # TODO: Update docs to specify meters, document that x is up, y is straight into air, and z is left-right
    # TODO: try moving focal length out, etc.

    # The normal is pointing down
    v_target_normal = Vxyz((0, 0, -1))

    # Pointing angle is pointing straight down (Uxyz(0,0,-1))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

    # Save optic objects
    plots = StandardPlotOutput()
    plots.optic_measured = mirror_measured
    plots.optic_reference = mirror_reference

    # Update visualization parameters
    plots.options_slope_vis.clim = 20  # maximum slope at edge of mirror (we know focal length and radius of mirror, )
    plots.options_slope_vis.resolution = 0.001
    plots.options_slope_vis.quiver_scale = 2000  # TODO: can this be set to auto and computed by default?
    plots.options_slope_vis.quiver_density = 0.05  # TODO: can this be set to auto and computed by default?

    plots.options_slope_deviation_vis.clim = 15
    plots.options_slope_deviation_vis.resolution = 0.001
    # TODO: Document why the quiver scale and denisty doesn't need to be updated. (Our magnitude of error is about 15 mrad, CSP mirrors are typically 7-10)

    plots.options_curvature_vis.resolution = 0.001

    plots.options_ray_trace_vis.enclosed_energy_max_semi_width = 1
    plots.options_ray_trace_vis.hist_extent = 0.05  # image of light source reflection
    plots.options_ray_trace_vis.hist_bin_res = 0.00075  # TODO: maybe also have a auto default to compute defaults
    plots.options_ray_trace_vis.ray_trace_optic_res = 0.01  # TODO: maybe also have a auto default to compute defaults

    plots.options_file_output.to_save = True
    plots.options_file_output.number_in_name = False
    plots.options_file_output.output_dir = dir_save_cur

    # Define ray trace parameters
    plots.params_ray_trace.source = source
    plots.params_ray_trace.v_target_center = v_target_center
    plots.params_ray_trace.v_target_normal = v_target_normal

    # Create standard output plots
    plots.plot()


if __name__ == '__main__':
    example_laptop()
