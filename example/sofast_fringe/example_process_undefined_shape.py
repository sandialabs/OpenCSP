from os.path import join, dirname

import matplotlib

from opencsp.app.sofast.lib.visualize_setup import visualize_setup
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example(dir_save):
    """Example SOFAST script

    Performs processing of previously collected Sofast data of single facet mirror.
    1. Loads saved "single-facet" SOFAST collection data
    2. Processes data with SOFAST (without using facet file)
    3. Prints best-fit parabolic focal lengths
    4. Plots slope magnitude, physical setup
    """
    # Define sample data directory
    sample_data_dir = join(opencsp_code_dir(), 'test/data/measurements_sofast_fringe/')

    # Directory Setup
    file_measurement = join(sample_data_dir, 'measurement_facet.h5')
    file_camera = join(sample_data_dir, 'camera.h5')
    file_display = join(sample_data_dir, 'display_distorted_2d.h5')
    file_calibration = join(sample_data_dir, 'image_calibration.h5')

    # Load data
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)

    # Define surface definition (parabolic surface)
    surface = Surface2DParabolic(initial_focal_lengths_xy=(300.0, 300.0), robust_least_squares=True, downsample=10)

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, camera, display)
    sofast.params.mask_keep_largest_area = True

    # Process
    sofast.process_optic_undefined(surface)

    # Calculate focal length from parabolic fit
    surf_coefs = sofast.data_characterization_facet[0].surf_coefs_facet
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    lt.info(f'Facet xy focal lengths (meters): ' f'{focal_lengths_xy[0]:.3f}, {focal_lengths_xy[1]:.3f}')

    # Get optic representation
    facet: Facet = sofast.get_optic()

    # Generate plots
    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control_m = rca.meters()

    # Visualize setup
    fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, title='')
    spatial_ori: SpatialOrientation = sofast.data_geometry_facet[0].spatial_orientation
    visualize_setup(display, camera, spatial_ori.v_screen_optic_screen, spatial_ori.r_optic_screen, ax=fig_record.axis)
    fig_record.save(dir_save, 'physical_setup_layout', 'png')

    # Plot slope map
    fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
    facet.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, 'slope_magnitude', 'png')

    # Save data
    sofast.save_to_hdf(f'{dir_save}/data_undefined.h5')


def example_driver():
    # Define save dir
    save_path = join(dirname(__file__), 'data/output/single_facet')
    ft.create_directories_if_necessary(save_path)

    # Set up logger
    lt.logger(join(save_path, 'log.txt'), lt.log.INFO)

    example(save_path)


if __name__ == '__main__':
    example_driver()
