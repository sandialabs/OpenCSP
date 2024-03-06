import os
from os.path import join

import matplotlib

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.lib.visualize_setup import visualize_setup
from opencsp.common.lib.deflectometry.Display import Display
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.app.sofast.lib.Sofast import Sofast
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlMirror as rcm


def example_driver():
    """Example SOFAST script

    Performs processing of previously collected Sofast data of single facet mirror.
        1. Loads saved "single-facet" SOFAST collection data
        2. Processes data with SOFAST
        3. Prints best-fit parabolic focal lengths
        4. Plots slope magnitude, physical setup
    """
    # Define sample data directory
    sample_data_dir = join(
        opencsp_code_dir(), 'test/data/sofast_measurements/'
    )

    # Directory Setup
    file_measurement = join(sample_data_dir, 'measurement_facet.h5')
    file_camera = join(sample_data_dir, 'camera.h5')
    file_display = join(sample_data_dir, 'display_distorted_2d.h5')
    file_calibration = join(sample_data_dir, 'calibration.h5')
    file_facet = join(sample_data_dir, 'Facet_NSTTF.json')

    # Define save dir
    dir_save = join(os.path.dirname(__file__), 'data/output/single_facet')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # Load data
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    facet_data = FacetData.load_from_json(file_facet)

    # Define surface definition (parabolic surface)
    surface_data = dict(
        surface_type='parabolic',
        initial_focal_lengths_xy=(300.0, 300),
        robust_least_squares=True,
        downsample=10,
    )

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, camera, display)

    # Process
    sofast.process_optic_singlefacet(facet_data, surface_data)

    # Calculate focal length from parabolic fit
    if surface_data['surface_type'] == 'parabolic':
        surf_coefs = sofast.data_characterization_facet[0].surf_coefs_facet
        focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
        print('Parabolic fit focal lengths:')
        print(f'  X {focal_lengths_xy[0]:.3f} m')
        print(f'  Y {focal_lengths_xy[1]:.3f} m')

    # Get optic representation
    facet: Facet = sofast.get_optic()

    # Generate plots
    figure_control = rcfg.RenderControlFigure(
        tile_array=(1, 1), tile_square=True)
    mirror_control = rcm.RenderControlMirror(
        centroid=True, surface_normals=True, norm_res=1
    )
    axis_control_m = rca.meters()

    # Visualize setup
    fig_record = fm.setup_figure_for_3d_data(
        figure_control, axis_control_m, title='')
    spatial_ori: SpatialOrientation = sofast.data_geometry_facet[0].spatial_orientation
    visualize_setup(
        display,
        camera,
        spatial_ori.v_screen_optic_screen,
        spatial_ori.r_optic_screen,
        ax=fig_record.axis,
    )
    fig_record.save(dir_save, 'physical_setup_layout', 'png')

    # Plot slope map
    fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
    facet.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, 'slope_magnitude', 'png')

    # Save data
    sofast.save_data_to_hdf(f'{dir_save}/data_singlefacet.h5')


if __name__ == '__main__':
    example_driver()
