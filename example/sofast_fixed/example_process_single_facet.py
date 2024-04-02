from os.path import join, dirname

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.common.lib.camera.Camera import Camera
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_process_single_facet():
    """Example Sofast script that processes a SofastFixed measurement of a single facet mirror
    1. Load saved single facet SofastFixed collection data
    2. Process data with SofastFixed
    3. Log best-fit parabolic focal lengnths
    4. Plot slope magnitude
    5. Save slope data as HDF file
    """
    # General setup
    # =============

    # Define save dir
    dir_save = join(dirname(__file__), 'data/output/single_facet')
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define sample data directory
    dir_data_sofast = join(opencsp_code_dir(), 'test/data/sofast_fixed')
    dir_data_common = join(opencsp_code_dir(), 'test/data/sofast_common')

    # Directory setup
    file_meas = join(dir_base, 'measurement_printed_target_1.h5')
    file_camera = join(dir_base, "calibration_files/camera.h5")
    file_dot_locs = join(dir_base, 'dot_locations_printed_target.h5')
    file_ori = join(dir_base, 'spatial_orientation.h5')
    file_facet = join(dir_base, "calibration_files/Facet_NSTTF.json")

    surface_data = dict(
        surface_type='parabolic', initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1
    )

    # Load data
    camera = Camera.load_from_hdf(file_camera)
    facet_data = DefinitionFacet.load_from_json(file_facet)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
    orientation = SpatialOrientation.load_from_hdf(file_ori)
    measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

    # Instantiate class
    fixed_pattern = ProcessSofastFixed(orientation, camera, fixed_pattern_dot_locs, facet_data)
    fixed_pattern.load_measurement_data(measurement)

    # Process
    fixed_pattern.process_single_facet_optic(surface_data)

    # Print focal lengths
    surf_coefs = fixed_pattern.data_slope_solver.surf_coefs_facet
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    print('Parabolic fit focal lengths:')
    print(f'  X {focal_lengths_xy[0]:.3f} m')
    print(f'  Y {focal_lengths_xy[1]:.3f} m')

    # Plot slope image
    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control_m = rca.meters()

    fig_mng = fm.setup_figure(figure_control, axis_control_m, title='')
    mirror = fixed_pattern.get_mirror('bilinear')
    mirror.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_mng.axis)
    fig_mng.save(save_dir, 'orthorectified_slope_magnitude', 'png')

    # Save data
    fixed_pattern.save_to_hdf(join(save_dir, 'calculations.h5'))


if __name__ == '__main__':
    example_process_single_facet()
