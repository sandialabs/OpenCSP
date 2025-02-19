from os.path import join, dirname

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_process_facet():
    """Processes a SofastFixed measurement of a single facet mirror

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
    file_meas = join(dir_data_sofast, 'data_measurement/measurement_facet.h5')
    file_camera = join(dir_data_common, 'camera_sofast.h5')
    file_dot_locs = join(dir_data_sofast, 'data_measurement/fixed_pattern_dot_locations.h5')
    file_ori = join(dir_data_common, 'spatial_orientation.h5')
    file_facet = join(dir_data_common, 'Facet_NSTTF.json')

    # 1. Load saved single facet SofastFixed collection data
    # ======================================================
    camera = Camera.load_from_hdf(file_camera)
    facet_data = DefinitionFacet.load_from_json(file_facet)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
    orientation = SpatialOrientation.load_from_hdf(file_ori)
    measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

    # 2. Process data with SofastFixed
    # ================================

    # Instantiate SofastFixed class and load measurement data
    fixed_pattern = ProcessSofastFixed(orientation, camera, fixed_pattern_dot_locs)
    fixed_pattern.load_measurement_data(measurement)

    # Process
    surface = Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1)
    fixed_pattern.process_single_facet_optic(facet_data, surface, measurement.origin, (0, 0))

    # 3. Log best-fit parabolic focal lengnths
    # ========================================
    surf_coefs = fixed_pattern.slope_solvers[0].surface.surf_coefs
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    lt.info(f'Parabolic fit xy focal lengths: {focal_lengths_xy[0]:.3f}, {focal_lengths_xy[1]:.3f} m')

    # 4. Plot slope magnitude
    # =======================
    mirror = fixed_pattern.get_optic('bilinear')

    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control_m = rca.meters()
    fig_mng = fm.setup_figure(figure_control, axis_control_m, title='')
    mirror.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_mng.axis)
    fig_mng.save(dir_save, 'orthorectified_slope_magnitude', 'png')

    # 5. Save slope data as HDF file
    # ==============================
    fixed_pattern.save_to_hdf(join(dir_save, 'calculations.h5'))


if __name__ == '__main__':
    example_process_facet()
