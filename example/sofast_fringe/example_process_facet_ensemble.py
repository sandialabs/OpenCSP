from os.path import join, dirname

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_process_facet_ensemble():
    """Example Sofast script

    Performs processing of previously collected Sofast data of multi facet mirror ensemble:
    1. Load saved facet ensemble Sofast collection data
    2. Processes data with Sofast
    3. Log best-fit parabolic focal lengths
    4. Plot slope magnitude
    5. Plot 3d representation of facet ensemble
    6. Save slope data as HDF5 file
    """
    # General setup
    # =============

    # Define save dir
    dir_save = join(dirname(__file__), 'data/output/facet_ensemble')
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define sample data directory
    dir_data_sofast = join(opencsp_code_dir(), 'test/data/sofast_fringe')
    dir_data_common = join(opencsp_code_dir(), 'test/data/sofast_common')

    # Directory setup
    file_measurement = join(dir_data_sofast, 'data_measurement/measurement_ensemble.h5')
    file_camera = join(dir_data_common, 'camera_sofast_downsampled.h5')
    file_display = join(dir_data_common, 'display_distorted_2d.h5')
    file_orientation = join(dir_data_common, 'spatial_orientation.h5')
    file_calibration = join(dir_data_sofast, 'data_measurement/image_calibration.h5')
    file_facet = join(dir_data_common, 'Facet_lab_6x4.json')
    file_ensemble = join(dir_data_common, 'Ensemble_lab_6x4.json')

    # 1. Load saved single facet Sofast collection data
    # =================================================
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    orientation = SpatialOrientation.load_from_hdf(file_orientation)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    ensemble_data = DefinitionEnsemble.load_from_json(file_ensemble)

    # 2. Process data with Sofast
    # ===========================

    # Define facet data
    facet_data = [DefinitionFacet.load_from_json(file_facet)] * ensemble_data.num_facets

    # Define surface data
    surface_data = [
        Surface2DParabolic(initial_focal_lengths_xy=(100.0, 100.0), robust_least_squares=False, downsample=20)
    ] * ensemble_data.num_facets

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, orientation, camera, display)

    # Update search parameters
    sofast.params.mask_hist_thresh = 0.83
    sofast.params.geometry_params.perimeter_refine_perpendicular_search_dist = 10.0
    sofast.params.geometry_params.facet_corns_refine_frac_keep = 1.0
    sofast.params.geometry_params.facet_corns_refine_perpendicular_search_dist = 3.0
    sofast.params.geometry_params.facet_corns_refine_step_length = 5.0

    # Process
    sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    # 3. Log best-fit parabolic focal lengths
    # =======================================
    for idx in range(sofast.num_facets):
        surf_coefs = sofast.data_characterization_facet[idx].surf_coefs_facet
        focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
        lt.info(f'Facet {idx:d} xy focal lengths (meters): {focal_lengths_xy[0]:.3f}, {focal_lengths_xy[1]:.3f}')

    # 4. Plot slope magnitude
    # =======================

    # Get optic representation
    ensemble: FacetEnsemble = sofast.get_optic()

    # Generate plots
    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    mirror_control = rcm.RenderControlMirror(centroid=True, surface_normals=True, norm_res=1)
    axis_control_m = rca.meters()

    # Plot slope map
    fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
    ensemble.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, 'slope_magnitude', 'png')

    # 5. Plot 3d representation of facet ensemble
    # ===========================================
    fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, title='Facet Ensemble')
    ensemble.draw(fig_record.view, mirror_control)
    fig_record.axis.axis('equal')
    fig_record.save(dir_save, 'facet_ensemble', 'png')

    # 6. Save slope data as HDF5 file
    # ===============================
    sofast.save_to_hdf(f'{dir_save}/data_multifacet.h5')


if __name__ == '__main__':
    example_process_facet_ensemble()
