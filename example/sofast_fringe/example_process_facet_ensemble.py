from os.path import join, dirname
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


def example_process_facet_ensemble():
    """Performs processing of previously collected Sofast data
    of multi facet mirror ensemble:

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

    base_dir = join(opencsp_code_dir(), 'test/data/sofast_fringe')

    # Directory Setup
    file_dataset = join(base_dir, 'data_expected_facet_ensemble/data.h5')
    file_measurement = join(base_dir, 'data_measurement/measurement_ensemble.h5')

    # Load data
    camera = Camera.load_from_hdf(file_dataset)
    display = DisplayShape.load_from_hdf(file_dataset)
    orientation = SpatialOrientation.load_from_hdf(file_dataset)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)

    # Load sofast params
    datasets = [
        'DataSofastInput/sofast_params/mask_hist_thresh',
        'DataSofastInput/sofast_params/mask_filt_width',
        'DataSofastInput/sofast_params/mask_filt_thresh',
        'DataSofastInput/sofast_params/mask_thresh_active_pixels',
        'DataSofastInput/sofast_params/mask_keep_largest_area',
        'DataSofastInput/sofast_params/perimeter_refine_axial_search_dist',
        'DataSofastInput/sofast_params/perimeter_refine_perpendicular_search_dist',
        'DataSofastInput/sofast_params/facet_corns_refine_step_length',
        'DataSofastInput/sofast_params/facet_corns_refine_perpendicular_search_dist',
        'DataSofastInput/sofast_params/facet_corns_refine_frac_keep',
    ]
    params = load_hdf5_datasets(datasets, file_dataset)

    # Calibrate measurement
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = ProcessSofastFringe(measurement, orientation, camera, display)

    # Update parameters
    sofast.params.mask.hist_thresh = params['mask_hist_thresh']
    sofast.params.mask.filt_width = params['mask_filt_width']
    sofast.params.mask.filt_thresh = params['mask_filt_thresh']
    sofast.params.mask.thresh_active_pixels = params['mask_thresh_active_pixels']
    sofast.params.mask.keep_largest_area = params['mask_keep_largest_area']

    sofast.params.geometry.perimeter_refine_axial_search_dist = params['perimeter_refine_axial_search_dist']
    sofast.params.geometry.perimeter_refine_perpendicular_search_dist = params[
        'perimeter_refine_perpendicular_search_dist'
    ]
    sofast.params.geometry.facet_corns_refine_step_length = params['facet_corns_refine_step_length']
    sofast.params.geometry.facet_corns_refine_perpendicular_search_dist = params[
        'facet_corns_refine_perpendicular_search_dist'
    ]
    sofast.params.geometry.facet_corns_refine_frac_keep = params['facet_corns_refine_frac_keep']

    # Load ensemble data
    datasets = [
        'DataSofastInput/optic_definition/ensemble/ensemble_perimeter',
        'DataSofastInput/optic_definition/ensemble/r_facet_ensemble',
        'DataSofastInput/optic_definition/ensemble/v_centroid_ensemble',
        'DataSofastInput/optic_definition/ensemble/v_facet_locations',
    ]
    ensemble_data = load_hdf5_datasets(datasets, file_dataset)
    ensemble_data = DefinitionEnsemble(
        Vxyz(ensemble_data['v_facet_locations']),
        [Rotation.from_rotvec(r) for r in ensemble_data['r_facet_ensemble']],
        ensemble_data['ensemble_perimeter'],
        Vxyz(ensemble_data['v_centroid_ensemble']),
    )

    facet_data = []
    for idx in range(len(ensemble_data.r_facet_ensemble)):
        datasets = [
            f'DataSofastInput/optic_definition/facet_{idx:03d}/v_centroid_facet',
            f'DataSofastInput/optic_definition/facet_{idx:03d}/v_facet_corners',
        ]
        data = load_hdf5_datasets(datasets, file_dataset)
        facet_data.append(DefinitionFacet(Vxyz(data['v_facet_corners']), Vxyz(data['v_centroid_facet'])))

    # Load surface data
    surfaces = []
    for idx in range(len(facet_data)):
        datasets = [
            f'DataSofastInput/surface_params/facet_{idx:03d}/downsample',
            f'DataSofastInput/surface_params/facet_{idx:03d}/initial_focal_lengths_xy',
            f'DataSofastInput/surface_params/facet_{idx:03d}/robust_least_squares',
        ]
        data = load_hdf5_datasets(datasets, file_dataset)
        data['robust_least_squares'] = bool(data['robust_least_squares'])
        surfaces.append(Surface2DParabolic(**data))

    # Update search parameters
    # sofast.params.mask_hist_thresh = 0.83
    # sofast.params.geometry.perimeter_refine_perpendicular_search_dist = 15.0
    # sofast.params.geometry.facet_corns_refine_frac_keep = 1.0
    # sofast.params.geometry.facet_corns_refine_perpendicular_search_dist = 3.0
    # sofast.params.geometry.facet_corns_refine_step_length = 5.0

    # Process
    sofast.process_optic_multifacet(facet_data, ensemble_data, surfaces)

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
    facet_control = rcf.RenderControlFacet(
        draw_mirror_curvature=True, mirror_styles=mirror_control, draw_outline=False, draw_surface_normal=True
    )
    facet_ensemble_control = rcfe.RenderControlFacetEnsemble(default_style=facet_control, draw_outline=True)
    axis_control_m = rca.meters()

    # Plot slope map
    fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
    ensemble.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, 'slope_magnitude', 'png')

    # 5. Plot 3d representation of facet ensemble
    # ===========================================
    fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, title='Facet Ensemble')
    ensemble.draw(fig_record.view, facet_ensemble_control)
    fig_record.axis.axis('equal')
    fig_record.save(dir_save, 'facet_ensemble', 'png')

    # 6. Save slope data as HDF5 file
    # ===============================
    sofast.save_to_hdf(f'{dir_save}/data_multifacet.h5')


if __name__ == '__main__':
    example_process_facet_ensemble()
