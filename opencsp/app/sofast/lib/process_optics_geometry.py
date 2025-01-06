"""Library of functions used to process the geometry of a deflectometry setup."""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.ParamsMaskCalculation import ParamsMaskCalculation
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt


def process_singlefacet_geometry(
    facet_data: DefinitionFacet,
    mask_raw: ndarray,
    v_measure_point_facet: Vxyz,
    dist_optic_screen: float,
    orientation: SpatialOrientation,
    camera: Camera,
    params: ParamsOpticGeometry = ParamsOpticGeometry(),
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for single facet

    Parameters
    ----------
    facet_data : DefinitionFacet
        DefinitionFacet object
    mask_raw : ndarray
        Raw calculated mask
    v_measure_point_facet : Vxyz
        Measure point location on facet, meters
    dist_optic_screen : float
        Optic to screen distance, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    params : ParamsOpticGeometry, optional
        ParamsOpticGeometry object, by default ParamsOpticGeometry()
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calcualtions specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    if debug.debug_active:
        lt.debug('process_optics_geometry debug on.')
    else:
        lt.debug('process_optics_geometry debug off.')

    # Create data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = cdc.CalculationDataGeometryFacet()
    data_image_processing_facet = cdc.CalculationImageProcessingFacet()
    data_error = cdc.CalculationError()

    # Make copy of orientation
    ori = copy(orientation)

    # Get optic data
    v_facet_corners: Vxyz = facet_data.v_facet_corners  # Corners of facet in facet coordinates
    v_centroid_facet: Vxyz = facet_data.v_facet_centroid  # Centroid of facet in facet coordinates

    # Save mask raw
    data_image_processing_general.mask_raw = mask_raw

    # Plot mask
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw, cmap='gray')
        plt.title('Raw Mask')

    # Find edges of mask
    v_edges_image = ip.edges_from_mask(mask_raw)
    data_image_processing_general.v_edges_image = v_edges_image

    # Find centroid of processed mask
    v_mask_centroid_image = ip.centroid_mask(mask_raw)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Plot centroid
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        plt.scatter(*v_mask_centroid_image.data, marker='x')
        plt.title('Mask Centroid')

    # Find expected position of optic centroid
    v_cam_optic_centroid_cam_exp = sp.t_from_distance(
        v_mask_centroid_image, dist_optic_screen, camera, ori.v_cam_screen_cam
    )
    data_geometry_general.v_cam_optic_centroid_cam_exp = v_cam_optic_centroid_cam_exp

    # Plot expected centroid
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        plt.scatter(
            *camera.project(v_cam_optic_centroid_cam_exp, Rotation.identity(), Vxyz((0, 0, 0))).data, marker='.'
        )
        plt.title('Expected Optic Centroid')

    # Find expected orientation of optic
    r_cam_optic_exp = sp.r_from_position(v_cam_optic_centroid_cam_exp, ori.v_cam_screen_cam)
    data_geometry_general.r_optic_cam_exp = r_cam_optic_exp.inv()

    # Find expected position of optic origin
    v_cam_optic_cam_exp = v_cam_optic_centroid_cam_exp - v_centroid_facet.rotate(r_cam_optic_exp.inv())
    data_geometry_general.v_cam_optic_cam_exp = v_cam_optic_cam_exp

    # Find expected optic loop in pixels
    v_optic_corners_image_exp = camera.project(v_facet_corners, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
    loop_optic_image_exp = LoopXY.from_vertices(v_optic_corners_image_exp)
    data_image_processing_general.loop_optic_image_exp = loop_optic_image_exp

    # Plot expected optic corners
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        _plot_labeled_points(v_optic_corners_image_exp)
        plt.title('Expected Optic Corners')

    # Refine locations of optic corners with mask
    prs = [params.perimeter_refine_axial_search_dist, params.perimeter_refine_perpendicular_search_dist]
    loop_facet_image_refine = ip.refine_mask_perimeter(loop_optic_image_exp, v_edges_image, *prs)
    data_image_processing_facet.loop_facet_image_refine = loop_facet_image_refine

    # Plot refined optic corners
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        _plot_labeled_points(loop_facet_image_refine.vertices)
        plt.title('Refined Optic Corners')

    # Create fitted mask
    vx = np.arange(mask_raw.shape[1])
    vy = np.arange(mask_raw.shape[0])
    mask_fitted = loop_facet_image_refine.as_mask(vx, vy)
    data_image_processing_facet.mask_fitted = mask_fitted

    # Remove non-active pixels from mask
    mask_processed = np.logical_and(mask_fitted, mask_raw)
    data_image_processing_facet.mask_processed = mask_processed

    # Calculate R/T from found corners
    r_optic_cam_refine_1, v_cam_optic_cam_refine_1 = sp.calc_rt_from_img_pts(
        loop_facet_image_refine.vertices, v_facet_corners, camera
    )
    r_cam_optic_refine_1 = r_optic_cam_refine_1.inv()
    data_geometry_general.r_optic_cam_refine_1 = r_optic_cam_refine_1
    data_geometry_general.v_cam_optic_cam_refine_1 = v_cam_optic_cam_refine_1

    # Plot reprojected points 1
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        pts_reproj = camera.project(facet_data.v_facet_corners, r_cam_optic_refine_1.inv(), v_cam_optic_cam_refine_1)
        _plot_labeled_points(pts_reproj)
        plt.title('Reprojected Points 1')

    # Calculate refined measure point vector in optic coordinates
    v_measure_point_optic_cam_refine_1 = v_measure_point_facet.rotate(r_optic_cam_refine_1)

    # Refine V with measured optic to display distance
    v_cam_optic_cam_refine_2 = sp.refine_v_distance(
        v_cam_optic_cam_refine_1, dist_optic_screen, ori.v_cam_screen_cam, v_measure_point_optic_cam_refine_1
    )
    data_geometry_general.v_cam_optic_cam_refine_2 = v_cam_optic_cam_refine_2

    # Plot reprojected points 2
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        pts_reproj = camera.project(facet_data.v_facet_corners, r_cam_optic_refine_1.inv(), v_cam_optic_cam_refine_2)
        _plot_labeled_points(pts_reproj)
        plt.title('Reprojected Points 2')

    # Orient optic
    ori.orient_optic_cam(r_cam_optic_refine_1, v_cam_optic_cam_refine_2)

    # Calculate measure point pointing direction
    u_cam_measure_point_facet = Uxyz((ori.v_cam_optic_optic + v_measure_point_facet).data)
    data_geometry_facet.u_cam_measure_point_facet = u_cam_measure_point_facet

    # Calculate errors from using only facet corners
    error_dist_optic_screen_1 = sp.distance_error(
        ori.v_cam_screen_cam, v_cam_optic_cam_refine_1 + v_measure_point_optic_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_1 = error_dist_optic_screen_1
    error_reprojection_1 = sp.reprojection_error(
        camera, v_facet_corners, loop_facet_image_refine.vertices, r_optic_cam_refine_1, v_cam_optic_cam_refine_1
    )
    data_error.error_reprojection_1 = error_reprojection_1

    # Calculate errors after refining with measured distance
    error_dist_optic_screen_2 = sp.distance_error(
        ori.v_cam_screen_cam, v_cam_optic_cam_refine_2 + v_measure_point_optic_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_2 = error_dist_optic_screen_2
    error_reprojection_2 = sp.reprojection_error(
        camera, v_facet_corners, loop_facet_image_refine.vertices, r_optic_cam_refine_1, v_cam_optic_cam_refine_2
    )
    data_error.error_reprojection_2 = error_reprojection_2

    # Save other data
    data_geometry_facet.measure_point_screen_distance = dist_optic_screen
    data_geometry_facet.spatial_orientation = ori
    data_geometry_facet.v_align_point_facet = v_centroid_facet

    return (
        data_geometry_general,
        data_image_processing_general,
        [data_geometry_facet],
        [data_image_processing_facet],
        data_error,
    )


def process_undefined_geometry(
    mask_raw: ndarray,
    mask_keep_largest_area: bool,
    dist_optic_screen: float,
    orientation: SpatialOrientation,
    camera: Camera,
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for undefined deflectometry measurement

    Parameters
    ----------
    mask_raw : ndarray
        Raw calculated mask
    mask_keep_largest_area : bool
        To apply the "keep largest area" mask operation
    dist_optic_screen : float
        Optic centroid to screen distance, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calcualtions specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    if debug.debug_active:
        lt.debug('process_optics_geometry debug on, but is not yet supported for undefined mirrors.')

    # Define data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = cdc.CalculationDataGeometryFacet()
    data_image_processing_facet = cdc.CalculationImageProcessingFacet()
    data_error = None

    # Save mask raw
    data_image_processing_general.mask_raw = mask_raw

    # If enabled, keep only the largest mask area
    if mask_keep_largest_area:
        mask_raw_proc = ip.keep_largest_mask_area(mask_raw)
        mask_processed = np.logical_and(mask_raw, mask_raw_proc)
    else:
        mask_processed = mask_raw.copy()

    data_image_processing_facet.mask_processed = mask_processed

    # Find centroid of processed mask
    v_mask_centroid_image = ip.centroid_mask(mask_processed)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Find position of optic centroid in space
    v_cam_optic_cam = sp.t_from_distance(v_mask_centroid_image, dist_optic_screen, camera, orientation.v_cam_screen_cam)
    data_geometry_general.v_cam_optic_cam_exp = v_cam_optic_cam

    # Find orientation of optic
    r_cam_optic = sp.r_from_position(v_cam_optic_cam, orientation.v_cam_screen_cam)
    data_geometry_general.r_optic_cam_exp = r_cam_optic.inv()

    # Orient optic
    spatial_orientation = SpatialOrientation(orientation.r_cam_screen, orientation.v_cam_screen_cam)
    spatial_orientation.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

    # Calculate measure point pointing direction
    u_cam_measure_point_facet = Uxyz(spatial_orientation.v_cam_optic_optic.data)

    # Save processed optic data
    data_geometry_facet.u_cam_measure_point_facet = u_cam_measure_point_facet
    data_geometry_facet.measure_point_screen_distance = dist_optic_screen
    data_geometry_facet.spatial_orientation = spatial_orientation
    data_geometry_facet.v_align_point_facet = Vxyz((0, 0, 0))

    return (
        data_geometry_general,
        data_image_processing_general,
        [data_geometry_facet],
        [data_image_processing_facet],
        data_error,
    )


def process_multifacet_geometry(
    facet_data: list[DefinitionFacet],
    ensemble_data: DefinitionEnsemble,
    mask_raw: ndarray,
    v_meas_pt_ensemble: Vxyz,
    orientation: SpatialOrientation,
    camera: Camera,
    dist_optic_screen: float,
    params_geometry: ParamsOpticGeometry = ParamsOpticGeometry(),
    params_mask: ParamsMaskCalculation = ParamsMaskCalculation(),
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for multifacet deflectometry measurement

    Parameters
    ----------
    facet_data : DefinitionFacet
        Facet definition object
    ensemble_data : DefinitionEnsemble
        Ensemble definition object
    mask_raw : ndarray
        Raw calculated mask, shape (m, n) array of booleans
    v_meas_pt_ensemble : Vxyz
        Measure point lcoation on ensemble, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    dist_optic_screen : float
        Optic to screen distance, meters
    params_geometry : ParamsOpticGeometry, optional
        ParamsOpticGeometry object, by default ParamsOpticGeometry()
    params_mask : ParamsMaskCalculation, optional
        ParamsMaskCalculation object, by default ParamsMaskCalculation()
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calcualtions specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    if debug.debug_active:
        lt.debug('process_optics_geometry debug on.')

    # Get facet data
    v_facet_corners_facet: list = [
        f.v_facet_corners for f in facet_data
    ]  # Location of facet corners in facet coordinates
    v_facet_centroid_facet: list = [
        f.v_facet_centroid for f in facet_data
    ]  # Location of facet centroids in facet coordinates

    # Get ensemble data
    v_facet_locs_ensemble = (
        ensemble_data.v_facet_locations
    )  # Locations of facet origins relative to ensemble origin in ensemble coordinates
    r_facet_ensemble = ensemble_data.r_facet_ensemble  # Facet to ensemble rotation
    ensemble_corns_indices = ensemble_data.ensemble_perimeter  # [(facet_idx, facet_corner_idx), ...], integers
    v_centroid_ensemble = ensemble_data.v_centroid_ensemble  # Centroid of ensemble in ensemble coordinates

    # Get number of facets
    num_facets = len(v_facet_locs_ensemble)

    # Define data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = [cdc.CalculationDataGeometryFacet() for _ in range(num_facets)]
    data_image_processing_facet = [cdc.CalculationImageProcessingFacet() for _ in range(num_facets)]
    data_error = cdc.CalculationError()

    # Convert facet corners to ensemble coordinates
    v_ensemble_facet_corns = []
    for idx in range(num_facets):
        v_ensemble_facet_corns.append(
            v_facet_locs_ensemble[idx] + v_facet_corners_facet[idx].rotate(r_facet_ensemble[idx])
        )

    # Calculate ensemble corners in ensemble coordinates
    v_ensemble_corns_ensemble = []
    for idx_facet, idx_corn in ensemble_corns_indices:
        r_facet_ensemble_cur = r_facet_ensemble[idx_facet]
        v_ensemble_corns_ensemble.append(
            (
                v_facet_locs_ensemble[idx_facet]
                + v_facet_corners_facet[idx_facet][idx_corn].rotate(r_facet_ensemble_cur)
            ).data
        )
    v_ensemble_corns_ensemble = Vxyz(np.concatenate(v_ensemble_corns_ensemble, axis=1))

    # Concatenate all facet corners
    v_ensemble_facet_corns_all = Vxyz(np.concatenate([V.data for V in v_ensemble_facet_corns], axis=1))

    # Calculate raw mask
    data_image_processing_general.mask_raw = mask_raw

    # Plot mask
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw, cmap='gray')
        plt.title('Raw Mask')

    # Find edges of mask
    v_edges_image = ip.edges_from_mask(mask_raw)
    data_image_processing_general.v_edges_image = v_edges_image

    # Calculate centroid of mask
    v_mask_centroid_image = ip.centroid_mask(mask_raw)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Calculate expected position of ensemble centroid
    v_cam_ensemble_cent_cam_exp = sp.t_from_distance(
        v_mask_centroid_image, dist_optic_screen, camera, orientation.v_cam_screen_cam
    )
    data_geometry_general.v_cam_optic_centroid_cam_exp = v_cam_ensemble_cent_cam_exp

    # Calculate expected orientation of facet ensemble
    r_cam_ensemble_exp = sp.r_from_position(v_cam_ensemble_cent_cam_exp, orientation.v_cam_screen_cam)
    data_geometry_general.r_optic_cam_exp = r_cam_ensemble_exp.inv()

    # Calculate expected position of ensemble origin
    v_cam_ensemble_cam_exp = v_cam_ensemble_cent_cam_exp - v_centroid_ensemble.rotate(r_cam_ensemble_exp.inv())
    data_geometry_general.v_cam_optic_cam_exp = v_cam_ensemble_cam_exp

    # Project perimeter points
    v_ensemble_corners_exp_image = camera.project(
        v_ensemble_corns_ensemble, r_cam_ensemble_exp.inv(), v_cam_ensemble_cam_exp
    )
    loop_ensemble_exp = LoopXY.from_vertices(v_ensemble_corners_exp_image)
    data_image_processing_general.loop_optic_image_exp = loop_ensemble_exp

    # Plot expected perimeter points
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        _plot_labeled_points(v_ensemble_corners_exp_image)
        plt.title('Expected Perimeter Points')

    # Refine perimeter points
    args = [
        params_geometry.perimeter_refine_axial_search_dist,
        params_geometry.perimeter_refine_perpendicular_search_dist,
    ]
    loop_ensemble_image_refine = ip.refine_mask_perimeter(loop_ensemble_exp, v_edges_image, *args)
    data_image_processing_general.loop_optic_image_refine = loop_ensemble_image_refine

    # Plot refined perimeter points
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        _plot_labeled_points(loop_ensemble_image_refine.vertices)
        plt.title('Refined Perimeter Points')

    # Refine ensemble position/orientation with perimeter points
    r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1 = sp.calc_rt_from_img_pts(
        loop_ensemble_image_refine.vertices, v_ensemble_corns_ensemble, camera
    )
    data_geometry_general.r_optic_cam_refine_1 = r_ensemble_cam_refine_1
    data_geometry_general.v_cam_optic_cam_refine_1 = v_cam_ensemble_cam_refine_1

    # Calculate refined measure point vector in optic coordinates
    v_meas_pt_ensemble_cam_refine_1 = v_meas_pt_ensemble.rotate(r_ensemble_cam_refine_1)

    # Calculate expected location of all facet corners and centroids
    v_facet_corners_image_exp = [
        camera.project(P, r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1) for P in v_ensemble_facet_corns
    ]
    v_uv_facet_cent_exp = camera.project(v_facet_locs_ensemble, r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1)
    for idx in range(num_facets):
        data_image_processing_facet[idx].v_facet_corners_image_exp = v_facet_corners_image_exp[idx]
        data_image_processing_facet[idx].v_facet_centroid_image_exp = v_uv_facet_cent_exp[idx]

    # Refine facet corners
    args = [
        params_geometry.facet_corns_refine_step_length,
        params_geometry.facet_corns_refine_perpendicular_search_dist,
        params_geometry.facet_corns_refine_frac_keep,
    ]
    loops_facets_refined: list[LoopXY] = []
    for idx in range(num_facets):
        loop = ip.refine_facet_corners(v_facet_corners_image_exp[idx], v_uv_facet_cent_exp[idx], v_edges_image, *args)
        loops_facets_refined.append(loop)
        data_image_processing_facet[idx].loop_facet_image_refine = loop

        # Plot refined perimeter points
        if debug.debug_active:
            if idx == 0:
                fig = plt.figure()
                debug.figures.append(fig)
                plt.imshow(mask_raw)
                plt.title('Refined Facet Corners')
            loop.draw()

    # Concatenate all refined facet corners
    v_facet_corners_all_image_refine = []
    for loop in loops_facets_refined:
        v_facet_corners_all_image_refine.append(loop.vertices.data)
    v_facet_corners_all_image_refine = Vxy(np.concatenate(v_facet_corners_all_image_refine, axis=1))

    # Calculate fitted masks
    mask_fitted = np.zeros(mask_raw.shape + (num_facets,), dtype=bool)
    vx = np.arange(mask_raw.shape[1])
    vy = np.arange(mask_raw.shape[0])
    for idx in range(num_facets):
        mask_fitted[..., idx] = loops_facets_refined[idx].as_mask(vx, vy)
        data_image_processing_facet[idx].mask_fitted = mask_fitted[..., idx]

    # Calculate processed masks
    mask_processed = np.ones(mask_raw.shape + (num_facets,), dtype=bool)
    mask_processed *= mask_raw[..., np.newaxis]
    mask_processed = np.logical_and(mask_processed, mask_fitted)
    for idx in range(num_facets):
        mask = mask_processed[..., idx]
        # If enabled, keep largest mask area (fill holes) for each individual facet
        if params_mask.keep_largest_area:
            mask = ip.keep_largest_mask_area(mask)
        data_image_processing_facet[idx].mask_processed = mask

    # Refine R/T with all refined facet corners
    r_ensemble_cam_refine_2, v_cam_ensemble_cam_refine_2 = sp.calc_rt_from_img_pts(
        v_facet_corners_all_image_refine, v_ensemble_facet_corns_all, camera
    )
    r_cam_ensemble_refine_2 = r_ensemble_cam_refine_2.inv()
    data_geometry_general.r_optic_cam_refine_2 = r_ensemble_cam_refine_2
    data_geometry_general.v_cam_optic_cam_refine_2 = v_cam_ensemble_cam_refine_2

    # Calculate refined measure point location in optic coordinates vector
    v_meas_pt_ensemble_cam_refine_2 = v_meas_pt_ensemble.rotate(r_ensemble_cam_refine_2)

    # Refine T with measured distance
    v_cam_ensemble_cam_refine_3 = sp.refine_v_distance(
        v_cam_ensemble_cam_refine_2, dist_optic_screen, orientation.v_cam_screen_cam, v_meas_pt_ensemble_cam_refine_2
    )
    data_geometry_general.v_cam_optic_cam_refine_3 = v_cam_ensemble_cam_refine_3

    # Calculate error 1 (R/T calculated using only ensemble perimeter points)
    error_dist_optic_screen_1 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_1 + v_meas_pt_ensemble_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_1 = error_dist_optic_screen_1
    error_reprojection_1 = sp.reprojection_error(
        camera,
        v_ensemble_corns_ensemble,
        loop_ensemble_image_refine.vertices,
        r_ensemble_cam_refine_1,
        v_cam_ensemble_cam_refine_1,
    )
    data_error.error_reprojection_1 = error_reprojection_1

    # Calculate error 2 (R/T calculated using all facet corners)
    error_dist_optic_screen_2 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_2 + v_meas_pt_ensemble_cam_refine_2, dist_optic_screen
    )
    data_error.error_dist_optic_screen_2 = error_dist_optic_screen_2
    error_reprojection_2 = sp.reprojection_error(
        camera,
        v_ensemble_facet_corns_all,
        v_facet_corners_all_image_refine,
        r_ensemble_cam_refine_2,
        v_cam_ensemble_cam_refine_2,
    )
    data_error.error_reprojection_2 = error_reprojection_2

    # Calculate error 3 (T refined using measured distance)
    error_dist_optic_screen_3 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_3 + v_meas_pt_ensemble_cam_refine_2, dist_optic_screen
    )
    data_error.error_dist_optic_screen_3 = error_dist_optic_screen_3
    error_reprojection_3 = sp.reprojection_error(
        camera,
        v_ensemble_facet_corns_all,
        v_facet_corners_all_image_refine,
        r_cam_ensemble_refine_2,
        v_cam_ensemble_cam_refine_3,
    )
    data_error.error_reprojection_3 = error_reprojection_3

    # Spatially orient facets and the setup
    for idx in range(num_facets):
        # Calculate ensemble to facet vector in camera coordinates
        v_ensemble_facet_cam = v_facet_locs_ensemble[idx].rotate(r_ensemble_cam_refine_2)

        # Instantiate spatial orientation object
        facet_ori = SpatialOrientation(orientation.r_cam_screen, orientation.v_cam_screen_cam)

        # Orient facet
        r_cam_facet = r_facet_ensemble[idx].inv() * r_cam_ensemble_refine_2
        v_cam_facet_cam = v_cam_ensemble_cam_refine_3 + v_ensemble_facet_cam
        facet_ori.orient_optic_cam(r_cam_facet, v_cam_facet_cam)

        # Calculate facet measure point pointing direction (measure point defined as facet centroid)
        v_cam_meas_pt_facet = facet_ori.v_cam_optic_optic + v_facet_centroid_facet[idx]

        # Calculate facet measure point to screen distance
        v_cam_screen_optic = facet_ori.v_cam_screen_cam.rotate(facet_ori.r_cam_optic)
        dist = (v_cam_meas_pt_facet - v_cam_screen_optic).magnitude()[0]

        data_geometry_facet[idx].u_cam_measure_point_facet = Uxyz(v_cam_meas_pt_facet.data)
        data_geometry_facet[idx].measure_point_screen_distance = dist
        data_geometry_facet[idx].spatial_orientation = facet_ori
        data_geometry_facet[idx].v_align_point_facet = v_facet_centroid_facet[idx]

    return (
        data_geometry_general,
        data_image_processing_general,
        data_geometry_facet,
        data_image_processing_facet,
        data_error,
    )


def _plot_labeled_points(pts: Vxy) -> None:
    """Plots labeled points on axis for debugging"""
    plt.scatter(*pts.data)
    for idx, pt in enumerate(pts):
        plt.text(*pt.data, idx, color='k')
