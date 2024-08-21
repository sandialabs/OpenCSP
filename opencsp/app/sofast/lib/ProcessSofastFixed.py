"""Class that handles the processing of fixed pattern deflectometry data.
"""

from typing import Literal

import cv2 as cv
import numpy as np
from numpy import ndarray

from opencsp.app.sofast.lib.BlobIndex import BlobIndex
import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ParamsSofastFixed import ParamsSofastFixed
import opencsp.app.sofast.lib.process_optics_geometry as pr
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.tool.hdf5_tools import HDF5_SaveAbstract
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastFixed(HDF5_SaveAbstract):
    """Fixed Pattern Deflectrometry data processing class"""

    def __init__(
        self, orientation: SpatialOrientation, camera: Camera, fixed_pattern_dot_locs: DotLocationsFixedPattern
    ) -> 'ProcessSofastFixed':
        """Instantiates class

        Parameters
        ----------
        orientation : SpatialOrientaton
            SpatialOrientation object
        camera : Camera
            Camera object
        fixed_pattern_dot_locs : DotLocationsFixedPattern
            Image projection dictionary
        """
        self.orientation = orientation
        self.camera = camera
        self.fixed_pattern_dot_locs = fixed_pattern_dot_locs
        self.params = ParamsSofastFixed()
        self.optic_type: Literal['single_facet', 'multi_facet', 'undefined'] = None
        self.num_facets: int = None

        # Measurement data
        self.measurement: MeasurementSofastFixed

        # Define blob detector
        self.blob_detector: cv.SimpleBlobDetector_Params = cv.SimpleBlobDetector_Params()
        self.blob_detector.minDistBetweenBlobs = 2
        self.blob_detector.filterByArea = True
        self.blob_detector.minArea = 3
        self.blob_detector.maxArea = 30
        self.blob_detector.filterByCircularity = False
        self.blob_detector.filterByConvexity = False
        self.blob_detector.filterByInertia = False

        # Calculations
        self.data_facet: list[DefinitionFacet]
        self.data_ensemble: DefinitionEnsemble
        self.data_surface: list[Surface2DAbstract]
        self.data_ensemble: DefinitionEnsemble
        self.data_slope_solver: list[SlopeSolverData]
        self.data_geometry_general: cdc.CalculationDataGeometryGeneral
        self.data_image_proccessing_general: cdc.CalculationImageProcessingGeneral
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet]
        self.data_image_processing_facet: list[cdc.CalculationImageProcessingFacet]
        self.data_error: cdc.CalculationError
        self.blob_index: BlobIndex
        self.slope_solver: list[SlopeSolver]

    def find_blobs(self, pts_known: Vxy, xys_known: tuple[tuple[int, int]]) -> BlobIndex:
        """Finds blobs in image

        Parameters
        ----------
        pts_known : Vxy
            Length N, xy pixel location of known point(s) with known xy dot index locations
        xys_known : tuple[tuple[int]]
            Length N integer xy dot indices

        NOTE: N=number of facets
        """
        pts_blob = ip.detect_blobs(self.measurement.image, self.blob_detector)

        # Index blobs
        blob_index = BlobIndex(pts_blob, *self.fixed_pattern_dot_locs.dot_extent)
        blob_index.search_thresh = self.params.blob_search_thresh
        blob_index.search_perp_axis_ratio = self.params.search_perp_axis_ratio
        for pt_known, xy_known in zip(pts_known, xys_known):
            blob_index.run(pt_known, xy_known[0], xy_known[1])

        return blob_index

    def _calculate_mask(self) -> ndarray:
        # Calculate mask
        im_dark = self.measurement.image * 0
        images = np.concatenate((im_dark[..., None], self.measurement.image[..., None]), axis=2)
        params = [
            self.params.mask.hist_thresh,
            self.params.mask.filt_width,
            self.params.mask.filt_thresh,
            self.params.mask.thresh_active_pixels,
        ]
        mask = ip.calc_mask_raw(images, *params)

        if self.optic_type == 'multi_facet':
            lt.warn(
                '"keep_largest_area" mask processing option cannot be used '
                'for multifacet ensembles. This will be turned off.'
            )
            self.params.mask.keep_largest_area = False
        elif self.params.mask.keep_largest_area:
            mask = ip.keep_largest_mask_area(mask)

        return mask

    def load_measurement_data(self, measurement: MeasurementSofastFixed) -> None:
        """Saves measurement data in class

        Parameters
        ----------
        measurement: MeasurementSofastFixed
            Fixed pattern measurement object
        """
        self.measurement = measurement

    def _process_optic_singlefacet_geometry(self, blob_index: BlobIndex, mask_raw: np.ndarray) -> dict:
        # Process optic geometry (find mask corners, etc.)
        (
            self.data_geometry_general,
            self.data_image_proccessing_general,
            self.data_geometry_facet,  # list
            self.data_image_processing_facet,  # list
            self.data_error,
        ) = pr.process_singlefacet_geometry(
            self.data_facet[0],
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.measurement.dist_optic_screen,
            self.orientation,
            self.camera,
            self.params.geometry,
            self.params.debug_geometry,
        )

        # Get image points and blob indices
        pts_image, pts_index_xy = blob_index.get_data()

        # Define optic orientation w.r.t. camera
        rot_optic_cam = self.data_geometry_general.r_optic_cam_refine_1
        v_cam_optic_cam = self.data_geometry_general.v_cam_optic_cam_refine_2
        u_cam_measure_point_facet = self.data_geometry_facet[0].u_cam_measure_point_facet

        # Get screen/camera poses
        rot_cam_optic = rot_optic_cam.inv()
        rot_optic_screen = self.orientation.r_cam_screen * rot_optic_cam
        rot_screen_optic = rot_optic_screen.inv()

        v_optic_cam_optic = -v_cam_optic_cam.rotate(rot_cam_optic)
        v_cam_screen_optic = self.orientation.v_cam_screen_cam.rotate(rot_cam_optic)
        v_optic_screen_optic = v_optic_cam_optic + v_cam_screen_optic

        # Calculate xyz screen points
        v_screen_points_screen = self.fixed_pattern_dot_locs.xy_indices_to_screen_coordinates(pts_index_xy)
        v_screen_points_facet = v_optic_screen_optic + v_screen_points_screen.rotate(rot_screen_optic)

        # Calculate active pixel pointing
        u_pixel_pointing_cam = self.camera.vector_from_pixel(pts_image)
        u_pixel_pointing_facet = u_pixel_pointing_cam.rotate(rot_cam_optic).as_Vxyz()

        # Update debug data
        self.params.debug_slope_solver.optic_data = self.data_facet[0]

        # Construct surface kwargs
        return {
            'v_optic_cam_optic': v_optic_cam_optic,
            'u_active_pixel_pointing_optic': u_pixel_pointing_facet,
            'u_measure_pixel_pointing_optic': u_cam_measure_point_facet,
            'v_screen_points_facet': v_screen_points_facet,
            'v_optic_screen_optic': v_optic_screen_optic,
            'v_align_point_optic': self.data_facet[0].v_facet_centroid,
            'dist_optic_screen': self.measurement.dist_optic_screen,
            'debug': self.params.debug_slope_solver,
            'surface': self.data_surface[0],
        }

    def _process_optic_multifacet_geometry(self, blob_index: BlobIndex, mask_raw: np.ndarray) -> list[dict]:
        # Process optic geometry (find mask corners, etc.)
        (
            self.data_geometry_general,
            self.data_image_proccessing_general,
            self.data_geometry_facet,  # list
            self.data_image_processing_facet,  # list
            self.data_error,
        ) = pr.process_multifacet_geometry(
            self.data_facet,
            self.data_ensemble,
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.orientation,
            self.camera,
            self.measurement.dist_optic_screen,
            self.params.geometry,
            self.params.debug_geometry,
        )

        kwargs_list = []
        for idx_facet in range(self.num_facets):
            # Get image points and blob indices
            pts_image, pts_index_xy = blob_index.get_data()

            # Define optic orientation w.r.t. camera
            rot_facet_ensemble = self.data_ensemble.r_facet_ensemble[idx_facet]
            rot_ensemble_cam = self.data_geometry_general.r_optic_cam_refine_2
            rot_facet_cam = rot_ensemble_cam * rot_facet_ensemble

            v_cam_ensemble_cam = self.data_geometry_general.v_cam_optic_cam_refine_3
            v_ensemble_facet_ensemble = self.data_ensemble.v_facet_locations[idx_facet]
            v_ensemble_facet_cam = v_ensemble_facet_ensemble.rotate(rot_ensemble_cam)
            v_cam_facet_cam = v_cam_ensemble_cam + v_ensemble_facet_cam

            u_cam_measure_point_facet = self.data_geometry_facet[idx_facet].u_cam_measure_point_facet

            # Get screen/camera poses
            rot_cam_facet = rot_facet_cam.inv()
            rot_facet_screen = self.orientation.r_cam_screen * rot_facet_cam
            rot_screen_facet = rot_facet_screen.inv()

            v_facet_cam_facet = -v_cam_facet_cam.rotate(rot_cam_facet)
            v_cam_screen_facet = self.orientation.v_cam_screen_cam.rotate(rot_cam_facet)
            v_facet_screen_facet = v_facet_cam_facet + v_cam_screen_facet

            # Calculate xyz screen points
            v_screen_points_screen = self.fixed_pattern_dot_locs.xy_indices_to_screen_coordinates(pts_index_xy)
            v_screen_points_facet = v_facet_screen_facet + v_screen_points_screen.rotate(rot_screen_facet)

            # Calculate active pixel pointing
            u_pixel_pointing_cam = self.camera.vector_from_pixel(pts_image)
            u_pixel_pointing_facet = u_pixel_pointing_cam.rotate(rot_cam_facet).as_Vxyz()

            # Update debug data
            self.params.debug_slope_solver.optic_data = self.data_facet[idx_facet]

            # Construct list of surface kwargs
            kwargs_list.append(
                {
                    'v_optic_cam_optic': v_facet_cam_facet,
                    'u_active_pixel_pointing_optic': u_pixel_pointing_facet,
                    'u_measure_pixel_pointing_optic': u_cam_measure_point_facet,
                    'v_screen_points_facet': v_screen_points_facet,
                    'v_optic_screen_optic': v_facet_screen_facet,
                    'v_align_point_optic': self.data_geometry_facet[idx_facet].v_align_point_facet,
                    'dist_optic_screen': self.data_geometry_facet[idx_facet].measure_point_screen_distance,
                    'debug': self.params.debug_slope_solver,
                    'surface': self.data_surface[idx_facet],
                }
            )
        return kwargs_list

    def process_single_facet_optic(
        self, data_facet: DefinitionFacet, surface: Surface2DAbstract, pt_known: Vxy, xy_known: tuple[int, int]
    ) -> None:
        """Processes single facet optic. Saves data to self.data_slope_solver

        Parameters
        ----------
        data_facet : DefinitionFacet objec
            Facet definition
        surface : Surface2DAbstract
            Surface 2d class
        pt_known : Vxy
            Length 1, xy pixel location of known point(s) with known xy dot index locations
        xy_known : tuple[int, int]
            Integer xy dot indices
        """

        # Check inputs
        if len(pt_known) != 1:
            lt.error_and_raise(
                ValueError, f'Only 1 pt_known can be given for single facet processing but {len(pt_known):d} were given'
            )

        self.optic_type = 'single_facet'
        self.num_facets = 1
        self.data_facet = [data_facet.copy()]
        self.data_surface = [surface]

        # Find blobs
        self.blob_index = self.find_blobs(pt_known, (xy_known,))

        # Calculate mask
        mask_raw = self._calculate_mask()

        # Generate geometry and slope solver inputs
        kwargs = self._process_optic_singlefacet_geometry(self.blob_index, mask_raw)

        # Calculate slope
        slope_solver = SlopeSolver(**kwargs)
        slope_solver.fit_surface()
        slope_solver.solve_slopes()
        self.slope_solver = slope_solver
        self.data_slope_solver = [self.slope_solver.get_data()]

    def process_multi_facet_optic(
        self,
        data_facet: list[DefinitionFacet],
        surfaces: list[Surface2DAbstract],
        data_ensemble: DefinitionEnsemble,
        pts_known: Vxy,
        xys_known: tuple[tuple[int, int]],
    ) -> None:
        """Processes multi facet optic. Saves data to self.data_slope_solver

        Parameters
        ----------
        data_facet : list[DefinitionFacet]
            List of facet data objects.
        data_ensemble : DefinitionEnsemble
            Ensemble data object.
        surfaces : list[Surface2dAbstract]
            List of surface type definitions
        pts_known : Vxy
            Length N, xy pixel location of known point(s) with known xy dot index locations
        xys_known : tuple[tuple[int, int]]
            List of N integer xy dot indices corresponding to pts_known

        NOTE: N=number of facets
        """

        # Check inputs
        if len(data_facet) != len(surfaces) != len(pts_known) != len(xys_known):
            lt.error_and_raise(
                ValueError,
                'Length of data_facet does not equal length of data_surface'
                + f'data_facet={len(data_facet)}, surface_data={len(surfaces)}, '
                + f'pts_known={len(pts_known)}, xys_known={len(xys_known)}',
            )

        self.optic_type = 'multi_facet'
        self.num_facets = len(data_facet)
        self.data_facet = [d.copy() for d in data_facet]
        self.data_ensemble = data_ensemble.copy()
        self.data_surface = surfaces

        # Find blobs
        self.blob_index = self.find_blobs(pts_known, xys_known)

        # Calculate mask
        mask_raw = self._calculate_mask()

        # Generate geometry and slope solver inputs
        kwargs_list = self._process_optic_multifacet_geometry(self.blob_index, mask_raw)

        # Calculate slope
        self.slope_solver = []
        self.data_slope_solver = []
        for kwargs in kwargs_list:
            slope_solver = SlopeSolver(**kwargs)
            slope_solver.fit_surface()
            slope_solver.solve_slopes()
            self.slope_solver.append(slope_solver)
            self.data_slope_solver.append(self.slope_solver.get_data())

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given HDF5 file. Data is stored in CalculationsFixedPattern/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        # Sofast input parameters
        self.params.save_to_hdf(file, f'{prefix:s}DataSofastInput/')
        for idx, (data_facet, data_surface) in enumerate(zip(self.data_facet, self.data_surface)):
            data_surface.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx:3d}/')
            data_facet.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx:3d}/')

        # General
        self.data_error.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_geometry_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_image_proccessing_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')

        # Calculations
        for idx_facet in range(self.num_facets):
            self.data_slope_solver[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:3d}/'
            )
            self.data_geometry_facet[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:3d}/'
            )
            self.data_image_processing_facet[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:3d}/'
            )

        lt.info(f'SofastFixed data saved to: {file:s} with prefix: {prefix:s}')

    def get_mirror(
        self, interpolation_type: Literal['given', 'bilinear', 'clough_tocher', 'nearest'] = 'nearest'
    ) -> MirrorPoint:
        """Returns mirror object with slope data"""
        if self.optic_type in ['single_facet', 'undefined']:
            v_surf_pts = self.data_slope_solver[0].v_surf_points_facet
            v_normals_data = np.ones((3, len(v_surf_pts)))
            v_normals_data[:2, :] = self.data_slope_solver[0].slopes_facet_xy
            v_normals_data[:2, :] *= -1
            v_normals = Uxyz(v_normals_data)
            shape = RegionXY.from_vertices(self.data_facet[0].v_facet_corners.projXY())
            return MirrorPoint(v_surf_pts, v_normals, shape, interpolation_type)
        elif self.optic_type == 'multi_facet':
            # TODO: fill in for multifacet
            pass
