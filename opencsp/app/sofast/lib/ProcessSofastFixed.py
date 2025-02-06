"""Class that handles the processing of fixed pattern deflectometry data.
"""

import cv2 as cv
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.BlobIndex import BlobIndex, DebugBlobIndex
from opencsp.app.sofast.lib.calculation_data_classes import CalculationBlobAssignment
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ParamsSofastFixed import ParamsSofastFixed
from opencsp.app.sofast.lib.ProcessSofastAbstract import ProcessSofastAbstract
import opencsp.app.sofast.lib.process_optics_geometry as pr
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastFixed(ProcessSofastAbstract):
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
        super().__init__()

        self.orientation = orientation
        self.camera = camera
        self.measurement: MeasurementSofastFixed
        self.fixed_pattern_dot_locs = fixed_pattern_dot_locs
        self.params: ParamsSofastFixed = ParamsSofastFixed()

        # Define blob detector
        self.blob_detector: cv.SimpleBlobDetector_Params = cv.SimpleBlobDetector_Params()
        self.blob_detector.minDistBetweenBlobs = 2
        self.blob_detector.filterByArea = True
        self.blob_detector.minArea = 3
        self.blob_detector.maxArea = 30
        self.blob_detector.filterByCircularity = False
        self.blob_detector.filterByConvexity = False
        self.blob_detector.filterByInertia = False

        # Instantiate other data containers
        self.slope_solvers: list[SlopeSolver] = None
        self.blob_index: list[BlobIndex] = None
        self.debug_blob_index: DebugBlobIndex = DebugBlobIndex()
        self.data_calculation_blob_assignment: list[CalculationBlobAssignment] = None

    def _find_blobs(self, pt_known: Vxy, xy_known: tuple[int, int], loop: LoopXY = None) -> BlobIndex:
        # Finds and assigns indices to blobs in image
        # Parameters:
        # pt_known, Vxy, xy pixel location of known point in image, units of pixels.
        # xy_known, tuple[int, int], length 2 tuple. Integer xy dot indices
        # loop, LoopXY | None, only consider points inside this loop. If None, consider all points.
        # Returns:
        # BlobIndex class with found/assigned blobs

        # Find all blobs in image
        pts_blob = ip.detect_blobs(self.measurement.image, self.blob_detector)

        # Filter blobs if loop is given
        if loop is not None:
            mask = loop.is_inside(pts_blob)
            pts_blob = pts_blob[mask]

        # Index blobs
        blob_index = BlobIndex(pts_blob, *self.fixed_pattern_dot_locs.dot_extent)
        blob_index.search_thresh = self.params.blob_search_thresh
        blob_index.search_perp_axis_ratio = self.params.search_perp_axis_ratio
        blob_index.debug = self.debug_blob_index
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

        return mask

    def load_measurement_data(self, measurement: MeasurementSofastFixed) -> None:
        """Saves measurement data in class

        Parameters
        ----------
        measurement: MeasurementSofastFixed
            Fixed pattern measurement object
        """
        self.measurement = measurement

    def _process_optic_common_geometry(
        self, rot_optic_cam: Rotation, v_cam_optic_cam: Vxyz, idx_facet: int
    ) -> tuple[dict, CalculationBlobAssignment]:
        # Get blob index data for current facet
        pts_image, pts_index_xy = self.blob_index[idx_facet].get_data()

        # Get measure point unit vector
        u_cam_measure_point_facet = self.data_geometry_facet[idx_facet].u_cam_measure_point_facet

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

        # Check for nans returning from screen point calculation
        nan_mask = np.isnan(v_screen_points_screen.data).sum(0).astype(bool)
        active_point_mask: np.ndarray = np.logical_not(nan_mask)

        # Remove nans if any present
        if np.any(nan_mask):
            lt.warn(
                'ProcessSofastFixed._process_optic_common_geometry(): '
                f'{nan_mask.sum():d} / {nan_mask.size:d} screen points are undefined '
                f'for facet {idx_facet:d}. These data points will be removed.'
            )
            # Remove nan data points from screen points
            pts_image = pts_image[active_point_mask]
            pts_index_xy = pts_index_xy[active_point_mask]
            v_screen_points_facet = v_screen_points_facet[active_point_mask]

        # Make 2d mask of active points (w.r.t. BlobIndex internal 2d arrays)
        x_idx_mat, y_idx_mat = self.blob_index[idx_facet].pts_index_to_mat_index(pts_index_xy)
        active_point_mask_mat = np.zeros(self.blob_index[idx_facet].shape_yx_data_mat, dtype=bool)
        active_point_mask_mat[y_idx_mat, x_idx_mat] = True

        # Save Sofast Fixed calculation parameters (specific to Sofast Fixed calculations)
        data_calculation_blob_assignment = CalculationBlobAssignment(pts_image, pts_index_xy, active_point_mask_mat)

        # Calculate active pixel pointing
        u_pixel_pointing_cam = self.camera.vector_from_pixel(pts_image)
        u_pixel_pointing_facet = u_pixel_pointing_cam.rotate(rot_cam_optic).as_Vxyz()

        # Update debug data
        self.params.debug_slope_solver.optic_data = self.data_facet_def[idx_facet]

        # Construct surface kwargs
        kwargs = {
            'v_optic_cam_optic': v_optic_cam_optic,
            'u_active_pixel_pointing_optic': u_pixel_pointing_facet,
            'u_measure_pixel_pointing_optic': u_cam_measure_point_facet,
            'v_screen_points_facet': v_screen_points_facet,
            'v_optic_screen_optic': v_optic_screen_optic,
            'v_align_point_optic': self.data_geometry_facet[idx_facet].v_align_point_facet,
            'dist_optic_screen': self.data_geometry_facet[idx_facet].measure_point_screen_distance,
            'debug': self.params.debug_slope_solver,
            'surface': self.data_surfaces[idx_facet],
        }

        return kwargs, data_calculation_blob_assignment

    def _process_optic_singlefacet_geometry(self) -> dict:
        # Define optic orientation w.r.t. camera
        rot_optic_cam = self.data_geometry_general.r_optic_cam_refine_1
        v_cam_optic_cam = self.data_geometry_general.v_cam_optic_cam_refine_2

        # Calculate optic geometry for single facet (facet index = 0)
        kwargs, data_calculation_blob_assignment = self._process_optic_common_geometry(
            rot_optic_cam, v_cam_optic_cam, idx_facet=0
        )

        # Save blob assignment data
        self.data_calculation_blob_assignment = [data_calculation_blob_assignment]

        return kwargs

    def _process_optic_multifacet_geometry(self) -> list[dict]:
        # Clear blob assignment calculations data container
        self.data_calculation_blob_assignment = []

        # Loop through all facets and calcualte geometry data
        kwargs_list = []
        for idx_facet in range(self.num_facets):
            # Define optic orientation w.r.t. camera
            rot_facet_ensemble = self.data_ensemble_def.r_facet_ensemble[idx_facet]
            rot_ensemble_cam = self.data_geometry_general.r_optic_cam_refine_2
            rot_facet_cam = rot_ensemble_cam * rot_facet_ensemble

            v_cam_ensemble_cam = self.data_geometry_general.v_cam_optic_cam_refine_3
            v_ensemble_facet_ensemble = self.data_ensemble_def.v_facet_locations[idx_facet]
            v_ensemble_facet_cam = v_ensemble_facet_ensemble.rotate(rot_ensemble_cam)
            v_cam_facet_cam = v_cam_ensemble_cam + v_ensemble_facet_cam

            # Calculate optic geometry for all facets
            kwargs, data_calculation_blob_assignment = self._process_optic_common_geometry(
                rot_facet_cam, v_cam_facet_cam, idx_facet
            )
            kwargs_list.append(kwargs)
            self.data_calculation_blob_assignment.append(data_calculation_blob_assignment)

        return kwargs_list

    def process_single_facet_optic(
        self, data_facet_def: DefinitionFacet, surface: Surface2DAbstract, pt_known: Vxy, xy_known: tuple[int, int]
    ) -> None:
        """Processes single facet optic. Saves data to self.data_calculation_facet

        Parameters
        ----------
        data_facet_def : DefinitionFacet objec
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

        self.optic_type = 'single'
        self.num_facets = 1
        self.data_facet_def = [data_facet_def.copy()]
        self.data_surfaces = [surface]

        # Calculate mask
        mask_raw = self._calculate_mask()

        # If enabled, fill in holes in mask area
        if self.params.mask.keep_largest_area:
            mask_raw = ip.keep_largest_mask_area(mask_raw)

        # Process optic geometry (find mask corners, etc.)
        (
            self.data_geometry_general,
            self.data_image_processing_general,
            self.data_geometry_facet,  # list
            self.data_image_processing_facet,  # list
            self.data_error,
        ) = pr.process_singlefacet_geometry(
            self.data_facet_def[0],  # 0 because there is only one facet
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.measurement.dist_optic_screen,
            self.orientation,
            self.camera,
            self.params.geometry,
            self.params.debug_geometry,
        )

        # Find blobs
        self.blob_index = [self._find_blobs(pt_known, xy_known)]

        # Generate geometry and slope solver inputs
        kwargs = self._process_optic_singlefacet_geometry()

        # Calculate slope
        slope_solver = SlopeSolver(**kwargs)
        slope_solver.fit_surface()
        slope_solver.solve_slopes()
        self.slope_solvers = [slope_solver]
        self.data_calculation_facet = [slope_solver.get_data()]

    def process_multi_facet_optic(
        self,
        data_facet_def: list[DefinitionFacet],
        surfaces: list[Surface2DAbstract],
        data_ensemble_def: DefinitionEnsemble,
        pts_known: Vxy,
        xys_known: tuple[tuple[int, int]],
    ) -> None:
        """Processes multi facet optic. Saves data to self.data_calculation_facet

        Parameters
        ----------
        data_facet_def : list[DefinitionFacet]
            List of facet data objects.
        data_ensemble_def : DefinitionEnsemble
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
        if len(data_facet_def) != len(surfaces) != len(pts_known) != len(xys_known):
            lt.error_and_raise(
                ValueError,
                'ProcessSofastFixed: Length of data_facet_def does not equal length of data_surfaces'
                + f'data_facet_def={len(data_facet_def)}, surface_data={len(surfaces)}, '
                + f'pts_known={len(pts_known)}, xys_known={len(xys_known)}',
            )

        self.optic_type = 'multi'
        self.num_facets = len(data_facet_def)
        self.data_facet_def = [d.copy() for d in data_facet_def]
        self.data_ensemble_def = data_ensemble_def.copy()
        self.data_surfaces = surfaces

        # Calculate mask
        mask_raw = self._calculate_mask()

        # Process optic geometry (find mask corners, etc.)
        (
            self.data_geometry_general,
            self.data_image_processing_general,
            self.data_geometry_facet,  # list
            self.data_image_processing_facet,  # list
            self.data_error,
        ) = pr.process_multifacet_geometry(
            self.data_facet_def,
            self.data_ensemble_def,
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.orientation,
            self.camera,
            self.measurement.dist_optic_screen,
            self.params.geometry,
            self.params.mask,
            self.params.debug_geometry,
        )

        # Find blobs
        self.blob_index: list[BlobIndex] = []

        for idx_facet, geom in enumerate(self.data_image_processing_facet):
            loop = geom.loop_facet_image_refine
            self.debug_blob_index.name = f' - Facet {idx_facet:d}'
            self.blob_index.append(self._find_blobs(pts_known[idx_facet], xys_known[idx_facet], loop))

        # Generate geometry and slope solver inputs
        kwargs_list = self._process_optic_multifacet_geometry()

        # Calculate slope
        self.slope_solvers = []
        self.data_calculation_facet = []
        for kwargs in kwargs_list:
            slope_solver = SlopeSolver(**kwargs)
            slope_solver.fit_surface()
            slope_solver.solve_slopes()
            self.slope_solvers.append(slope_solver)
            self.data_calculation_facet.append(slope_solver.get_data())

        # Calculate facet pointing
        self._calculate_facet_pointing()

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Save default data
        super().save_to_hdf(file, prefix)

        # Calculations, one per facet
        for idx_facet in range(self.num_facets):
            # Save facet blob index calcluations (specific to Sofast Fixed)
            if self.data_calculation_blob_assignment is not None:
                self.data_calculation_blob_assignment[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
