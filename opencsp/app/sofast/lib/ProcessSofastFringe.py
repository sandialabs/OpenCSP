"""Controls the processing of Sofast measurement data
to calculate surface slopes.
"""

from typing import Literal

import numpy as np

import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ParamsSofastFringe import ParamsSofastFringe
import opencsp.app.sofast.lib.process_optics_geometry as po
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import HDF5_SaveAbstract
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastFringe(HDF5_SaveAbstract):
    """Class that processes measurement data captured by a SOFAST
    system. Computes optic surface slope and saves data to HDF5 format.

    Processing types
    ----------------
    Sofast can process three optic types:
    - Undefined optic - Sofast.process_optic_undefined(surface_data)
    - Single facet optic - Sofast.process_optic_singlefacet(facet_data, surface_data)
    - Multi-facet ensemble - Sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    Data classes
    ------------
    - surface_data : dict
        Defines surface fitting parameters. See SlopeSolver documentation for more information.
    - params : SofastParams
        Parameters specific to Sofast calculations (facet mask calculation, etc.)
    - params.geometry_params : ParamsOpticGeometry
        Parameters specific to finding boundaries of optics, etc.

    Internal Data Storage
    ---------------------
    Data is stored in the following fields:
        - data_ensemble_def - copy of ensemble definition user input
        - data_facet_def - copy of facet definition user input
        - data_surface_params - copy of surface definition user input
        - params - SofastParams class, internal sofast parameters
        - data_geometry_general - general optic geometric calculations
        - data_geometry_facet - facet specific geometric calculations
        - data_error - errors between optic/sceario definitions and internal calculations
        - data_image_processing_general - general optic image processing calculations
        - data_image_processing_facet - facet specific image processing calculations
        - data_characterization_facet - facet specific slope calculations in facet reference frame
        - data_characterization_ensemble - facet specific slope/pointing calculations in ensemble reference frame

    External Data Storage
    ---------------------
    When data is saved in an HDF file, data is stored in the following format.
    Not all the fields below are populated depending on the type of optic being processed.

    - DataSofastInput
        - optic_definintion
            - EnsembleDefinition
                - v_centroid_ensemble
                - v_facet_locations
                - ensemble_perimeter
                - r_facet_ensemble
            - facet_000
                - FacetDefinition
                    - v_centroid_facet
                    - v_facet_corners
                - surface_definition
                    - downsample
                    - initial_focal_lengths_xy
                    - robust_least_squares
                    - surface_type
        - SofastParams
            - mask_filt_thresh
            - mask_filt_width
            - mask_hist_thresh
            - mask_keep_largest_area
            - mask_thresh_active_pixels
            - ParamsOpticGeometry
                - facet_corns_refine_frac_keep
                - facet_corns_refine_perpendicular_search_dist
                - facet_corns_refine_step_length
                - perimeter_refine_axial_search_dist
                - perimeter_refine_perpendicular_search_dist
    - DataSofastCalculation
        - facet
            - facet_000
                - CalculationDataGeometryFacet
                    - SpatialOrientation
                        - r_cam_optic
                        - r_cam_screen
                        - v_cam_optic_cam
                        - v_cam_screen_cam
                    - u_cam_measure_point_facet
                    - measure_point_screen_distance
                    - u_pixel_pointing_facet
                    - v_align_point_facet
                    - v_screen_points_facet
                    - v_screen_points_screen
                    - v_screen_points_fractional_screens
                - CalculationEnsemble
                    - trans_facet_ensemble
                    - slopes_ensemble_xy
                    - v_surf_points_ensemble
                    - v_facet_pointing_ensemble
                - CalculationImageProcessingFacet
                    - loop_facet_image_refine
                    - mask_fitted
                    - mask_bad_pixels
                    - mask_processed
                    - v_facet_centroid_image_exp
                    - v_facet_corners_image_exp
                - SlopeSolverData
                    - slope_coefs_facet
                    - slopes_facet_xy
                    - surf_coefs_facet
                    - trans_alignment
                    - v_surf_points_facet
        - general
            - CalculationDataGeometryGeneral
                - r_optic_cam_exp
                - r_optic_cam_refine_1
                - r_optic_cam_refine_2
                - v_cam_optic_cam_exp
                - v_cam_optic_cam_refine_1
                - v_cam_optic_cam_refine_2
                - v_cam_optic_cam_refine_3
                - v_cam_optic_centroid_cam_exp
            - CalculationError
                - error_dist_optic_screen_1
                - error_dist_optic_screen_2
                - error_dist_optic_screen_3
                - error_reprojection_1
                - error_reprojection_2
                - error_reprojection_3
            - CalculationImageProcessingGeneral
                - loop_optic_image_exp
                - loop_optic_image_refine
                - mask_raw
                - v_edges_image
                - v_mask_centroid_image
    """

    def __init__(
        self, measurement: MeasurementSofastFringe, orientation: SpatialOrientation, camera: Camera, display: Display
    ) -> 'ProcessSofastFringe':
        """
        SOFAST processing class.

        Parameters
        ----------
        measurement : MeasurementSofastFringe
            MeasurementSofastFringe class to process.
        orientation : SpatialOrientation
            SpatialOrientation object
        camera : Camera
            Camera object used to capture data.
        display : Display
            Display object used to capture data.
        """
        # Store data
        self.measurement = measurement
        self.display = display
        self.camera = camera
        self.orientation = orientation

        # Define default calculation parameters
        self.params = ParamsSofastFringe()

        # Instantiate data containers
        self.num_facets: int = 0
        self.optic_type: Literal['undefined', 'single', 'multi'] = None
        self.data_facet_def: list[DefinitionFacet] = None
        self.data_ensemble_def: DefinitionEnsemble = None

        self.data_surfaces: list[Surface2DAbstract] = None

        self.data_geometry_general: cdc.CalculationDataGeometryGeneral = None
        self.data_image_processing_general: cdc.CalculationImageProcessingGeneral = None
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet] = None
        self.data_image_processing_facet: list[cdc.CalculationImageProcessingFacet] = None
        self.data_error: cdc.CalculationError = None

        self.data_characterization_facet: list[SlopeSolverData] = None
        self.data_characterization_ensemble: list[cdc.CalculationFacetEnsemble] = None

    def help(self) -> None:
        """Prints Sofast doc string"""
        print(self.__doc__)

    def process_optic_undefined(self, surface: Surface2DAbstract) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for undefined optical surface.

        Parameters
        ----------
        surface_data : Surface2DAbstract
            Surface type definition
        """
        # Process optic/setup geometry
        self._process_optic_undefined_geometry()

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes([surface])

    def process_optic_singlefacet(self, facet_data: DefinitionFacet, surface: Surface2DAbstract) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for single facet optic.

        Parameters
        ----------
        facet_data : DefinitionFacet
            Facet data object.
        surface_data : Surface2DAbstract
            Surface type definition.
        """
        # Process optic/setup geometry
        self._process_optic_singlefacet_geometry(facet_data)

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes([surface])

    def process_optic_multifacet(
        self, facet_data: list[DefinitionFacet], ensemble_data: DefinitionEnsemble, surfaces: list[Surface2DAbstract]
    ) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for multi-facet optic.

        Parameters
        ----------
        facet_data : list[DefinitionFacet]
            List of facet data objects.
        ensemble_data : DefinitionEnsemble
            Ensemble data object.
        surface_data : dict
            See Sofast documentation or Sofast.help() for more details.

        """
        # Check inputs
        if len(facet_data) != len(surfaces):
            lt.error_and_raise(
                ValueError,
                'Length of facet_data does not equal length of surfaces'
                f'facet_data={len(facet_data)}, surface_data={len(surfaces)}',
            )

        # Process optic/setup geometry
        self._process_optic_multifacet_geometry(facet_data, ensemble_data)

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes(surfaces)

        # Calculate facet pointing
        self._calculate_facet_pointing()

    def _process_optic_undefined_geometry(self) -> None:
        """
        Processes undefined optic data.

        """
        # Save number of facets
        self.num_facets = 1
        self.optic_type = 'undefined'

        # Calculate raw mask
        params = [
            self.params.mask.hist_thresh,
            self.params.mask.filt_width,
            self.params.mask.filt_thresh,
            self.params.mask.thresh_active_pixels,
        ]
        mask_raw = ip.calc_mask_raw(self.measurement.mask_images, *params)

        # Process optic geometry
        (
            self.data_geometry_general,
            self.data_image_processing_general,
            self.data_geometry_facet,
            self.data_image_processing_facet,
            self.data_error,
        ) = po.process_undefined_geometry(
            mask_raw,
            self.params.mask.keep_largest_area,
            self.measurement.dist_optic_screen,
            self.orientation,
            self.camera,
            self.params.debug_geometry,
        )

        # Save data
        self.data_facet_def = None
        self.data_ensemble_def = None

    def _process_optic_singlefacet_geometry(self, facet_data: DefinitionFacet) -> None:
        """
        Processes optic geometry for single facet.

        Parameters
        ----------
        facet_data : DefinitionFacet
            Facet data object.

        """
        # Save number of facets
        self.num_facets = 1
        self.optic_type = 'single'

        if self.params.debug_geometry.debug_active:
            lt.info('Sofast image processing debug on.')
        if self.params.debug_slope_solver.debug_active:
            lt.info('SlopeSolver debug on.')

        # Calculate raw mask
        params = [
            self.params.mask.hist_thresh,
            self.params.mask.filt_width,
            self.params.mask.filt_thresh,
            self.params.mask.thresh_active_pixels,
        ]
        mask_raw = ip.calc_mask_raw(self.measurement.mask_images, *params)

        # If enabled, keep only the largest mask area
        if self.params.mask.keep_largest_area:
            mask_raw2 = ip.keep_largest_mask_area(mask_raw)
            mask_raw = np.logical_and(mask_raw, mask_raw2)

        (
            self.data_geometry_general,
            self.data_image_processing_general,
            self.data_geometry_facet,
            self.data_image_processing_facet,
            self.data_error,
        ) = po.process_singlefacet_geometry(
            facet_data,
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.measurement.dist_optic_screen,
            self.orientation,
            self.camera,
            self.params.geometry,
            self.params.debug_geometry,
        )

        # Save data
        self.data_facet_def = [facet_data.copy()]
        self.data_ensemble_def = None

    def _process_optic_multifacet_geometry(
        self, facet_data: list[DefinitionFacet], ensemble_data: DefinitionEnsemble
    ) -> None:
        """Processes optic geometry for an ensemble of facets.

        Parameters
        ----------
        facet_data : list[DefinitionFacet]
            List of DefinitionFacet objects.
        ensemble_data : DefinitionEnsemble
            Ensemble data object.
        """
        # Get number of facets
        self.num_facets = ensemble_data.num_facets
        self.optic_type = 'multi'

        # Check inputs
        if len(facet_data) != self.num_facets:
            lt.error_and_raise(
                ValueError,
                f'Given length of facet data is {len(facet_data):d}'
                f'but ensemble_data expects {ensemble_data.num_facets:d} facets.',
            )

        # Calculate mask
        params = [
            self.params.mask.hist_thresh,
            self.params.mask.filt_width,
            self.params.mask.filt_thresh,
            self.params.mask.thresh_active_pixels,
        ]
        mask_raw = ip.calc_mask_raw(self.measurement.mask_images, *params)

        if self.params.mask.keep_largest_area:
            lt.warn(
                '"keep_largest_area" mask processing option cannot be used '
                'for multifacet ensembles. This will be turned off.'
            )
            self.params.mask.keep_largest_area = False

        (
            self.data_geometry_general,
            self.data_image_processing_general,
            self.data_geometry_facet,
            self.data_image_processing_facet,
            self.data_error,
        ) = po.process_multifacet_geometry(
            facet_data,
            ensemble_data,
            mask_raw,
            self.measurement.v_measure_point_facet,
            self.orientation,
            self.camera,
            self.measurement.dist_optic_screen,
            self.params.geometry,
            self.params.debug_geometry,
        )

        # Initialize data dictionaries
        self.data_facet_def = [f.copy() for f in facet_data]
        self.data_ensemble_def = ensemble_data.copy()

    def _process_display(self) -> None:
        """
        Calculates the reflected pixel locations on the display in screen
        and optic coordinates.

        """
        x_ims = self.measurement.fringe_images_x_calibrated
        y_ims = self.measurement.fringe_images_y_calibrated
        x_periods = self.measurement.fringe_periods_x
        y_periods = self.measurement.fringe_periods_y

        for idx_facet in range(self.num_facets):
            # Get current processed mask layer
            mask_processed = self.data_image_processing_facet[idx_facet].mask_processed
            ori = self.data_geometry_facet[idx_facet].spatial_orientation

            # Calculate pixel positions on screen (fractional screens)
            screen_xs = ip.unwrap_phase(x_ims[mask_processed, :].T, x_periods)
            screen_ys = ip.unwrap_phase(y_ims[mask_processed, :].T, y_periods)
            # Flip Y direction because screen is flipped in y direction
            screen_ys = 1.0 - screen_ys
            # Store screen points in Vxy
            v_screen_points_fractional_screens = Vxy((screen_xs, screen_ys))
            self.data_geometry_facet[idx_facet].v_screen_points_fractional_screens = v_screen_points_fractional_screens

            # Undistort screen points (display coordinates)
            v_screen_points_screen = self.display.interp_func(
                v_screen_points_fractional_screens
            )  # meters, display coordinates
            self.data_geometry_facet[idx_facet].v_screen_points_screen = v_screen_points_screen

            # Check for nans returning from screen point calculation
            nan_mask = np.isnan(v_screen_points_screen.data).sum(0).astype(bool)
            mask_bad_pixels = np.zeros(mask_processed.shape, dtype=bool)
            if np.any(nan_mask):
                lt.warn(
                    f'{nan_mask.sum():d} / {nan_mask.size:d} points are NANs in calculated '
                    'screen points for facet {idx_facet:d}. These data points will be removed.'
                )
                # Make mask of NANs
                mask_bad_pixels[mask_processed] = nan_mask
                # Update processed mask
                mask_processed[mask_bad_pixels] = False
                # Remove nan data points from screen points
                self.data_geometry_facet[idx_facet].v_screen_points_fractional_screens = (
                    v_screen_points_fractional_screens[np.logical_not(nan_mask)]
                )
                self.data_geometry_facet[idx_facet].v_screen_points_screen = v_screen_points_screen[
                    np.logical_not(nan_mask)
                ]
            # Save bad pixel mask
            self.data_image_processing_facet[idx_facet].mask_bad_pixels = mask_bad_pixels

            # Calculate pixel pointing directions (camera coordinates)
            u_pixel_pointing_cam = ip.calculate_active_pixels_vectors(mask_processed, self.camera)
            # Convert to optic coordinates
            u_pixel_pointing_facet = u_pixel_pointing_cam.rotate(ori.r_cam_optic)
            self.data_geometry_facet[idx_facet].u_pixel_pointing_facet = u_pixel_pointing_facet

            # Convert to optic coordinates
            v_screen_points_facet = ori.trans_screen_optic.apply(
                self.data_geometry_facet[idx_facet].v_screen_points_screen
            )
            self.data_geometry_facet[idx_facet].v_screen_points_facet = v_screen_points_facet

    def _solve_slopes(self, surfaces: list[Surface2DAbstract]) -> None:
        """
        Solves slopes of each active pixel for each facet.

        Parameters
        ----------
        surface_data : list[Surface2DAbstract]
            List of surface definition classes.
        """
        # Check inputs
        if self.data_geometry_facet is None:
            lt.error_and_raise(ValueError, 'Not all facets geometrically processed; cannot solve slopes.')

        # Loop through all input facets and solve slopes
        self.data_characterization_facet = []
        for facet_idx in range(self.num_facets):
            # Check debug status
            if self.params.debug_slope_solver.debug_active:
                self.params.debug_slope_solver.optic_data = self.data_facet_def[facet_idx]

            # Instantiate slope solver object
            kwargs = {
                'v_optic_cam_optic': self.data_geometry_facet[facet_idx].spatial_orientation.v_optic_cam_optic,
                'u_active_pixel_pointing_optic': self.data_geometry_facet[facet_idx].u_pixel_pointing_facet,
                'u_measure_pixel_pointing_optic': self.data_geometry_facet[facet_idx].u_cam_measure_point_facet,
                'v_screen_points_facet': self.data_geometry_facet[facet_idx].v_screen_points_facet,
                'v_optic_screen_optic': self.data_geometry_facet[facet_idx].spatial_orientation.v_optic_screen_optic,
                'v_align_point_optic': self.data_geometry_facet[facet_idx].v_align_point_facet,
                'dist_optic_screen': self.data_geometry_facet[facet_idx].measure_point_screen_distance,
                'surface': surfaces[facet_idx],
                'debug': self.params.debug_slope_solver,
            }

            # Instantiate slope solver
            slope_solver = SlopeSolver(**kwargs)

            # Perform surface fitting
            slope_solver.fit_surface()

            # Perform full slope solving
            slope_solver.solve_slopes()

            # Save slope data
            self.data_characterization_facet.append(slope_solver.get_data())

        # Save input surface parameters data
        self.data_surfaces = surfaces

    def _calculate_facet_pointing(self, reference: Literal['average'] | int = 'average') -> None:
        """
        Calculates facet pointing relative to the given reference.

        Parameters
        ----------
        reference : 'average' | int
            If 'average', the pointing reference is the average of all
            facet pointing directions. If, int, that facet index is assumed
            to have perfect pointing.
        """
        if self.data_characterization_facet is None:
            lt.error_and_raise(ValueError, 'Slopes must be solved first by running "solve_slopes".')
        if reference != 'average' and not isinstance(reference, int):
            lt.error_and_raise(ValueError, 'Given reference must be int or "average".')
        if isinstance(reference, int) and reference >= self.num_facets:
            lt.error_and_raise(
                ValueError, f'Given facet index, {reference:d}, is out of range of 0-{self.num_facets - 1:d}.'
            )

        # Instantiate data list
        self.data_characterization_ensemble = []

        trans_facet_ensemble_list = []
        v_pointing_matrix = np.zeros((3, self.num_facets))
        for idx in range(self.num_facets):
            # Get transformation from user-input and slope solving
            trans_1 = TransformXYZ.from_R_V(
                self.data_ensemble_def.r_facet_ensemble[idx], self.data_ensemble_def.v_facet_locations[idx]
            )
            trans_2 = self.data_characterization_facet[idx].trans_alignment
            # Calculate inverse of slope solving transform
            trans_2 = TransformXYZ.from_V(-trans_2.V) * TransformXYZ.from_R(trans_2.R.inv())
            # Create local to global transformation
            trans_facet_ensemble_list.append(trans_2 * trans_1)

            # Calculate pointing vector in ensemble coordinates
            v_pointing = Vxyz((0, 0, 1)).rotate(trans_facet_ensemble_list[idx].R)
            v_pointing_matrix[:, idx] = v_pointing.data.squeeze()

        # Calculate reference pointing direction
        if isinstance(reference, int):
            v_pointing_ref = Vxyz(v_pointing_matrix[:, reference])
        elif reference == 'average':
            v_pointing_ref = Vxyz(v_pointing_matrix.mean(1))
        # Calculate rotation to align pointing vectors
        r_align_pointing = v_pointing_ref.align_to(Vxyz((0, 0, 1)))
        trans_align_pointing = TransformXYZ.from_R(r_align_pointing)

        # Apply alignment rotation to total transformation
        trans_facet_ensemble_list = [trans_align_pointing * t for t in trans_facet_ensemble_list]

        # Calculate global slope and surface points
        for idx in range(self.num_facets):
            # Get slope data
            slopes = self.data_characterization_facet[idx].slopes_facet_xy  # facet coordinats

            # Calculate surface normals in local (facet) coordinates
            u_surf_norms = np.ones((3, slopes.shape[1]))
            u_surf_norms[:2] = -slopes
            u_surf_norms = Uxyz(u_surf_norms).as_Vxyz()

            # Apply rotation to normal vectors
            u_surf_norms_global = u_surf_norms.rotate(trans_facet_ensemble_list[idx].R)
            # Convert normal vectors to global (ensemble) slopes
            slopes_ensemble_xy = -u_surf_norms_global.data[:2] / u_surf_norms_global.data[2:]

            # Convert surface points to global (ensemble) coordinates
            v_surf_points_ensemble = trans_facet_ensemble_list[idx].apply(
                self.data_characterization_facet[idx].v_surf_points_facet
            )

            # Calculate pointing vectors in ensemble coordinates
            v_facet_pointing_ensemble = Vxyz((0, 0, 1)).rotate(trans_facet_ensemble_list[idx].R)

            data = cdc.CalculationFacetEnsemble(
                trans_facet_ensemble_list[idx], slopes_ensemble_xy, v_surf_points_ensemble, v_facet_pointing_ensemble
            )
            self.data_characterization_ensemble.append(data)

    def get_optic(
        self, interp_type: Literal['bilinear', 'clough_tocher', 'nearest'] = 'nearest'
    ) -> FacetEnsemble | Facet:
        """Returns the OpenCSP representation of the optic under test. For
        single facet data collects, returns a Facet object, and for multi-facet
        collects, returns a FacetEnsemble object. Each mirror is represented
        by a MirrorPoint object. Each mirror origin is co-located with its
        parent facet origin.

        Parameters
        ----------
        interp_type : {'bilinear', 'clough_tocher', 'nearest'}, optional
            Mirror interpolation type, by default 'nearest'

        Returns
        -------
        FacetEnsemble | Facet
            Optic object
        """
        facets = []
        trans_list = []
        for idx_mirror, data in enumerate(self.data_characterization_facet):
            # Get surface points
            pts: Vxyz = data.v_surf_points_facet
            # Get normals from slopes
            slopes: np.ndarray = data.slopes_facet_xy
            norm_data = np.ones((3, slopes.shape[1]))
            norm_data[:2] = -slopes
            norm_vecs = Uxyz(norm_data)
            # Get mirror shape
            if self.optic_type == 'undefined':
                # Find bounding box
                x1 = pts.x.min()
                x2 = pts.x.max()
                y1 = pts.y.min()
                y2 = pts.y.max()
                vertices = Vxy(([x1, x1, x2, x2], [y1, y2, y2, y1]))
            else:
                # Get optic region from optic definition
                vertices = self.data_facet_def[idx_mirror].v_facet_corners.projXY()
            shape = RegionXY.from_vertices(vertices)
            # Create mirror
            mirror = MirrorPoint(pts, norm_vecs, shape, interp_type)
            # Create facet
            facet = Facet(mirror)
            # Locate facet within ensemble
            if self.optic_type == 'multi':
                trans: TransformXYZ = self.data_characterization_ensemble[idx_mirror].trans_facet_ensemble
                trans_list.append(trans)
            # Save facets
            facets.append(facet)

        # Return optics
        if self.optic_type == 'multi':
            ensemble = FacetEnsemble(facets)
            ensemble.set_facet_transform_list(trans_list)
            return ensemble
        else:
            return facets[0]

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Log
        lt.info(f'Saving SofastFringe data to: {file:s}, in HDF5 folder: "{prefix:s}"')

        # One per measurement
        if self.data_error is not None:
            self.data_error.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_geometry_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_image_processing_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')

        # Sofast parameters
        self.params.save_to_hdf(file, f'{prefix:s}DataSofastInput/')

        # Facet definition
        if self.data_facet_def is not None:
            for idx_facet, facet_data in enumerate(self.data_facet_def):
                facet_data.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/')

        # Ensemble definition
        if self.data_ensemble_def is not None:
            self.data_ensemble_def.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/')

        # Surface definition
        for idx_facet, surface in enumerate(self.data_surfaces):
            surface.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/')

        # Calculations, one per facet
        for idx_facet in range(self.num_facets):
            # Save facet slope data
            self.data_characterization_facet[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
            )
            # Save facet geometry data
            self.data_geometry_facet[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
            )
            # Save facet image processing data
            self.data_image_processing_facet[idx_facet].save_to_hdf(
                file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
            )

            if self.data_characterization_ensemble:
                # Save ensemle data
                self.data_characterization_ensemble[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
