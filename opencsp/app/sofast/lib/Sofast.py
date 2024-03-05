"""Controls the processing of Sofast measurement data
to calculate surface slopes.
"""
from typing import Literal, Any
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.app.sofast.lib.SofastParams import SofastParams
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
import opencsp.common.lib.deflectometry.calculation_data_classes as cdc
from opencsp.common.lib.deflectometry.Display import Display
from opencsp.common.lib.deflectometry.EnsembleData import EnsembleData
from opencsp.common.lib.deflectometry.FacetData import FacetData
import opencsp.common.lib.deflectometry.image_processing as ip
import opencsp.common.lib.deflectometry.process_optics_geometry as po
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import save_hdf5_datasets


class Sofast:
    """Class that processes measurement data captured by a SOFAST
    system. Computes optic surface slope and saves data to HDF5 format.

    Debug Modes
    -----------
    - 0 - No outputs.
    - 1 - Just Sofast image processing plots
    - 2 - Just slope solving plots
    - 3 - All plots

    Processing types
    ----------------
    Sofast can process three optic types:
        - Undefined optic - Sofast.process_optic_undefined(surface_data)
        - Single facet optic - Sofast.process_optic_singlefacet(facet_data, surface_data)
        - Multi-facet ensemble - Sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    - surface_data : defines surface fitting parameters. See SlopeSolver
        documentation for more information.

    Internal Data Storage
    ---------------------
    Data is stored in the following fields:
        - data_ensemble_def - copy of ensemble definition user input
        - data_facet_def - copy of facet definition user input
        - data_surface_params - copy of surface definition user input
        - params - internal sofast parameters
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
            - ensemble
                - v_centroid_ensemble
                - v_facet_locations
                - ensemble_perimeter
                - r_facet_ensemble
            - facet_000
                - v_centroid_facet
                - v_facet_corners
        - sofast_params
            - facet_corns_refine.fraction_keep
            - facet_corns_refine.perpendicular_search_dist
            - facet_corns_refine.step_length
            - mask.filter_thresh
            - mask.filter_width
            - mask.hist_thresh
            - mask.keep_largest_area
            - mask.thresh_active_pixels
            - perimeter_refine.axial_search_dist
            - perimeter_refine.perpendicular_search_dist
        - surface_params
            - facet_000
                - downsample
                - initial_focal_lengths_xy
                - robust_least_squares
                - surface_type
    - DataSofastCalculation
        - geometry
            - optic_general
                - r_optic_cam
                - r_optic_cam_exp
                - r_optic_cam_refine_1
                - r_optic_cam_refine_2
                - v_cam_optic_cam
                - v_cam_optic_cam_exp
                - v_cam_optic_cam_refine_1
                - v_cam_optic_cam_refine_2
                - v_cam_optic_cam_refine_3
                - v_cam_optic_centroid_cam_exp
            - facet_000
                - u_cam_measure_point_facet
                - measure_point_screen_distance
                - u_pixel_pointing_facet
                - v_align_point_facet
                - v_screen_points_facet
                - v_screen_points_screen
                - v_screen_points_fractional_screens
        - error
            - error_optic_screen_dist_1
            - error_optic_screen_dist_2
            - error_optic_screen_dist_3
            - error_reprojection_1
            - error_reprojection_2
            - error_reprojection_3
        - image_processing
            - optic_general
                - v_facet_centroid_image_exp
                - v_facet_corners_image_exp
                - v_mask_centroid_image
                - mask_raw
                - loop_optic_image_exp
                - loop_optic_image_refine
                - v_edges_image
            - facet_000
                - loop_facet_image_refine
                - mask_fitted
                - mask_processed
        - facet
            - facet_000
                - v_surf_points_facet
                - slope_coefs_facet
                - slopes_facet_xy
                - surf_coefs_facet
                - trans_alignment
        - ensemble
            - facet_000
                - trans_facet_ensemble
                - slopes_ensemble_xy
                - v_surf_points_ensemble
                - v_facet_pointing_ensemble
    """

    def __init__(
        self, measurement: Measurement, camera: Camera, display: Display
    ) -> 'Sofast':
        """
        SOFAST processing class.

        Parameters
        ----------
        measurement : Measurement
            Measurement class to process.
        camera : Camera
            Camera object used to capture data.
        display : Display
            Display object used to capture data.

        """
        # Store data
        self.measurement = measurement
        self.display = display
        self.camera = camera
        self.orientation = SpatialOrientation(
            display.r_cam_screen, display.v_cam_screen_cam
        )

        # Define default calculation parameters
        self.params = SofastParams()

        # Instantiate data containers
        self.num_facets: int = 0
        self.optic_type: Literal['undefined', 'single', 'multi']
        self.data_facet_def: list[FacetData]
        self.data_ensemble_def: EnsembleData

        self.data_surface_params: list[dict]

        self.data_geometry_general: cdc.CalculationDataGeometryGeneral
        self.data_image_processing_general: cdc.CalculationsImageProcessingGeneral
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet]
        self.data_image_processing_facet: list[cdc.CalculationsImageProcessingFacet]
        self.data_error: cdc.CalculationError

        self.data_characterization_facet: list[SlopeSolverData]
        self.data_characterization_ensemble: list[dict]

    def help(self) -> None:
        """Prints Sofast doc string"""
        print(self.__doc__)

    @staticmethod
    def _check_surface_data(surf_data: dict) -> None:
        """Checks that all necessary fields are present in surface data dict"""
        if 'surface_type' not in surf_data.keys():
            raise ValueError('Missing "surface_type" key in surface_data dictionary.')

        if surf_data['surface_type'] == 'parabolic':
            fields_exp = [
                'surface_type',
                'initial_focal_lengths_xy',
                'robust_least_squares',
                'downsample',
            ]
        elif surf_data['surface_type'] == 'plano':
            fields_exp = ['surface_type', 'robust_least_squares', 'downsample']
        else:
            raise ValueError(
                f'Given surface type {surf_data["surface_type"]} is not supported.'
            )

        for k in surf_data.keys():
            if k in fields_exp:
                idx = fields_exp.index(k)
                fields_exp.pop(idx)
            else:
                raise ValueError(
                    f'Unrecognized field, {k}, in surface_data dictionary.'
                )

        if len(fields_exp) > 0:
            raise ValueError(f'Missing fields in surface_data dictionary: {fields_exp}')

    def process_optic_undefined(self, surface_data: dict) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for undefined optical surface.

        Parameters
        ----------
        surface_data : dict
            See Sofast documentation or Sofast.help() for more details.

        """
        # Check input data
        self._check_surface_data(surface_data)

        # Process optic/setup geometry
        self._process_optic_undefined_geometry()

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes([surface_data])

    def process_optic_singlefacet(
        self, facet_data: FacetData, surface_data: dict
    ) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for single facet optic.

        Parameters
        ----------
        facet_data : FacetData
            Facet data object.
        surface_data : dict
            See Sofast documentation or Sofast.help() for more details.

        """
        # Check input data
        self._check_surface_data(surface_data)

        # Process optic/setup geometry
        self._process_optic_singlefacet_geometry(facet_data)

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes([surface_data])

    def process_optic_multifacet(
        self,
        facet_data: list[FacetData],
        ensemble_data: EnsembleData,
        surface_data: list[dict],
    ) -> None:
        """
        Processes optic geometry, screen intersection points, and solves
        for slops for multi-facet optic.

        Parameters
        ----------
        facet_data : list[FacetData]
            List of facet data objects.
        ensemble_data : EnsembleData
            Ensemble data object.
        surface_data : dict
            See Sofast documentation or Sofast.help() for more details.

        """
        # Check inputs
        if len(facet_data) != len(surface_data):
            raise ValueError(
                f'Length of facet_data does not equal length of surface data; facet_data={len(facet_data)}, surface_data={len(surface_data)}'
            )
        list(map(self._check_surface_data, surface_data))

        # Process optic/setup geometry
        self._process_optic_multifacet_geometry(facet_data, ensemble_data)

        # Process display ray intersection points
        self._process_display()

        # Solve slopes
        self._solve_slopes(surface_data)

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
            self.params.mask_hist_thresh,
            self.params.mask_filt_width,
            self.params.mask_filt_thresh,
            self.params.mask_thresh_active_pixels,
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
            self.params.mask_keep_largest_area,
            self.measurement.optic_screen_dist,
            self.orientation,
            self.camera,
            self.params.geometry_data_debug,
        )

        # Save data
        self.data_facet_def = None
        self.data_ensemble_def = None

    def _process_optic_singlefacet_geometry(self, facet_data: FacetData) -> None:
        """
        Processes optic geometry for single facet.

        Parameters
        ----------
        facet_data : FacetData
            Facet data object.

        """
        # Save number of facets
        self.num_facets = 1
        self.optic_type = 'single'

        if self.params.geometry_data_debug.debug_active:
            print('Sofast image processing debug on.')
        if self.params.slope_solver_data_debug.debug_active:
            print('SlopeSolver debug on.')

        # Calculate raw mask
        params = [
            self.params.mask_hist_thresh,
            self.params.mask_filt_width,
            self.params.mask_filt_thresh,
            self.params.mask_thresh_active_pixels,
        ]
        mask_raw = ip.calc_mask_raw(self.measurement.mask_images, *params)

        # If enabled, keep only the largest mask area
        if self.params.mask_keep_largest_area:
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
            self.measurement.measure_point,
            self.measurement.optic_screen_dist,
            self.orientation,
            self.camera,
            self.params.geometry_params,
            self.params.geometry_data_debug,
        )

        # Save data
        self.data_facet_def = [facet_data.copy()]
        self.data_ensemble_def = None

    def _process_optic_multifacet_geometry(
        self, facet_data: list[FacetData], ensemble_data: EnsembleData
    ) -> None:
        """
        Processes optic geometry for an ensemble of facets.

        Parameters
        ----------
        facet_data : list[FacetData]
            List of FacetData objects.
        ensemble_data : EnsembleData
            Ensemble data object.

        """
        # Get number of facets
        self.num_facets = ensemble_data.num_facets
        self.optic_type = 'multi'

        # Check inputs
        if len(facet_data) != self.num_facets:
            raise ValueError(
                f'Given length of facet data is {len(facet_data):d} but ensemble_data expects {ensemble_data.num_facets:d} facets.'
            )

        # Calculate mask
        params = [
            self.params.mask_hist_thresh,
            self.params.mask_filt_width,
            self.params.mask_filt_thresh,
            self.params.mask_thresh_active_pixels,
        ]
        mask_raw = ip.calc_mask_raw(self.measurement.mask_images, *params)

        if self.params.mask_keep_largest_area:
            warnings.warn(
                '"keep_largest_area" mask processing option cannot be used for multifacet ensembles. This will be turned off.',
                stacklevel=2,
            )
            self.params.mask_keep_largest_area = False

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
            self.measurement.measure_point,
            self.orientation,
            self.camera,
            self.measurement.optic_screen_dist,
            self.params.geometry_params,
            self.params.geometry_data_debug,
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
            self.data_geometry_facet[
                idx_facet
            ].v_screen_points_fractional_screens = v_screen_points_fractional_screens

            # Undistort screen points (display coordinates)
            v_screen_points_screen = self.display.interp_func(
                v_screen_points_fractional_screens
            )  # meters, display coordinates
            self.data_geometry_facet[
                idx_facet
            ].v_screen_points_screen = v_screen_points_screen

            # Check for nans returning from screen point calculation
            nan_mask = np.isnan(v_screen_points_screen.data).sum(0).astype(bool)
            mask_bad_pixels = np.zeros(mask_processed.shape, dtype=bool)
            if np.any(nan_mask):
                warnings.warn(
                    f'{nan_mask.sum():d} / {nan_mask.size:d} points are NANs in calculated screen points for facet {idx_facet:d}. These data points will be removed.',
                    stacklevel=2,
                )
                # Make mask of NANs
                mask_bad_pixels[mask_processed] = nan_mask
                # Update processed mask
                mask_processed[mask_bad_pixels] = False
                # Remove nan data points from screen points
                self.data_geometry_facet[
                    idx_facet
                ].v_screen_points_fractional_screens = v_screen_points_fractional_screens[
                    np.logical_not(nan_mask)
                ]
                self.data_geometry_facet[
                    idx_facet
                ].v_screen_points_screen = v_screen_points_screen[
                    np.logical_not(nan_mask)
                ]
            # Save bad pixel mask
            self.data_image_processing_facet[
                idx_facet
            ].mask_bad_pixels = mask_bad_pixels

            # Calculate pixel pointing directions (camera coordinates)
            u_pixel_pointing_cam = ip.calculate_active_pixels_vectors(
                mask_processed, self.camera
            )
            # Convert to optic coordinates
            u_pixel_pointing_facet = u_pixel_pointing_cam.rotate(ori.r_cam_optic)
            self.data_geometry_facet[
                idx_facet
            ].u_pixel_pointing_facet = u_pixel_pointing_facet

            # Convert to optic coordinates
            v_screen_points_facet = ori.trans_screen_optic.apply(
                self.data_geometry_facet[idx_facet].v_screen_points_screen
            )
            self.data_geometry_facet[
                idx_facet
            ].v_screen_points_facet = v_screen_points_facet

    def _solve_slopes(self, surface_data: list[dict]) -> None:
        """
        Solves slopes of each active pixel for each facet.

        Parameters
        ----------
        surface_data : list[dict]
            List containing one dictionary for each facet being processed.
            See SlopeSolver documentation for details.

        """
        # Check inputs
        if self.data_geometry_facet is None:
            raise ValueError('Not all facets geometrically processed.')

        # Loop through all input facets and solve slopes
        self.data_characterization_facet = []
        for facet_idx in range(self.num_facets):
            # Check debug status
            if self.params.slope_solver_data_debug.debug_active:
                self.params.slope_solver_data_debug.optic_data = self.data_facet_def[
                    facet_idx
                ]

            # Instantiate slope solver object
            kwargs = {
                'v_optic_cam_optic': self.data_geometry_facet[
                    facet_idx
                ].spatial_orientation.v_optic_cam_optic,
                'u_active_pixel_pointing_optic': self.data_geometry_facet[
                    facet_idx
                ].u_pixel_pointing_facet,
                'u_measure_pixel_pointing_optic': self.data_geometry_facet[
                    facet_idx
                ].u_cam_measure_point_facet,
                'v_screen_points_facet': self.data_geometry_facet[
                    facet_idx
                ].v_screen_points_facet,
                'v_optic_screen_optic': self.data_geometry_facet[
                    facet_idx
                ].spatial_orientation.v_optic_screen_optic,
                'v_align_point_optic': self.data_geometry_facet[
                    facet_idx
                ].v_align_point_facet,
                'dist_optic_screen': self.data_geometry_facet[
                    facet_idx
                ].measure_point_screen_distance,
                'surface_data': surface_data[facet_idx],
                'debug': self.params.slope_solver_data_debug,
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
        self.data_surface_params = [s.copy() for s in surface_data]

    def _calculate_facet_pointing(
        self, reference: Literal['average'] | int = 'average'
    ) -> None:
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
            raise ValueError('Slopes must be solved first by running "solve_slopes".')
        if reference != 'average' and not isinstance(reference, int):
            raise ValueError('Given reference must be int or "average".')
        if isinstance(reference, int) and reference >= self.num_facets:
            raise ValueError(
                f'Given facet index, {reference:d}, is out of range of 0-{self.num_facets - 1:d}.'
            )

        # Instantiate data dictionary
        self.data_characterization_ensemble = [{} for idx in range(self.num_facets)]

        trans_facet_ensemble_list = []
        v_pointing_matrix = np.zeros((3, self.num_facets))
        for idx in range(self.num_facets):
            # Get transformation from user-input and slope solving
            trans_1 = TransformXYZ.from_R_V(
                self.data_ensemble_def.r_facet_ensemble[idx],
                self.data_ensemble_def.v_facet_locations[idx],
            )
            trans_2 = self.data_characterization_facet[idx].trans_alignment
            # Calculate inverse of slope solving transform
            trans_2 = TransformXYZ.from_V(-trans_2.V) * TransformXYZ.from_R(
                trans_2.R.inv()
            )
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
        trans_facet_ensemble_list = [
            trans_align_pointing * t for t in trans_facet_ensemble_list
        ]

        # Calculate global slope and surface points
        for idx in range(self.num_facets):
            # Get slope data
            slopes = self.data_characterization_facet[
                idx
            ].slopes_facet_xy  # facet coordinats

            # Calculate surface normals in local (facet) coordinates
            u_surf_norms = np.ones((3, slopes.shape[1]))
            u_surf_norms[:2] = -slopes
            u_surf_norms = Uxyz(u_surf_norms).as_Vxyz()

            # Apply rotation to normal vectors
            u_surf_norms_global = u_surf_norms.rotate(trans_facet_ensemble_list[idx].R)
            # Convert normal vectors to global (ensemble) slopes
            slopes_ensemble_xy = (
                -u_surf_norms_global.data[:2] / u_surf_norms_global.data[2:]
            )

            # Convert surface points to global (ensemble) coordinates
            v_surf_points_ensemble = trans_facet_ensemble_list[idx].apply(
                self.data_characterization_facet[idx].v_surf_points_facet
            )

            # Calculate pointing vectors in ensemble coordinates
            v_facet_pointing_ensemble = Vxyz((0, 0, 1)).rotate(
                trans_facet_ensemble_list[idx].R
            )

            self.data_characterization_ensemble[idx].update(
                {
                    'trans_facet_ensemble': trans_facet_ensemble_list[idx],
                    'slopes_ensemble_xy': slopes_ensemble_xy,
                    'v_surf_points_ensemble': v_surf_points_ensemble,
                    'v_facet_pointing_ensemble': v_facet_pointing_ensemble,
                }
            )

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
            # Locate facet
            if self.optic_type == 'multi':
                trans: TransformXYZ = self.data_characterization_ensemble[idx_mirror][
                    'trans_facet_ensemble'
                ]
                facet.set_position_in_space(trans.V, trans.R)
            # Save facets
            facets.append(facet)

        # Return optics
        if self.optic_type == 'multi':
            return FacetEnsemble(facets)
        else:
            return facets[0]

    def save_data_to_hdf(self, file: str) -> None:
        """
        Saves all processed data to HDF file. This includes the following.

        Parameters
        ----------
        file : str
            HDF file name.

        """
        values = []
        names = []

        def save_dict_data(d: dict, prefix: str) -> None:
            if d is not None:
                for v, n in zip(d.values(), d.keys()):
                    values.append(v)
                    names.append(f'{prefix}/{n}')

        def save_list_data(l: list[dict], prefix: str) -> None:
            if l is not None:
                for idx, d in enumerate(l):
                    for v, n in zip(d.values(), d.keys()):
                        values.append(v)
                        names.append(f'{prefix}/facet_{idx:03d}/{n}')

        # Save facet definitions in dictionary
        if self.data_facet_def is not None:
            data_facet_def_dict = []
            for data in self.data_facet_def:
                data_facet_def_dict.append(
                    {
                        'v_facet_corners': data.v_facet_corners,
                        'v_centroid_facet': data.v_facet_centroid,
                    }
                )
        else:
            data_facet_def_dict = None

        # Save ensemble definitions in dictionary
        if self.data_ensemble_def is not None:
            data_ensemble_def_dict = {
                'ensemble_perimeter': self.data_ensemble_def.ensemble_perimeter,
                'r_facet_ensemble': np.array(
                    [r.as_rotvec() for r in self.data_ensemble_def.r_facet_ensemble]
                ),
                'v_centroid_ensemble': self.data_ensemble_def.v_centroid_ensemble,
                'v_facet_locations': self.data_ensemble_def.v_facet_locations,
            }
        else:
            data_ensemble_def_dict = None

        # Save sofast params in dictionary
        data_params_dict = self.params.__dict__
        data_params_dict.pop('debug')

        # Collect all data
        save_dict_data(
            data_ensemble_def_dict, 'DataSofastInput/optic_definition/ensemble'
        )
        save_list_data(data_facet_def_dict, 'DataSofastInput/optic_definition')
        save_dict_data(data_params_dict, 'DataSofastInput/sofast_params')
        save_list_data(self.data_surface_params, 'DataSofastInput/surface_params')
        save_dict_data(
            self.data_geometry_general, 'DataSofastCalculation/geometry/general'
        )
        save_list_data(self.data_geometry_facet, 'DataSofastCalculation/geometry')
        save_dict_data(self.data_error, 'DataSofastCalculation/error')
        save_dict_data(
            self.data_image_processing_general,
            'DataSofastCalculation/image_processing/general',
        )
        save_list_data(
            self.data_image_processing_facet, 'DataSofastCalculation/image_processing'
        )
        save_list_data(self.data_characterization_facet, 'DataSofastCalculation/facet')
        save_list_data(
            self.data_characterization_ensemble, 'DataSofastCalculation/ensemble'
        )

        # Format data
        values_formatted = []
        names_formatted = []
        for _ in range(len(values)):
            name = names.pop(0)
            value = values.pop(0)

            if not isinstance(value, SpatialOrientation):
                values_formatted.append(self._format_for_hdf(value))
                names_formatted.append(name)

        # Save all data to HDF file
        save_hdf5_datasets(values_formatted, names_formatted, file)

    @staticmethod
    def _format_for_hdf(data: Any) -> Any:
        if isinstance(data, LoopXY):
            value = data.vertices.data
        elif isinstance(data, Vxy):
            value = data.data.squeeze()
        elif isinstance(data, Vxyz):
            value = data.data.squeeze()
        elif isinstance(data, Rotation):
            value = data.as_rotvec()
        elif isinstance(data, TransformXYZ):
            value = data.matrix
        elif isinstance(data, bool):
            value = int(data)
        elif isinstance(data, np.ndarray) and data.dtype is bool:
            value = data.astype(int)
        elif isinstance(data, list):
            value = list(map(Sofast._format_for_hdf, data))
        else:
            value = data

        return value
