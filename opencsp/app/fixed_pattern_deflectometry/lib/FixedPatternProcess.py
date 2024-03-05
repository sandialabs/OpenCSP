"""Class that handles the processing of fixed pattern deflectometry data.
"""
from typing import Literal

import cv2 as cv
import numpy as np
from   numpy import ndarray

from   opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternDotLocations import FixedPatternDotLocations
from   opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternMeasurement import FixedPatternMeasurement
from   opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternProcessParams import FixedPatternProcessParams
from   opencsp.common.lib.camera.Camera import Camera
from   opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from   opencsp.common.lib.deflectometry.BlobIndex import BlobIndex
import opencsp.common.lib.deflectometry.calculation_data_classes as cdc
from   opencsp.common.lib.deflectometry.FacetData import FacetData
import opencsp.common.lib.deflectometry.image_processing as ip
import opencsp.common.lib.deflectometry.process_optics_geometry as pr
from   opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from   opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from   opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from   opencsp.common.lib.geometry.RegionXY import RegionXY
from   opencsp.common.lib.geometry.Uxyz import Uxyz


class FixedPatternProcess:
    """Fixed Pattern Deflectrometry data processing class"""

    def __init__(self,
                 orientation: SpatialOrientation,
                 camera: Camera,
                 fixed_pattern_dot_locs: FixedPatternDotLocations,
                 facet_data: FacetData) -> 'FixedPatternProcess':
        """Instantiates class

        Parameters
        ----------
        orientation : SpatialOrientaton
            SpatialOrientation object
        camera : Camera
            Camera object
        fixed_pattern_dot_locs : FixedPatternDotLocations
            Image projection dictionary
        facet_data : FacetData
            FacetData object
        """
        self.orientation = orientation
        self.camera = camera
        self.fixed_pattern_dot_locs = fixed_pattern_dot_locs
        self.facet_data = facet_data
        self.params = FixedPatternProcessParams()

        # Measurement data
        self.measurement: FixedPatternMeasurement

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
        self.data_slope_solver: SlopeSolverData
        self.data_geometry_general: cdc.CalculationDataGeometryGeneral
        self.data_image_proccessing_general: cdc.CalculationsImageProcessingGeneral
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet]
        self.data_image_processing_facet: list[cdc.CalculationsImageProcessingFacet]
        self.data_error: cdc.CalculationError

    def find_blobs(self) -> BlobIndex:
        """Finds blobs in image"""
        pts_blob = ip.detect_blobs(self.measurement.image, self.blob_detector)

        # Index blobs
        blob_index = BlobIndex(pts_blob, *self.fixed_pattern_dot_locs.dot_extent)
        blob_index.verbose = False
        blob_index.search_thresh = self.params.blob_search_thresh
        blob_index.search_perp_axis_ratio = self.params.search_perp_axis_ratio
        blob_index.run(self.measurement.origin)

        return blob_index

    def calculate_mask(self) -> ndarray:
        """Calculate mask image

        Parameters
        ----------
        image : ndarray
            Measurement image
        """
        # Calculate mask
        im_dark = self.measurement.image * 0
        images = np.concatenate((im_dark[..., None], self.measurement.image[..., None]), axis=2)
        params = [
            self.params.mask_hist_thresh,
            self.params.mask_filt_width,
            self.params.mask_filt_thresh,
            self.params.mask_thresh_active_pixels,
        ]
        mask = ip.calc_mask_raw(images, *params)

        if self.params.mask_keep_largest_area:
            mask = ip.keep_largest_mask_area(mask)

        return mask

    def generate_geometry(self,
                          blob_index: BlobIndex,
                          mask_raw: np.ndarray) -> dict:
        """Generates blob dataset from sofast dataset. 

        Parameters
        ----------
        blob_index : BlobIndex
            BlobIndex object with all blobs assigned
        mask_raw : ndarray
            Mask array

        Returns
        -------
        dict
            Key word argument dictionary for SlopeSolver (does not include "surface_data")

        Sets Attributes
        ---------------
        self.data_geometry_general
        self.data_image_proccessing_general
        self.data_geometry_facet
        self.data_image_processing_facet
        self.data_error
        """
        pts_image, pts_index_xy = blob_index.get_data()

        # Process optic geometry
        (self.data_geometry_general,
         self.data_image_proccessing_general,
         self.data_geometry_facet,
         self.data_image_processing_facet,
         self.data_error) = pr.process_singlefacet_geometry(self.facet_data, mask_raw, self.measurement.v_measure_point_facet,
                                                            self.measurement.dist_optic_screen, self.orientation, self.camera, debug=self.params.geometry_data_debug)

        # Define optic orientation w.r.t. camera
        rot_optic_cam = self.data_geometry_general.r_optic_cam_refine_1
        v_cam_optic_cam = self.data_geometry_general.v_cam_optic_cam_refine_2
        u_cam_measure_point_facet = self.data_geometry_facet[0].u_cam_measure_point_facet

        # Get screen-camera pose
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

        self.params.slope_solver_data_debug.optic_data = self.facet_data

        return {
            'v_optic_cam_optic': v_optic_cam_optic,
            'u_active_pixel_pointing_optic': u_pixel_pointing_facet,
            'u_measure_pixel_pointing_optic': u_cam_measure_point_facet,
            'v_screen_points_facet': v_screen_points_facet,
            'v_optic_screen_optic': v_optic_screen_optic,
            'v_align_point_optic': self.facet_data.v_facet_centroid,
            'dist_optic_screen': self.measurement.dist_optic_screen,
            'debug': self.params.slope_solver_data_debug
        }

    def load_measurement_data(self, measurement: FixedPatternMeasurement) -> None:
        """Saves measurement data in class

        Parameters
        ----------
        measurement: FixedPatternMeasurement
            Fixed pattern measurement object
        """
        self.measurement = measurement

    def process_single_facet_optic(self, surface_data: dict) -> None:
        """Processes single facet optic. Sets attribute
        self.data_slope_solver
        """
        # Find blobs
        blob_index = self.find_blobs()

        # Calculate mask
        mask_raw = self.calculate_mask()
        mask_raw = ip.keep_largest_mask_area(mask_raw)

        # Generate geometry and slope solver inputs
        kwargs = self.generate_geometry(blob_index, mask_raw)

        # Add surface fitting parameters
        kwargs.update({'surface_data': surface_data})

        # Calculate slope
        slope_solver = SlopeSolver(**kwargs)
        slope_solver.fit_surface()
        slope_solver.solve_slopes()
        self.data_slope_solver = slope_solver.get_data()

    def save_to_hdf(self, file: str):
        """Saves data to given HDF5 file. Data is stored in CalculationsFixedPattern/...

        Parameters
        ----------
        file : str
            HDF file to save to
        """
        self.data_slope_solver.save_to_hdf(file, 'CalculationsFixedPattern/Facet_000/')
        self.data_geometry_general.save_to_hdf(file, 'CalculationsFixedPattern/')
        self.data_image_proccessing_general.save_to_hdf(file, 'CalculationsFixedPattern/')
        self.data_geometry_facet[0].save_to_hdf(file, 'CalculationsFixedPattern/Facet_000/')
        self.data_image_processing_facet[0].save_to_hdf(file, 'CalculationsFixedPattern/Facet_000/')
        self.data_error.save_to_hdf(file, 'CalculationsFixedPattern/')

    def get_mirror(self, interpolation_type: Literal['given', 'bilinear', 'clough_tocher', 'nearest'] = 'nearest') -> MirrorPoint:
        """Returns mirror object with slope data"""
        v_surf_pts = self.data_slope_solver.v_surf_points_facet
        v_normals_data = np.ones((3, len(v_surf_pts)))
        v_normals_data[:2, :] = -self.data_slope_solver.slopes_facet_xy
        v_normals = Uxyz(v_normals_data)
        shape = RegionXY.from_vertices(self.facet_data.v_facet_corners.projXY())
        return MirrorPoint(v_surf_pts, v_normals, shape, interpolation_type)
