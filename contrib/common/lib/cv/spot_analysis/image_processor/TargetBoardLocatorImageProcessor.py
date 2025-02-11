import copy
import dataclasses
import cv2 as cv
import numpy as np
import os

from numpy._typing._array_like import NDArray

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import contrib.common.lib.cv.PerspectiveTransform as pt
import contrib.common.lib.cv.RegionDetector as rd
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.LineXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TargetBoardLocatorImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Locates the target board by identifying the rectangular outline. Output
    images are the perspective transformed target board region, similar to an
    orthographic correction.

    Any rectangular target board can be located with this class, as long as the
    target board is roughly orthoganol to and square with the camera. Small
    amounts of rotation and/or skew relative to the camera frame are ok.

    Other notes::
        - The location in the reference image is applied to all input images
        - It is recommended this this processor be used after
          ConvolutionImageProcessor(kernel='gaussian'), and probably also after
          AverageByGroupImageProcessor().
    """

    def __init__(
        self,
        reference_image_dir_or_file: str | None,
        cropped_x1x2y1y2: list[int] | None,
        target_width_meters: float,
        target_height_meters: float,
        canny_edges_gradient: int,
        canny_non_edges_gradient: int,
        edge_coarse_width=30,
        canny_test_gradients: list[tuple[int, int]] = None,
        debug_target_locating: bool = False,
    ):
        """
        Parameters
        ----------
        reference_image_dir_or_file : str
            The reference image in which the target board is clearly visible. It
            is assumed that the target board will be in the same location in all
            future images. Can be None and set at a later time.
        cropped_x1x2y1y2 : list[int] | None
            If set, then this value indicates how the input images have been
            cropped up to this point. The reference images used to find the
            target board will also be cropped using these values.
        target_width_meters : float
            Width of the target board in meters.
        target_height_meters : float
            Height of the target board in meters.
        canny_edges_gradient : int
            The threshold to use for gradients that definitely belong to edges.
        canny_non_edges_gradient : int
            The threshold to use for gradients that probably don't belong to edges.
        edge_coarse_width : int, optional
            The total width/height of the search window for the
            top/right/bottom/left edges, by default 30
        canny_test_gradients : list[tuple[int, int]], optional
            If set then display candidate canny edge images with the given list
            of edges and non-edges gradients. Useful for finding good edge
            threshold values quickly when starting an analysis with new data.
            Default is None.
        debug_target_locating : bool, optional
            If True, then the target locating steps from the
            :py:class:`.RegionDetector` class will be displayed upon first image
            evaluation.
        """
        super().__init__()

        # register inputs
        self._reference_image_dir_or_file = None
        self.cropped_x1x2y1y2 = cropped_x1x2y1y2
        self.target_width_meters = target_width_meters
        self.target_height_meters = target_height_meters
        self.canny_edges_gradient = canny_edges_gradient
        self.canny_non_edges_gradient = canny_non_edges_gradient
        self.edge_coarse_width = edge_coarse_width
        self.canny_test_gradients = canny_test_gradients
        self.debug_target_locating = debug_target_locating

        # geometry values in the image
        self.detector: rd.RegionDetector = rd.RegionDetector(
            self.edge_coarse_width, self.canny_edges_gradient, self.canny_non_edges_gradient, self.canny_test_gradients
        )
        self.edges: dict[str, l2.LineXY] = None
        self._corners: dict[str, p2.Pxy] = None
        self.region: reg2.RegionXY = None
        self.transform: pt.PerspectiveTransform = None

        # explainer images
        self.region_detector_cacheables: list[CacheableImage] = []

        # assignments
        self.reference_image_dir_or_file = reference_image_dir_or_file

    @classmethod
    def from_corners(
        cls, corners: dict[str, p2.Pxy], target_width_meters: float, target_height_meters: float
    ) -> "TargetBoardLocatorImageProcessor":
        """
        Creates an instance of this class using the given corner pixels as the
        corners of the target board in the source images. Does not require a
        reference image.

        This method is useful for re-creating the
        TargetBoardLocatorImageProcessor class for further analysis after the
        initial computation, or for manually identified target board corners.

        Parameters
        ----------
        corners : dict[str, p2.Pxy]
            The corners that identify the target board in the reference image.
            Should contain values for top-left 'tl', top-right 'tr',
            bottom-right 'br', and bottom-left 'bl'.
        target_width_meters : float
            Width of the target board in meters.
        target_height_meters : float
            Height of the target board in meters.
        """
        # validate the input
        corner_names = ['tl', 'tr', 'br', 'bl']
        for corner_name in corner_names:
            if corner_name not in corners:
                lt.error_and_raise(
                    ValueError,
                    f"In TargetBoardLocatorImageProcessor.from_corners(): "
                    + f"'corners' parameter should have keys {corner_names}, but instead has {list(corners.keys())}",
                )

        # make sure the dict is in the correct order
        corners_in_order = {}
        for corner_name in corner_names:
            corners_in_order[corner_name] = corners[corner_name]
        corners = corners_in_order

        # build the target board locator
        ret = cls(
            reference_image_dir_or_file=None,
            cropped_x1x2y1y2=None,
            target_width_meters=target_width_meters,
            target_height_meters=target_height_meters,
            canny_edges_gradient=0,
            canny_non_edges_gradient=0,
        )
        ret.corners = corners
        ret.edges = {
            'top': l2.LineXY.from_two_points(corners['tl'], corners['tr']),
            'bottom': l2.LineXY.from_two_points(corners['bl'], corners['br']),
            'left': l2.LineXY.from_two_points(corners['tl'], corners['bl']),
            'right': l2.LineXY.from_two_points(corners['tr'], corners['br']),
        }
        ret.region = reg2.RegionXY.from_vertices(p2.Pxy.from_list(corners.values()))

        return ret

    @property
    def corners(self) -> dict[str, p2.Pxy] | None:
        return self._corners

    @corners.setter
    def corners(self, val: dict[str, p2.Pxy]):
        lt.info(f"In TargetBoardLocatorImageProcessor: corners are {val}")

        # set the corners
        self._corners = val

        # build the transform used to isolate the target board from the rest of the image
        meters_x = [0, self.target_width_meters, self.target_width_meters, 0]  # tl, tr, br, bl
        meters_y = [0, 0, self.target_height_meters, self.target_height_meters]
        meters_tltrbrbl = p2.Pxy((meters_x, meters_y))
        pixels_tltrbrbl = [self.corners['tl'], self.corners['tr'], self.corners['br'], self.corners['bl']]
        self.transform = pt.PerspectiveTransform(pixels_tltrbrbl, meters_tltrbrbl)

    @property
    def reference_image_dir_or_file(self) -> str:
        return self._reference_image_dir_or_file

    @reference_image_dir_or_file.setter
    def reference_image_dir_or_file(self, reference_image_dir_or_file: str):
        # ignore if the value hasn't changed
        if (self._reference_image_dir_or_file is None and reference_image_dir_or_file is None) or (
            self._reference_image_dir_or_file == reference_image_dir_or_file
        ):
            return

        # validate input
        if not ft.file_exists(reference_image_dir_or_file, error_if_exists_as_dir=False) and not ft.directory_exists(
            reference_image_dir_or_file, error_if_exists_as_file=False
        ):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in TargetBoardLocatorImageProcessor(): "
                + f"image or directory {reference_image_dir_or_file} does not exist!",
            )

        # normalize input
        reference_image_dir_or_file = ft.norm_path(reference_image_dir_or_file)

        # keep a consistent state
        if self.transform is not None:
            lt.warn(
                "Warning in setter for TargetBoardLocatorImageProcessor.reference_image_dir_or_file: "
                + "discarding previously computed target board locations."
            )
            self.transform = None
            self.edges = None
            self.corners = None
            self.region = None
            self.detector = None

        # assign the value
        self._reference_image_dir_or_file = reference_image_dir_or_file

    def _find_rectangle_in_reference_image(self):
        if self.transform is not None:
            return

        # import here to avoid cyclic references
        from opencsp.common.lib.cv.spot_analysis.image_processor import AverageByGroupImageProcessor
        from opencsp.common.lib.cv.spot_analysis.image_processor import ConvolutionImageProcessor
        from opencsp.common.lib.cv.spot_analysis.image_processor import CroppingImageProcessor
        from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis

        # Create necessary directories
        cache_dir = os.path.join(orp.opencsp_temporary_dir(), "target_board_locator")
        ft.create_directories_if_necessary(cache_dir)

        # Compile a list of all reference images
        if os.path.isdir(self.reference_image_dir_or_file):
            image_filenames = ft.files_in_directory(self.reference_image_dir_or_file, files_only=True)
            image_files: list[str] = []
            for filename in image_filenames:
                file_path_name_ext = os.path.join(self.reference_image_dir_or_file, filename)
                image_files.append(ft.norm_path(file_path_name_ext))
        elif os.path.isfile(self.reference_image_dir_or_file):
            image_files = [ft.norm_path(self.reference_image_dir_or_file)]
        elif self.reference_image_dir_or_file is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in TargetBoardLocatorImageProcessor._find_rectangle_in_reference_image(): "
                + f"must assign the reference_image_dir_or_file first, but is None.",
            )
        else:
            lt.error_and_raise(
                FileNotFoundError,
                "Error in TargetBoardLocatorImageProcessor._find_rectangle_in_reference_image(): "
                + f"reference_image_dir_or_file \"{self.reference_image_dir_or_file}\" is neither a directory or a file.",
            )

        # Verify that all the reference images exist
        for image_file in image_files:
            if not ft.file_exists(image_file):
                lt.error_and_raise(
                    FileNotFoundError,
                    "Error in TargetBoardLocatorImageProcessor._find_rectangle_in_reference_image(): "
                    + f"image {image_file} does not exist!",
                )

        # Load (and average) the reference images
        image_processors = {
            'Avg': AverageByGroupImageProcessor(lambda o: 0, lambda l: None),
            'Conv': ConvolutionImageProcessor(kernel="gaussian", diameter=3),
        }
        if self.cropped_x1x2y1y2:
            image_processors['Crop'] = CroppingImageProcessor(*self.cropped_x1x2y1y2)

        sa = SpotAnalysis("averager", list(image_processors.values()))
        sa.set_primary_images(image_files)
        preprocessed_operable = next(iter(sa))
        preprocessed_image = preprocessed_operable.primary_image.nparray

        # Get the target board region
        width, height = preprocessed_image.shape[1], preprocessed_image.shape[0]
        image_center = (width / 2, height / 2)
        debug_canny_settings = self.canny_test_gradients is not None
        canny, boundary_pixels = self.detector.find_boundary_pixels_in_image(
            preprocessed_image,
            approx_center_pixel=image_center,
            debug_canny_settings=debug_canny_settings,
            debug_blob_analysis=self.debug_target_locating,
            debug_ray_projection=self.debug_target_locating,
        )

        self.edges, self.corners, self.region = self.detector.find_rectangular_region(
            boundary_pixels,
            canny,
            debug_edge_groups=self.debug_target_locating,
            debug_edge_assignment=self.debug_target_locating,
        )

        # Get the explainer images
        self.region_detector_cacheables = [CacheableImage(v[1]) for v in self.detector.summary_visualizations]

    def _isolate_and_annotate_target(
        self, image: np.ndarray, image_processor_notes: list[tuple[str, list[str]]]
    ) -> tuple[np.ndarray, np.ndarray]:
        # visualize the edges of the target
        annotated_target = image.copy()
        annotated_target = self.detector.visualize_edges_corners(
            annotated_target, self.edges, self.corners, thickness=1
        )

        # visualize the center of the target
        target_center_meters = p2.Pxy((self.target_width_meters / 2, self.target_height_meters / 2))
        target_center_pixels = self.transform.meters_to_pixels(target_center_meters)
        target_center_int = (int(target_center_pixels.x[0]), int(target_center_pixels.y[0]))
        annotated_target = cv.drawMarker(
            annotated_target, target_center_int, color=color.magenta().rgb_255(), markerSize=5
        )

        # extract the target
        isolated_target = image.copy()
        isolated_target = self.transform.transform_image(isolated_target)
        annotated_target = self.transform.transform_image(annotated_target, buffer_width_px=10)

        return isolated_target, annotated_target

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray.squeeze()

        # get the target from the image
        self._find_rectangle_in_reference_image()
        isolated_target, annotated_target = self._isolate_and_annotate_target(image, operable.image_processor_notes)
        isolated_cacheable, annotated_cacheable = CacheableImage(isolated_target), CacheableImage(annotated_target)

        # get the visualizations explaining how we found the target board
        my_algorithm_images = self.region_detector_cacheables

        # get the visualization images that show this operable's extracted
        # target board images
        my_visualization_images = [annotated_cacheable]

        visualization_images = copy.copy(operable.visualization_images)
        visualization_images[self] = my_visualization_images
        algorithm_images = copy.copy(operable.algorithm_images)
        algorithm_images[self] = my_algorithm_images
        ret = dataclasses.replace(
            operable,
            primary_image=isolated_cacheable,
            visualization_images=visualization_images,
            algorithm_images=algorithm_images,
        )
        return [ret]
