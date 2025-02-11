import copy
import dataclasses
import enum

import cv2
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
from opencsp.common.lib.geometry.RegionXY import RegionXY
from contrib.common.lib.geometry.RectXY import RectXY
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.file_tools as ft


class SIPNotEnoughPointsAction(enum.Enum):
    RAISE_RUNTIME_ERROR = 1
    DISCARD_IMAGE = 2
    IGNORE_KEEP_UNSTABILIZED = 3


class StabilizationImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Stabilizes images from one frame to the next, such that the features are
    kept in consistent locations between images.
    """

    def __init__(
        self,
        stabilization_frame_range=1,
        reference_image_path: str = None,
        feature_region: RegionXY | np.ndarray = None,
        stabilization_region: RectXY = None,
        min_points=3,
        not_enough_points_action=SIPNotEnoughPointsAction.RAISE_RUNTIME_ERROR,
        include_debug_images=False,
    ):
        """
        Parameters
        ----------
        stabilization_frame_range : int, optional
            Number of images to look back at for stabilization. Ignored if
            reference_image_path is set. Default 1.
        reference_image_path : str, optional
            If set, then stabilization_frame_range is ignored and instead this
            image is used as the anchor point that all other images are compared
            to. By default None.
        feature_region : RegionXY | np.ndarray, optional
            Area to search for features to stabilize on. Can be either a region
            of contained pixels, or a mask where 1's are included in the search
            and 0's aren't included. By default None.
        stabilization_region: Rect, optional
            If set, then images are cropped to this region before features are
            identified. The resulting transforms are applied to the full image.
            Defualt is None.
        min_points : int, optional
            Minimum number of valid tracking features needed before image
            transformation will be done. The not_enough_points_action will be
            taken for images that don't meet this criteria. By default 3.
        not_enough_points_action : _type_, optional
            What to do with images that don't meet the min_points criteria, by
            default SIPNotEnoughPointsAction.RAISE_RUNTIME_ERROR
        include_debug_images : bool, optional
            If True, then two operables will be returned for every operable that
            matches, where the first operable will be the unstabilized image
            with annotated features and the second will be stabilized. By
            default False.
        """
        super().__init__()

        self.stabilization_frame_range = stabilization_frame_range
        self.previous_operables: list[SpotAnalysisOperable] = []

        self.reference_image_path: str = reference_image_path
        self.reference_operable: SpotAnalysisOperable = None
        self.waiting_operables: list[SpotAnalysisOperable] = []

        self.feature_region: RegionXY | np.ndarray = feature_region
        self.dense_feature_region: np.ndarray = None
        self.stabilization_region: RectXY = stabilization_region
        self.images_to_features: dict[CacheableImage, np.ndarray] = {}
        self.images_to_transform: dict[CacheableImage, np.ndarray] = {}
        self.min_points = min_points
        self.not_enough_points_action = not_enough_points_action
        self.include_debug_images = include_debug_images

        if self.reference_image_path is not None:
            if not ft.file_exists(self.reference_image_path):
                lt.error_and_raise(
                    FileNotFoundError,
                    "Error in StabilizationImageProcessor(): "
                    + f"reference_image_path {reference_image_path} does not exist!",
                )

    def apply_stabilization_region(self, image: np.ndarray) -> np.ndarray:
        image = np.copy(image)

        if self.stabilization_region is None:
            return image

        else:
            r = self.stabilization_region
            image = image[int(r.top) : int(r.bottom), int(r.left) : int(r.right)]
            return image

    def unapply_stabilization_region(self, features: np.ndarray) -> np.ndarray:
        features = np.copy(features)

        if self.stabilization_region is None:
            return features

        else:
            if features.ndim > 2 and features.shape[2] == 2:
                features[:, :, 0] += self.stabilization_region.left
                features[:, :, 1] += self.stabilization_region.top
            else:
                features[:, 0] += self.stabilization_region.left
                features[:, 1] += self.stabilization_region.top
            return features

    def get_feature_region(self, image: np.ndarray) -> np.ndarray:
        if self.feature_region is None:
            return np.ones_like(image)

        elif isinstance(self.feature_region, np.ndarray):
            return self.feature_region

        elif isinstance(self.feature_region, RegionXY):
            if self.dense_feature_region is None:
                reg: RegionXY = self.feature_region
                contained = reg.as_mask(np.arange(image.shape[1]), np.arange(image.shape[0]))
                self.dense_feature_region = np.zeros_like(image)
                self.dense_feature_region[np.where(contained)] = 1
            return self.dense_feature_region

        else:
            lt.error_and_raise(
                TypeError,
                "Error in StabilizationIamgeProcessor.get_feature_region(): "
                + f"expected roi to be a RegionXY or Numpy.NDArray, but is a {type(self.feature_region)}!",
            )

    def find_good_features(self, operable: SpotAnalysisOperable) -> np.ndarray:
        cacheable = operable.primary_image
        image = cacheable.nparray

        # find features to track
        try:
            features = self.images_to_features[cacheable]
        except KeyError:
            mask = self.get_feature_region(image)
            image = self.apply_stabilization_region(image)
            mask = self.apply_stabilization_region(mask)

            features = cv2.goodFeaturesToTrack(
                image, maxCorners=500, qualityLevel=0.3, minDistance=30, blockSize=21, mask=mask
            )

            self.images_to_features[cacheable] = features

        return features

    def draw_features(self, image: np.ndarray, features: np.ndarray, color=None) -> SpotAnalysisOperable:
        if color is None:
            color = (255, 0, 0)

        # convert to colored image
        annotated = np.copy(image)
        if annotated.ndim < 3 or annotated.shape[2] < 3:
            annotated = cv2.cvtColor(annotated, getattr(cv2, f"COLOR_GRAY2RGB"))

        # add a circle for each feature
        for i in range(features.shape[0]):
            location = tuple(features[i][0].astype(np.int_).tolist())
            cv2.circle(annotated, location, radius=11, color=color, thickness=2)

        return annotated

    def stabilize_relative_image(
        self, new_operable: SpotAnalysisOperable, old_operable: SpotAnalysisOperable, cummulative_stabilization=False
    ) -> tuple[SpotAnalysisOperable, SpotAnalysisOperable]:
        # find features to track
        old_features = self.find_good_features(old_operable)

        # track the features
        new_features = copy.copy(old_features)
        statuses = np.zeros((len(new_features)), dtype=np.uint8)
        new_image = self.apply_stabilization_region(new_operable.primary_image.nparray)
        old_image = self.apply_stabilization_region(old_operable.primary_image.nparray)
        cv2.calcOpticalFlowPyrLK(old_image, new_image, old_features, new_features, statuses)

        # filter to valid points
        from_points = old_features[np.where(statuses == 1)[0]]
        to_points = new_features[np.where(statuses == 1)[0]]

        # create the annotated image from the tracked features
        annotated_operable = None
        if self.include_debug_images:
            annotated_image = np.copy(new_operable.primary_image.nparray)
            annotated_image = self.draw_features(
                annotated_image, self.unapply_stabilization_region(to_points), (0, 0, 255)
            )
            annotated_cacheable = CacheableImage(annotated_image)
            annotated_operable = dataclasses.replace(new_operable, primary_image=annotated_cacheable)

        # check if tracking worked
        if np.sum(statuses) < self.min_points:
            old_image_name_ext = old_operable.get_primary_path_nameext()[1]
            new_image_name_ext = new_operable.get_primary_path_nameext()[1]
            errmsg = (
                "Error in StabilizationImageProcessor.stabilize_relative_image(): "
                + f"Not enough stabilization points have been found in {new_image_name_ext} relative to {old_image_name_ext}!"
            )

            if self.not_enough_points_action == SIPNotEnoughPointsAction.RAISE_RUNTIME_ERROR:
                lt.error_and_raise(RuntimeError, errmsg)
            elif self.not_enough_points_action == SIPNotEnoughPointsAction.DISCARD_IMAGE:
                lt.info(errmsg)
                return None, None
            elif self.not_enough_points_action == SIPNotEnoughPointsAction.IGNORE_KEEP_UNSTABILIZED:
                return annotated_operable, new_operable
            else:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in StabilizationImageProcessor.stabilize_relative_image(): "
                    + f"unknown value {self.not_enough_points_action=}",
                )

        # find the transform
        transform, _ = cv2.estimateAffine2D(to_points, from_points)
        dx, dy = transform[0, 2], transform[1, 2]
        if cummulative_stabilization:
            dx_cummulative, dy_cummulative = dx, dy
            try:
                old_transform = self.images_to_transform[old_operable.primary_image]
                dx_cummulative = dx + old_transform[0, 2]
                dy_cummulative = dy + old_transform[1, 2]
            except KeyError:
                pass
            transform = np.array([[1, 0, dx_cummulative], [0, 1, dy_cummulative]])
            print(f"rel: (dx,dy)={dx,dy}, abs: (dx,dy)={dx_cummulative,dy_cummulative}")
        else:
            transform = np.array([[1, 0, dx], [0, 1, dy]])
            print(f"(dx,dy)={dx,dy}")
        self.images_to_transform[new_operable.primary_image] = transform
        # transform[0, 2] = dy
        # transform[1, 2] = dx

        # apply the transform
        new_image = np.copy(new_operable.primary_image.nparray)
        shape_yx = (new_image.shape[1], new_image.shape[0])
        stabilized_image = cv2.warpAffine(new_image, transform, shape_yx)

        # draw the updated features
        transformed_to_points = np.zeros_like(to_points)
        for i in range(to_points.shape[0]):
            location = self.unapply_stabilization_region(to_points[i])
            rotated = np.matmul(location, transform[:2, :2])
            dx, dy = transform[0, 2], transform[1, 2]
            translated = rotated + transform[:, 2]
            transformed_to_points[i] = translated
        stabilized_image = self.draw_features(stabilized_image, transformed_to_points)

        stabilized_cacheable = CacheableImage(stabilized_image)
        image_processor_notes = copy.copy(new_operable.image_processor_notes) + [
            (self.name, [f"transform {transform}"])
        ]
        stabilized_operable = dataclasses.replace(
            new_operable, primary_image=stabilized_cacheable, image_processor_notes=image_processor_notes
        )

        return annotated_operable, stabilized_operable

    def process_waiting_operables(self):
        ret: list[SpotAnalysisOperable] = []

        for operable in self.waiting_operables:
            if self.include_debug_images:
                annotated, stabilized = self.stabilize_relative_image(operable, self.reference_operable)
                if stabilized is not None:
                    ret.append(annotated)
                    ret.append(stabilized)
            else:
                _, stabilized = self.stabilize_relative_image(operable, self.reference_operable)
                if stabilized is not None:
                    ret.append(stabilized)
        self.waiting_operables.clear()

        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.reference_image_path is not None:
            if ft.join(*operable.get_primary_path_nameext()) == ft.norm_path(self.reference_image_path):
                self.reference_operable = operable
                ret = self.process_waiting_operables()

                features = self.find_good_features(operable)
                annotated_image = self.draw_features(operable.primary_image.nparray, features)
                cacheable = CacheableImage(annotated_image)
                annotated = dataclasses.replace(operable, primary_image=annotated_image)

                ret.append(annotated)

            else:
                self.waiting_operables.append(operable)
                if self.reference_operable is not None:
                    ret = self.process_waiting_operables()
                else:
                    ret = []

        else:
            if len(self.previous_operables) == 0:
                features = self.find_good_features(operable)
                annotated_image = self.draw_features(operable.primary_image.nparray, features)
                cacheable = CacheableImage(annotated_image)
                annotated = dataclasses.replace(operable, primary_image=annotated_image)

                ret = [annotated]

                self.previous_operables.append(operable)

            else:
                if len(self.previous_operables) >= self.stabilization_frame_range:
                    comparison_operable = self.previous_operables[
                        len(self.previous_operables) - self.stabilization_frame_range
                    ]
                else:
                    comparison_operable = self.previous_operables[0]

                _, stabilized = self.stabilize_relative_image(
                    operable, comparison_operable, cummulative_stabilization=True
                )
                if stabilized is not None:
                    ret = [stabilized]
                else:
                    ret = []

                self.previous_operables.append(operable)
                if len(self.previous_operables) > self.stabilization_frame_range:
                    self.previous_operables.pop(0)

        return ret
