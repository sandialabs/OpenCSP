import copy
import dataclasses
from typing import Callable, TYPE_CHECKING

import cv2 as cv
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


if TYPE_CHECKING:
    # import here to avoid cyclic import loop
    from contrib.common.lib.cv.annotations.MomentsAnnotation import MomentsAnnotation


class MomentsImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Uses OpenCV.moments() to get the moments for an image.

    The moments will be assigned to the image with a
    :py:class:`MomentsAnnotation`. The moments can be used to find the image
    centroid and orientation, among other uses.
    """

    def __init__(
        self,
        include_visualization: bool = False,
        centroid_style: rcps.RenderControlPointSeq = None,
        rotation_style: rcps.RenderControlPointSeq = None,
    ):
        """
        Parameters
        ----------
        include_visualization : bool, optional
            True to add a visualization image with a marker at the centroid and
            arrow indicating the orientation. By default False.
        centroid_style: RenderControlPointSeq, optional
            Style used for render the centroid point. By default
            RenderControlPointSeq.defualt(color=magenta).
        rotation_style: RenderControlPointSeq, optional
            Style used for render the rotation line. By default centroid_style.
        """
        super().__init__()

        self.include_visualization = include_visualization
        self.centroid_style = centroid_style
        self.rotation_style = rotation_style

    def calc_moments(self, operable: SpotAnalysisOperable) -> "MomentsAnnotation":
        """Get the moments for the primary image of the given operable."""
        cacheable_image = operable.primary_image
        image = cacheable_image.nparray

        # import here to avoid cyclic import loop
        from contrib.common.lib.cv.annotations.MomentsAnnotation import MomentsAnnotation

        # convert to grayscale
        if (image.ndim > 2) and (image.shape[2] > 1):
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Only uint8 and float32 images are accepted by cv.moments(). Note that
        # this is slightly different than what the documentation says as of
        # 12/9/2024, which claims only int32 and float32 are supported.
        if np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        else:
            image = image.astype(np.uint8)

        # get the moments
        moments = cv.moments(image)

        return MomentsAnnotation(moments, self.centroid_style, self.rotation_style)

    def build_vis(self, operable: SpotAnalysisOperable, moments: "MomentsAnnotation") -> CacheableImage:
        image = operable.primary_image.nparray

        # render
        vis_image = moments.render_to_image(image)

        # build the cacheable image
        cacheable_vis_image = CacheableImage.from_single_source(vis_image)

        return cacheable_vis_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        moments = self.calc_moments(operable)
        annotations = copy.copy(operable.annotations)
        annotations.append(moments)
        ret = dataclasses.replace(operable, annotations=annotations)

        if self.include_visualization:
            vis_image = self.build_vis(operable, moments)
            visualization_images = copy.copy(operable.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            visualization_images[self].append(vis_image)
            ret = dataclasses.replace(ret, visualization_images=visualization_images)

        return [ret]
