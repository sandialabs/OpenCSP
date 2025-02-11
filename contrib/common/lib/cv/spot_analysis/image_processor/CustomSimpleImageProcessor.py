import dataclasses
from typing import Callable

import cv2
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.image_tools as it


class CustomSimpleImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    A simple image processor that allows the user to modify the primary image
    based on the input operable.
    """

    def __init__(self, custom_function: Callable[[SpotAnalysisOperable], np.ndarray]):
        """
        Parameters
        ----------
        custom_function : Callable[[SpotAnalysisOperable], np.ndarray]
            The function to be provided that takes in an operable and returns a
            modified image. The returned value will be compared to the input
            image and, if the same, then will be interpretted as a null
            operator.
        """
        super().__init__()

        self.custom_function = custom_function

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        processed_image = self.custom_function(operable)

        if np.array_equal(operable.primary_image.nparray, processed_image):
            return [operable]

        else:
            cacheable = CacheableImage(processed_image)
            ret = dataclasses.replace(operable, primary_image=cacheable)
            return [ret]
