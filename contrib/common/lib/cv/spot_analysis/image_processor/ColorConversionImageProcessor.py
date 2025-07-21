import dataclasses
from typing import Callable

import cv2 as cv
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class ColorConversionImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Uses OpenCV to convert the color of the input images.
    """

    def __init__(self, conversion: int | Callable[[SpotAnalysisOperable], int] = cv.COLOR_RGB2GRAY):
        """
        Parameters
        ----------
        conversion : str
            The type of the conversion to do. Default is RGB -> Grayscale.
        """
        super().__init__()

        # register parameters
        self.conversion = conversion

    def convert_primary_image(self, operable: SpotAnalysisOperable) -> np.ndarray:
        """
        Converts the color space for the primary image of the given operable.
        """
        conversion = self.conversion if isinstance(self.conversion, int) else self.conversion(operable)

        new_image = cv.cvtColor(operable.primary_image.nparray, conversion)

        return new_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        new_primary = self.convert_primary_image(operable)

        new_primary_cacheable = CacheableImage(new_primary)
        new_operable = dataclasses.replace(operable, primary_image=new_primary_cacheable)

        return [new_operable]
