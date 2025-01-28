import dataclasses

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class NullImageSubtractionImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Subtracts the NULL supporting image from the primary image, if there is an associated NULL image.

    Suggested use is by either assigning supporting images manually, or by using the
    SupportingImagesCollectorImageProcessor.
    """

    def __init__(self, default_null_image: np.ndarray = None):
        super().__init__()

        self.default_null_image = default_null_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        null_image: np.ndarray = None

        # validate the input
        if (ImageType.NULL not in operable.supporting_images) or (operable.supporting_images[ImageType.NULL] is None):
            if self.default_null_image is None:
                lt.warning(
                    "Warning in NullImageSubtractionImageProcessor._execute(): "
                    + f"skipping subtraction of null image for {operable.best_primary_pathnameext}. "
                    + "Given image does not have an associated NULL supporting image."
                )
                return [operable]
            else:
                null_image = self.default_null_image
        else:
            null_image = operable.supporting_images[ImageType.NULL].nparray

        # Get the primary image with the null image subtracted.
        # We convert the primary image to type int64 so that we have negative values available. If left as a uint8, then
        # subtracting below 0 would cause values to wrap around, instead.
        primary_image = operable.primary_image.nparray.astype(np.int64)
        null_image = null_image.astype(np.int64)
        new_primary_image = np.clip(primary_image - null_image, 0, np.max(primary_image))
        new_primary_image = new_primary_image.astype(operable.primary_image.nparray.dtype)

        # Create and return the updated operable
        new_primary_cacheable = CacheableImage(new_primary_image)
        new_operable = dataclasses.replace(operable, primary_image=new_primary_cacheable)
        return [new_operable]
