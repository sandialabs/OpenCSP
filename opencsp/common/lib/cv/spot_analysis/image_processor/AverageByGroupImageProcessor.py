import dataclasses
from typing import Callable

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import (
    AbstractAggregateImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class AverageByGroupImageProcessor(AbstractAggregateImageProcessor):
    def __init__(
        self,
        images_group_assigner: Callable[[SpotAnalysisOperable], int],
        group_execution_trigger: Callable[[list[tuple[SpotAnalysisOperable, int]]], int | None] = None,
        *vargs,
        **kwargs,
    ):
        """
        Averages the values from groups of images into a single image. All images must have the same shape.
        """
        super().__init__(images_group_assigner, group_execution_trigger, *vargs, **kwargs)

    def _execute_aggregate(
        self, group: int, operables: list[SpotAnalysisOperable], is_last: bool
    ) -> list[SpotAnalysisOperable]:
        # Initialize the image to return.
        lt.debug(f"In AverageByGroupImageProcessor._execute_aggregate(): averaging {len(operables)} images")
        averaged_image = np.array(operables[0].primary_image.nparray).astype(np.int64)

        # build the average image
        for operable in operables[1:]:
            other_image = operable.primary_image.nparray
            if averaged_image.shape != other_image.shape:
                lt.error_and_raise(
                    ValueError,
                    "Error in AverageByGroupImageProcessor._execute_aggregate(): "
                    + f"first image in group has a different shape {averaged_image.shape} than another image's shape {other_image.shape}. "
                    + f"First image is '{operables[0].primary_image_source_path}', current image is '{operable.primary_image_source_path}'.",
                )
            averaged_image += other_image
        averaged_image = averaged_image.astype(np.float_)
        averaged_image /= len(operables)
        averaged_image = averaged_image.astype(operables[0].primary_image.nparray.dtype)

        # collect the list of images that were averaged
        image_names = [operable.primary_image_source_path for operable in operables]

        # build the return operable from the first operable
        averaged_cacheable = CacheableImage(averaged_image, source_path=operables[0].primary_image.source_path)
        ret = dataclasses.replace(operables[0], primary_image=averaged_cacheable)
        ret.image_processor_notes.append(("AverageByGroupImageProcessor", f"averaged_images: {image_names}"))

        return [ret]
