import dataclasses
import numpy as np

from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImagesProcessor
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class LogScaleImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """Converts the input images into a log scale."""

    def __init__(self, max_value_input=0, cummulative_max_value_input=False, max_value_output=65535):
        """A video processor that adjusts the scale of input images to a log scale.

        Args::
            max_value_input: int
                The expected maximum value in the input images. For example,
                images from a 12-bit camera will be expected to cap out at 4095.
                If zero (auto), then this value is adjusted during execution to
                the maximum value of the current (and all previous) primary
                images. Auto will use the population_statistics, if available
                from a preceeding PopulationStatisticsImageProcessor. Defaults
                to 0.
            cummulative_max_value_input: bool
                If True, then the "max_value_input" will be set to the maximum
                of the current AND all previous primary images. If False, then
                only the maximum value of the current primary image will be
                used. Defaults to False.
            max_value_output: int
                The maximum value to re-scale the max_value_input to. Everything
                less than max_value_input will be less than this number in the
                output images. Defaults to 65535.
        """
        super().__init__("LogScaleImages")
        self.original_max_value_input = max_value_input
        self.max_value_input = max_value_input
        self.cummulative_max_value_input = cummulative_max_value_input
        self.max_value_output = max_value_output

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[np.ndarray]:
        primary_image: np.ndarray = operable.primary_image.nparray
        current_max_value_input = operable.max_popf

        # update input maximum values to the largest observed value
        if self.original_max_value_input == 0:
            if self.cummulative_max_value_input:
                if operable.population_statistics != None:
                    self.max_value_input = current_max_value_input
                else:
                    self.max_value_input = np.max([self.max_value_input, current_max_value_input])
            else:
                self.max_value_input = current_max_value_input

        # determine the necessary output bit depth
        for data_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if np.iinfo(data_type).max >= self.max_value_output:
                break
        if primary_image.dtype != data_type:
            primary_image = primary_image.astype(data_type)

        # log and rescale the image
        log_image = np.log(primary_image + 1)
        log_max = np.max(log_image)
        target_max_val = self.max_value_output * (current_max_value_input / self.max_value_input)
        scalar = target_max_val / log_max
        processed_image = scalar * log_image

        ret = dataclasses.replace(operable, primary_image=processed_image)
        return [ret]
