import dataclasses
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisPopulationStatistics import SpotAnalysisPopulationStatistics
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


@dataclasses.dataclass
class _RollingWindowOperableStats:
    operable: SpotAnalysisOperable
    sum_per_color: np.ndarray
    dims: np.ndarray


class PopulationStatisticsImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Generates statistics for groups of images.

    A group of images is held until enough have been seen to generate statistics off of. Once the required number of
    images has been reached, then images will start being released one at a time with the statistics for the group
    up until that point.

    Some use cases for this class could include automatically determining the maximum pixel value during streaming
    to select an appropriate bit depth, using the rolling average for exposure calibration, or leveling all images
    by subtracting the gloal pixel minimum.
    """

    def __init__(self, min_pop_size=1, target_rolling_window_size=1, initial_min: int = None, initial_max: int = None):
        """
        Parameters
        ----------
        min_pop_size : int, optional
            The minimum number of images that must be seen before any images (and their statistics) are released to the
            next image processor. -1 to wait for all images. By default 1
        target_rolling_window_size : int, optional
            Number of images used to determine rolling averages. The first N-1 images are not held back while waiting
            for this target. By default 1
        initial_min : int, optional
            Initial value used to estimate the population minimum. If None, then the minimum of the first image seen is
            used. By default None
        initial_max : int, optional
            Initial value used to estimage the population maximum. If None, then the maximum of the first image seen is
            used. By default None
        """
        super().__init__()

        if min_pop_size > 0:
            if target_rolling_window_size > min_pop_size:
                lt.warn(
                    "Warning in PopulationStatisticsImageProcessor: "
                    + "trimming target_rolling_window_size to be <= min_pop_size! "
                    + f"({target_rolling_window_size=}, {min_pop_size=})"
                )
                target_rolling_window_size = min_pop_size

        self.min_pop_size = min_pop_size
        """ The number of operables that must be analyzed before allowing any
        through for further processing. -1 if all images are required to be
        analyzed before continuing. """
        self.target_rolling_window_size = target_rolling_window_size
        """ The number of operables that are held for calculating rolling
        average statistics. Truncated to be <= min_pop_size. """
        self.curr_stats: SpotAnalysisPopulationStatistics = None
        """ The current statistics, which get updated with each image seen. None
        if min_pop_size hasn't been met yet. """
        self.initial_min = [initial_min] if initial_min is not None else None
        self.initial_max = [initial_max] if initial_max is not None else None
        self.initial_operables: list[SpotAnalysisOperable] = []
        """ The initial operables gathered while waiting for min_pop_size. """
        self.rolling_window_operables: list[SpotAnalysisOperable] = []
        """ The last N operables seen, for the purpose of gathering statistics
        on a rolling window of imagess. """
        self._rolling_window_stats: list[_RollingWindowOperableStats] = []
        """ Optimization to require only one operable per call to
        _calculate_rolling_window(). """

    def _calculate_rolling_window(
        self,
        curr_stats: SpotAnalysisPopulationStatistics,
        operable: SpotAnalysisOperable,
        window: list[SpotAnalysisOperable],
    ):
        """Analyze the given operables to generate rolling window statistics."""
        ret: SpotAnalysisPopulationStatistics = dataclasses.replace(curr_stats)

        # remove operables no longer in the window
        to_remove = []
        for stat in self._rolling_window_stats:
            if stat.operable not in window:
                to_remove.append(stat)
        for stat in to_remove:
            self._rolling_window_stats.remove(stat)

        # add this new operable
        found_operable = False
        for stat in self._rolling_window_stats:
            if stat.operable == operable:
                found_operable = True
                break
        if not found_operable:
            image = operable.primary_image.nparray
            dims, _ = it.dims_and_nchannels(image)
            sum_per_color = np.sum(image, axis=(0, 1))
            self._rolling_window_stats.append(_RollingWindowOperableStats(operable, sum_per_color, dims))

        # calculate statistics
        first_stat = self._rolling_window_stats[0]
        rolling_colors_sum = np.array(first_stat.sum_per_color)
        rolling_pixels_cnt = first_stat.dims[0] * first_stat.dims[1]

        for stat in self._rolling_window_stats[1:]:
            rolling_colors_sum += stat.sum_per_color
            rolling_pixels_cnt += stat.dims[0] * stat.dims[1]

        ret.avgf_rolling_window = rolling_colors_sum / rolling_pixels_cnt
        ret.window_size = len(self._rolling_window_stats)

        return ret

    def _calculate_cummulative(self, curr_stats: SpotAnalysisPopulationStatistics, operable: SpotAnalysisOperable):
        """Analyze the given operable and update the cummulative statistics."""
        ret: SpotAnalysisPopulationStatistics = dataclasses.replace(curr_stats)

        image = operable.primary_image.nparray
        min_colors, max_colors = it.min_max_colors(image)

        if ret.minf is not None:
            ret.minf = np.min([ret.minf, min_colors], axis=0)
        else:
            ret.minf = min_colors

        if ret.maxf is not None:
            ret.maxf = np.max([ret.maxf, max_colors], axis=0)
        else:
            ret.maxf = max_colors

        ret.population_size += 1

        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        ret: list[SpotAnalysisOperable] = []
        self.rolling_window_operables.append(operable)

        if self.curr_stats == None:
            # We haven't hit the minimum population size yet as of the previous call to _execute().
            # Maybe we'll reach the minimum population size this time?
            if (len(self.initial_operables) < self.min_pop_size - 1) or (self.min_pop_size == -1):
                if not is_last:
                    # We still haven't reached the minimum population size.
                    self.initial_operables.append(operable)
                    return []
                else:
                    pass

            # We've reached the minimum population size (or the end of the images stream, as indicated by is_last).
            self.curr_stats = SpotAnalysisPopulationStatistics(minf=self.initial_min, maxf=self.initial_max)
            for prior_operable in self.initial_operables:
                self.curr_stats = self._calculate_rolling_window(
                    self.curr_stats, prior_operable, self.rolling_window_operables
                )
                self.curr_stats = self._calculate_cummulative(self.curr_stats, prior_operable)
                ret.append(dataclasses.replace(prior_operable, population_statistics=self.curr_stats))

        # do some calculations
        self.curr_stats = self._calculate_rolling_window(self.curr_stats, operable, self.rolling_window_operables)
        self.curr_stats = self._calculate_cummulative(self.curr_stats, operable)
        ret.append(dataclasses.replace(operable, population_statistics=self.curr_stats))

        # release operables that we no longer need
        self.initial_operables.clear()
        while len(self.rolling_window_operables) > self.target_rolling_window_size - 1:
            self.rolling_window_operables.pop(0)

        return ret
