import numpy as np
import numpy.testing as nptest
import os
import unittest
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import (
    PopulationStatisticsImageProcessor,
)

import opencsp.common.lib.tool.file_tools as ft


class TestPopulationStatisticsImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "PopulationStatisticsImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "PopulationStatisticsImageProcessor")

        im1 = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
        im2 = np.array([[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]])
        im3 = np.array([[[0, 1, 2], [1, 2, 3]], [[2, 3, 4], [3, 4, 5]]])
        self.images = [CacheableImage(im) for im in [im1, im2, im3]]
        self.operables = [SpotAnalysisOperable(im) for im in self.images]

        self.processor = PopulationStatisticsImageProcessor(min_pop_size=1)
        self.processor._allowed_memory_footprint = pow(2, 30)

    def test_min_max_pop1(self):
        stats = self.processor.process_image(self.operables[0], is_last=False)[0].population_statistics
        nptest.assert_array_equal(stats.minf, np.array([1, 1, 1]))
        nptest.assert_array_equal(stats.maxf, np.array([1, 1, 1]))

        stats = self.processor.process_image(self.operables[1], is_last=False)[0].population_statistics
        nptest.assert_array_equal(stats.minf, np.array([1, 1, 1]))
        nptest.assert_array_equal(stats.maxf, np.array([2, 2, 2]))

        stats = self.processor.process_image(self.operables[2], is_last=True)[0].population_statistics
        nptest.assert_array_equal(stats.minf, np.array([0, 1, 1]))
        nptest.assert_array_equal(stats.maxf, np.array([3, 4, 5]))

    def test_min_max_pop3(self):
        self.processor.min_pop_size = 3

        # Results won't be returned until the min_pop_size has been reached
        empty1 = self.processor.process_image(self.operables[0], is_last=False)
        empty2 = self.processor.process_image(self.operables[1], is_last=False)
        # this is the last one, but by not setting is_last we can test the min_pop_size attribute
        operables = self.processor.process_image(self.operables[2], is_last=False)
        stats = [operable.population_statistics for operable in operables]

        self.assertEqual(len(empty1), 0)
        self.assertEqual(len(empty2), 0)
        self.assertEqual(len(operables), 3)

        nptest.assert_array_equal(stats[0].minf, np.array([1, 1, 1]))
        nptest.assert_array_equal(stats[0].maxf, np.array([1, 1, 1]))
        nptest.assert_array_equal(stats[1].minf, np.array([1, 1, 1]))
        nptest.assert_array_equal(stats[1].maxf, np.array([2, 2, 2]))
        nptest.assert_array_equal(stats[2].minf, np.array([0, 1, 1]))
        nptest.assert_array_equal(stats[2].maxf, np.array([3, 4, 5]))

    def test_pop_neg1(self):
        self.processor.min_pop_size = -1

        # Results won't be returned until is_last has been reached
        empty1 = self.processor.process_image(self.operables[0], is_last=False)
        empty2 = self.processor.process_image(self.operables[1], is_last=False)
        operables = self.processor.process_image(self.operables[2], is_last=True)

        self.assertEqual(len(empty1), 0)
        self.assertEqual(len(empty2), 0)
        self.assertEqual(len(operables), 3)

    def test_is_last(self):
        self.processor.min_pop_size = 1e10

        # Results won't be returned until is_last has been reached
        empty1 = self.processor.process_image(self.operables[0], is_last=False)
        empty2 = self.processor.process_image(self.operables[1], is_last=False)
        operables = self.processor.process_image(self.operables[2], is_last=True)

        self.assertEqual(len(empty1), 0)
        self.assertEqual(len(empty2), 0)
        self.assertEqual(len(operables), 3)

    def test_avg_pop1(self):
        stats = self.processor.process_image(self.operables[0], is_last=False)[0].population_statistics
        nptest.assert_array_almost_equal(stats.avgf_rolling_window, np.array([1, 1, 1]))

        stats = self.processor.process_image(self.operables[1], is_last=False)[0].population_statistics
        nptest.assert_array_almost_equal(stats.avgf_rolling_window, np.array([2, 2, 2]))

        stats = self.processor.process_image(self.operables[2], is_last=True)[0].population_statistics
        expected = np.array([0 + 1 + 2 + 3, 1 + 2 + 3 + 4, 2 + 3 + 4 + 5]) / 4
        nptest.assert_array_almost_equal(stats.avgf_rolling_window, expected)

    def test_avg_pop3(self):
        self.processor.min_pop_size = 3

        # Results won't be returned until min_pop_size has been reached
        empty1 = self.processor.process_image(self.operables[0], is_last=False)
        empty2 = self.processor.process_image(self.operables[1], is_last=False)
        operables = self.processor.process_image(self.operables[2], is_last=True)
        stats = [operable.population_statistics for operable in operables]

        nptest.assert_array_almost_equal(stats[0].avgf_rolling_window, np.array([1, 1, 1]))
        nptest.assert_array_almost_equal(stats[1].avgf_rolling_window, np.array([1.5, 1.5, 1.5]))
        expected = np.array([3 + 4 + 5 + 6, 4 + 5 + 6 + 7, 5 + 6 + 7 + 8]) / (3 * 4)
        nptest.assert_array_almost_equal(stats[2].avgf_rolling_window, expected)


if __name__ == "__main__":
    unittest.main()
