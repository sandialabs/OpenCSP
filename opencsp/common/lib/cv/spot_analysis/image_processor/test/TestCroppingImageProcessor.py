import numpy as np
import os
import unittest
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor

import opencsp.common.lib.tool.file_tools as ft


class TestCroppingImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "CroppingImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "CroppingImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_crop_by_region(self):
        tenbyfive = CacheableImage(np.arange(50).reshape((5, 10)))
        # [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        #  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #  [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
        expected = np.array([[21, 22, 23, 24, 25, 26, 27, 28], [31, 32, 33, 34, 35, 36, 37, 38]])
        operable = SpotAnalysisOperable(tenbyfive, "tenbyfive")

        # crop with a static value
        processor = CroppingImageProcessor.by_region((1, 9, 2, 4))
        result = processor.process_operable(operable)[0]
        cropped_image = result.primary_image.nparray
        np.testing.assert_array_equal(cropped_image, expected)

        # crop with a callable
        processor = CroppingImageProcessor.by_region(lambda op: (1, 9, 2, 4))
        result = processor.process_operable(operable)[0]
        cropped_image = result.primary_image.nparray
        np.testing.assert_array_equal(cropped_image, expected)

    def test_crop_by_center_and_size(self):
        tenbyfive = CacheableImage(np.arange(50).reshape((5, 10)))
        # [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        #  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #  [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
        expected = np.array([[21, 22, 23, 24]])
        operable = SpotAnalysisOperable(tenbyfive, "tenbyfive")

        # crop with static and dynamic values
        for center in [(3, 2), (lambda op: (3, 2))]:
            for width_height in [(4, 1), (lambda op: (4, 1))]:
                processor = CroppingImageProcessor.by_center_and_size(center, width_height)
                result = processor.process_operable(operable)[0]
                cropped_image = result.primary_image.nparray
                np.testing.assert_array_equal(cropped_image, expected)

    def test_bad_input_raises_error(self):
        tenbyfive = np.arange(50).reshape((5, 10))
        
        # bad left
        with self.assertRaises(ValueError):
            processor = CroppingImageProcessor.by_region((-1, 9, 2, 4))
        with self.assertRaises(ValueError):
            processor = CroppingImageProcessor.by_region((9, 9, 2, 4))

        # bad top
        with self.assertRaises(ValueError):
            processor = CroppingImageProcessor.by_region((1, 9, -1, 4))
        with self.assertRaises(ValueError):
            processor = CroppingImageProcessor.by_region((1, 9, 4, 4))
        
        # bad right
        processor = CroppingImageProcessor.by_region((1, 11, 2, 4))
        with self.assertRaises(ValueError):
            processor.process_operable(SpotAnalysisOperable(tenbyfive))

        # bad bottom
        processor = CroppingImageProcessor.by_region((1, 9, 2, 6))
        with self.assertRaises(ValueError):
            processor.process_operable(SpotAnalysisOperable(tenbyfive))

if __name__ == "__main__":
    unittest.main()
