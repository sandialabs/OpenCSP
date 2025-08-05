import os
import unittest

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor import ConvolutionImageProcessor
import opencsp.common.lib.tool.file_tools as ft


class TestConvolutionImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ConvolutionImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "ConvolutionImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        self.ones = np.ones((5, 5))
        """5x5 array of 1's"""
        self.tfive_arr = np.arange(1, 26).reshape((5, 5))
        """5x5 array of values between 1 and 25"""

    def test_validate_initialization(self):
        with self.assertRaises(ValueError):
            ConvolutionImageProcessor(kernel="not a valid kernel")
        with self.assertRaises(ValueError):
            ConvolutionImageProcessor(diameter=-3)
        with self.assertRaises(ValueError):
            ConvolutionImageProcessor(diameter=0)
        with self.assertRaises(ValueError):
            ConvolutionImageProcessor(diameter=2)

    def test_box(self):
        processor = ConvolutionImageProcessor(kernel="box", diameter=3)
        # fmt: off
        expected = np.array([    
            [ 3,  4,  5,  6,  6],
            [ 6,  7,  8,  9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [20, 20, 21, 22, 23] 
        ])                       
        # fmt: on

        # simply test the filter function
        actual = processor._box_filter(self.tfive_arr)
        np.testing.assert_array_equal(expected, actual)

        # test the processor
        cacheable = CacheableImage.from_single_source(self.tfive_arr)
        operable = SpotAnalysisOperable(cacheable, primary_image_source_path="test_box")
        result = processor.process_operable(operable, False)[0]
        np.testing.assert_array_equal(expected, result.primary_image.nparray)

    def test_box_large_diameter(self):
        processor = ConvolutionImageProcessor(kernel="box", diameter=7)
        cacheable = CacheableImage.from_single_source(self.tfive_arr)
        operable = SpotAnalysisOperable(cacheable, primary_image_source_path="test_box_large_diameter")

        with self.assertRaises(RuntimeError):
            processor.process_operable(operable)

    def test_gaussian(self):
        processor = ConvolutionImageProcessor(kernel="gaussian", diameter=3)
        # fmt: off
        expected = np.array([    
            [ 3,  3,  4,  5,  6],
            [ 6,  7,  8,  9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [20, 21, 22, 23, 23] 
        ])                       
        # fmt: on

        # simply test the filter function
        actual = processor._gaussian_filter(self.tfive_arr)
        np.testing.assert_array_equal(expected, actual)

        # test the processor
        cacheable = CacheableImage.from_single_source(self.tfive_arr)
        operable = SpotAnalysisOperable(cacheable, primary_image_source_path="test_gaussian")
        result = processor.process_operable(operable, False)[0]
        np.testing.assert_array_equal(expected, result.primary_image.nparray)

    def test_gaussian_large_diameter(self):
        processor = ConvolutionImageProcessor(kernel="gaussian", diameter=7)
        cacheable = CacheableImage.from_single_source(self.tfive_arr)
        operable = SpotAnalysisOperable(cacheable, primary_image_source_path="test_gaussian_large_diameter")

        with self.assertRaises(RuntimeError):
            processor.process_operable(operable)


if __name__ == "__main__":
    unittest.main()
