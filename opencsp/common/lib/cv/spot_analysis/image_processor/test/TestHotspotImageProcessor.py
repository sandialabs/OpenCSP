import os
import unittest

import numpy.testing as npt

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft


class TestHotspotImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "HotspotImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "HotspotImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_internal_shapes_int(self):
        processor = HotspotImageProcessor(desired_shape=21)
        # assume that factor = 2
        # start = odd(21 * 2) = odd(42) = 43
        # assume that reduction = min(10, 21 / 3) = 6
        expected = [43, 37, 31, 25, 21]
        self.assertEqual(expected, processor.internal_shapes)

    def test_internal_shapes_tuple(self):
        processor = HotspotImageProcessor(desired_shape=(3, 21))
        # assume that factor = 2
        # start_x = odd(21 * 2) = odd(42) = 43
        # start_y = odd(3 * 2) = odd(6) = 7
        # assume that reduction = min(10, 7 / 3) = 1
        expected_x = [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21]
        expected_y = [7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        expected = [tuple(v) for v in zip(expected_y, expected_x)]
        self.assertEqual(expected, processor.internal_shapes)

    def test_hotspot_location(self):
        # process the example
        example_image_path = ft.join(self.data_dir, "09W01_inpainted.png")
        example_image = CacheableImage.from_single_source(example_image_path)
        example_operable = SpotAnalysisOperable(example_image, example_image_path)
        processor = HotspotImageProcessor(41, record_visualization=True)
        result = processor.process_operable(example_operable, is_last=True)[0]

        # get the results
        actual_position: HotspotAnnotation = result.annotations[0]
        actual_image = result.visualization_images[processor][0]
        actual_image_path = ft.join(self.out_dir, self._testMethodName + "_hotspot_location_visualization.png")
        actual_image.to_image().save(actual_image_path)

        # verify the results
        self.assertEqual(actual_position.origin.astuple(), (812.0, 1258.0))
        expected_image_path = ft.join(self.data_dir, "hotspot_location_visualization.png")
        expected_image = CacheableImage.from_single_source(expected_image_path)
        npt.assert_allclose(actual_image.nparray, expected_image.nparray, atol=2)


if __name__ == "__main__":
    unittest.main()
