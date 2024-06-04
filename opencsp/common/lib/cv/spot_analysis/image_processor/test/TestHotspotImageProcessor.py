import os
import unittest
from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor

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
        # assume that factor = 2.5
        # start = odd(21 * 2.5) = odd(52.5) = 53
        # assume that reduction = 6
        expected = [53, 47, 41, 35, 29, 23, 21]
        self.assertEqual(expected, processor.internal_shapes)

    def test_internal_shapes_tuple(self):
        processor = HotspotImageProcessor(desired_shape=(3, 21))
        # assume that factor = 2.5
        # start_x = odd(21 * 2.5) = odd(52.5) = 53
        # start_y = odd(3 * 2.5) = odd(7.5) = 9
        # assume that reduction = 6
        expected_x = [53, 47, 41, 35, 29, 23, 21]
        expected_y = [9, 3, 3, 3, 3, 3, 3]
        expected = [tuple(v) for v in zip(expected_y, expected_x)]
        self.assertEqual(expected, processor.internal_shapes)


if __name__ == '__main__':
    unittest.main()
