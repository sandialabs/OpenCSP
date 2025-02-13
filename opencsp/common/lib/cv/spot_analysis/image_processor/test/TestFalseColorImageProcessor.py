import numpy as np
import numpy.testing as nptest
import os
from PIL import Image
import unittest
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import FalseColorImageProcessor

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestFalseColorImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FalseColorImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "FalseColorImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_jet_large(self):
        large_grayscale_image = np.arange(1530, dtype=np.int16)
        large_grayscale_image = np.expand_dims(large_grayscale_image, axis=1)
        large_grayscale_image = np.broadcast_to(large_grayscale_image, (1530, 1530))
        self.assertEqual(large_grayscale_image[0, 0], 0)
        self.assertEqual(large_grayscale_image[0, 1529], 0)
        self.assertEqual(large_grayscale_image[1529, 0], 1529)
        self.assertEqual(large_grayscale_image[1529, 1529], 1529)

        processor = FalseColorImageProcessor(map_type="large")
        operable = processor.process_image(SpotAnalysisOperable(large_grayscale_image))[0]
        actual_result = operable.primary_image.nparray
        actual_path_name_ext = os.path.join(self.out_dir, "test_jet_large.png")
        it.numpy_to_image(actual_result, "clip").save(actual_path_name_ext)

        expected_path_name_ext = os.path.join(self.data_dir, "test_jet_large.png")
        expected_result = np.asarray(Image.open(expected_path_name_ext))

        all_colors = [str(color) for color in actual_result[:, :]]
        self.assertEqual(np.unique(all_colors).size, 1530)
        nptest.assert_array_equal(actual_result, expected_result)

    def test_jet_human(self):
        large_grayscale_image = np.arange(1020, dtype=np.int16)
        large_grayscale_image = np.expand_dims(large_grayscale_image, axis=1)
        large_grayscale_image = np.broadcast_to(large_grayscale_image, (1020, 1020))
        self.assertEqual(large_grayscale_image[0, 0], 0)
        self.assertEqual(large_grayscale_image[0, 1019], 0)
        self.assertEqual(large_grayscale_image[1019, 0], 1019)
        self.assertEqual(large_grayscale_image[1019, 1019], 1019)

        processor = FalseColorImageProcessor(map_type="human")
        operable = processor.process_image(SpotAnalysisOperable(large_grayscale_image))[0]
        actual_result = operable.primary_image.nparray
        actual_path_name_ext = os.path.join(self.out_dir, "test_jet_human.png")
        it.numpy_to_image(actual_result, "clip").save(actual_path_name_ext)

        expected_path_name_ext = os.path.join(self.data_dir, "test_jet_human.png")
        expected_result = np.asarray(Image.open(expected_path_name_ext))

        all_colors = [str(color) for color in actual_result[:, :]]
        self.assertEqual(np.unique(all_colors).size, 1020)
        nptest.assert_array_equal(actual_result, expected_result)


if __name__ == "__main__":
    unittest.main()
