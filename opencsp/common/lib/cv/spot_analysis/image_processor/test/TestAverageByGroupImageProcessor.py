import os
import re
import unittest

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AverageByGroupImageProcessor import (
    AverageByGroupImageProcessor,
)
import opencsp.common.lib.tool.file_tools as ft


class TestAverageByGroupImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "AverageByGroupImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "AverageByGroupImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        # # generate test data
        # np.save(os.path.join(self.data_dir, "a1"), np.array([10]), allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "a2"), np.array([20]), allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "b1"), np.array([30]), allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "b2"), np.array([40]), allow_pickle=False)
        self.a1_img = CacheableImage(os.path.join(self.data_dir, "a1.npy"))
        self.a2_img = CacheableImage(os.path.join(self.data_dir, "a2.npy"))
        self.b1_img = CacheableImage(os.path.join(self.data_dir, "b1.npy"))
        self.b2_img = CacheableImage(os.path.join(self.data_dir, "b2.npy"))
        self.a1 = SpotAnalysisOperable(self.a1_img, primary_image_source_path="a1.npy")
        self.a2 = SpotAnalysisOperable(self.a2_img, primary_image_source_path="a2.npy")
        self.b1 = SpotAnalysisOperable(self.b1_img, primary_image_source_path="b1.npy")
        self.b2 = SpotAnalysisOperable(self.b2_img, primary_image_source_path="b2.npy")

        self.assigner = AverageByGroupImageProcessor.group_by_name(re.compile(r"^([a-z]).*"))
        self.triggerer = AverageByGroupImageProcessor.group_trigger_on_change()
        self.processor = AverageByGroupImageProcessor(self.assigner, self.triggerer)

    @staticmethod
    def always_trigger(image_groups: list[tuple[SpotAnalysisOperable, int]], *vargs) -> int:
        return image_groups[0][1]

    def test_single_image(self):
        processor_always_trigger = AverageByGroupImageProcessor(self.assigner, self.always_trigger)

        operables = processor_always_trigger.process_image(self.a1)
        self.assertEqual(1, len(operables))

        operable = operables[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.a1_img.nparray.astype(np.float_))

    def test_two_images_one_group(self):
        operables = self.processor.process_image(self.a1)
        self.assertEqual(0, len(operables))
        operables = self.processor.process_image(self.a2)
        self.assertEqual(0, len(operables))
        operables = self.processor.process_image(self.b1)  # changing names should trigger an execution
        self.assertEqual(1, len(operables))

        operable = operables[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, np.array([15], dtype=np.float_))

    def test_two_images_two_groups(self):
        operables = self.processor.process_image(self.a1)
        self.assertEqual(0, len(operables))
        operables_a = self.processor.process_image(self.b1)  # changing names should trigger an execution
        self.assertEqual(1, len(operables_a))
        operables_b = self.processor.process_image(self.a2)  # trigger again, to get a second pair of results
        self.assertEqual(1, len(operables_b))

        operable = operables_a[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.a1_img.nparray.astype(np.float_))
        operable = operables_b[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.b1_img.nparray.astype(np.float_))

    def test_three_images_two_groups(self):
        operables = self.processor.process_image(self.a1)
        self.assertEqual(0, len(operables))
        operables = self.processor.process_image(self.a2)
        self.assertEqual(0, len(operables))
        operables_a = self.processor.process_image(self.b1)  # changing names should trigger an execution
        self.assertEqual(1, len(operables_a))
        operables_b = self.processor.process_image(self.a1)  # trigger again, to get a second pair of results
        self.assertEqual(1, len(operables_b))

        operable = operables_a[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, np.array([15], dtype=np.float_))
        operable = operables_b[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.b1_img.nparray.astype(np.float_))

    def test_four_images_two_groups(self):
        operables = self.processor.process_image(self.a1)
        self.assertEqual(0, len(operables))
        operables = self.processor.process_image(self.a2)
        self.assertEqual(0, len(operables))
        operables_a = self.processor.process_image(self.b1)  # changing names should trigger an execution
        self.assertEqual(1, len(operables_a))
        operables = self.processor.process_image(self.b2)
        self.assertEqual(0, len(operables))
        operables_b = self.processor.process_image(self.a1)  # trigger again, to get a second pair of results
        self.assertEqual(1, len(operables_b))

        operable = operables_a[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, np.array([15], dtype=np.float_))
        operable = operables_b[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, np.array([35], dtype=np.float_))

    def test_four_images_alternating_group(self):
        operables = self.processor.process_image(self.a1)
        self.assertEqual(0, len(operables))
        operables_a1 = self.processor.process_image(self.b1)
        self.assertEqual(1, len(operables_a1))
        operables_b1 = self.processor.process_image(self.a2)
        self.assertEqual(1, len(operables_b1))
        operables_a2 = self.processor.process_image(self.b2)
        self.assertEqual(1, len(operables_a2))
        operables_b2 = self.processor.process_image(self.a1)  # trigger again, to get a second pair of results
        self.assertEqual(1, len(operables_b2))

        operable = operables_a1[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.a1_img.nparray.astype(np.float_))
        operable = operables_a2[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.a2_img.nparray.astype(np.float_))
        operable = operables_b1[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.b1_img.nparray.astype(np.float_))
        operable = operables_b2[0]
        np.testing.assert_array_equal(operable.primary_image.nparray, self.b2_img.nparray.astype(np.float_))


if __name__ == "__main__":
    unittest.main()
