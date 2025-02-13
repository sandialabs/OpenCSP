import copy
import dataclasses
import random
import unittest

import numpy as np

import opencsp.common.lib.cv.CacheableImage as ci
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable as sao
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class DoNothingImageProcessor(AbstractSpotAnalysisImageProcessor):
    def _execute(self, operable: sao.SpotAnalysisOperable, is_last: bool) -> list[sao.SpotAnalysisOperable]:
        return [operable]


class SetOnesImageProcessor(AbstractSpotAnalysisImageProcessor):
    def _execute(self, operable: sao.SpotAnalysisOperable, is_last: bool) -> list[sao.SpotAnalysisOperable]:
        img = copy.copy(operable.primary_image.nparray)
        img[:, :] = 1
        ret = dataclasses.replace(operable, primary_image=img)
        return [ret]


class test_AbstractSpotAnalysisImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, "data/input", name.split("test_")[-1])
        cls.out_dir = ft.join(path, "data/output", name.split("test_")[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split(".")[-1]

        self.example_cache_path = ft.join(self.in_dir, "example_image.npy")
        self.cacheable_image = ci.CacheableImage(cache_path=self.example_cache_path)
        self.example_operable = sao.SpotAnalysisOperable(self.cacheable_image)

        self.example_operables: list[sao.SpotAnalysisOperable] = []
        self.num_example_operables = random.randint(0, 10)
        for i in range(self.num_example_operables):
            ci_i = ci.CacheableImage(cache_path=self.example_cache_path)
            sao_i = sao.SpotAnalysisOperable(ci_i)
            self.example_operables.append(sao_i)

        self.example_operables_gte2: list[sao.SpotAnalysisOperable] = []
        self.num_example_operables_gte2 = self.num_example_operables + 2
        for i in range(self.num_example_operables_gte2):
            ci_i = ci.CacheableImage(cache_path=self.example_cache_path)
            sao_i = sao.SpotAnalysisOperable(ci_i)
            self.example_operables_gte2.append(sao_i)

    def test_name(self):
        """
        Verify that the auto-assigned name is the class name, and that it can be
        overwritten with a specific name.
        """
        try:
            instance = DoNothingImageProcessor()
            self.assertEqual(instance.name, "DoNothingImageProcessor")
            instance = DoNothingImageProcessor("Other Name")
            self.assertEqual(instance.name, "Other Name")

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_name(): "
                + f"failed for operables {self.example_operables=}"
            )
            raise

    def test_finished(self):
        """
        Verify that finished is True only when no images have been assigned, or
        when all images have been processed.
        """
        try:
            # 0
            instance = DoNothingImageProcessor()
            self.assertTrue(instance.finished)

            # 1
            instance.assign_inputs([self.example_operable])
            self.assertFalse(instance.finished)
            for result in instance:
                pass
            self.assertTrue(instance.finished)

            # > 1
            instance.assign_inputs(self.example_operables_gte2)
            for result in instance:
                pass
            self.assertTrue(instance.finished)

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_finished(): "
                + f"failed for operables {self.example_operables_gte2=}"
            )
            raise

    def test_0_operables(self):
        """Verify that we see the expected behavior when attempting to run with 0 input."""
        instance = DoNothingImageProcessor()
        nprocessed = 0

        # test assignment
        instance.assign_inputs([])

        # test processing
        for result in instance:
            nprocessed += 1
        self.assertEqual(0, nprocessed)

        # test running
        results = instance.run([])
        self.assertEqual(len(results), 0)

    def test_iterator_finishes_all(self):
        try:
            instance = DoNothingImageProcessor()
            nprocessed = 0

            # test with an assignment of a few operables
            instance.assign_inputs(self.example_operables)
            for result in instance:
                nprocessed += 1
            self.assertEqual(nprocessed, self.num_example_operables)

            # test with an assignment of an additional "two" operables
            instance.assign_inputs([self.example_operable, self.example_operable])
            for result in instance:
                nprocessed += 1
            self.assertEqual(nprocessed, self.num_example_operables + 2)

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_iterator_finishes_all(): "
                + f"failed for operables {self.example_operables=}"
            )
            raise

    def test_run(self):
        """Verify that run() touches all the operables"""
        try:
            # sanity check: no pixels are equal to 1
            for operable in self.example_operables:
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

            # process all images
            instance = SetOnesImageProcessor()
            results = instance.run(self.example_operables)

            # verify the input operables haven't been touched
            for operable in self.example_operables:
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

            # verify all pixels in the new operables are equal to 1
            for operable in results:
                self.assertTrue(np.all(operable.primary_image.nparray == 1))

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_run(): "
                + f"failed for operables {self.example_operables=}"
            )
            raise

    def test_process_operable(self):
        """Verify that process_operable() updates the pixels"""
        try:
            for operable in self.example_operables:
                # sanity check: no pixels are equal to 1
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

                # process the operable
                instance = SetOnesImageProcessor()
                result = instance.process_operable(operable, is_last=True)

                # verify the input operable hasn't been touched
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

                # verify all pixels in the new operable are equal to 1
                self.assertTrue(np.all(result[0].primary_image.nparray == 1))

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_process_operable(): "
                + f"failed for operables {self.example_operables=}"
            )
            raise

    def test_process_images(self):
        """Verify that process_images() updates the pixels"""
        try:
            for operable in self.example_operables:
                # sanity check: no pixels are equal to 1
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

                # process the image
                instance = SetOnesImageProcessor()
                result = instance.process_images([operable.primary_image])

                # verify the input image hasn't been touched
                self.assertTrue(np.all(operable.primary_image.nparray != 1))

                # verify all pixels in the new image are equal to 1
                self.assertTrue(np.all(result[0].nparray == 1))

        except:
            lt.error(
                "Error in test_AbstractSpotAnalysisImageProcessor.test_process_images(): "
                + f"failed for operables {self.example_operables=}"
            )
            raise

    def test_set_previous_operables(self):
        """Verify that image processors append themselves to the operable's history."""
        operable_1 = self.example_operable
        processor_1 = SetOnesImageProcessor()
        operable_2 = processor_1.process_operable(operable_1, is_last=True)[0]
        processor_2 = SetOnesImageProcessor()
        operable_3 = processor_2.process_operable(operable_2, is_last=True)[0]

        # verify we got different operables as return values
        self.assertNotEqual(operable_1, operable_2)
        self.assertNotEqual(operable_1, operable_3)
        self.assertNotEqual(operable_2, operable_3)

        # verify each operable's history
        self.assertEqual(operable_1.previous_operables, (None, None))
        self.assertEqual(operable_2.previous_operables[0], [operable_1])
        self.assertEqual(operable_2.previous_operables[1], processor_1)
        self.assertEqual(operable_3.previous_operables[0], [operable_2])
        self.assertEqual(operable_3.previous_operables[1], processor_2)

        # sanity check - do nothing processors don't add themselves to the history
        processor_3 = DoNothingImageProcessor()
        operable_4 = processor_3.process_operable(operable_3, is_last=True)[0]
        self.assertEqual(operable_3, operable_4)
        self.assertEqual(operable_4.previous_operables[0], [operable_2])
        self.assertEqual(operable_4.previous_operables[1], processor_2)


if __name__ == "__main__":
    unittest.main()
