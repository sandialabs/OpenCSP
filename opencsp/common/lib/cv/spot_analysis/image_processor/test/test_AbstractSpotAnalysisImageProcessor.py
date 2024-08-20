import copy
import dataclasses
import unittest

import numpy as np

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.cv.CacheableImage as ci
import opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor as asaip
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable as sao


class DoNothingImageProcessor(asaip.AbstractSpotAnalysisImagesProcessor):
    def _execute(self, operable: sao.SpotAnalysisOperable, is_last: bool) -> list[sao.SpotAnalysisOperable]:
        return [operable]


class SetOnesImageProcessor(asaip.AbstractSpotAnalysisImagesProcessor):
    def _execute(self, operable: sao.SpotAnalysisOperable, is_last: bool) -> list[sao.SpotAnalysisOperable]:
        img = copy.copy(operable.primary_image.nparray)
        img[:, :] = 1
        ret = dataclasses.replace(operable, primary_image=img)
        return [ret]


class test_AbstractSpotAnalysisImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, 'data/input', name.split('test_')[-1])
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, '*')
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

        self.example_cache_path = ft.join(self.in_dir, "example_image.npy")
        self.cacheable_image = ci.CacheableImage(cache_path=self.example_cache_path)
        self.example_operable = sao.SpotAnalysisOperable(self.cacheable_image)

        self.example_operables: list[sao.SpotAnalysisOperable] = []
        for i in range(3):
            ci_i = ci.CacheableImage(cache_path=self.example_cache_path)
            sao_i = sao.SpotAnalysisOperable(ci_i)
            self.example_operables.append(sao_i)

    def test_name(self):
        """
        Verify that the auto-assigned name is the class name, and that it can be
        overwritten with a specific name.
        """
        instance = DoNothingImageProcessor()
        self.assertEqual(instance.name, "DoNothingImageProcessor")
        instance = DoNothingImageProcessor("Other Name")
        self.assertEqual(instance.name, "Other Name")

    def test_finished(self):
        """
        Verify that finished is True only when no images have been assigned, or
        when all images have been processed.
        """
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
        instance.assign_inputs(self.example_operables)
        self.assertFalse(instance.finished)
        for result in instance:
            pass
        self.assertTrue(instance.finished)

    def test_iterator_finishes_all(self):
        instance = DoNothingImageProcessor()
        nprocessed = 0

        # test with an assignment of a few operables
        instance.assign_inputs(self.example_operables)
        for result in instance:
            nprocessed += 1
        self.assertEqual(nprocessed, 3)

        # test with an assignment of an additional "two" operables
        instance.assign_inputs([self.example_operable, self.example_operable])
        for result in instance:
            nprocessed += 1
        self.assertEqual(nprocessed, 5)

    def test_run(self):
        """Verify that run() touches all the operables"""
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


if __name__ == '__main__':
    unittest.main()
