from functools import cache
import os
import random
import unittest

import cv2 as cv
import numpy as np
import numpy.testing as npt

import contrib.common.lib.cv.annotations.MomentsAnnotation as ma
from contrib.common.lib.cv.spot_analysis.image_processor import MomentsImageProcessor
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.image_tools as it


class TestMomentsImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "Moments")
        cls.out_dir = os.path.join(path, "data", "output", "Moments")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self) -> None:
        self.processor = MomentsImageProcessor(include_visualization=True)

    @cache
    def _get_operable(self, img_name_ext: str) -> SpotAnalysisOperable:
        img_path_name_ext = ft.join(self.data_dir, img_name_ext)
        cacheable = CacheableImage.from_single_source(img_path_name_ext)
        operable = SpotAnalysisOperable(primary_image=cacheable)
        return operable

    def _get_moments(self, operable: SpotAnalysisOperable) -> ma.MomentsAnnotation:
        moments = list(filter(lambda a: isinstance(a, ma.MomentsAnnotation), operable.annotations))
        if len(moments) != 1:
            lt.error_and_raise(
                RuntimeError,
                f"Error in {self._testMethodName}: "
                + f"annotations list should have 1 MomentsAnnotation but instead has {len(moments)}!",
            )
        return moments[0]

    def test_centroid(self):
        operable_centered = self._get_operable("centered.png")
        operable_offset = self._get_operable("offset_50_75.png")

        result_centered = self.processor.process_operable(operable_centered, is_last=True)[0]
        moments_centered = self._get_moments(result_centered)
        result_offset = self.processor.process_operable(operable_offset, is_last=True)[0]
        moments_offset = self._get_moments(result_offset)

        self.assertAlmostEqual(100, moments_centered.cX, delta=0.1)
        self.assertAlmostEqual(100, moments_centered.cY, delta=0.1)
        self.assertAlmostEqual(50, moments_offset.cX, delta=0.1)
        self.assertAlmostEqual(75, moments_offset.cY, delta=0.1)

        # save visualizations for human verification
        eval_sets = [("centered.png", result_centered), ("offset_50_75.png", result_offset)]
        for img_name_ext, result in eval_sets:
            image = result.visualization_images[self.processor][0].to_image()
            image_path_name_ext = ft.join(self.out_dir, img_name_ext)
            image.save(image_path_name_ext)

    def test_rotation_angle_2d(self):
        operable_centered = self._get_operable("centered.png")
        operable_a15 = self._get_operable("angle_15.png")
        operable_a45 = self._get_operable("angle_45.png")
        operable_a90 = self._get_operable("angle_90.png")
        operable_a135 = self._get_operable("angle_135.png")

        result_centered = self.processor.process_operable(operable_centered, is_last=True)[0]
        moments_centered = self._get_moments(result_centered)
        zero_angle = min(moments_centered.rotation_angle_2d, np.pi - moments_centered.rotation_angle_2d)
        result_a15 = self.processor.process_operable(operable_a15, is_last=True)[0]
        moments_a15 = self._get_moments(result_a15)
        result_a45 = self.processor.process_operable(operable_a45, is_last=True)[0]
        moments_a45 = self._get_moments(result_a45)
        result_a90 = self.processor.process_operable(operable_a90, is_last=True)[0]
        moments_a90 = self._get_moments(result_a90)
        result_a135 = self.processor.process_operable(operable_a135, is_last=True)[0]
        moments_a135 = self._get_moments(result_a135)

        self.assertAlmostEqual(0, np.rad2deg(zero_angle), delta=0.1)
        self.assertAlmostEqual(15, np.rad2deg(moments_a15.rotation_angle_2d), delta=0.1)
        self.assertAlmostEqual(45, np.rad2deg(moments_a45.rotation_angle_2d), delta=0.1)
        self.assertAlmostEqual(90, np.rad2deg(moments_a90.rotation_angle_2d), delta=0.1)
        self.assertAlmostEqual(135, np.rad2deg(moments_a135.rotation_angle_2d), delta=0.1)

        # save visualizations for human verification
        eval_sets = [
            ("centered.png", result_centered),
            ("angle_15.png", result_a15),
            ("angle_45.png", result_a45),
            ("angle_90.png", result_a90),
            ("angle_135.png", result_a135),
        ]
        for img_name_ext, result in eval_sets:
            image = result.visualization_images[self.processor][0].to_image()
            image_path_name_ext = ft.join(self.out_dir, img_name_ext)
            image.save(image_path_name_ext)


if __name__ == '__main__':
    unittest.main()
