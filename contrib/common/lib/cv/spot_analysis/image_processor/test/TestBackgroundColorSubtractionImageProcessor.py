import os
import random
import unittest

import cv2 as cv
import numpy as np
import numpy.testing as npt
from PIL import Image

from contrib.common.lib.cv.spot_analysis.image_processor import BackgroundColorSubtractionImageProcessor
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestBackgroundColorSubtractionImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "BackgroundColorSubtraction")
        cls.out_dir = os.path.join(path, "data", "output", "BackgroundColorSubtraction")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def _visualize_three_point_regions(
        self, image: np.ndarray, three_points: p2.Pxy, three_points_regions: list[reg2.RegionXY]
    ) -> Image.Image:
        (height, width), nchannels = it.dims_and_nchannels(image)

        # draw the regions
        img = np.zeros_like(image)
        for i in range(3):
            mask = three_points_regions[i].as_mask(np.arange(width), np.arange(height))
            img += 50 * (i + 1) * mask.astype(img.dtype)

        # draw the points
        pt2coord = lambda index: (int(three_points.x[index]), int(three_points.y[index]))
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = cv.circle(img, pt2coord(0), 7, color=(255, 255, 0))
        img = cv.circle(img, pt2coord(1), 7, color=(0, 255, 255))
        img = cv.circle(img, pt2coord(2), 7, color=(255, 0, 255))

        return img

    def test_three_point_regions(self):
        self.setUpClass()
        processor = BackgroundColorSubtractionImageProcessor()

        # square image
        image = np.ndarray((100, 100), dtype=np.uint8)
        three_points, three_points_regions = processor._get_three_point_regions(image)
        vis = self._visualize_three_point_regions(image, three_points, three_points_regions)
        Image.fromarray(vis).save(ft.join(self.out_dir, "test_three_point_regions_square.png"))
        expected_vis = np.array(Image.open(ft.join(self.data_dir, "test_three_point_regions_square.png")))
        npt.assert_array_equal(np.array([0, 100, 50]), three_points.x)
        npt.assert_array_equal(np.array([0, 0, 100]), three_points.y)
        npt.assert_array_equal(expected_vis, vis)

        # short rectangular
        image = np.ndarray((100, 150), dtype=np.uint8)
        three_points, three_points_regions = processor._get_three_point_regions(image)
        vis = self._visualize_three_point_regions(image, three_points, three_points_regions)
        Image.fromarray(vis).save(ft.join(self.out_dir, "test_three_point_regions_short_rectangular.png"))
        expected_vis = np.array(Image.open(ft.join(self.data_dir, "test_three_point_regions_short_rectangular.png")))
        npt.assert_array_equal(np.array([0, 150, 75]), three_points.x)
        npt.assert_array_equal(np.array([0, 0, 100]), three_points.y)
        npt.assert_array_equal(expected_vis, vis)

        # rectangular
        image = np.ndarray((100, 200), dtype=np.uint8)
        three_points, three_points_regions = processor._get_three_point_regions(image)
        vis = self._visualize_three_point_regions(image, three_points, three_points_regions)
        Image.fromarray(vis).save(ft.join(self.out_dir, "test_three_point_regions_rectangular.png"))
        expected_vis = np.array(Image.open(ft.join(self.data_dir, "test_three_point_regions_rectangular.png")))
        npt.assert_array_equal(np.array([0, 200, 100]), three_points.x)
        npt.assert_array_equal(np.array([0, 0, 100]), three_points.y)
        npt.assert_array_equal(expected_vis, vis)

        # long rectangular
        image = np.ndarray((100, 250), dtype=np.uint8)
        three_points, three_points_regions = processor._get_three_point_regions(image)
        vis = self._visualize_three_point_regions(image, three_points, three_points_regions)
        Image.fromarray(vis).save(ft.join(self.out_dir, "test_three_point_regions_long_rectangular.png"))
        expected_vis = np.array(Image.open(ft.join(self.data_dir, "test_three_point_regions_long_rectangular.png")))
        npt.assert_array_equal(np.array([0, 250, 125]), three_points.x)
        npt.assert_array_equal(np.array([0, 0, 100]), three_points.y)
        npt.assert_array_equal(expected_vis, vis)

    def test_build_background_image(self):
        self.setUpClass()
        processor = BackgroundColorSubtractionImageProcessor()

        image = np.ndarray((100, 100), dtype=np.uint8)
        background = processor.build_background_image(image, solid_background_color=[150])
        Image.fromarray(background).save(ft.join(self.out_dir, "test_build_background_image_solid.png"))

        image = np.ndarray((100, 100), dtype=np.uint8)
        background = processor.build_background_image(image, gradient_tl_tr_bl_br=[[50, 100, 150, 200]])
        Image.fromarray(background).save(ft.join(self.out_dir, "test_build_background_image_gradient.png"))

        image = np.ndarray((100, 100), dtype=np.uint8)
        background = processor.build_background_image(image, background_plane_tl_tr_bm=[[50, 100, 150]])
        Image.fromarray(background).save(ft.join(self.out_dir, "test_build_background_image_plane.png"))


if __name__ == "__main__":
    TestBackgroundColorSubtractionImageProcessor().test_build_background_image()
