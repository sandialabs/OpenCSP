import os
import unittest

import numpy as np

from contrib.common.lib.cv.RegionDetector import RegionDetector
from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from opencsp.common.lib.cv.spot_analysis.image_processor import AverageByGroupImageProcessor, ConvolutionImageProcessor
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestRegionDetector(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "RegionDetector")
        self.out_dir = os.path.join(path, "data", "output", "RegionDetector")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def load_image(self, image_path_name_ext: str):
        # Load (and average) the reference images
        image_processors = {
            'Avg': AverageByGroupImageProcessor(lambda o: 0, lambda l: None),
            'Conv': ConvolutionImageProcessor(kernel="gaussian", diameter=3),
        }

        sa = SpotAnalysis("averager", list(image_processors.values()))
        sa.set_primary_images([image_path_name_ext])
        preprocessed_operable = next(iter(sa))
        preprocessed_image = preprocessed_operable.primary_image.nparray

        return preprocessed_image

    @unittest.skip("Skipping until test images have been approved for release")
    def test_dark(self):
        image = self.load_image(ft.join(self.data_dir, "dark.jpg"))
        # canny_test_gradients=[(30, 10), (30, 20), (30, 30), (30, 40), (30, 50), (30, 60)]
        detector = RegionDetector(edge_coarse_width=10, canny_edges_gradient=30, canny_non_edges_gradient=20)
        canny, boundary_pixels = detector.find_boundary_pixels_in_image(
            image,
            approx_center_pixel=p2.Pxy([1626 / 2, 1236 / 2]),
            # debug_canny_settings=True,
        )
        edges, corners, region = detector.find_rectangular_region(boundary_pixels, canny)

        self.assertAlmostEqual(corners["tl"].x[0], 517, delta=5)
        self.assertAlmostEqual(corners["tl"].y[0], 257, delta=5)
        self.assertAlmostEqual(corners["tr"].x[0], 1107, delta=5)
        self.assertAlmostEqual(corners["tr"].y[0], 268, delta=5)
        self.assertAlmostEqual(corners["br"].x[0], 1094, delta=5)
        self.assertAlmostEqual(corners["br"].y[0], 855, delta=5)
        self.assertAlmostEqual(corners["bl"].x[0], 503, delta=5)
        self.assertAlmostEqual(corners["bl"].y[0], 842, delta=5)

    @unittest.skip("Skipping until test images have been approved for release")
    def test_very_dark(self):
        image = self.load_image(ft.join(self.data_dir, "very_dark.jpg"))

        # canny_test_gradients=[(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7)]
        detector = RegionDetector(edge_coarse_width=10, canny_edges_gradient=10, canny_non_edges_gradient=5)
        canny, boundary_pixels = detector.find_boundary_pixels_in_image(
            image,
            approx_center_pixel=p2.Pxy([1626 / 2, 1236 / 2]),
            # debug_canny_settings=True,
        )
        edges, corners, region = detector.find_rectangular_region(boundary_pixels, canny)

        self.assertAlmostEqual(corners["tl"].x[0], 636, delta=5)
        self.assertAlmostEqual(corners["tl"].y[0], 288, delta=5)
        self.assertAlmostEqual(corners["tr"].x[0], 1125, delta=5)
        self.assertAlmostEqual(corners["tr"].y[0], 290, delta=5)
        self.assertAlmostEqual(corners["br"].x[0], 1121, delta=5)
        self.assertAlmostEqual(corners["br"].y[0], 776, delta=5)
        self.assertAlmostEqual(corners["bl"].x[0], 633, delta=5)
        self.assertAlmostEqual(corners["bl"].y[0], 773, delta=5)


if __name__ == '__main__':
    unittest.main()
