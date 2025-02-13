"""Unit test for ImageColor class
"""

import matplotlib.pyplot as plt
import numpy as np
import unittest

from opencsp.app.target.target_color.lib.ImageColor import ImageColor
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Vxy import Vxy


class TestImageColor(unittest.TestCase):
    """Tests the ImageColor class"""

    def setUp(self) -> None:
        """Creates test RGB image and loads into ImageColor"""
        # Create RGB image with six bands
        # White, gray, black, red, off-red 1, off-red
        rgb = np.zeros((600, 400, 3)).astype(np.uint8)

        # White
        rgb[:100] = 255
        # Gray
        rgb[100:200] = 128
        # Black
        rgb[200:300] = 0
        # Red 1
        rgb[300:, :, 0] = 255
        # Off 2
        rgb[400:500, :, 1] = 5
        # Off 3
        rgb[500:600, :, 1] = 10
        # Create ImageColor from test image
        self.image = ImageColor(rgb)

    def tearDown(self) -> None:
        # Make sure we release all matplotlib resources.
        plt.close("all")

    def test_normalize(self) -> None:
        """Tests image color normalization of all patches"""
        # White
        actual = self.image.image_norm[50, 200]
        desired = np.ones(3) / np.sqrt(3)
        np.testing.assert_allclose(actual, desired)
        # Gray
        actual = self.image.image_norm[150, 200]
        desired = np.ones(3) / np.sqrt(3)
        np.testing.assert_allclose(actual, desired)
        # Black
        actual = np.isnan(self.image.image_norm[250, 200])
        desired = np.ones(3, dtype=bool)
        np.testing.assert_array_equal(actual, desired)
        # Red 1
        actual = self.image.image_norm[350, 200]
        desired = np.array([1, 0, 0], dtype=float)
        np.testing.assert_array_equal(actual, desired)
        # Red 2
        actual = self.image.image_norm[450, 200]
        desired = np.array([255, 5, 0], dtype=float)
        desired /= np.linalg.norm(desired)
        np.testing.assert_allclose(actual, desired)
        # Red 3
        actual = self.image.image_norm[550, 200]
        desired = np.array([255, 10, 0], dtype=float)
        desired /= np.linalg.norm(desired)
        np.testing.assert_allclose(actual, desired)

    def test_cropping_image(self):
        """Tests cropping of image"""
        # Create test image
        rgb = np.ones((50, 50, 3), dtype=np.uint8)
        image = ImageColor(rgb)

        # Crop image
        shape_init = image.shape
        region = LoopXY.from_vertices(Vxy(np.array([[40, 0, 20, 40], [50, 50, 0, 0]])))
        image.crop_image(region)
        shape_post = image.shape

        # Test pre/post shape
        np.testing.assert_array_equal(shape_init, (50, 50))
        np.testing.assert_array_equal(shape_post, (50, 40))

        # Test nans in upper left corner
        actual = np.isnan(image.image[0, 0])
        desired = np.array([1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(actual, desired)

    def test_smooth_image(self):
        """Tests smoothing of the image"""
        # Create test image
        rgb = np.zeros((50, 50, 3), dtype=np.float32)
        rgb[25, 25] = np.ones(3, dtype=np.float32)
        image = ImageColor(rgb)

        # Smooth image
        n = 10
        ker = np.ones((n, n), dtype=np.float32) / (n**2)
        image.smooth_image(ker)

        # Test smoothed value
        actual = image.image[25, 25]
        desired = np.ones(3, dtype=np.float32) * 0.01
        np.testing.assert_array_equal(actual, desired)

    def test_plotting(self):
        """Tests plotting of the images"""
        self.image.plot_normalized()
        self.image.plot_unprocessed()

    def test_matching(self):
        """Tests pixel matching in images"""
        # Test white/gray
        rgb = np.array([1, 1, 1], dtype=float)
        thresh = 1e-8
        actual = self.image.match_mask(rgb, thresh)
        desired = np.zeros(self.image.shape, dtype=bool)
        desired[:200] = True
        np.testing.assert_array_equal(actual, desired)

        # Test Red 1
        rgb = np.array([1, 0, 0], dtype=float)
        thresh = 1e-8
        actual = self.image.match_mask(rgb, thresh)
        desired = np.zeros(self.image.shape, dtype=bool)
        desired[300:400] = True
        np.testing.assert_array_equal(actual, desired)

        # Test Red 2 (0.01961 radians)
        rgb = np.array([1, 0, 0], dtype=float)
        thresh = 0.02
        actual = self.image.match_mask(rgb, thresh)
        desired = np.zeros(self.image.shape, dtype=bool)
        desired[300:500] = True
        np.testing.assert_array_equal(actual, desired)

        # Test Red 3 (0.03919 radians)
        rgb = np.array([1, 0, 0], dtype=float)
        thresh = 0.04
        actual = self.image.match_mask(rgb, thresh)
        desired = np.zeros(self.image.shape, dtype=bool)
        desired[300:600] = True
        np.testing.assert_array_equal(actual, desired)
