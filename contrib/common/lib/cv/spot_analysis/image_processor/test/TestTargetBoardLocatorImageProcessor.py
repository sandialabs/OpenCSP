import os
import random
import unittest

import numpy as np
import numpy.testing as npt
from PIL import Image

from contrib.common.lib.cv.spot_analysis.image_processor import TargetBoardLocatorImageProcessor
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestTargetBoardLocatorImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "TargetBoardLocator")
        cls.out_dir = os.path.join(path, "data", "output", "TargetBoardLocator")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self) -> None:
        pass

    def test_target_board_location(self):
        processor = TargetBoardLocatorImageProcessor(
            reference_image_dir_or_file=ft.join(self.data_dir, "reference_target_board"),
            cropped_x1x2y1y2=None,
            target_width_meters=2.44,
            target_height_meters=2.44,
            canny_edges_gradient=10,
            canny_non_edges_gradient=15,
            # canny_test_gradients=[(5,5),(5,10),(5,15),(5,20),(10,5),(10,10),(10,15),(10,20)],
            # debug_target_locating=True
        )
        lighted = Image.open(ft.join(self.data_dir, "09W01.png"))
        result = processor.process_images([lighted])[0]
        result.save(ft.join(self.out_dir, self._testMethodName + ".png"))

        corners = processor.corners
        # fmt: off
        self.assertAlmostEqual(corners["tl"].x[0], 35.7295165,   delta=2)  # max delta for (max - min) in 100 runs: 0.31794104
        self.assertAlmostEqual(corners["tl"].y[0], 29.26274199,  delta=2)  # max delta for (max - min) in 100 runs: 0.07478309
        self.assertAlmostEqual(corners["tr"].x[0], 625.25789369, delta=2)  # max delta for (max - min) in 100 runs: 0.31775339
        self.assertAlmostEqual(corners["tr"].y[0], 45.87442367,  delta=2)  # max delta for (max - min) in 100 runs: 0.0335551
        self.assertAlmostEqual(corners["br"].x[0], 605.83119206, delta=2)  # max delta for (max - min) in 100 runs: 0.30441256
        self.assertAlmostEqual(corners["br"].y[0], 633.02880323, delta=2)  # max delta for (max - min) in 100 runs: 0.01701056
        self.assertAlmostEqual(corners["bl"].x[0], 16.30514864,  delta=2)  # max delta for (max - min) in 100 runs: 0.49065163
        self.assertAlmostEqual(corners["bl"].y[0], 616.74420236, delta=2)  # max delta for (max - min) in 100 runs: 0.05482708
        # fmt: on

    def test_perspective_transform(self):
        corners = {
            "tl": p2.Pxy([35.7295165, 29.26274199]),
            "tr": p2.Pxy([625.25789369, 45.87442367]),
            "br": p2.Pxy([605.83119206, 633.02880323]),
            "bl": p2.Pxy([16.30514864, 616.74420236]),
        }
        processor = TargetBoardLocatorImageProcessor.from_corners(
            corners, target_width_meters=2.44, target_height_meters=2.44
        )
        lighted = Image.open(ft.join(self.data_dir, "09W01.png"))
        result = processor.process_images([lighted])[0]
        result.save(ft.join(self.out_dir, self._testMethodName + ".png"))

        expected = Image.open(ft.join(self.data_dir, "09W01_transformed.png"))
        npt.assert_array_equal(np.array(result), np.array(expected))


if __name__ == "__main__":
    unittest.main()
