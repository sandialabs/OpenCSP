import os
import random
import unittest

import numpy as np
import numpy.testing as npt
from PIL import Image

from contrib.common.lib.cv.spot_analysis.image_processor import TargetBoardLocatorImageProcessor
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
            canny_edges_gradient=30,
            canny_non_edges_gradient=20,
            # debug_target_locating=True
        )
        lighted = Image.open(ft.join(self.data_dir, "09W01.JPG"))
        result = processor.process_images([lighted])[0]
        result.to_image().save(ft.join(self.out_dir, self._testMethodName + ".png"))

        corners = processor.corners
        self.assertAlmostEqual(corners["tl"].x[0], 517, delta=20)
        self.assertAlmostEqual(corners["tl"].y[0], 257, delta=20)
        self.assertAlmostEqual(corners["tr"].x[0], 1107, delta=20)
        self.assertAlmostEqual(corners["tr"].y[0], 268, delta=20)
        self.assertAlmostEqual(corners["br"].x[0], 1094, delta=20)
        self.assertAlmostEqual(corners["br"].y[0], 855, delta=20)
        self.assertAlmostEqual(corners["bl"].x[0], 503, delta=20)
        self.assertAlmostEqual(corners["bl"].y[0], 842, delta=20)

    def test_target_board_location_cropped(self):
        # crop the input image
        crop_size = random.randint(1, 200)
        crop = [crop_size, 1626 + 1 - crop_size, crop_size, 1236 + 1 - crop_size]
        lighted = Image.open(ft.join(self.data_dir, "09W01.JPG"))
        lighted_cropped = lighted.crop([crop[0], crop[2], crop[1], crop[3]])

        # evaluate
        processor = TargetBoardLocatorImageProcessor(
            reference_image_dir_or_file=ft.join(self.data_dir, "reference_target_board"),
            cropped_x1x2y1y2=crop,
            target_width_meters=2.44,
            target_height_meters=2.44,
            canny_edges_gradient=30,
            canny_non_edges_gradient=20,
            #  debug_target_locating=True
        )
        result = processor.process_images([lighted_cropped])[0]
        result.to_image().save(ft.join(self.out_dir, self._testMethodName + ".png"))

        # verify
        corners = processor.corners
        self.assertAlmostEqual(corners["tl"].x[0], 517 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["tl"].y[0], 257 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["tr"].x[0], 1107 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["tr"].y[0], 268 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["br"].x[0], 1094 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["br"].y[0], 855 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["bl"].x[0], 503 - crop_size, delta=20, msg=f"failed for {crop_size=}")
        self.assertAlmostEqual(corners["bl"].y[0], 842 - crop_size, delta=20, msg=f"failed for {crop_size=}")

    def test_perspective_transform(self):
        corners = {
            "tl": p2.Pxy([519.42333545, 256.22223199]),
            "tr": p2.Pxy([1108.33624737, 271.21012117]),
            "br": p2.Pxy([1091.97009466, 857.37629342]),
            "bl": p2.Pxy([501.9556769, 840.95732619]),
        }
        processor = TargetBoardLocatorImageProcessor.from_corners(
            corners, target_width_meters=2.44, target_height_meters=2.44
        )
        lighted = Image.open(ft.join(self.data_dir, "09W01.JPG"))
        result = processor.process_images([lighted])[0]
        result.to_image().save(ft.join(self.out_dir, self._testMethodName + ".png"))

        expected = Image.open(ft.join(self.data_dir, "09W01_transformed.png"))
        npt.assert_allclose(result.nparray, np.array(expected), atol=2)


if __name__ == "__main__":
    unittest.main()
