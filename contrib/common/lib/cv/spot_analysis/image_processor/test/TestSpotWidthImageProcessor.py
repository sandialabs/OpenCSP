import os
import random
import unittest

import numpy as np
import numpy.testing as npt
from PIL import Image

from contrib.common.lib.cv.spot_analysis.image_processor import SpotWidthImageProcessor
import opencsp.common.lib.geometry.angle as geo_angle
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestSpotWidthImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "SpotWidthImageProcessor")
        cls.out_dir = os.path.join(path, "data", "output", "SpotWidthImageProcessor")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self) -> None:
        pass

    def test__find_closest_coordinate_to_angle(self):
        processor = SpotWidthImageProcessor()
        angles = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
        coordinates = p2.Pxy([[1, 0, -1, 0], [0, 1, 0, -1]])

        for idx, angle_range in [
            (0, (-np.pi / 4, np.pi / 4)),
            (1, (np.pi / 4, np.pi * 3 / 4)),
            (2, (np.pi * 3 / 4, np.pi * 5 / 4)),
            (3, (np.pi * 5 / 4, -np.pi / 4)),
        ]:
            expected_coord = coordinates[idx]
            deg = np.pi / 180
            for target_angle in np.arange(angle_range[0] + deg, angle_range[1] - deg, deg):
                actual_coord = processor._find_closest_coordinate_to_angle(target_angle, angles, coordinates)
                npt.assert_array_equal(
                    expected_coord.data,
                    actual_coord.data,
                    f"For angle {target_angle:0.2f} ({np.rad2deg(target_angle)} degrees): "
                    + f"actual coordinate {actual_coord.astuple()} does not match expected coordinate {expected_coord.astuple()}",
                )


if __name__ == '__main__':
    unittest.main()
