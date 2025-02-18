import os
import random
import unittest

import numpy as np
import numpy.testing as npt
from PIL import Image

from contrib.common.lib.cv.spot_analysis.image_processor import EnclosedEnergyImageProcessor
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestEnclosedEnergyImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "EnclosedEnergy")
        cls.out_dir = os.path.join(path, "data", "output", "EnclosedEnergy")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self) -> None:
        self.example_processor = EnclosedEnergyImageProcessor(
            calc_inner_radius_limit=10, calc_radius_resolution=2, calc_outer_radius_limit=20
        )

    def test_interpolation_resolution_2(self):
        max_radius = 40
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [12, 14, 16, 18, 20])
        self.assertEqual(interpolated_radii, [11, 13, 15, 17, 19])

    def test_interpolation_inner_limit_zero(self):
        self.example_processor.calc_inner_radius_limit = 0
        max_radius = 40
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, list(range(1, 21, 1)))
        self.assertEqual(interpolated_radii, [])

    def test_interpolation_outer_limit_zero(self):
        self.example_processor.calc_outer_radius_limit = 0
        max_radius = 40
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + list(range(12, 40 + 1, 2)))
        self.assertEqual(interpolated_radii, list(range(11, 40 + 1, 2)))

    def test_interpolation_resolution_zero(self):
        self.example_processor.calc_radius_resolution = 0
        max_radius = 40
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, list(range(1, 21)))
        self.assertEqual(interpolated_radii, [])

    def test_interpolation_inner_limit_greater_than_outer_limit(self):
        self.example_processor.calc_inner_radius_limit = 15
        self.example_processor.calc_outer_radius_limit = 10
        max_radius = 40
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, list(range(1, 16)))
        self.assertEqual(interpolated_radii, [])

    def test_interpolation_max_radius_less_than_inner_limit(self):
        self.example_processor.calc_inner_radius_limit = 15
        max_radius = 10
        direct_radii, interpolated_radii = self.example_processor._determine_interpolated_radii(max_radius)
        self.assertEqual(direct_radii, list(range(1, 11)))
        self.assertEqual(interpolated_radii, [])

    # def test_enclosed_energy(self):
    #     # process the example image
    #     processor = EnclosedEnergyImageProcessor()
    #     inpainted_image_path = ft.join(self.data_dir, "09W01_inpainted.png")
    #     inpainted_image = CacheableImage.from_single_source(inpainted_image_path)
    #     inpainted_operable = SpotAnalysisOperable(inpainted_image, inpainted_image_path)
    #     result = processor.process_operable(inpainted_operable, is_last=True)[0]

    #     # gather the results
    #     enclosed_energy_sums = result.image_processor_notes[0][1]
    #     enclosed_energy_plot = result.visualization_images[processor][0]

    #     # verify the results
    #     print(str(enclosed_energy_sums))
    #     enclosed_energy_plot.save_image(ft.join(self.data_dir, self._testMethodName+".png"))


if __name__ == "__main__":
    unittest.main()
