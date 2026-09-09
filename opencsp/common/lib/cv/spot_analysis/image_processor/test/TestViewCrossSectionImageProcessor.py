import numpy as np
import numpy.testing as npt
from PIL import Image
import unittest

from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from contrib.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft

skip_msg = """
There's an issue where matplotlib.close() doesn't fully release the plots from
memory. In our case that is causing the plots generated with a single test case
to differ from plots generated as the 2nd+ test case. I (BGB) think the
solution is here:
https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots"""


class TestViewCrossSectionImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, "data/input", name.split("Test")[-1])
        cls.out_dir = ft.join(path, "data/output", name.split("Test")[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def _build_image_comparison_error_msg(self, expected_path, actual_path):
        return (
            f"Expected images to match but they don't. Compare the images with:\n"
            + f"python contrib/app/ImageDiff/image_diff.py {expected_path} {actual_path}"
        )

    # @unittest.skip(skip_msg) Can run one of the tests
    def test_transform_coordinates_tblocator(self):
        corners = {
            "tl": p2.Pxy([35.7295165, 29.26274199]),
            "tr": p2.Pxy([625.25789369, 45.87442367]),
            "br": p2.Pxy([605.83119206, 633.02880323]),
            "bl": p2.Pxy([16.30514864, 616.74420236]),
        }

        processors = {
            "target_board": TargetBoardLocatorImageProcessor.from_corners(
                corners, target_width_meters=2.44, target_height_meters=2.44
            ),
            "vcross_sec": ViewCrossSectionImageProcessor((1200, 1200), plot_title=""),
        }

        sa = SpotAnalysis("test_targetboard_coords", list(processors.values()))
        sa.set_primary_images([ft.join(self.in_dir, "09W01.png")])

        load_dir = ft.join(self.in_dir, "test_transform_coordinates_tblocator")
        save_dir = ft.join(self.out_dir, "test_transform_coordinates_tblocator")
        ft.create_directories_if_necessary(save_dir)
        ft.delete_files_in_directory(save_dir, "*.png")
        imgs_to_compare: list[tuple[str, str]] = []
        for result in sa:
            for i, visualization_image in enumerate(result.visualization_images[processors["vcross_sec"]]):
                load_path = ft.join(load_dir, f"vis_{i}.png")
                save_path = ft.join(save_dir, f"vis_{i}.png")
                visualization_image.to_image().save(save_path)
                imgs_to_compare.append((load_path, save_path))

        for expected_path, actual_path in imgs_to_compare:
            expected_img = Image.open(expected_path)
            actual_img = Image.open(actual_path)
            err_msg = self._build_image_comparison_error_msg(expected_path, actual_path)
            npt.assert_array_equal(np.array(expected_img), np.array(actual_img), err_msg)

        csproc: ViewCrossSectionImageProcessor = processors["vcross_sec"]
        csproc.close_figures()

    @unittest.skip(skip_msg)
    def test_transform_coordinates_crop(self):
        processors = {
            "cropping": CroppingImageProcessor.by_region((100, 500, 200, 400)),
            "vcross_sec": ViewCrossSectionImageProcessor((234, 122), plot_title=""),
        }

        sa = SpotAnalysis("test_targetboard_coords", list(processors.values()))
        sa.set_primary_images([ft.join(self.in_dir, "09W01.png")])

        load_dir = ft.join(self.in_dir, "test_transform_coordinates_crop")
        save_dir = ft.join(self.out_dir, "test_transform_coordinates_crop")
        ft.create_directories_if_necessary(save_dir)
        ft.delete_files_in_directory(save_dir, "*.png")
        imgs_to_compare: list[tuple[str, str]] = []
        for result in sa:
            for i, visualization_image in enumerate(result.visualization_images[processors["vcross_sec"]]):
                load_path = ft.join(load_dir, f"vis_{i}.png")
                save_path = ft.join(save_dir, f"vis_{i}.png")
                visualization_image.to_image().save(save_path)
                imgs_to_compare.append((load_path, save_path))

        for expected_path, actual_path in imgs_to_compare:
            expected_img = Image.open(expected_path)
            actual_img = Image.open(actual_path)
            err_msg = self._build_image_comparison_error_msg(expected_path, actual_path)
            npt.assert_array_equal(np.array(expected_img), np.array(actual_img), err_msg)

        csproc: ViewCrossSectionImageProcessor = processors["vcross_sec"]
        csproc.close_figures()


if __name__ == "__main__":
    unittest.main()
