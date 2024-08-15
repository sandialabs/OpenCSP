import sys
import unittest

import numpy as np
from PIL import Image

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class test_CacheableImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, 'data/input', name.split('test_')[-1])
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.in_dir)
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, '*')
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

        # example image with color quadrants r, g, b, y
        self.example_array = np.zeros((40, 40, 3), dtype=np.uint8)
        self.example_array[:20, :20, 0] = 255
        self.example_array[20:, :20, 1] = 255
        self.example_array[:20, 20:, 2] = 255
        self.example_array[20:, 20:, :2] = 255

        # same as the example image, but as a numpy file
        self.example_cache_path = ft.join(self.in_dir, "example_image.npy")

        # same as the example image, but as an image file
        self.example_source_path = ft.join(self.in_dir, "example_image.png")

        # non-existant values
        self.noexist_cache_path = ft.join(self.out_dir, "noexist.npy")
        self.noexist_source_path = ft.join(self.out_dir, "noexist.png")

    def test_init_valid(self):
        """Test all valid combinations of CacheableImage constructor parameters."""
        # fmt: off
        valid_combinations = [
            [ self.example_array, None,                    None                     ],
            [ self.example_array, self.example_cache_path, None                     ],
            [ self.example_array, self.example_cache_path, self.example_source_path ],
            [ self.example_array, self.example_cache_path, self.noexist_source_path ],
            [ self.example_array, self.noexist_cache_path, None                     ],
            [ self.example_array, self.noexist_cache_path, self.example_source_path ],
            [ self.example_array, None,                    self.example_source_path ],
            [ None,               self.example_cache_path, None                     ],
            [ None,               self.example_cache_path, self.example_source_path ],
            [ None,               self.example_cache_path, self.noexist_source_path ],
            [ None,               self.noexist_cache_path, self.example_source_path ],
            [ None,               None,                    self.example_source_path ],
        ]
        # fmt: on

        for valid_combination in valid_combinations:
            try:
                CacheableImage(*valid_combination)
            except Exception:
                lt.error(
                    "Encountered exception with the following valid combination of constructor parameters:\n"
                    + f"\tarray = {type(valid_combination[0])}\n"
                    + f"\tcache_path = {valid_combination[1]}\n"
                    + f"\tsource_path = {valid_combination[2]}\n"
                )
                raise

    def test_init_invalid(self):
        """Test all invalid combinations of CacheableImage constructor parameters."""
        with self.assertRaises(ValueError):
            CacheableImage(None, None, None)

        # fmt: off
        invalid_combinations = [
            [ None,               None,                    self.noexist_source_path ],
            [ None,               self.noexist_cache_path, None                     ],
            [ None,               self.noexist_cache_path, self.noexist_source_path ],
            [ self.example_array, None,                    self.noexist_source_path ],
            [ self.example_array, self.noexist_cache_path, self.noexist_source_path ],
        ]
        # fmt: on

        for invalid_combination in invalid_combinations:
            with self.assertRaises(FileNotFoundError):
                CacheableImage(*invalid_combination)
                lt.error(
                    "Expected exception for the following invalid combination of constructor parameters:\n"
                    + f"\tarray = {type(invalid_combination[0])}\n"
                    + f"\tcache_path = {invalid_combination[1]}\n"
                    + f"\tsource_path = {invalid_combination[2]}\n"
                )

    def test_size(self):
        """
        Verifies that the size() built-in returns the correct value, and that
        the sum of all CacheableImages returns the correct value.
        """
        # cacheable images exist from other tests, include their sizes as well
        existing_sizes = CacheableImage.all_cacheable_images_size()

        # something is happening under the hood that causes the reference to the
        # example array to be larger
        delta = 40

        # one cacheable image
        ci1 = CacheableImage(self.example_array, None, self.example_source_path)
        example_image = None
        self.assertAlmostEqual(sys.getsizeof(ci1), sys.getsizeof(self.example_array), delta=delta)
        self.assertAlmostEqual(
            sys.getsizeof(ci1), CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )
        example_image = ci1.to_image()
        self.assertAlmostEqual(
            sys.getsizeof(ci1), sys.getsizeof(self.example_array) + sys.getsizeof(example_image), delta=delta
        )
        self.assertAlmostEqual(
            sys.getsizeof(ci1), CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )

        # multiple cacheable images
        ci2 = CacheableImage(self.example_array, None, self.example_source_path)
        ci3 = CacheableImage(self.example_array, None, self.example_source_path)
        ci2.to_image()
        ci3.to_image()
        self.assertAlmostEqual(
            sys.getsizeof(ci1) * 3, CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )


if __name__ == '__main__':
    unittest.main()
