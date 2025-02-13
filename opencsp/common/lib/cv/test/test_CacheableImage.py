import sys
import unittest

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class test_CacheableImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, "data/input", name.split("test_")[-1])
        cls.out_dir = ft.join(path, "data/output", name.split("test_")[-1])
        ft.create_directories_if_necessary(cls.in_dir)
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split(".")[-1]

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

        # de-register all cacheable images
        while CacheableImage.lru() is not None:
            pass

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
            [ self.example_array, self.noexist_cache_path, self.noexist_source_path ],
            [ self.example_array, None,                    self.example_source_path ],
            [ self.example_array, None,                    self.noexist_source_path ],
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
                    f"Encountered exception in {self.test_name} with the following valid combination of constructor parameters:\n"
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
        ]
        # fmt: on

        for invalid_combination in invalid_combinations:
            with self.assertRaises(FileNotFoundError):
                CacheableImage(*invalid_combination)
                lt.error(
                    f"Expected exception in {self.test_name} for the following invalid combination of constructor parameters:\n"
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

        implementation_overhead = 48
        # Note that this number is stemming from the CPython implementation of
        # objects and the memory those objects require for their book-keeping.
        # Therefore this number could be different on every system and could
        # also change with python versions.

        delta = 2 * implementation_overhead
        # Also, we don't actually care that much what the
        # implementation-specific number is, so let's just make the buffer a
        # little bigger.

        # one cacheable image
        ci1 = CacheableImage(self.example_array, None, self.example_source_path)
        # sys.getsizeof(ci1): 4976
        example_image = None
        self.assertAlmostEqual(sys.getsizeof(ci1), sys.getsizeof(self.example_array), delta=delta)
        self.assertAlmostEqual(
            sys.getsizeof(ci1),
            4800,
            delta=1000,
            msg="Sanity check that the memory usage is roughly proportional to the size of the image failed.",
        )
        self.assertAlmostEqual(
            sys.getsizeof(ci1), CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )
        example_image = ci1.to_image()
        # sys.getsizeof(ci1): 9808
        self.assertAlmostEqual(
            sys.getsizeof(ci1), sys.getsizeof(self.example_array) + it.getsizeof_approx(example_image), delta=delta
        )
        self.assertAlmostEqual(
            sys.getsizeof(ci1),
            9600,
            delta=2000,
            msg="Sanity check that the memory usage is roughly proportional to the size of the image failed.",
        )
        self.assertAlmostEqual(
            sys.getsizeof(ci1), CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )

        # multiple cacheable images
        ci2 = CacheableImage(self.example_array, None, self.example_source_path)
        ci3 = CacheableImage(self.example_array, None, self.example_source_path)
        ci2.to_image()
        ci3.to_image()
        # CacheableImage.all_cacheable_images_size(): 29424
        self.assertAlmostEqual(
            sys.getsizeof(ci1) * 3, CacheableImage.all_cacheable_images_size() - existing_sizes, delta=delta
        )

    def test_cache(self):
        """Test all valid combinations of CacheableImage constructor parameters."""
        default_cache_file = ft.join(self.out_dir, f"{self.test_name}.npy")
        noexist_cache_file = ft.join(self.out_dir, f"{self.test_name}_no_exist.npy")

        # fmt: off
        valid_combinations = [
        #     np array,           .np cache file,          .png source file,         expected .np cache file
            [ self.example_array, None,                    None,                     default_cache_file      ],
            [ self.example_array, self.example_cache_path, None,                     self.example_cache_path ],
            [ self.example_array, None,                    self.noexist_source_path, default_cache_file      ],
            [ self.example_array, self.example_cache_path, self.example_source_path, None                    ],
            [ self.example_array, noexist_cache_file,      None,                     noexist_cache_file      ],
            [ self.example_array, self.example_cache_path, self.noexist_source_path, None                    ],
            [ self.example_array, noexist_cache_file,      self.example_source_path, None                    ],
            [ self.example_array, noexist_cache_file,      self.noexist_source_path, noexist_cache_file      ],
            [ self.example_array, None,                    self.example_source_path, None                    ],
            [ None,               self.example_cache_path, None,                     None                    ],
            [ None,               self.example_cache_path, self.example_source_path, None                    ],
            [ None,               self.example_cache_path, self.noexist_source_path, None                    ],
            [ None,               noexist_cache_file,      self.example_source_path, None                    ],
            [ None,               None,                    self.example_source_path, None                    ],
        ]
        # fmt: on

        for valid_combination in valid_combinations:
            # setup
            err_msg = (
                f"Error encountered in {self.test_name} with the following valid combination of constructor parameters:\n"
                + f"\tarray = {type(valid_combination[0])}\n"
                + f"\tcache_path = {valid_combination[1]}\n"
                + f"\tsource_path = {valid_combination[2]}\n"
            )

            try:
                # setup
                should_create_cache_file = valid_combination[3]
                valid_combination = valid_combination[:3]
                ft.delete_file(default_cache_file, error_on_not_exists=False)
                ft.delete_file(noexist_cache_file, error_on_not_exists=False)

                # create the cacheable image
                cacheable = CacheableImage(*valid_combination)

                # check memory usage
                cacheable.nparray
                cacheable.to_image()
                self.assertGreaterEqual(sys.getsizeof(cacheable), sys.getsizeof(self.example_array), msg=err_msg)

                # verify that cacheing works
                self.assertFalse(ft.file_exists(default_cache_file), msg=err_msg)
                cacheable.cache(default_cache_file)
                if should_create_cache_file is not None:
                    self.assertTrue(ft.file_exists(should_create_cache_file), msg=err_msg)
                else:
                    self.assertFalse(ft.file_exists(default_cache_file), msg=err_msg)
                    self.assertFalse(ft.file_exists(noexist_cache_file), msg=err_msg)

                # check memory usage
                self.assertAlmostEqual(0, sys.getsizeof(cacheable), delta=cacheable._expected_cached_size, msg=err_msg)

                # verify that loading from the cache works
                uncached_array = cacheable.nparray
                self.assertGreaterEqual(sys.getsizeof(cacheable), sys.getsizeof(self.example_array), msg=err_msg)
                np.testing.assert_array_equal(self.example_array, uncached_array)

                # cache and delete the cache file
                # loading from the cache should fail
                cacheable.cache(default_cache_file)
                if should_create_cache_file is not None:
                    self.assertTrue(ft.file_exists(should_create_cache_file), msg=err_msg)
                    ft.delete_file(should_create_cache_file)
                    with self.assertRaises(Exception, msg=err_msg):
                        cacheable.nparray
                else:
                    self.assertFalse(ft.file_exists(default_cache_file), msg=err_msg)
                    self.assertFalse(ft.file_exists(noexist_cache_file), msg=err_msg)

            except Exception:
                lt.error(err_msg)
                raise

    def test_lru(self):
        """Verifies that the Least Recently Used functionality works as expected"""

        # create three cacheable images
        # new LRU: 1, 2, 3
        c1 = CacheableImage(self.example_array)
        self.assertEqual(c1, CacheableImage.lru(False))
        c2 = CacheableImage(self.example_array)
        self.assertEqual(c1, CacheableImage.lru(False))
        c3 = CacheableImage(self.example_array)
        self.assertEqual(c1, CacheableImage.lru(False))

        # get the value from the 1st image, then the 2nd image
        # LRU: 2, 3, 1
        c1.nparray
        self.assertEqual(c2, CacheableImage.lru(False))
        # LRU: 3, 1, 2
        c2.nparray
        self.assertEqual(c3, CacheableImage.lru(False))

        # cache the 1st image, check that this doesn't change the lru
        # LRU: 3, 2, 1
        c1.cache(ft.join(self.out_dir, f"{self.test_name}_c1.npy"))
        self.assertEqual(c3, CacheableImage.lru(False))

        # get the value of the 3rd image, check that the 2nd is now the LRU
        # LRU: 2, 1, 3
        c3.nparray
        self.assertEqual(c2, CacheableImage.lru(False))

        # deregister the 2nd, then 1st, then 3rd
        self.assertEqual(c2, CacheableImage.lru(True))
        # LRU: 1, 3
        self.assertEqual(c1, CacheableImage.lru(True))
        # LRU: 3
        self.assertEqual(c3, CacheableImage.lru(True))
        # LRU:
        self.assertEqual(None, CacheableImage.lru(True))

        # get the value for and deregister the 1st, then 2nd, then 3rd
        # LRU: 1
        c1.nparray
        self.assertEqual(c1, CacheableImage.lru(True))
        # LRU: 2
        c2.nparray
        self.assertEqual(c2, CacheableImage.lru(True))
        # LRU: 3
        c3.nparray
        self.assertEqual(c3, CacheableImage.lru(True))

        # check that there are no more registered cacheable images
        self.assertEqual(None, CacheableImage.lru(False))

    def test_save_image(self):
        """Verify that after saving the image, cacheing no longer creates a cache file"""
        cache_file = ft.join(self.out_dir, f"{self.test_name}.npy")
        image_file = ft.join(self.out_dir, f"{self.test_name}.png")
        ci = CacheableImage(self.example_array, source_path=image_file)
        in_memory_size = sys.getsizeof(ci)

        # Sanity test: cacheing without saving the image creates a cache file.
        # This is in preparation for the "finale".
        self.assertFalse(ft.file_exists(cache_file))
        ci.cache(cache_file)
        self.assertTrue(ft.file_exists(cache_file))
        cached_size = sys.getsizeof(ci)
        self.assertLess(cached_size, in_memory_size)

        # re-load the data
        ci.nparray
        in_memory_size = sys.getsizeof(ci)
        self.assertGreater(in_memory_size, cached_size)

        # delete the cache file
        ft.delete_file(cache_file)
        self.assertFalse(ft.file_exists(cache_file))

        # save to the image file
        self.assertFalse(ft.file_exists(image_file))
        ci.save_image(image_file)
        self.assertTrue(ft.file_exists(image_file))
        self.assertEqual(image_file, ci.source_path)

        # Finale: cacheing should not re-create the cache file because now the
        # image file exists
        self.assertFalse(ft.file_exists(cache_file))
        ci.cache(cache_file)
        self.assertFalse(ft.file_exists(cache_file))
        cached_size = sys.getsizeof(ci)
        self.assertLess(cached_size, in_memory_size)

    def test_cache_memlimit0(self):
        """Check that cacheing doesn't halt forever when the memory limit is 0."""
        default_cache_file = ft.join(self.out_dir, f"{self.test_name}.npy")
        cache_path_gen = lambda: default_cache_file

        # create the cacheable image
        ci1 = CacheableImage(cache_path=self.example_cache_path)
        self.assertAlmostEqual(0, sys.getsizeof(ci1), delta=ci1._expected_cached_size)

        # verify we're not cached yet
        ci1.nparray
        self.assertAlmostEqual(40 * 40 * 3, sys.getsizeof(ci1), delta=ci1._expected_cached_size)

        # verify the memory limit is working
        ci1.cache_images_to_disk_as_necessary(1e10, cache_path_gen)
        self.assertAlmostEqual(40 * 40 * 3, sys.getsizeof(ci1), delta=ci1._expected_cached_size)

        # check that a memory limit of 0 is accepted
        ci1.cache_images_to_disk_as_necessary(0, cache_path_gen)
        self.assertAlmostEqual(0, sys.getsizeof(ci1), delta=ci1._expected_cached_size)


if __name__ == "__main__":
    unittest.main()
