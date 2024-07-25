import numpy as np
import numpy.testing as nptest
import os
import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestImageTools(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = ft.join(path, "data", "input", "image_tools")
        self.out_dir = ft.join(path, "data", "output", "image_tools")

    def test_numpy_to_image_truncate(self):
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = it.numpy_to_image(arr8i, rescale_or_clip='truncate')
        im16i = it.numpy_to_image(arr16i, rescale_or_clip='truncate')
        im8f = it.numpy_to_image(arr8f, rescale_or_clip='truncate')
        im16f = it.numpy_to_image(arr16f, rescale_or_clip='truncate')

        nptest.assert_array_equal(np.asarray(im8i), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16i), np.array([[0, 255, 255]]))
        nptest.assert_array_equal(np.asarray(im8f), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16f), np.array([[0, 255, 255]]))

    def test_numpy_to_image_rescale(self):
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = it.numpy_to_image(arr8i, rescale_or_clip='rescale')
        im16i = it.numpy_to_image(arr16i, rescale_or_clip='rescale')
        im8f = it.numpy_to_image(arr8f, rescale_or_clip='rescale')
        im16f = it.numpy_to_image(arr16f, rescale_or_clip='rescale')

        nptest.assert_array_equal(np.asarray(im8i), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16i), np.array([[0, 127, 255]]))
        nptest.assert_array_equal(np.asarray(im8f), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16f), np.array([[0, 127, 255]]))

    def test_dims_and_nchannels(self):
        self.assertEqual(((1, 1), 1), it.dims_and_nchannels(np.array([[0]])))
        self.assertEqual(((2, 2), 1), it.dims_and_nchannels(np.array([[0, 0], [0, 0]])))
        self.assertEqual(((1, 1), 3), it.dims_and_nchannels(np.array([[[0, 0, 0]]])))

        # not enough/too many dimensions
        self.assertRaises(ValueError, it.dims_and_nchannels, np.array([0]))
        self.assertRaises(ValueError, it.dims_and_nchannels, np.array([[[[0]]]]))

    def test_min_max(self):
        arr1: np.ndarray = np.array([[[0], [1], [2], [5]]])
        arr3: np.ndarray = np.array([[[0, 1, 2], [1, 2, 3], [2, 4, 2], [5, 1, 2]]])

        nptest.assert_array_equal(it.min_max_colors(arr1)[0], np.array([0]))
        nptest.assert_array_equal(it.min_max_colors(arr1)[1], np.array([5]))
        nptest.assert_array_equal(it.min_max_colors(arr3)[0], np.array([0, 1, 2]))
        nptest.assert_array_equal(it.min_max_colors(arr3)[1], np.array([5, 4, 3]))

    def test_image_files_in_directory(self):
        all_image_files = it.image_files_in_directory(self.data_dir)
        self.assertIn("a.png", all_image_files)
        self.assertIn("b.PNG", all_image_files)
        self.assertIn("c.jpg", all_image_files)
        self.assertNotIn("d.txt", all_image_files)

        png_image_files = it.image_files_in_directory(self.data_dir, ["png"])
        self.assertIn("a.png", png_image_files)
        self.assertIn("b.PNG", png_image_files)
        self.assertNotIn("c.jpg", png_image_files)
        self.assertNotIn("d.txt", png_image_files)

        jpg_image_files = it.image_files_in_directory(self.data_dir, ["jpg"])
        self.assertNotIn("a.png", jpg_image_files)
        self.assertNotIn("b.PNG", jpg_image_files)
        self.assertIn("c.jpg", jpg_image_files)
        self.assertNotIn("d.txt", jpg_image_files)

        png_jpg_image_files = it.image_files_in_directory(self.data_dir, ["png", "jpg"])
        self.assertIn("a.png", png_jpg_image_files)
        self.assertIn("b.PNG", png_jpg_image_files)
        self.assertIn("c.jpg", png_jpg_image_files)
        self.assertNotIn("d.txt", png_jpg_image_files)


if __name__ == '__main__':
    unittest.main()
