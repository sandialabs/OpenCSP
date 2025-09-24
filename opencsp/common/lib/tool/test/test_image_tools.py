import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import PIL.Image as Image

import opencsp.common.lib.cv.CacheableImage as ci
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class TestImageTools(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = ft.join(path, "data", "input", "image_tools")
        self.out_dir = ft.join(path, "data", "output", "image_tools")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_numpy_to_image_truncate(self):
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = it.numpy_to_image(arr8i, rescale_or_clip="truncate")
        im16i = it.numpy_to_image(arr16i, rescale_or_clip="truncate")
        im8f = it.numpy_to_image(arr8f, rescale_or_clip="truncate")
        im16f = it.numpy_to_image(arr16f, rescale_or_clip="truncate")

        nptest.assert_array_equal(np.asarray(im8i), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16i), np.array([[0, 255, 255]]))
        nptest.assert_array_equal(np.asarray(im8f), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16f), np.array([[0, 255, 255]]))

    def test_numpy_to_image_rescale(self):
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = it.numpy_to_image(arr8i, rescale_or_clip="rescale")
        im16i = it.numpy_to_image(arr16i, rescale_or_clip="rescale")
        im8f = it.numpy_to_image(arr8f, rescale_or_clip="rescale")
        im16f = it.numpy_to_image(arr16f, rescale_or_clip="rescale")

        nptest.assert_array_equal(np.asarray(im8i), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16i), np.array([[0, 127, 255]]))
        nptest.assert_array_equal(np.asarray(im8f), np.array([[0, 125, 255]]))
        nptest.assert_array_equal(np.asarray(im16f), np.array([[0, 127, 255]]))

    def test_dims_and_nchannels(self):
        self.assertEqual(((1, 1), 1), it.dims_and_nchannels(np.array([[0]])))
        self.assertEqual(((2, 2), 1), it.dims_and_nchannels(np.array([[0, 0], [0, 0]])))
        self.assertEqual(((1, 1), 3), it.dims_and_nchannels(np.array([[[0, 0, 0]]])))
        self.assertEqual(((480, 640), 3), it.dims_and_nchannels(np.array(Image.new("RGB", (640, 480)))))

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

    def test_getsizeof_approx(self):
        """
        Verifies that the returned size of the image in memory is somewhere in
        the ballpark for how much memory we're expecting.
        """
        implementation_overhead = 48
        # The delta accounts for the extra bytes of memory used for the python
        # object overhead. This number is going to be system and implementation
        # specific, and so might change with different python versions.

        delta = 2 * implementation_overhead
        # Also, we don't actually care that much what the
        # implementation-specific number is, so let's just make the buffer a
        # little bigger.

        img = Image.new("RGB", (40, 40))
        self.assertAlmostEqual(it.getsizeof_approx(img), 40 * 40 * 3, delta=delta)
        if 40 * 40 * 3 + implementation_overhead != it.getsizeof_approx(img):
            lt.warn(
                "Warning in test_image_tools.test_getsizeof_approx(): "
                + "40*40*3 + implementation_overhead != it.getsizeof_approx(img) "
                + f"({40*40*3} + {implementation_overhead} != {it.getsizeof_approx(img)})"
            )

    def test_to_image(self):
        # Test with a random NumPy array
        arr = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        np_image = it.to_image(arr, output_type="numpy")
        pil_image = it.to_image(arr, output_type="pillow")

        nptest.assert_array_equal(np_image, arr)
        self.assertIsInstance(pil_image, Image.Image)

        # Test with a PIL Image
        pil_image_input = Image.fromarray(arr)
        np_image_from_pil = it.to_image(pil_image_input, output_type="numpy")
        pil_image_from_pil = it.to_image(pil_image_input, output_type="pillow")

        nptest.assert_array_equal(np_image_from_pil, arr)
        self.assertIs(pil_image_from_pil, pil_image_input)

        # Test with a file path
        file_path = ft.join(self.out_dir, "test_to_image.png")
        pil_image_from_pil.save(file_path)
        np_image_from_file = it.to_image(file_path, output_type="numpy")
        pil_image_from_file = it.to_image(file_path, output_type="pillow")

        self.assertIsInstance(np_image_from_file, np.ndarray)
        self.assertIsInstance(pil_image_from_file, Image.Image)

        # Test with a RenderControlFigureRecord
        figure_record = rcfr.RenderControlFigureRecord("Test Figure", "Test Title", "Test Caption", 1, plt.figure())
        np_image_from_record = it.to_image(figure_record, output_type="numpy")
        pil_image_from_record = it.to_image(figure_record, output_type="pillow")

        self.assertIsInstance(np_image_from_record, np.ndarray)
        self.assertIsInstance(pil_image_from_record, Image.Image)

        # Test with a CacheableImage
        cacheable_image = ci.CacheableImage(array=arr)
        np_image_from_cacheable = it.to_image(cacheable_image, output_type="numpy")
        pil_image_from_cacheable = it.to_image(cacheable_image, output_type="pillow")

        nptest.assert_array_equal(np_image_from_cacheable, arr)
        self.assertIsInstance(pil_image_from_cacheable, Image.Image)

        # Test with unsupported type
        with self.assertRaises(ValueError):
            it.to_image(12345, output_type="numpy")


if __name__ == "__main__":
    unittest.main()
