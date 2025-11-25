import cv2 as cv
import numpy as np
import PIL.Image as Image
import sympy
import unittest

from opencsp.common.lib.cv.PerspectiveTransform import PerspectiveTransform
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestPerspectiveTransform(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = ft.join(path, "data", "input", "PerspectiveTransform")
        self.out_dir = ft.join(path, "data", "output", "PerspectiveTransform")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def create_images(self):
        """Creates the test images."""
        path, _, _ = ft.path_components(__file__)
        data_dir = ft.join(path, "data", "input", "PerspectiveTransform")

        warped_image = np.full((105, 105, 3), fill_value=255, dtype=np.uint8)
        warped_image = cv.line(warped_image, (20, 20), (90, 10), color=(0, 0, 0), thickness=2)
        warped_image = cv.line(warped_image, (90, 10), (99, 99), color=(0, 0, 0), thickness=2)
        warped_image = cv.line(warped_image, (99, 99), (5, 85), color=(0, 0, 0), thickness=2)
        warped_image = cv.line(warped_image, (5, 85), (20, 20), color=(0, 0, 0), thickness=2)
        warped_image = ConvolutionImageProcessor(kernel="gaussian", diameter=3).process_images([warped_image])[0]
        Image.fromarray(warped_image).save(ft.join(data_dir, "warped_image.png"))

    def test_image_dewarp(self):
        """Regression test to verify that the image transform still works."""
        # corners:   TL  TR  BR   BL
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))
        warped_image = np.array(Image.open(ft.join(self.data_dir, "warped_image.png")))
        expected_dewarped_image = np.array(Image.open(ft.join(self.data_dir, "dewarped_image.png")))

        dewarped_image = persp_xform.transform_image(warped_image)

        Image.fromarray(dewarped_image).save(ft.join(self.out_dir, "dewarped_image.png"))
        np.testing.assert_array_almost_equal(dewarped_image, expected_dewarped_image)
        (height, width), nchannels = it.dims_and_nchannels(dewarped_image)
        self.assertEqual(width, 1000, "Expected the dewarped image to be 1000 pixels wide, to represent 1 full meter.")
        self.assertEqual(height, 1000, "Expected the dewarped image to be 1000 pixels high, to represent 1 full meter.")

    def test_image_dewarp_with_border(self):
        """Regression test to verify that the border option still works."""
        # corners:   TL  TR  BR   BL
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))
        warped_image = np.array(Image.open(ft.join(self.data_dir, "warped_image.png")))
        expected_dewarped_image = np.array(Image.open(ft.join(self.data_dir, "dewarped_image_with_border.png")))

        dewarped_image = persp_xform.transform_image(warped_image, buffer_width_px=5)

        Image.fromarray(dewarped_image).save(ft.join(self.out_dir, "dewarped_image_with_border.png"))
        np.testing.assert_array_almost_equal(dewarped_image, expected_dewarped_image)

    def test_image_dewarp_full(self):
        """Regression test to verify that the full image option still works."""
        # corners:   TL  TR  BR   BL
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))
        warped_image = np.array(Image.open(ft.join(self.data_dir, "warped_image.png")))
        expected_dewarped_image = np.array(Image.open(ft.join(self.data_dir, "dewarped_image_full.png")))

        dewarped_image = persp_xform.transform_image(warped_image, full_image=True)

        Image.fromarray(dewarped_image).save(ft.join(self.out_dir, "dewarped_image_full.png"))
        np.testing.assert_array_almost_equal(dewarped_image, expected_dewarped_image)

    def test_pixels_to_meters(self):
        """Tests that transformed locations can be determined on a single-point basis."""
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))

        for i, (px, py) in enumerate(zip(pixel_xs, pixel_ys)):
            mx, my = list(zip(meters_xs, meters_ys))[i]
            txy = persp_xform.pixels_to_meters(p2.Pxy([px, py]))
            self.assertAlmostEqual(
                txy.x[0],
                mx,
                msg=f"Forward transform ({px}, {py}) -> {txy.astuple()} instead of the expected ({mx}, {my})",
            )
            self.assertAlmostEqual(
                txy.y[0],
                my,
                msg=f"Forward transform ({px}, {py}) -> {txy.astuple()} instead of the expected ({mx}, {my})",
            )

    def test_coordinates_conversion_forward(self):
        """Tests that transformed locations can be determined using sympy."""
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))
        tx, ty = persp_xform.pixels_to_meters_conversions()

        for i, (px, py) in enumerate(zip(pixel_xs, pixel_ys)):
            mx, my = list(zip(meters_xs, meters_ys))[i]
            x, y = sympy.symbols("x y")
            txy = tx.evalf(subs={x: px, y: py}), ty.evalf(subs={x: px, y: py})
            self.assertAlmostEqual(
                txy[0],
                mx,
                msg=f"Forward conversion ({px}, {py}) -> {txy} instead of the expected ({mx}, {my})",
                delta=1e-5,
            )
            self.assertAlmostEqual(
                txy[1],
                my,
                msg=f"Forward conversion ({px}, {py}) -> {txy} instead of the expected ({mx}, {my})",
                delta=1e-5,
            )

    def test_meters_to_pixels(self):
        """Tests that original locations can be determined on a single-point basis."""
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))

        for i, (mx, my) in enumerate(zip(meters_xs, meters_ys)):
            px, py = list(zip(pixel_xs, pixel_ys))[i]
            txy = persp_xform.meters_to_pixels(p2.Pxy([mx, my]))
            self.assertAlmostEqual(
                txy.x[0],
                px,
                msg=f"Backward transform ({mx}, {my}) -> {txy.astuple()} instead of the expected ({px}, {py})",
            )
            self.assertAlmostEqual(
                txy.y[0],
                py,
                msg=f"Backward transform ({mx}, {my}) -> {txy.astuple()} instead of the expected ({px}, {py})",
            )

    def test_coordinates_conversion_backward(self):
        """Tests that transformed locations can be determined using sympy."""
        pixel_xs = [20, 90, 99, 5]
        pixel_ys = [20, 10, 99, 85]
        meters_xs = [0, 1, 1, 0]
        meters_ys = [0, 0, 1, 1]

        persp_xform = PerspectiveTransform(p2.Pxy((pixel_xs, pixel_ys)), p2.Pxy((meters_xs, meters_ys)))
        tx, ty = persp_xform.meters_to_pixels_conversions()

        for i, (mx, my) in enumerate(zip(meters_xs, meters_ys)):
            px, py = list(zip(pixel_xs, pixel_ys))[i]
            x, y = sympy.symbols("x y")
            txy = tx.evalf(subs={x: mx, y: my}), ty.evalf(subs={x: mx, y: my})
            self.assertAlmostEqual(
                txy[0],
                px,
                msg=f"Forward conversion ({mx}, {my}) -> {txy} instead of the expected ({px}, {py})",
                delta=1e-5,
            )
            self.assertAlmostEqual(
                txy[1],
                py,
                msg=f"Forward conversion ({mx}, {my}) -> {txy} instead of the expected ({px}, {py})",
                delta=1e-5,
            )


if __name__ == "__main__":
    # t = TestPerspectiveTransform()
    # t.create_images()
    unittest.main()
