from os.path import join, dirname
import unittest

import matplotlib.pyplot as plt
import matplotlib.testing.compare as mplt
import numpy as np

from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Uxyz import Uxyz
import opencsp.common.lib.tool.file_tools as ft


class TestVisualizeOrthorectifiedSlopeAbstract(unittest.TestCase):
    """Tests orthorectified plots of
    - Slope
    - Slope error
    - Curvature

    NOTE: To update the unit test data, run the test and copy the .PNG files from
    the data/output folder to the data/input folder. Run the test again to confirm passing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Set up directories
        cls.dir_output = join(dirname(__file__), "data/output/VisualizeOrthorectifiedSlopeAbstract")
        ft.create_directories_if_necessary(cls.dir_output)
        cls.dir_input = join(dirname(__file__), "data/input/VisualizeOrthorectifiedSlopeAbstract")
        ft.create_directories_if_necessary(cls.dir_input)

        # Define optic shape
        shape = RegionXY.from_vertices(Vxy(([0.6, -0.6, -0.6, 0.6], [-0.6, -0.6, 0.6, 0.6])))

        # Calculate surface xyz points
        xv = yv = np.arange(-0.6, 0.7, 0.02)
        X, Y = np.meshgrid(xv, yv)
        Z = np.zeros(X.shape)
        surface_points = Pxyz((X, Y, Z))

        # Calculate normal vectors
        nvecs = np.ones((3, len(surface_points)))
        nvecs[0] = np.sin(2 * np.pi * X).flatten() * 0.05
        nvecs[1] = np.sin(2 * np.pi * Y).flatten() * 0.05
        normal_vectors = Uxyz(nvecs)

        # Create mirror object
        cls.test_mirror_bilinear = MirrorPoint(surface_points, normal_vectors, shape, "bilinear")
        cls.test_mirror_nearest = MirrorPoint(surface_points, normal_vectors, shape, "nearest")

        # Create reference optic
        nvecs_flat = np.ones((3, len(surface_points)))
        nvecs_flat[0:2] = 0
        normal_vectors_flat = Uxyz(nvecs_flat)
        cls.reference = MirrorPoint(surface_points, normal_vectors_flat, shape, "nearest")

    def _get_axes(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(6, 6))
        return fig, fig.gca()

    def test_plot_slope_error_magnitude_linear(self):
        # Setup files
        file_out = join(self.dir_output, "slope_error_linear.png")
        file_in = join(self.dir_input, "slope_error_linear.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_bilinear.plot_orthorectified_slope_error(
            self.reference, 0.005, "magnitude", 100, ax, 0.1, 1000, "black"
        )
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_slope_error_magnitude_nearest(self):
        # Setup files
        file_out = join(self.dir_output, "slope_error_nearest.png")
        file_in = join(self.dir_input, "slope_error_nearest.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_nearest.plot_orthorectified_slope_error(
            self.reference, 0.005, "magnitude", 100, ax, 0.1, 1000, "black"
        )
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_slope_magnitude_linear(self):
        # Setup files
        file_out = join(self.dir_output, "slope_linear.png")
        file_in = join(self.dir_input, "slope_linear.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_bilinear.plot_orthorectified_slope(0.005, "magnitude", 100, ax, 0.1, 1000, "black")
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_slope_magnitude_nearest(self):
        # Setup files
        file_out = join(self.dir_output, "slope_nearest.png")
        file_in = join(self.dir_input, "slope_nearest.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_nearest.plot_orthorectified_slope(0.005, "magnitude", 100, ax, 0.1, 1000, "black")
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_curvature_magnitude_linear(self):
        # Setup files
        file_out = join(self.dir_output, "curvature_linear.png")
        file_in = join(self.dir_input, "curvature_linear.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_bilinear.plot_orthorectified_curvature(0.005, "combined", 100, ax)
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_curvature_magnitude_nearest(self):
        # Setup files
        file_out = join(self.dir_output, "curvature_nearest.png")
        file_in = join(self.dir_input, "curvature_nearest.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_nearest.plot_orthorectified_curvature(0.005, "combined", 100, ax)
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_curvature_magnitude_linear_processing_smooth(self):
        # Setup files
        file_out = join(self.dir_output, "curvature_linear_smooth.png")
        file_in = join(self.dir_input, "curvature_linear_smooth.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_bilinear.plot_orthorectified_curvature(
            0.005, "combined", 100, ax, processing=["smooth"], smooth_kernel_width=5
        )
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def test_plot_curvature_magnitude_linear_processing_smooth_log(self):
        # Setup files
        file_out = join(self.dir_output, "curvature_linear_smooth_log.png")
        file_in = join(self.dir_input, "curvature_linear_smooth_log.png")
        # Create image
        fig, ax = self._get_axes()
        self.test_mirror_bilinear.plot_orthorectified_curvature(
            0.005, "combined", 100, ax, processing=["smooth", "log"], smooth_kernel_width=5
        )
        fig.savefig(file_out, dpi=300)
        plt.close(fig)
        # Test
        self._compare_actual_expected_images(file_out, file_in)

    def _compare_actual_expected_images(self, actual_location: str, expected_location: str, tolerance=0.2) -> bool:
        """Tests if image files match."""
        self.assertIsNone(mplt.compare_images(expected_location, actual_location, tolerance))

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
