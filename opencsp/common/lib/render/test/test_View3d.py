import copy
import unittest

import opencsp.common.lib.tool.file_tools as ft

import numpy as np
from PIL import Image

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlHeatmap as rcheat
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.file_tools as ft


class test_View3d(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, "data/input", name.split("test_")[-1])
        cls.out_dir = ft.join(path, "data/output", name.split("test_")[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split(".")[-1]

    def setup_figure(self) -> rcfr.RenderControlFigureRecord:
        # clear existing figures
        fm.reset_figure_management()

        # setup the new figure
        axis_control = rca.meters(grid=False)
        figure_control = rcfg.RenderControlFigure()
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure(
            figure_control,
            axis_control,
            view_spec_2d,
            title=self.test_name,
            code_tag=f"{__file__}.{self.test_name}()",
            equal=False,
        )

        return fig_record

    def test_draw_image(self):
        fm.reset_figure_management()
        color_image = ft.join(self.in_dir, f"{self.test_name}_color.png")
        grayscale_image = ft.join(self.in_dir, f"{self.test_name}_grayscale.png")

        # setup
        axis_control = rca.meters(grid=False)
        figure_control = rcfg.RenderControlFigure(tile=False, figsize=(2, 2))
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure(
            figure_control,
            axis_control,
            view_spec_2d,
            title=self.test_name,
            code_tag=f"{__file__}.{self.test_name}()",
            equal=True,
        )

        # draw
        x = np.arange(0, np.pi * 2, 0.1)
        y = np.sin(x)
        fig_record.view.draw_pq((x * 100 / (np.pi * 2), (y + 1.0) * 80))
        fig_record.view.draw_image(color_image, (20, 40), (60, 80))
        fig_record.view.draw_image(grayscale_image, (70, 10), (20, 40), cmap="viridis", draw_on_top=False)

        # save
        fig_record.view.show()
        actual_img = Image.fromarray(fig_record.to_array())
        actual_img.save(ft.join(self.out_dir, f"{self.test_name}_actual.png"))
        fig_record.close()

        # compare
        expected_img = Image.open(ft.join(self.in_dir, f"{self.test_name}_expected.png"))
        actual_image = np.array(actual_img)
        expected_image = np.array(expected_img)
        np.testing.assert_array_equal(actual_image, expected_image)

    def test_draw_heatmap2d(self):
        # build the heatmap
        linear = np.arange(0, 256, 1)
        square = linear.reshape((16, 16))

        # with single points
        fig_record = self.setup_figure()
        fig_record.view.draw_heatmap_2d(square, style=rcheat.RenderControlHeatmap(cmap="inferno"))
        fig_record.view.show(equal=True)
        image_heatmap = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_heatmap).save(ft.join(self.out_dir, f"{self.test_name}.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_heatmap)

    def test_draw_xyz_text(self):
        """Verify that text gets drawn to the graph. Also test all the other options for drawing."""
        # draw a box, for reference
        fig_record = self.setup_figure()
        fig_record.view.draw_pq_list([(-1, -1), (1, -1), (1, 1), (-1, 1)], close=True)

        # basic text drawing
        fig_record.view.draw_xyz_text((0, 0, 0), "center", style=rctxt.default(color="black"))
        fig_record.view.draw_xyz_text((-1, -1, 0), "south-west", style=rctxt.default(color="black"))
        fig_record.view.draw_xyz_text((1, -1, 0), "south-east", style=rctxt.default(color="black"))
        fig_record.view.draw_xyz_text((1, 1, 0), "north-east", style=rctxt.default(color="black"))
        fig_record.view.draw_xyz_text((-1, 1, 0), "north-west", style=rctxt.default(color="black"))

        # text options
        fig_record.view.draw_xyz_text(
            (-1, 0, 0), "vertical", style=rctxt.RenderControlText(color="orange", rotation=np.pi / 2)
        )
        fig_record.view.draw_xyz_text(
            (1, 0, 0), "upside-down", style=rctxt.RenderControlText(color="orange", rotation=np.pi)
        )
        fig_record.view.draw_xyz_text(
            (0, 0.1, 0),
            "left aligned",
            style=rctxt.RenderControlText(color="orange", horizontalalignment="left", fontsize="small"),
        )
        fig_record.view.draw_xyz_text(
            (0, -0.1, 0),
            "right aligned",
            style=rctxt.RenderControlText(color="orange", horizontalalignment="right", fontsize="large"),
        )
        fig_record.view.draw_xyz_text(
            (-0.5, 0, 0),
            "top aligned",
            style=rctxt.RenderControlText(color="orange", verticalalignment="top", fontstyle="italic"),
        )
        fig_record.view.draw_xyz_text(
            (0.5, 0, 0),
            "bottom aligned",
            style=rctxt.RenderControlText(color="orange", verticalalignment="bottom", fontweight="bold"),
        )

        # render
        fig_record.view.show(equal=True)
        image_text = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_text).save(ft.join(self.out_dir, f"{self.test_name}.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_text)

    def test_draw_xyz(self):
        """Verify that the various accepted input arguments produce the same output."""
        style = rcps.RenderControlPointSeq("None", marker=".")

        # with single points
        fig_record = self.setup_figure()
        fig_record.view.draw_xyz((1, 1), style=style)
        fig_record.view.draw_xyz((2, 2), style=style)
        fig_record.view.draw_xyz((2, 1), style=style)
        fig_record.view.draw_xyz((1, 2), style=style)
        fig_record.view.show(equal=True)
        image_single_points = fig_record.to_array()
        fig_record.close()

        # with lists
        x = [1, 2, 2, 1]
        y = [1, 1, 2, 2]
        fig_record = self.setup_figure()
        fig_record.view.draw_xyz((x, y), style=style)
        fig_record.view.show(equal=True)
        image_xy_lists = fig_record.to_array()
        fig_record.close()

        # with a numpy array
        arr = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
        fig_record = self.setup_figure()
        fig_record.view.draw_xyz(arr, style=style)
        fig_record.view.show(equal=True)
        image_arr = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_single_points).save(ft.join(self.out_dir, f"{self.test_name}_single_points.png"))
        Image.fromarray(image_xy_lists).save(ft.join(self.out_dir, f"{self.test_name}_xy_lists.png"))
        Image.fromarray(image_arr).save(ft.join(self.out_dir, f"{self.test_name}_arr.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_single_points)
        np.testing.assert_array_equal(expected, image_xy_lists)
        np.testing.assert_array_equal(expected, image_arr)

    def test_draw_xyz_list_style_options(self):
        """Tests all the various render options for RenderControlPointSeq."""
        kwargs_default_other = {
            "linestyle": ("-", "--"),
            "linewidth": (1, 2),
            "color": ("b", "r"),
            "marker": ("X", "+"),
            "markersize": (6, 10),
            "markeredgecolor": (None, "r"),
            "markeredgewidth": (None, 3),
            "markerfacecolor": (None, "r"),
            "markeralpha": (1.0, 0.5),
        }
        defaults = {kwarg: kwargs_default_other[kwarg][0] for kwarg in kwargs_default_other}
        others = {kwarg: kwargs_default_other[kwarg][1] for kwarg in kwargs_default_other}

        # setup the plot
        fig_record = self.setup_figure()

        # plot all the options
        for i, kwarg in enumerate(kwargs_default_other):
            style_options = copy.copy(defaults)
            style_options[kwarg] = others[kwarg]
            style = rcps.RenderControlPointSeq(**style_options)
            fig_record.view.draw_xyz_list([(i, 0, 0), (i, 1, 0)], style=style)
            fig_record.view.draw_xyz_text(
                (i, -0.5, 0),
                kwarg,
                rctxt.RenderControlText(verticalalignment="top", fontsize="small", color="black", rotation=np.pi / 2),
            )
        fig_record.view.show(equal=True)
        image_style_options = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_style_options).save(ft.join(self.out_dir, f"{self.test_name}.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_style_options)

    def test_draw_pq_text(self):
        """Verify that text gets drawn to the graph. Also test all the other options for drawing."""
        # draw a box, for reference
        fig_record = self.setup_figure()
        fig_record.view.draw_pq_list([(-1, -1), (1, -1), (1, 1), (-1, 1)], close=True)

        # basic text drawing
        fig_record.view.draw_pq_text((0, 0), "center", style=rctxt.default(color="black"))
        fig_record.view.draw_pq_text((-1, -1), "south-west", style=rctxt.default(color="black"))
        fig_record.view.draw_pq_text((1, -1), "south-east", style=rctxt.default(color="black"))
        fig_record.view.draw_pq_text((1, 1), "north-east", style=rctxt.default(color="black"))
        fig_record.view.draw_pq_text((-1, 1), "north-west", style=rctxt.default(color="black"))

        # text options
        fig_record.view.draw_pq_text(
            (-1, 0), "vertical", style=rctxt.RenderControlText(color="orange", rotation=np.pi / 2)
        )
        fig_record.view.draw_pq_text(
            (1, 0), "upside-down", style=rctxt.RenderControlText(color="orange", rotation=np.pi)
        )
        fig_record.view.draw_pq_text(
            (0, 0.1),
            "left aligned",
            style=rctxt.RenderControlText(color="orange", horizontalalignment="left", fontsize="small"),
        )
        fig_record.view.draw_pq_text(
            (0, -0.1),
            "right aligned",
            style=rctxt.RenderControlText(color="orange", horizontalalignment="right", fontsize="large"),
        )
        fig_record.view.draw_pq_text(
            (-0.5, 0),
            "top aligned",
            style=rctxt.RenderControlText(color="orange", verticalalignment="top", fontstyle="italic"),
        )
        fig_record.view.draw_pq_text(
            (0.5, 0),
            "bottom aligned",
            style=rctxt.RenderControlText(color="orange", verticalalignment="bottom", fontweight="bold"),
        )

        # render
        fig_record.view.show(equal=True)
        image_text = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_text).save(ft.join(self.out_dir, f"{self.test_name}.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_text)

    def test_draw_pq_list_arrows(self):
        fig_record = self.setup_figure()

        # draw
        square_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        arrow_style = rcps.RenderControlPointSeq(marker="arrow", markersize=0.1)
        fig_record.view.draw_pq_list(square_corners, close=True, style=arrow_style)
        fig_record.view.show(equal=True, block=False)
        actual = fig_record.to_array()
        fig_record.close()

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, actual)

        # save
        img = Image.fromarray(actual)
        img.save(ft.join(self.out_dir, f"{self.test_name}.png"))

    def test_draw_pq_list_gradient(self):
        """
        Verify that a nice purple-yellow 'viridis' gradient is used to draw the
        four sides of the square.
        """
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # draw with gradient
        fig_record: rcfr.RenderControlFigureRecord = self.setup_figure()
        fig_record.view.draw_pq_list(square, close=True, gradient=True)

        # render
        fig_record.view.show(equal=True)
        image_gradient = fig_record.to_array()
        fig_record.close()

        # save the output
        Image.fromarray(image_gradient).save(ft.join(self.out_dir, f"{self.test_name}.png"))

        # load and compare
        expected = np.array(Image.open(ft.join(self.in_dir, f"{self.test_name}.png")))
        np.testing.assert_array_equal(expected, image_gradient)


if __name__ == "__main__":
    unittest.main()
