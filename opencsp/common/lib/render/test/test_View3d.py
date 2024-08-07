import unittest

import opencsp.common.lib.tool.file_tools as ft

import numpy as np
from PIL import Image

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft


class test_View3d(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, 'data/input', name.split('test_')[-1])
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, '*')
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

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

    def test_plot_arrows(self):
        fig_record = self.setup_figure()

        # draw
        square_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        arrow_style = rcps.RenderControlPointSeq(marker='arrow', markersize=0.1)
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

    def test_draw_xyz(self):
        """Verify that the various accepted input arguments produce the same output."""
        style = rcps.RenderControlPointSeq('None', marker='.')

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


if __name__ == '__main__':
    unittest.main()
