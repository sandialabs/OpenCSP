import unittest

import opencsp.common.lib.tool.file_tools as ft

import numpy as np
from PIL import Image

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
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

    def test_plot_arrows(self):
        fm.reset_figure_management()

        # setup
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


if __name__ == '__main__':
    unittest.main()
