import sys
import unittest

import matplotlib.pyplot as plt
from PIL import Image

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render.test.lib.RenderControlFigureRecordInfSave as rcfr_is
import opencsp.common.lib.tool.file_tools as ft

is_original_call = "--funcname" in sys.argv
""" Because we call this file again but with arguments, we need to know if
this was the original call as from unittest or if this was called from one
of the unit test methods. """


class test_figure_management(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, 'data/input', name.split('test_')[-1])
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        if is_original_call:
            ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

    def tearDown(self):
        # Make sure we release all matplotlib resources.
        plt.close('all')

    def assert_exists(self, figs_txts, cnt):
        """Verifies that cnt images and text files exist."""
        if len(figs_txts) == 2:
            figs, txts = figs_txts
        else:
            figs, txts, failed = figs_txts
        self.assertEqual(len(figs), cnt, f"Incorrect number of image files!")
        self.assertEqual(len(txts), cnt, f"Incorrect number of text files!")
        for f in figs:
            self.assertTrue(ft.file_exists(f), f"Image file \"{f}\" does not exist!")
        for f in txts:
            self.assertTrue(ft.file_exists(f), f"Text file \"{f}\" does not exist!")

    def test_save_all_figures_line(self):
        """Test that saving a single figure (aka one image) succeeds."""
        name = "line_1"
        fm.reset_figure_management()

        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        fig_record = fm.setup_figure(figure_control, name=name, code_tag=f"{__file__}.test_save_all_figures_line()")
        view = fig_record.view
        line = list(range(100))
        view.draw_p_list(line)

        figs_txts = fm.save_all_figures(self.out_dir)
        self.assert_exists(figs_txts, 1)

    def test_save_all_figures_two_lines(self):
        """Test that saving two figures (aka two images) succeeds."""
        names = ["upper", "lower"]
        fm.reset_figure_management()

        lines = [[100] * 100, [0] * 100]
        for i in range(2):
            figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
            fig_record = fm.setup_figure(
                figure_control, name=names[i], code_tag=f"{__file__}.test_save_all_figures_two_lines()"
            )
            view = fig_record.view
            line = lines[i]
            view.draw_p_list(line)

        figs_txts = fm.save_all_figures(self.out_dir)
        self.assert_exists(figs_txts, 2)

    def _figure_manager_timeout_1(self):
        """Helper method. Generate a figure manager and populate it with one figure record that will never finish saving."""
        name = "line_3"
        fm.reset_figure_management()

        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        fig_old = fm.setup_figure(figure_control, name=name, code_tag=f"{__file__}._figure_manager_timeout_1()")

        # replace the figure record with one that will never finish saving
        fig_record = rcfr_is.RenderControlFigureRecordInfSave(
            name=fig_old.name,
            title=fig_old.title,
            caption=fig_old.caption,
            figure_num=fig_old.figure_num,
            figure=fig_old.figure,
        )
        fig_record.view = fig_old.view
        fm.fig_record_list.clear()
        fm.fig_record_list.append(fig_record)

        # continue with setting up the figure
        view = fig_record.view
        line = list(range(100))
        view.draw_p_list(line)

        return fm

    def test_upper_left_xy_no_exception(self):
        """
        Verify that figure_management._setup_figure() with the figure control
        parameter "upper_left_xy" set doesn't raise an exception.
        """
        # TODO how to test that the window has actually been located correctly?
        axis_control = rca.meters()
        figure_control = rcfg.RenderControlFigure(tile=False, upper_left_xy=(100, 100))
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure(
            figure_control,
            axis_control,
            view_spec_2d,
            title=self.test_name,
            code_tag=f"{__file__}.{self.test_name}",
            equal=False,
        )
        fig_record.view.show()
        fig_record.close()

    def test_maximize_no_exception(self):
        """
        Verify that figure_management._setup_figure() with the figure control
        parameter "maximize" set doesn't raise an exception.
        """
        # TODO how to test that the window has actually been maximized?
        axis_control = rca.meters()
        figure_control = rcfg.RenderControlFigure(tile=False, maximize=True)
        view_spec_2d = vs.view_spec_xy()
        try:
            fig_record = fm.setup_figure(
                figure_control,
                axis_control,
                view_spec_2d,
                title=self.test_name,
                code_tag=f"{__file__}.{self.test_name}",
                equal=False,
            )
            fig_record.view.show()
            fig_record.close()
        except Exception as ex:
            ubi8_msg = '_tkinter.TclError: bad argument "zoomed": must be normal, iconic, or withdrawn'
            if ubi8_msg in str(ex):
                # TODO how to make this test work on ubi8?
                self.skipTest("Window 'maximize' state doesn't working on our ubi8 test docker image.")

    def test_save_figsize(self):
        """Verify that the size of the saved figure is as given in the save parameters."""
        # create and save the figure with pixel sizes:
        # small:   900 x 600
        # regular: 1800 x 1200
        # large:   2700 x 1800
        axis_control = rca.meters()
        figure_control = rcfg.RenderControlFigure(tile=False, figsize=(3, 2))
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure(
            figure_control,
            axis_control,
            view_spec_2d,
            title=self.test_name,
            code_tag=f"{__file__}.{self.test_name}",
            equal=False,
        )
        # fig_record.view.show()
        fig_record.save(self.out_dir, f"{self.test_name}_small", format="png", dpi=300, close_after_save=False)
        fig_record.save(self.out_dir, f"{self.test_name}_regular", format="png", dpi=600, close_after_save=False)
        fig_record.save(self.out_dir, f"{self.test_name}_large", format="png", dpi=900, close_after_save=False)
        fig_record.view.show()
        fig_record.close()

        # load the images and verify their size in pixels
        with Image.open(ft.join(self.out_dir, f"{self.test_name}_small_xy.png")) as img_small:
            self.assertEqual(img_small.width, 900)
            self.assertEqual(img_small.height, 600)
        with Image.open(ft.join(self.out_dir, f"{self.test_name}_regular_xy.png")) as img_regular:
            self.assertEqual(img_regular.width, 1800)
            self.assertEqual(img_regular.height, 1200)
        with Image.open(ft.join(self.out_dir, f"{self.test_name}_large_xy.png")) as img_large:
            self.assertEqual(img_large.width, 2700)
            self.assertEqual(img_large.height, 1800)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog=__file__.rstrip(".py"), description='Testing figure management')
    parser.add_argument('--funcname', help="Calls the given function")
    args = parser.parse_args()
    func_name = args.funcname

    if func_name != None and func_name in [
        "save_all_figures_fail_and_raise_executor",
        "save_all_figures_fail_no_raise_executor",
    ]:
        tfm = test_figure_management()
        tfm.__getattribute__(func_name)()
    else:
        unittest.main()
