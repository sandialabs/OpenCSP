import os
import subprocess
import sys
import time
import unittest

import matplotlib.pyplot as plt

import opencsp.common.lib.opencsp_path.opencsp_root_path as root_path
import opencsp.common.lib.process.subprocess_tools as st
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.test.lib.RenderControlFigureRecordInfSave as rcfr_is
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

is_original_call = "--funcname" in sys.argv
""" Because we call this file again but with arguments, we need to know if
this was the original call as from unittest or if this was called from one
of the unit test methods. """


class test_figure_management(unittest.TestCase):
    dir_in = os.path.join('common', 'lib', 'render', 'test', 'data', 'input', 'figure_management')
    dir_out = os.path.join('common', 'lib', 'render', 'test', 'data', 'output', 'figure_management')

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.dir_in = test_figure_management.dir_in
        self.dir_out = test_figure_management.dir_out

    @classmethod
    def setUpClass(cls) -> None:
        ret = super().setUpClass()
        ft.create_directories_if_necessary(cls.dir_out)
        if is_original_call:
            ft.delete_files_in_directory(cls.dir_out, "*")
        return ret

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

        figs_txts = fm.save_all_figures(self.dir_out)
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

        figs_txts = fm.save_all_figures(self.dir_out)
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
