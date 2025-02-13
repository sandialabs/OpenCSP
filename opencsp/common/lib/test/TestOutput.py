"""
Base class supporting testing routines that generate output, including plots, csv files, etc.

Derived classes will call test routines that write output to an "actual_output" directory, and these 
files will be compared against known-good files in a sibling "expected_output" directory.

Supports both interactive execution and automatic execution via pytest.
"""

import inspect
import os
import unittest

import matplotlib.pyplot as plt

import opencsp.common.lib.opencsp_path.opencsp_root_path as ort
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestOutput(unittest.TestCase):
    @classmethod
    def setUpClass(
        cls,
        source_file_body: str,
        figure_prefix_root: str,
        interactive: bool = None,
        verify: bool = None,
        output_path=os.path.dirname(__file__),
    ):
        # Capture input.
        cls.source_file_body = source_file_body
        cls.figure_prefix_root = figure_prefix_root
        cls.interactive = interactive
        cls.verify = verify
        cls.output_path = output_path

        # Set the location to save files.
        cls.expected_output_dir = os.path.join(cls.output_path, "data", "input", cls.source_file_body)
        cls.actual_output_dir = os.path.join(cls.output_path, "data", "output", cls.source_file_body)

        # Setup log reporting.
        log_file_dir_body_ext = os.path.join(cls.actual_output_dir, cls.source_file_body + ".log")
        print("log_file_dir_body_ext = ", log_file_dir_body_ext)
        lt.logger(log_file_dir_body_ext, delete_existing_log=True)

        # Clear actual output directory, so we don't interpret previously-written files as being output from this run.
        if ft.directory_exists(cls.actual_output_dir):
            lt.info('Clearing files in directory "' + cls.actual_output_dir + '"...')
            ft.delete_files_in_directory(cls.actual_output_dir, "*")
        else:
            ft.create_directories_if_necessary(cls.actual_output_dir)

        # Set the figure and axis control for all figures.
        lt.info("Initializing render control structures...")
        cls.figure_control: RenderControlFigure = rcfg.RenderControlFigure(tile_array=(2, 1), tile_square=True)
        cls.figure_control_large: RenderControlFigure = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=False)
        cls.axis_control_m: RenderControlAxis = rca.meters()

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

    def tearDown(self) -> None:
        lt.info("Closing created plot windows...")
        plt.close("all")

    def start_test(self, close=True):
        # Tag to enable users inspecting the output data to find the calling routine, based on the saved metadata.
        test_routine_name = inspect.stack()[1][
            3
        ]  # See https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
        self.code_tag = self.source_file_body + "." + test_routine_name + "()"
        lt.info("\n\nStarting " + self.code_tag + "...")

        # Reset rendering.
        lt.info("Closing existing plot windows...")
        if close:
            plt.close("all")
        lt.info("Resetting figure management structure...")
        fm.reset_figure_management()

    def figure_prefix(self, figure_num: int) -> None:
        # Figure numbers are required because figure titles may be identical, which would cause
        # saved filenames to be identical.  This causes a problem, because we want to compare a
        # gainst matching known-good files.  But if the filenames are re-used, we will compare
        # against the wrong content.
        #
        # Previously we attempted to automatically calculate figure numbers using a TestOutput
        # data member, but this failed because there was a problem achieving correct behavior
        # in updating a data member of the TestOutput class when a given test routine exited
        # when running pytest.  (Interactive execution worked fine.)
        #
        # While I didn't solve this, it makes sense since each test should operate independently
        # without needing the output from previous tests.  This is semantically appropriate due
        # to the need for independent test computations (one test should not depend on the output
        # from another test), and also because pytest can be configured so that individual tests
        # run in parallel, making their execution order unpredictable.
        #
        # So automatically generating consecutive figure numbers seems fundamentally flawed,
        # because we cannot guarantee the order of test execution.  But we still need some means
        # of assuring that (a) each figure filename is unique, and (b) filenames are consistent
        # regardless of test execution order.  The solution I chose is to hard-code figure numbers,
        # relying on the calling code to keep them unique.
        #
        return self.figure_prefix_root + "{0:03d}".format(figure_num)

    def show_save_and_check_figure(self, fig_record: RenderControlFigureRecord, dpi=600) -> None:
        """
        Once a figure is drawn, this routine wraps up the test.
        """
        # Show the figure, save it to disk, and verify that it matches expectations.
        stest.show_save_and_check_figure(
            fig_record, self.actual_output_dir, self.expected_output_dir, self.verify, show_figs=True, dpi=dpi
        )
        # Clear.
        if not self.interactive:
            plt.close("all")

    def save_and_check_image(self, image, dpm, output_file_body, output_ext) -> None:
        """
        Once an image is generated, this routine wraps up the test.
        """
        # Save the image to disk, and verify that it matches expectations.
        stest.save_and_check_image(
            image, dpm, self.actual_output_dir, self.expected_output_dir, output_file_body, output_ext, self.verify
        )
        # Clear.
        if not self.interactive:
            plt.close("all")
