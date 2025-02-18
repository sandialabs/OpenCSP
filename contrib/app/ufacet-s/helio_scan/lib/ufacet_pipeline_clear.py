"""
Utilities for clearing output from the UFACET pipeline.



"""

import os

import opencsp.common.lib.tool.file_tools as ft


def prepare_render_directory(output_render_dir, delete_suffix, render_control):
    """
    Sets up the directory for preparing the rendering output for  given step.
    Creates the directory if it does not exist.
    If the directory exists and already contains previous output. and if the render_control indicates,
    then deletes previous output files matching the specified regular expression.
    (Note that directories are not removed.  See the routine delete_files_in_directory() for an explanation.)
    """
    # Create the output render directory if necessary.
    ft.create_directories_if_necessary(output_render_dir)

    # Clear previous rendering output.
    # Here are three scenarios:
    #   1. We have recalculated the main result, and so old renderins need to be elminated to avoid inconsistency.
    #   2. We have improved the rendering code, and want to replace previous rendering output with updated versions.
    #   3. We have added new rendering capability, and want to add new rendering output, without taking the time
    #      to re-generate everything.
    # For cases 1 and 2, we want to clear out previous rendering results.
    # For case 3, we want to leave previous results in place, and selectively generate the new material using the
    # render control flags.
    if (not ft.directory_is_empty(output_render_dir)) and (render_control.clear_previous == True):
        print("In prepare_render_directory(), deleting previous render files from:", output_render_dir)
        ft.delete_files_in_directory(output_render_dir, ("*" + delete_suffix))
