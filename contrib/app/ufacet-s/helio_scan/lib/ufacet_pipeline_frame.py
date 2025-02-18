"""
Utilities for managing frames in the UFACET pipeline.



"""

from cv2 import cv2 as cv
import os

import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.math_tools as mt
import opencsp.app.ufacets.helio_scan.lib.ufacet_pipeline_clear as upc


def frame_id_str_given_frame_id(frame_id: int, input_frame_id_format: str):
    """Our extracted video file names are of the form:
       VideoFileBody.nnnnnn.JPG
    where the "nnnnnn" is a zero-padded numerical string such as "001258" and the
    length of the zero-padded string is given by the frame_id_format.
    An example frame_id_format is: "06d"

    Parameters
    ----------
        frame_id: int
        input_frame_id_format: str
            How to format the frame_id. Example "06d"

    Returns
    -------
        nnnnnn: str
            The frame id, formatted as a string"""
    frame_id_format_str = "{0:" + input_frame_id_format + "}"
    return frame_id_format_str.format(frame_id)


def frame_id_given_frame_id_str(frame_id_str):
    # Our extracted video file names are of the form:
    #    VideoFileBody.nnnnnn.JPG
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    return int(frame_id_str)


def frame_file_body_ext_given_frame_id(input_video_body, frame_id, input_frame_id_format):
    """Our extracted video file names are of the form:
       VideoFileBody.nnnnnn.JPG
    where the "nnnnnn" is a zero-padded numerical string such as "001258" and the
    length of the zero-padded string is given by the frame_id_format.
    An example frame_id_format is: "06d"

    Parameters
    ----------
        input_video_body: str
            Example "DJI_427t_428_429"
        frame_id: int
        input_frame_id_format: str
            How to format the frame_id. Example "06d"

    Returns
    -------
        body_ext: str
            Example "DJI_427t_428_429.000001.JPG" """
    frame_id_str = frame_id_str_given_frame_id(frame_id, input_frame_id_format)
    return frame_file_body_ext_given_frame_id_str(input_video_body, frame_id_str)


def frame_file_body_ext_given_frame_id_str(input_video_body: str, frame_id_str: str):
    """Our extracted video file names are of the form:
       VideoFileBody.nnnnnn.JPG
    where the "nnnnnn" is a zero-padded numerical string such as "001258" and the
    length of the zero-padded string is given by the frame_id_format.
    An example frame_id_format is: "06d"

    Parameters
    ----------
        input_video_body: str
            Example "DJI_427t_428_429"
        frame_id_str: str
            Example "000001" as from frame_id_str_given_frame_id(...)

    Returns
    -------
        body_ext: str
            Example "DJI_427t_428_429.000001.JPG" """
    return input_video_body + "." + frame_id_str + ".JPG"


def frame_id_str_given_prefix_number_KeyWord_body_ext(prefix_number_keyword_body_ext, keyword):
    # For multiple items (key corners, key frame tracks, ...), our canonical file names are of the form:
    #    VideoFileBody_nnnnnn_<keyword>_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258" and <keyword>
    # is a standard word such as "corners", "track", etc.
    #
    # Check extension.
    components = prefix_number_keyword_body_ext.split(".")
    if len(components) < 2:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" did not separate into at least two components when split by "."'
        )
        print(msg)
        raise ValueError(msg)
    ext = components[-1]
    body = components[-2]
    if ext != "csv":
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" did not end with extension ".csv"'
        )
        print(msg)
        raise ValueError(msg)
    # Break the filename into components.
    components = body.split("_")
    if len(components) < 4:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" file body did not separate into at least four components when split by "_"'
        )
        print(msg)
        raise ValueError(msg)
    fxnl_str = components[-1]
    keyword_str = components[-2]
    number_str = components[-3]
    if fxnl_str != "fnxl":
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" body did not end with substring "fnxl"'
        )
        print(msg)
        raise ValueError(msg)
    if keyword_str != keyword:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" did not include second-to-last substring "'
            + str(keyword)
            + '"'
        )
        print(msg)
        raise ValueError(msg)
    # Determine if the "number_str" component is a valid integer.
    if not mt.string_is_integer(number_str):
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_KeyWord_body_ext(), input prefix_number_keyword_body_ext="'
            + str(prefix_number_keyword_body_ext)
            + '" has a third-to-last substring "'
            + str(number_str)
            + '" that is not a valid integer.'
        )
        print(msg)
        raise ValueError(msg)
    # Return the number string, without parsing.
    return number_str


def frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(
    prefix_number_adjective_keyword_body_ext, adjective, keyword
):
    # For multiple items (key corners, key frame tracks, ...), our canonical file names are of the form:
    #    VideoFileBody_nnnnnn_<adjective>_<keyword>_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258" and <keyword>
    # is a standard word such as "corners", "track", etc.
    #
    # Check extension.
    components = prefix_number_adjective_keyword_body_ext.split(".")
    if len(components) < 2:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" did not separate into at least two components when split by "."'
        )
        print(msg)
        raise ValueError(msg)
    ext = components[-1]
    body = components[-2]
    if ext != "csv":
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" did not end with extension ".csv"'
        )
        print(msg)
        raise ValueError(msg)
    # Break the filename into components.
    components = body.split("_")
    if len(components) < 5:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" file body did not separate into at least five components when split by "_"'
        )
        print(msg)
        raise ValueError(msg)
    fxnl_str = components[-1]
    keyword_str = components[-2]
    adjective_str = components[-3]
    number_str = components[-4]
    if fxnl_str != "fnxl":
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" body did not end with substring "fnxl"'
        )
        print(msg)
        raise ValueError(msg)
    if keyword_str != keyword:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" did not include second-to-last substring "'
            + str(keyword)
            + '"'
        )
        print(msg)
        raise ValueError(msg)
    if adjective_str != adjective:
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" did not include second-to-last substring "'
            + str(adjective)
            + '"'
        )
        print(msg)
        raise ValueError(msg)
    # Determine if the "number_str" component is a valid integer.
    if not mt.string_is_integer(number_str):
        msg = (
            'ERROR: In frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(), input prefix_number_adjective_keyword_body_ext="'
            + str(prefix_number_adjective_keyword_body_ext)
            + '" has a third-to-last substring "'
            + str(number_str)
            + '" that is not a valid integer.'
        )
        print(msg)
        raise ValueError(msg)
    # Return the number string, without parsing.
    return number_str


def frame_id_str_given_key_corners_body_ext(key_corners_body_ext):
    # Our key corners FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_corners_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    #
    return frame_id_str_given_prefix_number_KeyWord_body_ext(key_corners_body_ext, "corners")


def frame_id_given_key_corners_body_ext(key_corners_body_ext):
    # Our key corners FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_corners_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    return frame_id_given_frame_id_str(frame_id_str_given_key_corners_body_ext(key_corners_body_ext))


def frame_id_str_given_key_projected_tracks_body_ext(key_track_body_ext):
    # Our frame track FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_track_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    #
    return frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(key_track_body_ext, "projected", "tracks")


def frame_id_given_key_projected_tracks_body_ext(key_track_body_ext):
    # Our key frame track FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_track_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    return frame_id_given_frame_id_str(frame_id_str_given_key_projected_tracks_body_ext(key_track_body_ext))


def frame_id_str_given_key_confirmed_tracks_body_ext(key_track_body_ext):
    # Our frame track FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_track_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    #
    return frame_id_str_given_prefix_number_Adjective_KeyWord_body_ext(key_track_body_ext, "confirmed", "tracks")


def frame_id_given_key_confirmed_tracks_body_ext(key_track_body_ext):
    # Our key frame track FrameNameXyList file names are of the form:
    #    VideoFileBody_nnnnnn_track_fxnl.csv
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    return frame_id_given_frame_id_str(frame_id_str_given_key_confirmed_tracks_body_ext(key_track_body_ext))


def frame_id_str_given_frame_file_body_ext(frame_file_body_ext):
    # Our extracted video file names are of the form:
    #    VideoFileBody.nnnnnn.JPG
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    #
    # Break the filename into components.
    components = frame_file_body_ext.split(".")
    if len(components) < 3:
        msg = (
            'ERROR: In frame_id_str_given_frame_file_body_ext(), input frame_file_body_ext="'
            + str(frame_file_body_ext)
            + '" did not separate into at least three components when split by "."'
        )
        print(msg)
        raise ValueError(msg)
    ext = components[-1]
    number_str = components[-2]
    # Determine if the "number" component is a valid integer.
    if not mt.string_is_integer(number_str):
        msg = (
            'ERROR: In frame_id_str_given_frame_file_body_ext(), input frame_file_body_ext="'
            + str(frame_file_body_ext)
            + '" has a second-to-last substring "'
            + str(number_str)
            + '" that is not a valid integer.'
        )
        print(msg)
        raise ValueError(msg)
    # Return the number string, without parsing.
    return number_str


def frame_id_given_frame_file_body_ext(frame_file_body_ext):
    # Our extracted video file names are of the form:
    #    VideoFileBody.nnnnnn.JPG
    # where the "nnnnnn" is a zero-padded numerical string such as "001258"
    return frame_id_given_frame_id_str(frame_id_str_given_frame_file_body_ext(frame_file_body_ext))


def draw_example_frame(
    input_full_frame_dir, input_frame_file, output_render_dir, render_control, include_figure_idx_in_filename=False
):
    # Load frame.
    input_dir_body_ext = os.path.join(
        input_full_frame_dir, input_frame_file
    )  # input_frame_file includes the extension.
    frame_img = cv.imread(input_dir_body_ext)
    ip.plot_image_figure(
        frame_img,
        rgb=False,
        title=input_frame_file,
        corners_color_pair_list=None,
        context_str="draw_example_frame()",
        save=True,
        output_dir=output_render_dir,
        output_body=input_frame_file + "_fig",
        dpi=render_control.example_frame_dpi,
        include_figure_idx_in_filename=include_figure_idx_in_filename,
    )


def draw_example_frames(
    input_full_frame_dir,  # Directory containing the full list of full-size frames.
    output_render_dir,  # Directory to place the sample rame figures.
    render_control,  # Render control, indicating whether to clear previous output, etc.
    delete_suffix=".png",  # File suffix characterizing previos files to delete.
    n_intervals=10,  # Number of intervals to divide the time line.  Must be at least 1.
    # Including first and last frame, (n+1) figures will be generated.
    include_figure_idx_in_filename=False,
):  # Whether to include the figure index in the output figure files.
    """
    Given a pointer to a directory containing a large number of full-size frames, selects a regular sampling
    of the frames and generates smaller figure versions, writing to the designated output directory.
    """
    # Fetch list of all frame filenames.
    input_frame_file_list = os.listdir(input_full_frame_dir)
    input_frame_file_list.sort()

    # Identify example frames.
    n_frames = len(input_frame_file_list)
    if n_frames > 0:
        # There are frames to draw from.
        first_idx = 0
        last_idx = n_frames - 1
        step = int(n_frames / n_intervals)
        n_intermediate = n_intervals - 1
        sample_idx_list = [first_idx]
        if n_intermediate > 0:
            for idx in range(1, (n_intermediate + 1)):
                sample_idx = idx * step
                sample_idx_list.append(sample_idx)
        sample_idx_list.append(last_idx)

        # Create sample frame figures.
        if len(sample_idx_list) > 0:
            # Prepare directory.
            upc.prepare_render_directory(output_render_dir, delete_suffix, render_control)

            # Draw example frames.
            for sample_idx in sample_idx_list:
                draw_example_frame(
                    input_full_frame_dir,
                    input_frame_file_list[sample_idx],
                    output_render_dir,
                    render_control,
                    include_figure_idx_in_filename=include_figure_idx_in_filename,
                )
