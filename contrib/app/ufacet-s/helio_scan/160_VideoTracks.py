"""
Converting individual key frame corner tracks into tracks for the whole video, incorporating all key frame track results.



"""

from cv2 import cv2 as cv
import os
import sys

import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import lib.FrameNameXyList as fnxl
import opencsp.common.lib.render_control.RenderControlVideoTracks as rcvt
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import lib.ufacet_pipeline_clear as upc
import lib.ufacet_pipeline_frame as upf
import opencsp.common.lib.render.video_manipulation as vm


class VideoTracks:
    """
    Class for converting individual key frame corner tracks into tracks for the whole video.

    """

    def __init__(
        self,
        # Execution control.
        force_construction,  # Recompute even if results already exist.
        specific_frame_ids,  # List of frame_ids to process, e.g. [777, 1126, ...].  For all frames, set None.
        include_all_frames,  # Whether to include all frames in the output FrameNameXyList and video, or just those with found corners.
        single_processor,  # Execute multi-processor steps with a single processor.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        # Input/output sources.
        input_video_dir_body_ext,  # Where to find the video file.
        input_key_projected_tracks_dir,  # Where to find files, one per key frame, with frame pair ids and associated [hel_name, projected_corners] pairs.
        input_key_confirmed_tracks_dir,  # Where to find files, one per key frame, with frame pair ids and associated [hel_name, confirmed_corners] pairs.
        input_frame_dir,  # Where to read full frames, for rendering.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting plots showing final video tracks.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        # Render control.
        render_control_projected,  # Flags to control rendering on this run, for the projected video.
        render_control_confirmed,
    ):  # Flags to control rendering on this run, for the confirmed video.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In VideoTracks.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In VideoTracks.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In VideoTracks.__init__(), null output_render_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        # Execution control.
        self.force_construction = force_construction
        self.specific_frame_ids = specific_frame_ids
        self.include_all_frames = include_all_frames
        self.single_processor = single_processor
        self.log_dir_body_ext = log_dir_body_ext
        # Input/output sources.
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_key_projected_tracks_dir = input_key_projected_tracks_dir
        self.input_key_confirmed_tracks_dir = input_key_confirmed_tracks_dir
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.render_control_projected = render_control_projected
        self.render_control_confirmed = render_control_confirmed

        # Found video tracks file names.
        # Projected.
        self.video_projected_tracks_body = self.input_video_body + "_video_projected_tracks_fnxl"
        self.video_projected_tracks_body_ext = self.video_projected_tracks_body + ".csv"
        self.video_projected_tracks_dir_body_ext = os.path.join(
            self.output_data_dir, self.video_projected_tracks_body_ext
        )
        # Confirmed.
        self.video_confirmed_tracks_body = self.input_video_body + "_video_confirmed_tracks_fnxl"
        self.video_confirmed_tracks_body_ext = self.video_confirmed_tracks_body + ".csv"
        self.video_confirmed_tracks_dir_body_ext = os.path.join(
            self.output_data_dir, self.video_confirmed_tracks_body_ext
        )

        # Output construction frame directories.
        self.output_construction_projected_dir = os.path.join(self.output_construction_dir, "projected")
        self.output_construction_confirmed_dir = os.path.join(self.output_construction_dir, "confirmed")

        # Output video file name.
        # Projected.
        self.output_video_projected_body = self.input_video_body + "_video_projected_tracks"
        self.output_video_projected_dir_body = os.path.join(self.output_render_dir, self.output_video_projected_body)
        # Confirmed.
        self.output_video_confirmed_body = self.input_video_body + "_video_confirmed_tracks"
        self.output_video_confirmed_dir_body = os.path.join(self.output_render_dir, self.output_video_confirmed_body)

        # Summary statistics file name.
        # Projected.
        self.dict_projected_body = self.input_video_body + "_video_projected_tracks_statistics"
        self.dict_projected_body_ext = self.dict_projected_body + ".csv"
        self.dict_projected_dir_body_ext = os.path.join(self.output_data_dir, self.dict_projected_body_ext)
        # Confirmed.
        self.dict_confirmed_body = self.input_video_body + "_video_confirmed_tracks_statistics"
        self.dict_confirmed_body_ext = self.dict_confirmed_body + ".csv"
        self.dict_confirmed_dir_body_ext = os.path.join(self.output_data_dir, self.dict_confirmed_body_ext)

        # Heliostats per video frame file name.
        # Projected.
        self.phpvf_body = self.input_video_body + "_projected_heliostats_per_video_frame"
        self.phpvf_body_ext = self.phpvf_body + ".csv"
        self.phpvf_dir_body_ext = os.path.join(self.output_data_dir, self.phpvf_body_ext)
        # Confirmed.
        self.chpvf_body = self.input_video_body + "_confirmed_heliostats_per_video_frame"
        self.chpvf_body_ext = self.chpvf_body + ".csv"
        self.chpvf_dir_body_ext = os.path.join(self.output_data_dir, self.chpvf_body_ext)

        # Points per video frame file name.
        # Projected.
        self.pppvf_body = self.input_video_body + "_projected_points_per_video_frame"
        self.pppvf_body_ext = self.pppvf_body + ".csv"
        self.pppvf_dir_body_ext = os.path.join(self.output_data_dir, self.pppvf_body_ext)
        # Confirmed.
        self.cppvf_body = self.input_video_body + "_confirmed_points_per_video_frame"
        self.cppvf_body_ext = self.cppvf_body + ".csv"
        self.cppvf_dir_body_ext = os.path.join(self.output_data_dir, self.cppvf_body_ext)

        # Load key frame track files.
        # Projected.
        print(
            "In VideoTracks.__init__(), reading found key frame projected tracks directory: ",
            self.input_key_projected_tracks_dir,
        )
        key_projected_tracks_body_ext_list = ft.files_in_directory(self.input_key_projected_tracks_dir)
        self.key_projected_tracks_dict = {}
        for key_projected_tracks_body_ext in key_projected_tracks_body_ext_list:
            key_frame_id_str = upf.frame_id_str_given_key_projected_tracks_body_ext(key_projected_tracks_body_ext)
            key_frame_id = upf.frame_id_given_frame_id_str(key_frame_id_str)
            if (self.specific_frame_ids is None) or (key_frame_id in self.specific_frame_ids):
                key_projected_tracks_dir_body_ext = os.path.join(
                    self.input_key_projected_tracks_dir, key_projected_tracks_body_ext
                )
                print(
                    "In KeyTracks.__init__(), reading key frame projected tracks file:     ",
                    key_projected_tracks_dir_body_ext,
                )
                key_projected_track_fnxl = fnxl.FrameNameXyList()
                key_projected_track_fnxl.load(key_projected_tracks_dir_body_ext)
                key_projected_track_fnxl_sorted_frame_id_list = (
                    key_projected_track_fnxl.sorted_frame_id_list()
                )  # Save repeat calls to this.
                # Store results.
                key_frame_dict = {}
                self.key_projected_tracks_dict[key_frame_id] = key_frame_dict
                self.key_projected_tracks_dict[key_frame_id]["key_frame_id_str"] = key_frame_id_str
                self.key_projected_tracks_dict[key_frame_id][
                    "key_projected_track_body_ext"
                ] = key_projected_tracks_body_ext
                self.key_projected_tracks_dict[key_frame_id][
                    "key_projected_track_dir_body_ext"
                ] = key_projected_tracks_dir_body_ext
                self.key_projected_tracks_dict[key_frame_id]["key_projected_track_fnxl"] = key_projected_track_fnxl
                self.key_projected_tracks_dict[key_frame_id]["key_projected_track_n_frames"] = len(
                    key_projected_track_fnxl_sorted_frame_id_list
                )
                self.key_projected_tracks_dict[key_frame_id]["key_projected_track_min_frame_id"] = (
                    key_projected_track_fnxl_sorted_frame_id_list[0]
                )
                self.key_projected_tracks_dict[key_frame_id]["key_projected_track_max_frame_id"] = (
                    key_projected_track_fnxl_sorted_frame_id_list[-1]
                )
        # Confirm what was read.
        print("In KeyTracks.__init__(), found key frame projected tracks dictionary:")
        dt.print_dict_of_dicts(self.key_projected_tracks_dict, max_value_2_length=200, indent_1=4)
        # Confirmed.
        print(
            "In VideoTracks.__init__(), reading found key frame confirmed tracks directory: ",
            self.input_key_confirmed_tracks_dir,
        )
        key_confirmed_tracks_body_ext_list = ft.files_in_directory(self.input_key_confirmed_tracks_dir)
        self.key_confirmed_tracks_dict = {}
        for key_confirmed_tracks_body_ext in key_confirmed_tracks_body_ext_list:
            key_frame_id_str = upf.frame_id_str_given_key_confirmed_tracks_body_ext(key_confirmed_tracks_body_ext)
            key_frame_id = upf.frame_id_given_frame_id_str(key_frame_id_str)
            if (self.specific_frame_ids is None) or (key_frame_id in self.specific_frame_ids):
                key_confirmed_tracks_dir_body_ext = os.path.join(
                    self.input_key_confirmed_tracks_dir, key_confirmed_tracks_body_ext
                )
                print(
                    "In KeyTracks.__init__(), reading key frame confirmed tracks file:     ",
                    key_confirmed_tracks_dir_body_ext,
                )
                key_confirmed_track_fnxl = fnxl.FrameNameXyList()
                key_confirmed_track_fnxl.load(key_confirmed_tracks_dir_body_ext)
                key_confirmed_track_fnxl_sorted_frame_id_list = (
                    key_confirmed_track_fnxl.sorted_frame_id_list()
                )  # Save repeat calls to this.
                # Store results.
                key_frame_dict = {}
                self.key_confirmed_tracks_dict[key_frame_id] = key_frame_dict
                self.key_confirmed_tracks_dict[key_frame_id]["key_frame_id_str"] = key_frame_id_str
                self.key_confirmed_tracks_dict[key_frame_id][
                    "key_confirmed_track_body_ext"
                ] = key_confirmed_tracks_body_ext
                self.key_confirmed_tracks_dict[key_frame_id][
                    "key_confirmed_track_dir_body_ext"
                ] = key_confirmed_tracks_dir_body_ext
                self.key_confirmed_tracks_dict[key_frame_id]["key_confirmed_track_fnxl"] = key_confirmed_track_fnxl
                self.key_confirmed_tracks_dict[key_frame_id]["key_confirmed_track_n_frames"] = len(
                    key_confirmed_track_fnxl_sorted_frame_id_list
                )
                self.key_confirmed_tracks_dict[key_frame_id]["key_confirmed_track_min_frame_id"] = (
                    key_confirmed_track_fnxl_sorted_frame_id_list[0]
                )
                self.key_confirmed_tracks_dict[key_frame_id]["key_confirmed_track_max_frame_id"] = (
                    key_confirmed_track_fnxl_sorted_frame_id_list[-1]
                )
        # Confirm what was read.
        print("In KeyTracks.__init__(), found key frame confirmed tracks dictionary:")
        dt.print_dict_of_dicts(self.key_confirmed_tracks_dict, max_value_2_length=200, indent_1=4)

        # Fetch a list of all frame ids in the video (not just key frames).
        # The corresponding frame_ids are not necessarily in sequential order, because
        # we previously removed spurious duplicate frames.
        self.all_frame_file_list = ft.files_in_directory(self.input_frame_dir, sort=True)
        # Confirm what was read.
        max_print_files = 12
        print("In VideoTracks.__init__(), self.all_frame_file_list:")
        for frame_file in self.all_frame_file_list[0 : min(max_print_files, len(self.all_frame_file_list))]:
            print("In VideoTracks.__init__()   ", frame_file)
        print("...")

        # Combine key frame tracks into a single video FrameNameXyList object, archiving the result.
        self.construct_and_save_video_tracks()

        # Load found tracks.
        self.read_video_tracks()

        # Load summary data.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT

    def construct_and_save_video_tracks(self):
        # Check if tracks have already been found.
        if (
            self.force_construction
            or (not ft.directory_exists(self.output_data_dir))
            or ft.directory_is_empty(self.output_data_dir)
        ):
            # We haven't generated yet.
            self.generated_video_projected_tracks = False
            self.generated_video_confirmed_tracks = False
            # Projected.
            self.construct_and_save_video_tracks_aux(self.key_projected_tracks_dict, "projected")
            self.generated_video_projected_tracks = True
            # Confirmed.
            self.construct_and_save_video_tracks_aux(self.key_confirmed_tracks_dict, "confirmed")
            self.generated_video_confirmed_tracks = True

    def construct_and_save_video_tracks_aux(self, key_tracks_dict, projected_or_confirmed_str):
        print(
            "In VideoTracks.construct_and_save_video_tracks_aux(), constructing "
            + projected_or_confirmed_str
            + " video tracks..."
        )

        # Construct an initial FrameNameXyList object for the video tracks.
        video_tracks_fnxl = fnxl.FrameNameXyList()

        # Fill the FrameNameXyList with all frame ids, so that when we render, we render the full video.
        if self.include_all_frames:
            for frame_file_body_ext in self.all_frame_file_list:
                frame_id = upf.frame_id_given_frame_file_body_ext(frame_file_body_ext)
                video_tracks_fnxl.add_list_of_name_xy_lists(frame_id, [])

        # Add key frame tracks.
        # (The selection of which key frames to consider has already been taken into account, by only including key frames of interest in the key_tracks_dict.)
        # Determine which key frames to process.
        key_frame_ids_to_process = dt.sorted_keys(key_tracks_dict)  # Already pruned to key frame ids of interest.
        for key_frame_id in key_frame_ids_to_process:
            # Fetch the FrameNameXyList object for holding the tracks for this key frame.
            key_frame_dict = key_tracks_dict[key_frame_id]
            key_track_fnxl = key_frame_dict["key_" + projected_or_confirmed_str + "_track_fnxl"]
            for (
                frame_id
            ) in key_track_fnxl.sorted_frame_id_list():  # Use sorted list so status output is easier to understand.
                video_tracks_fnxl.merge_list_of_name_xy_lists(
                    frame_id,
                    key_track_fnxl.list_of_name_xy_lists(frame_id),
                    warn_if_common_name=True,
                    skip_if_common_name=True,
                    error_if_common_name=False,
                )

        # Summarize construction result.
        print(
            "In VideoTracks.construct_and_save_video_tracks_aux(), constructed "
            + projected_or_confirmed_str
            + " video_tracks_fnxl:"
        )
        video_tracks_fnxl.print(indent=4)

        # Write video tracks file.
        self.save_video_tracks(video_tracks_fnxl, projected_or_confirmed_str)

        # Write summary information.
        self.save_data(video_tracks_fnxl, projected_or_confirmed_str)

    # WRITE RESULT

    def save_video_tracks(self, video_tracks_fnxl, projected_or_confirmed_str):
        # Filenames.
        # Write the full-video FrameNameXyList object.
        if projected_or_confirmed_str == "projected":
            video_tracks_dir_body_ext = self.video_projected_tracks_dir_body_ext
        elif projected_or_confirmed_str == "confirmed":
            video_tracks_dir_body_ext = self.video_confirmed_tracks_dir_body_ext
        else:
            msg = (
                'In VideoTracks.save_video_tracks(), encountered projected_or_confirmed_str="'
                + str(projected_or_confirmed_str)
                + '" which was neither "projected" or "confirmed".'
            )
            print("ERROR: " + msg)
            raise ValueError(msg)
        print("In VideoTracks.save_video_tracks(), writing video track file: ", video_tracks_dir_body_ext)
        ft.create_directories_if_necessary(self.output_data_dir)
        video_tracks_fnxl.save(video_tracks_dir_body_ext)

    def save_data(self, video_tracks_fnxl, projected_or_confirmed_str):
        # Filenames.
        if projected_or_confirmed_str == "projected":
            dict_body = self.dict_projected_body
            hpvf_body = self.phpvf_body
            hpvf_body_ext = self.phpvf_body_ext
            ppvf_body = self.pppvf_body
            ppvf_body_ext = self.pppvf_body_ext
        elif projected_or_confirmed_str == "confirmed":
            dict_body = self.dict_confirmed_body
            hpvf_body = self.chpvf_body
            hpvf_body_ext = self.chpvf_body_ext
            ppvf_body = self.cppvf_body
            ppvf_body_ext = self.cppvf_body_ext
        else:
            msg = (
                'In VideoTracks.save_data(), encountered projected_or_confirmed_str="'
                + str(projected_or_confirmed_str)
                + '" which was neither "projected" or "confirmed".'
            )
            print("ERROR: " + msg)
            raise ValueError(msg)
        # Statistics.
        summary_dict = {}
        summary_dict["n_video_track_frames"] = video_tracks_fnxl.number_of_frames()
        print(
            "In VideoTracks.save_data(), writing video frame " + projected_or_confirmed_str + " summary statistics..."
        )
        ft.write_dict_file(
            "video " + projected_or_confirmed_str + " tracks summary statistics",
            self.output_data_dir,
            dict_body,
            summary_dict,
        )
        # Heliostats per video frame.
        heliostats_per_video_frame_dict = video_tracks_fnxl.heliostats_per_frame()
        print(
            "In VideoTracks.save_data(), writing " + projected_or_confirmed_str + " heliostats per video frame:",
            os.path.join(self.output_data_dir, hpvf_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, hpvf_body, heliostats_per_video_frame_dict)
        # Points per video frame.
        points_per_key_frame_dict = video_tracks_fnxl.points_per_frame()
        print(
            "In VideoTracks.save_data(), writing points per video frame:    ",
            os.path.join(self.output_data_dir, ppvf_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, ppvf_body, points_per_key_frame_dict)

    # LOAD RESULT

    def read_video_tracks(self):
        # Projected.
        print(
            "In VideoTracks.read_video_tracks(), reading video projected tracks file: ",
            self.video_projected_tracks_dir_body_ext,
        )
        self.video_projected_tracks_fnxl = fnxl.FrameNameXyList()
        self.video_projected_tracks_fnxl.load(self.video_projected_tracks_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_video_tracks(), video projected tracks read:")
        self.video_projected_tracks_fnxl.print(max_keys=12, max_value_length=200, indent=4)
        # Confirmed.
        print(
            "In VideoTracks.read_video_tracks(), reading video confirmed tracks file: ",
            self.video_confirmed_tracks_dir_body_ext,
        )
        self.video_confirmed_tracks_fnxl = fnxl.FrameNameXyList()
        self.video_confirmed_tracks_fnxl.load(self.video_confirmed_tracks_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_video_tracks(), video confirmed tracks read:")
        self.video_confirmed_tracks_fnxl.print(max_keys=12, max_value_length=200, indent=4)

    def read_data(self):
        # Projected.
        # Statistics.
        print("In VideoTracks.read_data(), reading projected frame statistics: ", self.dict_projected_dir_body_ext)
        self.projected_frame_statistics_dict = ft.read_dict(self.dict_projected_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), projected frame statistics read:")
        dt.print_dict(self.projected_frame_statistics_dict, indent=4)
        # Heliostats per video frame.
        print("In VideoTracks.read_data(), reading projected heliostats per video frame: ", self.phpvf_dir_body_ext)
        self.phpvf_dict = ft.read_dict(self.phpvf_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), projected heliostats per video frame read:")
        dt.print_dict(self.phpvf_dict, max_keys=12, max_value_length=200, indent=4)
        # Points per video frame.
        print("In VideoTracks.read_data(), reading projected points per video frame: ", self.pppvf_dir_body_ext)
        self.pppvf_dict = ft.read_dict(self.pppvf_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), projected points per video frame read:")
        dt.print_dict(self.pppvf_dict, max_keys=7, max_value_length=200, indent=4)
        # Confirmed.
        # Statistics.
        print("In VideoTracks.read_data(), reading confirmed frame statistics: ", self.dict_confirmed_dir_body_ext)
        self.confirmed_frame_statistics_dict = ft.read_dict(self.dict_confirmed_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), confirmed frame statistics read:")
        dt.print_dict(self.confirmed_frame_statistics_dict, indent=4)
        # Heliostats per video frame.
        print("In VideoTracks.read_data(), reading confirmed heliostats per video frame: ", self.phpvf_dir_body_ext)
        self.phpvf_dict = ft.read_dict(self.phpvf_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), confirmed heliostats per video frame read:")
        dt.print_dict(self.phpvf_dict, max_keys=12, max_value_length=200, indent=4)
        # Points per video frame.
        print("In VideoTracks.read_data(), reading confirmed points per video frame: ", self.pppvf_dir_body_ext)
        self.pppvf_dict = ft.read_dict(self.pppvf_dir_body_ext)
        # Confirm what was read.
        print("In VideoTracks.read_data(), confirmed points per video frame read:")
        dt.print_dict(self.pppvf_dict, max_keys=7, max_value_length=200, indent=4)

    # RENDER RESULT

    def render(self):
        print(
            "In VideoTracks.render(), self.output_construction_projected_dir=", self.output_construction_projected_dir
        )  # ?? SCFFOLDING RCB -- TEMPORARY
        print(
            "In VideoTracks.render(), self.output_construction_confirmed_dir=", self.output_construction_confirmed_dir
        )  # ?? SCFFOLDING RCB -- TEMPORARY
        # Projected.
        if (
            self.render_control_projected.draw_video_tracks and self.generated_video_projected_tracks
        ):  # Don't render unless we generated.
            self.render_aux(
                self.video_projected_tracks_fnxl,
                self.output_construction_projected_dir,
                self.output_video_projected_dir_body,
                self.render_control_projected,
                "projected",
            )
        # Confirmed.
        if (
            self.render_control_confirmed.draw_video_tracks and self.generated_video_confirmed_tracks
        ):  # Don't render unless we generated.
            self.render_aux(
                self.video_confirmed_tracks_fnxl,
                self.output_construction_confirmed_dir,
                self.output_video_confirmed_dir_body,
                self.render_control_confirmed,
                "confirmed",
            )

    def render_aux(
        self,
        video_tracks_fnxl,
        output_construction_dir,
        output_video_dir_body,
        render_control,
        projected_or_confirmed_str,
    ):
        print("In VideoTracks.render_aux(), rendering video " + projected_or_confirmed_str + " tracks...")
        print(
            "In VideoTracks.render_aux(), output_construction_dir=", output_construction_dir
        )  # ?? SCFFOLDING RCB -- TEMPORARY
        # Descriptive strings.
        title_name = projected_or_confirmed_str.capitalize() + " Corners"
        context_str = "VideoTracks.render_aux()"
        # Required suffix strings.
        fig_suffix = "_video_" + projected_or_confirmed_str + "_tracks_fig"
        delete_suffix = ".JPG" + fig_suffix + ".png"
        # Prepare directory for frames.
        upc.prepare_render_directory(output_construction_dir, delete_suffix, render_control)
        # Setup annotation styles.
        style_dict = {}
        style_dict["point_seq"] = rcps.marker(
            marker=render_control.video_tracks_points_marker,
            markersize=render_control.video_tracks_points_markersize,
            color=render_control.video_tracks_points_color,
        )
        style_dict["text"] = rctxt.RenderControlText(
            horizontalalignment=render_control.video_tracks_label_horizontalalignment,
            verticalalignment=render_control.video_tracks_label_verticalalignment,
            fontsize=render_control.video_tracks_label_fontsize,
            fontstyle=render_control.video_tracks_label_fontstyle,
            fontweight=render_control.video_tracks_label_fontweight,
            color=render_control.video_tracks_label_color,
        )
        # Draw the frames.
        video_tracks_fnxl.draw_frames(
            self.single_processor,
            self.log_dir_body_ext,
            title_name,
            context_str,
            fig_suffix,
            self.input_video_body,
            self.input_frame_dir,
            self.input_frame_id_format,
            output_construction_dir,
            dpi=render_control.video_tracks_dpi,
            close_xy_list=True,
            style_dict=style_dict,
            crop=render_control.video_tracks_crop,
        )

        print("In VideoTracks.render_aux() for " + projected_or_confirmed_str + " tracks, draw_frames() has returned.")

        # Prepare directory for video.
        ft.create_directories_if_necessary(self.output_render_dir)
        # Construct the video.
        print("In VideoTracks.render_aux(), constructing video of " + projected_or_confirmed_str + " tracks...")
        vm.construct_video(output_construction_dir, output_video_dir_body)
        print("In VideoTracks.render_aux(), " + projected_or_confirmed_str + " tracks video construction finished.")
        print()

        # Check to see if all the frames have been constructed.
        all_written_frame_file_list = ft.files_in_directory(self.output_construction_dir, sort=True)
        # Confirm what was read.
        max_print_files = 12
        print("In VideoTracks.render_aux(), self.all_written_frame_file_list:")
        for frame_file in all_written_frame_file_list[0 : min(max_print_files, len(all_written_frame_file_list))]:
            print("In VideoTracks.render_aux()   ", frame_file)
        print("...")
        print("In VideoTracks.render_aux(), len(self.all_frame_file_list) =", len(self.all_frame_file_list))
        print("In VideoTracks.render(), len(all_written_frame_file_list)  =", len(all_written_frame_file_list))
        print()


if __name__ == "__main__":
    # Execution control.
    force_construction = False
    # specific_frame_ids           = [3896,4569,5417,6076]  # Heliostats 11W6, 11W7, and 11W8 are seen in multiple passes.
    specific_frame_ids = None
    include_all_frames = True  # False #True
    single_processor = True  # False
    # log_dir_body_ext               = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_160_VideoTracks/mavic_zoom/log/VideoTracks_log.txt'
    # # Input/output sources.
    # input_video_dir_body_ext       = experiment_dir() + '2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4'
    # input_key_projected_tracks_dir = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_150_KeyTracks/mavic_zoom/data/key_frame_projected_tracks/'
    # input_key_confirmed_tracks_dir = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_150_KeyTracks/mavic_zoom/data/key_frame_confirmed_tracks/'
    # input_frame_dir                = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/'
    # input_frame_id_format          = '06d' # Note different from format used in ffmpeg call, which is '.%06d'
    # output_data_dir                = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_160_VideoTracks/mavic_zoom/data/'
    # output_render_dir              = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_160_VideoTracks/mavic_zoom/render/'
    # output_construction_dir        = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/Small_160c_VideoTracks/mavic_zoom/'
    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/160_VideoTracks/mavic_zoom/log/VideoTracks_log.txt"
    )
    # Input/output sources.
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_key_projected_tracks_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/150_KeyTracks/mavic_zoom/data/key_frame_projected_tracks/"
    )
    input_key_confirmed_tracks_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/150_KeyTracks/mavic_zoom/data/key_frame_confirmed_tracks/"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/160_VideoTracks/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/160_VideoTracks/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/160c_VideoTracks/mavic_zoom/"
    )
    # Render control.
    # render_control_projected     = rcvt.default(color='m')
    # render_control_confirmed     = rcvt.default(color='c')
    render_control_projected = rcvt.fast()  # Don't draw frames.
    render_control_confirmed = rcvt.fast()  # Don't draw frames.

    key_frames_object = VideoTracks(  # Execution control.
        force_construction,
        specific_frame_ids,
        include_all_frames,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        input_video_dir_body_ext,
        input_key_projected_tracks_dir,
        input_key_confirmed_tracks_dir,
        input_frame_dir,
        input_frame_id_format,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        # Render control.
        render_control_projected,
        render_control_confirmed,
    )
