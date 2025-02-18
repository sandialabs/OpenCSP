"""
Converting full video corner tracks into tracks per heliostat, removing camera distortion.



"""

from cv2 import cv2 as cv
import os

import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import lib.FrameNameXyList as fnxl
import lib.NameFrameXyList as nfxl
import opencsp.common.lib.render_control.RenderControlHeliostatTracks as rcht
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import lib.ufacet_pipeline_clear as upc
import lib.ufacet_pipeline_frame as upf
import opencsp.common.lib.render.video_manipulation as vm


class HeliostatTracks:
    """
    Class for converting full video corner tracks into tracks per heliostat, without camera distortion.

    """

    def __init__(
        self,
        # Execution control.
        force_construction,  # Recompute even if results already exist.
        specific_frame_ids,  # List of frame_ids to process, e.g. [777, 1126, ...].  For all frames, set None.
        single_processor,  # Execute multi-processor steps with a single processor.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        # Input/output sources.
        input_video_dir_body_ext,  # Where to find the video file.
        input_video_projected_tracks_dir_body_ext,  # Where read to the full video projected tracks file, encoding a FrameNameXyList object.
        input_video_confirmed_tracks_dir_body_ext,  # Where read to the full video confirmed tracks file, encoding a FrameNameXyList object.
        input_frame_dir,  # Where to read full frames, for rendering.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting plots showing final heliostat tracks.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        # Render control.
        render_control_projected,  # Flags to control rendering on this run, for the projected data.
        render_control_confirmed,
    ):  # Flags to control rendering on this run, for the confirmed data.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In HeliostatTracks.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In HeliostatTracks.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In HeliostatTracks.__init__(), null output_render_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        # Execution control.
        self.force_construction = force_construction
        self.specific_frame_ids = specific_frame_ids
        self.single_processor = single_processor
        self.log_dir_body_ext = log_dir_body_ext
        # Input/output sources.
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_video_projected_tracks_dir_body_ext = input_video_projected_tracks_dir_body_ext
        self.input_video_confirmed_tracks_dir_body_ext = input_video_confirmed_tracks_dir_body_ext
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.render_control_projected = render_control_projected
        self.render_control_confirmed = render_control_confirmed

        # Found heliostat tracks file name.
        # Projected.
        self.heliostat_projected_tracks_body = self.input_video_body + "_heliostat_projected_tracks_nfxl"
        self.heliostat_projected_tracks_body_ext = self.heliostat_projected_tracks_body + ".csv"
        self.heliostat_projected_tracks_dir_body_ext = os.path.join(
            self.output_data_dir, self.heliostat_projected_tracks_body_ext
        )
        # Confirmed.
        self.heliostat_confirmed_tracks_body = self.input_video_body + "_heliostat_confirmed_tracks_nfxl"
        self.heliostat_confirmed_tracks_body_ext = self.heliostat_confirmed_tracks_body + ".csv"
        self.heliostat_confirmed_tracks_dir_body_ext = os.path.join(
            self.output_data_dir, self.heliostat_confirmed_tracks_body_ext
        )

        # Output directories for heliostat corner tracks files.
        self.output_heliostat_projected_corner_tracks_dir = os.path.join(
            self.output_data_dir, "projected_corner_tracks"
        )
        self.output_heliostat_confirmed_corner_tracks_dir = os.path.join(
            self.output_data_dir, "confirmed_corner_tracks"
        )
        self.output_heliostat_undistorted_projected_corner_tracks_dir = os.path.join(
            self.output_data_dir, "undistorted_projected_corner_tracks"
        )
        self.output_heliostat_undistorted_confirmed_corner_tracks_dir = os.path.join(
            self.output_data_dir, "undistorted_confirmed_corner_tracks"
        )

        # Output construction frame directories.
        self.output_construction_projected_dir = os.path.join(self.output_construction_dir, "projected")
        self.output_construction_confirmed_dir = os.path.join(self.output_construction_dir, "confirmed")

        # Summary statistics file name.
        # Projected.
        self.dict_projected_body = self.input_video_body + "_heliostat_projected_tracks_statistics"
        self.dict_projected_body_ext = self.dict_projected_body + ".csv"
        self.dict_projected_dir_body_ext = os.path.join(self.output_data_dir, self.dict_projected_body_ext)
        # Conifrmed.
        self.dict_confirmed_body = self.input_video_body + "_heliostat_confirmed_tracks_statistics"
        self.dict_confirmed_body_ext = self.dict_confirmed_body + ".csv"
        self.dict_confirmed_dir_body_ext = os.path.join(self.output_data_dir, self.dict_confirmed_body_ext)

        # Video frames per heliostat dictionary name.
        # Projected.
        self.vfpph_body = self.input_video_body + "_video_projected_frames_per_heliostat"
        self.vfpph_body_ext = self.vfpph_body + ".csv"
        self.vfpph_dir_body_ext = os.path.join(self.output_data_dir, self.vfpph_body_ext)
        # Confirmed.
        self.vfpch_body = self.input_video_body + "_video_confirmed_frames_per_heliostat"
        self.vfpch_body_ext = self.vfpch_body + ".csv"
        self.vfpch_dir_body_ext = os.path.join(self.output_data_dir, self.vfpch_body_ext)

        # Load video tracks files.
        # Projected.
        print(
            "In HeliostatTracks.__init__(), reading projected video tracks file: ",
            self.input_video_projected_tracks_dir_body_ext,
        )
        self.video_projected_tracks_fnxl = fnxl.FrameNameXyList()
        self.video_projected_tracks_fnxl.load(self.input_video_projected_tracks_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.__init__(), projected video tracks read:")
        self.video_projected_tracks_fnxl.print(max_keys=12, max_value_length=200, indent=4)
        # Confirmed.
        print(
            "In HeliostatTracks.__init__(), reading confirmed video tracks file: ",
            self.input_video_confirmed_tracks_dir_body_ext,
        )
        self.video_confirmed_tracks_fnxl = fnxl.FrameNameXyList()
        self.video_confirmed_tracks_fnxl.load(self.input_video_confirmed_tracks_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.__init__(), confirmed video tracks read:")
        self.video_confirmed_tracks_fnxl.print(max_keys=12, max_value_length=200, indent=4)

        # Fetch a list of all frame ids in the video (not just key frames).
        # The corresponding frame_ids are not necessarily in sequential order, because
        # we previously removed spurious duplicate frames.
        self.all_frame_file_list = ft.files_in_directory(self.input_frame_dir, sort=True)
        # Confirm what was read.
        max_print_files = 12
        print("In HeliostatTracks.__init__(), self.all_frame_file_list:")
        for frame_file in self.all_frame_file_list[0 : min(max_print_files, len(self.all_frame_file_list))]:
            print("In HeliostatTracks.__init__()   ", frame_file)
        print("...")

        # Convert each full video tracks FrameNameXyList organized by frame into a NameFrameXyList object,
        # organized by heliostat name, archiving the result.
        self.construct_and_save_heliostat_tracks()

        # Load found tracks.
        self.read_heliostat_tracks()

        # Load summary data.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT

    def construct_and_save_heliostat_tracks(self):
        # Check if tracks have already been found.
        if (
            self.force_construction
            or (not ft.directory_exists(self.output_data_dir))
            or ft.directory_is_empty(self.output_data_dir)
        ):
            # We haven't generated yet.
            self.generated_heliostat_projected_tracks = False
            self.generated_heliostat_confirmed_tracks = False
            # Projected.
            self.construct_and_save_heliostat_tracks_aux(self.video_projected_tracks_fnxl, "projected")
            self.generated_heliostat_projected_tracks = True
            # Confirmed.
            self.construct_and_save_heliostat_tracks_aux(self.video_confirmed_tracks_fnxl, "confirmed")
            self.generated_heliostat_confirmed_tracks = True

    def construct_and_save_heliostat_tracks_aux(self, video_tracks_fnxl, projected_or_confirmed_str):
        print(
            "In HeliostatTracks.construct_and_save_heliostat_tracks_aux(), constructing "
            + projected_or_confirmed_str
            + " heliostat tracks..."
        )

        # Construct an initial NameFrameXyList object for the heliostat tracks.
        heliostat_tracks_nfxl = nfxl.NameFrameXyList()

        # Add data from the video FrameNameXyList object to the new NameFrameXyList object.
        heliostat_tracks_nfxl.add_FrameNameXyList(video_tracks_fnxl)

        # Summarize construction result.
        print("In HeliostatTracks.construct_and_save_heliostat_tracks_aux(), constructed heliostat_tracks_nfxl:")
        heliostat_tracks_nfxl.print(indent=4)

        # Write heliostat tracks file.
        self.save_heliostat_tracks(heliostat_tracks_nfxl, projected_or_confirmed_str)

        # Write summary information.
        self.save_data(heliostat_tracks_nfxl, projected_or_confirmed_str)

        # Record that we generated the heliostat tracks.
        self.generated_heliostat_tracks = True

    # WRITE RESULT

    def save_heliostat_tracks(self, heliostat_tracks_nfxl, projected_or_confirmed_str):
        # Filenames.
        # Write the NameFrameXyList object.
        if projected_or_confirmed_str == "projected":
            heliostat_tracks_dir_body_ext = self.heliostat_projected_tracks_dir_body_ext
        elif projected_or_confirmed_str == "confirmed":
            heliostat_tracks_dir_body_ext = self.heliostat_confirmed_tracks_dir_body_ext
        else:
            msg = (
                'In HeliostatTracks.save_heliostat_tracks(), encountered projected_or_confirmed_str="'
                + str(projected_or_confirmed_str)
                + '" which was neither "projected" or "confirmed".'
            )
            print("ERROR: " + msg)
            raise ValueError(msg)
        print(
            "In HeliostatTracks.save_heliostat_tracks(), writing heliostat track file: ", heliostat_tracks_dir_body_ext
        )
        ft.create_directories_if_necessary(self.output_data_dir)
        heliostat_tracks_nfxl.save(heliostat_tracks_dir_body_ext)
        # Corner tracks for 3-d reconstruction.

    def save_data(self, heliostat_tracks_nfxl, projected_or_confirmed_str):
        # Filenames.
        if projected_or_confirmed_str == "projected":
            dict_body = self.dict_projected_body
            vfph_body = self.vfpph_body
            vfph_body_ext = self.vfpph_body_ext
        elif projected_or_confirmed_str == "confirmed":
            dict_body = self.dict_confirmed_body
            vfph_body = self.vfpch_body
            vfph_body_ext = self.vfpch_body_ext
        else:
            msg = (
                'In HeliostatTracks.save_data(), encountered projected_or_confirmed_str="'
                + str(projected_or_confirmed_str)
                + '" which was neither "projected" or "confirmed".'
            )
            print("ERROR: " + msg)
            raise ValueError(msg)
        # Statistics.
        summary_dict = {}
        summary_dict["n_heliostat_track_frames"] = heliostat_tracks_nfxl.number_of_frames()
        print(
            "In HeliostatTracks.save_data(), writing key frame " + projected_or_confirmed_str + " summary statistics..."
        )
        ft.write_dict_file(
            "heliostat " + projected_or_confirmed_str + " tracks summary statistics",
            self.output_data_dir,
            dict_body,
            summary_dict,
        )
        # Video frames per heliostat.
        video_frames_per_heliostat_dict = heliostat_tracks_nfxl.frames_per_heliostat()
        print(
            "In HeliostatTracks.save_data(), writing video frames per " + projected_or_confirmed_str + " heliostat:",
            os.path.join(self.output_data_dir, vfph_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, vfph_body, video_frames_per_heliostat_dict)

    # LOAD RESULT

    def read_heliostat_tracks(self):
        # Projected.
        print(
            "In HeliostatTracks.read_heliostat_tracks(), reading heliostat projected tracks file: ",
            self.heliostat_projected_tracks_dir_body_ext,
        )
        self.heliostat_projected_tracks_nfxl = nfxl.NameFrameXyList()
        self.heliostat_projected_tracks_nfxl.load(self.heliostat_projected_tracks_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_heliostat_tracks(), heliostat projected tracks read:")
        self.heliostat_projected_tracks_nfxl.print(max_keys=12, max_value_length=200, indent=4)
        # Confirmed.
        print(
            "In HeliostatTracks.read_heliostat_tracks(), reading heliostat confirmed tracks file: ",
            self.heliostat_confirmed_tracks_dir_body_ext,
        )
        self.heliostat_confirmed_tracks_nfxl = nfxl.NameFrameXyList()
        self.heliostat_confirmed_tracks_nfxl.load(self.heliostat_confirmed_tracks_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_heliostat_tracks(), heliostat confirmed tracks read:")
        self.heliostat_confirmed_tracks_nfxl.print(max_keys=12, max_value_length=200, indent=4)

    def read_data(self):
        # Projected.
        # Statistics.
        print("In HeliostatTracks.read_data(), reading projected frame statistics: ", self.dict_projected_dir_body_ext)
        self.projected_frame_statistics_dict = ft.read_dict(self.dict_projected_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_data(), projected frame statistics read:")
        dt.print_dict(self.projected_frame_statistics_dict, indent=4)
        # Heliostats per video frame.
        print("In HeliostatTracks.read_data(), reading projected heliostats per video frame: ", self.vfpph_dir_body_ext)
        self.vfpph_dict = ft.read_dict(self.vfpph_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_data(), projected heliostats per video frame read:")
        dt.print_dict(self.vfpph_dict, max_keys=12, max_value_length=200, indent=4)
        # Confirmed.
        # Statistics.
        print("In HeliostatTracks.read_data(), reading confirmed frame statistics: ", self.dict_confirmed_dir_body_ext)
        self.confirmed_frame_statistics_dict = ft.read_dict(self.dict_confirmed_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_data(), confirmed frame statistics read:")
        dt.print_dict(self.confirmed_frame_statistics_dict, indent=4)
        # Heliostats per video frame.
        print("In HeliostatTracks.read_data(), reading confirmed heliostats per video frame: ", self.vfpph_dir_body_ext)
        self.vfpph_dict = ft.read_dict(self.vfpph_dir_body_ext)
        # Confirm what was read.
        print("In HeliostatTracks.read_data(), confirmed heliostats per video frame read:")
        dt.print_dict(self.vfpph_dict, max_keys=12, max_value_length=200, indent=4)

    # RENDER RESULT

    def render(self):
        print(
            "In HeliostatTracks.render(), self.output_construction_projected_dir=",
            self.output_construction_projected_dir,
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
        print(
            "In HeliostatTracks.render(), self.output_construction_confirmed_dir=",
            self.output_construction_confirmed_dir,
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
        # Projected.
        if (
            self.render_control_projected.draw_heliostat_tracks and self.generated_heliostat_projected_tracks
        ):  # Don't render unless we generated.
            self.render_aux(
                self.heliostat_projected_tracks_nfxl,
                self.output_construction_projected_dir,
                self.render_control_projected,
                "projected",
            )
        # Confirmed.
        if (
            self.render_control_confirmed.draw_heliostat_tracks and self.generated_heliostat_confirmed_tracks
        ):  # Don't render unless we generated.
            self.render_aux(
                self.heliostat_confirmed_tracks_nfxl,
                self.output_construction_confirmed_dir,
                self.render_control_confirmed,
                "confirmed",
            )

    def render_aux(self, heliostat_tracks_nfxl, output_construction_dir, render_control, projected_or_confirmed_str):
        print("In HeliostatTracks.render_aux(), rendering heliostat " + projected_or_confirmed_str + " tracks...")
        print(
            "In HeliostatTracks.render_aux(), output_construction_dir=", output_construction_dir
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
        print("WARNING: In HeliostatTracks.render_aux(), not implemented yet.")


if __name__ == "__main__":
    # Execution control.
    force_construction = True  # False
    # specific_frame_ids                 = [3896,4569,5417,6076]  # Heliostats 11W6, 11W7, and 11W8 are seen in multiple passes.
    specific_frame_ids = None
    single_processor = True  # False

    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/log/HeliostatTracks_log.txt"
    )
    # Input/output sources.
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_video_projected_tracks_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_160_VideoTracks/mavic_zoom/data/DJI_427t_428_429_video_projected_tracks_fnxl.csv"
    )
    input_video_confirmed_tracks_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_160_VideoTracks/mavic_zoom/data/DJI_427t_428_429_video_confirmed_tracks_fnxl.csv"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    output_data_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/Small_170c_HeliostatTracks/mavic_zoom/"
    )

    # log_dir_body_ext                          = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/170_HeliostatTracks/mavic_zoom/log/HeliostatTracks_log.txt'
    # # Input/output sources.
    # input_video_dir_body_ext                  = experiment_dir() + '2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4'
    # input_video_projected_tracks_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/160_VideoTracks/mavic_zoom/data/DJI_427t_428_429_video_projected_tracks_fnxl.csv'
    # input_video_confirmed_tracks_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/160_VideoTracks/mavic_zoom/data/DJI_427t_428_429_video_confirmed_tracks_fnxl.csv'
    # input_frame_dir                           = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/'
    # input_frame_id_format                     = '06d' # Note different from format used in ffmpeg call, which is '.%06d'
    # output_data_dir                           = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/170_HeliostatTracks/mavic_zoom/data/'
    # output_render_dir                         = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/170_HeliostatTracks/mavic_zoom/render/'
    # output_construction_dir                   = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/170c_HeliostatTracks/mavic_zoom/'

    # Render control.
    render_control_projected = rcht.default(color="m")
    render_control_confirmed = rcht.default(color="c")
    # render_control_projected               = rcht.fast()  # Don't draw frames.
    # render_control_confirmed               = rcht.fast()  # Don't draw frames.

    key_frames_object = HeliostatTracks(  # Execution control.
        force_construction,
        specific_frame_ids,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        input_video_dir_body_ext,
        input_video_projected_tracks_dir_body_ext,
        input_video_confirmed_tracks_dir_body_ext,
        input_frame_dir,
        input_frame_id_format,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        # Render control.
        render_control_projected,
        render_control_confirmed,
    )
