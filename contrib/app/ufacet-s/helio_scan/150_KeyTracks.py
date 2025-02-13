"""
Tracking heliostat facet corners previously found in key frames of a UFACET scan video.



"""

import logging
from multiprocessing import Pool
import os
import sys

import lib.DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY

import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import lib.FrameNameXyList as fnxl
import lib.KeyFrameTrackSearch as kfts
import opencsp.common.lib.tool.log_tools as logt
import opencsp.common.lib.render_control.RenderControlKeyTracks as rckt
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import lib.ufacet_pipeline_clear as upc
import lib.ufacet_pipeline_frame as upf


class KeyTracks:
    """
    Class controlling facet corner tracking given corners found in key frames of a UFACET scan video.

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
        input_key_projected_corners_dir,  # Where to find files, one per key frame, with frame pair ids and associated [hel_name, projected_corners] pairs.
        input_key_confirmed_corners_dir,  # Where to find files, one per key frame, with frame pair ids and associated [hel_name, confirmed_corners] pairs.
        input_frame_dir,  # Where to read full frames, for rendering.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting plots showing final found corners.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        # Render control.
        render_control,
    ):  # Flags to control rendering on this run.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In KeyTracks.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In KeyTracks.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In KeyTracks.__init__(), null output_render_dir encountered.")

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
        self.input_key_projected_corners_dir = input_key_projected_corners_dir
        self.input_key_confirmed_corners_dir = input_key_confirmed_corners_dir
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.render_control = render_control

        # Found key tracks file names.
        self.key_frame_projected_tracks_dir = os.path.join(self.output_data_dir, "key_frame_projected_tracks")
        self.key_frame_confirmed_tracks_dir = os.path.join(self.output_data_dir, "key_frame_confirmed_tracks")

        # Summary statistics file name.
        self.dict_body = self.input_video_body + "_key_frames_with_corners_statistics"
        self.dict_body_ext = self.dict_body + ".csv"
        self.dict_dir_body_ext = os.path.join(self.output_data_dir, self.dict_body_ext)

        # File listing key frames with mismatched heliostats.
        self.mismatched_ids_body = self.input_video_body + "_mismatched_key_frame_ids"
        self.mismatched_ids_body_ext = self.mismatched_ids_body + ".txt"
        self.mismatched_ids_dir_body_ext = os.path.join(self.output_data_dir, self.mismatched_ids_body_ext)

        # Tracked frames per key frame file name.
        self.tfpkf_body = self.input_video_body + "_tracked_frames_per_key_frame"
        self.tfpkf_body_ext = self.tfpkf_body + ".csv"
        self.tfpkf_dir_body_ext = os.path.join(self.output_data_dir, self.tfpkf_body_ext)

        # Load key corners files.
        # Projected.
        print(
            "In KeyTracks.__init__(), reading found key projected corners directory: ",
            self.input_key_projected_corners_dir,
        )
        key_projected_corners_body_ext_list = ft.files_in_directory(self.input_key_projected_corners_dir)
        self.key_projected_corners_dict = {}
        for key_projected_corners_body_ext in key_projected_corners_body_ext_list:
            key_frame_id_str = upf.frame_id_str_given_key_corners_body_ext(key_projected_corners_body_ext)
            key_frame_id = upf.frame_id_given_frame_id_str(key_frame_id_str)
            if (self.specific_frame_ids is None) or (key_frame_id in self.specific_frame_ids):
                key_corners_dir_body_ext = os.path.join(
                    self.input_key_projected_corners_dir, key_projected_corners_body_ext
                )
                print(
                    "In KeyTracks.__init__(), reading found projected key corners file:      ", key_corners_dir_body_ext
                )
                key_frame_projected_corners_fnxl = fnxl.FrameNameXyList()
                key_frame_projected_corners_fnxl.load(key_corners_dir_body_ext)
                # Store results.
                key_frame_dict = {}
                self.key_projected_corners_dict[key_frame_id] = key_frame_dict
                self.key_projected_corners_dict[key_frame_id]["key_frame_id_str"] = key_frame_id_str
                self.key_projected_corners_dict[key_frame_id][
                    "key_projected_corners_body_ext"
                ] = key_projected_corners_body_ext
                self.key_projected_corners_dict[key_frame_id]["key_corners_dir_body_ext"] = key_corners_dir_body_ext
                self.key_projected_corners_dict[key_frame_id][
                    "key_frame_projected_corners_fnxl"
                ] = key_frame_projected_corners_fnxl
        # Confirm what was read.
        print("In KeyTracks.__init__(), found projected key corners dictionary:")
        dt.print_dict_of_dicts(self.key_projected_corners_dict, max_value_2_length=200)
        # Confirmed.
        print(
            "In KeyTracks.__init__(), reading found key confirmed corners directory: ",
            self.input_key_confirmed_corners_dir,
        )
        key_confirmed_corners_body_ext_list = ft.files_in_directory(self.input_key_confirmed_corners_dir)
        self.key_confirmed_corners_dict = {}
        for key_confirmed_corners_body_ext in key_confirmed_corners_body_ext_list:
            key_frame_id_str = upf.frame_id_str_given_key_corners_body_ext(key_confirmed_corners_body_ext)
            key_frame_id = upf.frame_id_given_frame_id_str(key_frame_id_str)
            if (self.specific_frame_ids is None) or (key_frame_id in self.specific_frame_ids):
                key_corners_dir_body_ext = os.path.join(
                    self.input_key_confirmed_corners_dir, key_confirmed_corners_body_ext
                )
                print(
                    "In KeyTracks.__init__(), reading found confirmed key corners file:      ", key_corners_dir_body_ext
                )
                key_frame_confirmed_corners_fnxl = fnxl.FrameNameXyList()
                key_frame_confirmed_corners_fnxl.load(key_corners_dir_body_ext)
                # Store results.
                key_frame_dict = {}
                self.key_confirmed_corners_dict[key_frame_id] = key_frame_dict
                self.key_confirmed_corners_dict[key_frame_id]["key_frame_id_str"] = key_frame_id_str
                self.key_confirmed_corners_dict[key_frame_id][
                    "key_confirmed_corners_body_ext"
                ] = key_confirmed_corners_body_ext
                self.key_confirmed_corners_dict[key_frame_id]["key_corners_dir_body_ext"] = key_corners_dir_body_ext
                self.key_confirmed_corners_dict[key_frame_id][
                    "key_frame_confirmed_corners_fnxl"
                ] = key_frame_confirmed_corners_fnxl
        # Confirm what was read.
        print("In KeyTracks.__init__(), found confirmed key corners dictionary:")
        dt.print_dict_of_dicts(self.key_confirmed_corners_dict, max_value_2_length=200)

        # Fetch a list of all frame ids in the video (not just key frames).
        # The corresponding frame_ids are not necessarily in sequential order, because
        # we previously removed spurious duplicate frames.
        self.all_frame_body_ext_list = ft.files_in_directory(self.input_frame_dir, sort=True)
        # Confirm what was read.
        max_print_files = 12
        print("In KeyTracks.__init__(), self.all_frame_body_ext_list:")
        for frame_file in self.all_frame_body_ext_list[0 : min(max_print_files, len(self.all_frame_body_ext_list))]:
            print("In KeyTracks.__init__()   ", frame_file)
        print("...")

        # Find facet tracks in key frames, archiving the result.
        self.find_and_save_key_tracks()

        # Load found tracks.
        self.read_key_tracks()

        # Load summary data.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT

    def find_and_save_key_tracks(self):
        # Check if tracks have already been found.
        if (
            self.force_construction
            or (not ft.directory_exists(self.output_data_dir))
            or ft.directory_is_empty(self.output_data_dir)
        ):
            print("In KeyTracks.find_and_save_key_tracks(), constructing key frame tracks...")

            # Determine which key frames to process.
            key_frame_ids_to_process = dt.sorted_keys(
                self.key_projected_corners_dict
            )  # Already pruned to key frame ids of interest.

            # Process each key frame_id.
            if self.single_processor:
                print("In KeyTracks.search_key_tracks(), starting key frame corner tracking (single processor)...")
                list_of_result_dicts = []
                for key_frame_id in key_frame_ids_to_process:
                    list_of_result_dicts.append(self.search_key_track(key_frame_id))
            else:
                print("In KeyTracks.search_key_tracks(), starting key frame corner tracking (multi-processor)...")
                logger = logt.multiprocessing_logger(self.log_dir_body_ext, level=logging.INFO)
                logger.info("================================= Execution =================================")
                with Pool(25) as pool:
                    list_of_result_dicts = pool.map(self.search_key_track, key_frame_ids_to_process)

            print(
                "In KeyTracks.search_key_tracks(), key frame corner tracking done.  len(list_of_result_dicts) =",
                len(list_of_result_dicts),
            )

            # Summarize search results.
            print("In KeyTracks.find_and_save_key_tracks(), key_frame_projected_track_fnxls:")
            for result_dict in list_of_result_dicts:
                key_frame_id = result_dict["key_frame_id"]
                key_frame_projected_track_fnxl = result_dict["key_frame_projected_track_fnxl"]
                print(
                    "    " + str(key_frame_id) + ":"
                )  # Using "str(key_frame_id)" is okay, because we don't want leading zeros.
                key_frame_projected_track_fnxl.print(max_value_length=200, indent=8)
            print("In KeyTracks.find_and_save_key_tracks(), key_frame_confirmed_track_fnxls:")
            for result_dict in list_of_result_dicts:
                key_frame_id = result_dict["key_frame_id"]
                key_frame_confirmed_track_fnxl = result_dict["key_frame_confirmed_track_fnxl"]
                key_frame_confirmed_track_fnxl = result_dict["key_frame_confirmed_track_fnxl"]
                print(
                    "    " + str(key_frame_id) + ":"
                )  # Using "str(key_frame_id)" is okay, because we don't want leading zeros.
                key_frame_confirmed_track_fnxl.print(max_value_length=200, indent=8)

            # Write found tracks files.
            self.save_key_tracks(list_of_result_dicts)

            # Write summary information.
            self.save_data(list_of_result_dicts)

            # Record that we generated the key corners.
            self.generated_key_tracks = True

        else:
            # We skipped generation.
            self.generated_key_tracks = False

    def search_key_track(self, key_frame_id):
        # Notify start.
        print("In KeyTracks.search_key_track(), fetching key frames for key_frame_id=" + str(key_frame_id) + "...")

        # Initialize logger.
        if not self.single_processor:
            # Don't make this a data member of self -- it will cause error: "can't pickle _thread.RLock objects"
            local_logger = logt.multiprocessing_logger(self.log_dir_body_ext, level=logging.INFO)
        else:
            local_logger = None

        try:
            # Input key frame corners.
            key_frame_projected_corners_fnxl = self.key_projected_corners_dict[key_frame_id][
                "key_frame_projected_corners_fnxl"
            ]
            key_frame_confirmed_corners_fnxl = self.key_confirmed_corners_dict[key_frame_id][
                "key_frame_confirmed_corners_fnxl"
            ]

            # Solar field parameters.
            specifications = Dspec.nsttf_specifications()  # ?? SCAFFOLDING RCB -- MAKE THIS GENERAL

            # Execution control.
            iterations = 3  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            canny_levels = ["medium", "light"]  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            solvePnPtype = "pnp"  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            # # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            # # Test runs were made with values of None and None for cam_matrix and dist_coeff.  However, study of the routine solvePNP() in utils.py
            # # and also routine confirm_corners() in KeyFrameTrackSearch.py indicate that the value of CameraMatrix and DistCoefs in utils.py are
            # # what is actually used throughout the calculation from beginning to end.
            # # Thus a recommended path forward is to:
            # #    (a) Set these values to utils.CameraMatrix and utils.DistCoeffs, as shown in the *untested* code lines below.
            # #    (b) Modify the code in confirm_corners() to not suggest that the internal variables mtx and dist (better names needed)
            # #        are not actually refined by OpenCV calls, but rather set to the default in utils.py and left that way.
            # # But first:  Verify that the call to  cv.projectPoints(points3d, rvec, tvec, mtx, dist) in utils.solavePNP() does not actually
            # # modify mtx and dist as a side effect.  If it does, then the above is incorrect.
            # cam_matrix=utils.CameraMatrix  # Note: Untested code!
            # dist_coeff=utils.DistCoeffs  # Note: Untested code!
            cam_matrix = None  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            dist_coeff = None  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
            confirm_type = ""  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS

            # Execute search.
            logt.info(
                local_logger,
                "In KeyTracks.search_key_track_aux(), searching for corners in key_frame_id="
                + str(key_frame_id)
                + "...",
            )
            search_result = kfts.KeyFrameTrackSearch(  # Execution control.
                iterations,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                canny_levels,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                solvePnPtype,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                cam_matrix,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                dist_coeff,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                confirm_type,  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
                # Input/output sources.
                self.input_video_body,
                key_frame_projected_corners_fnxl,
                key_frame_confirmed_corners_fnxl,
                specifications,  # ?? SCAFFOLDING RCB -- MAKE THIS GENERAL
                self.input_frame_dir,
                self.input_frame_id_format,
                self.all_frame_body_ext_list,
                output_construction_dir=self.output_construction_dir,
                # Render control.
                draw_track_images=True,
            )

            logt.info(
                local_logger,
                "In KeyTracks.search_key_track_aux(), corners done key_frame_id=" + str(key_frame_id) + ".",
            )
        except:
            error_type, error_instance, traceback = sys.exc_info()
            logt.log_and_raise_value_error(
                local_logger,
                "In KeyTracks.search_key_track(), Key Frame 1 Processing Exception: " + str(error_instance.args[0]),
            )

        # Assemble result dictionary.
        result_dict = {}
        result_dict["key_frame_id"] = key_frame_id
        result_dict["key_frame_projected_track_fnxl"] = search_result.key_frame_projected_track_fnxl
        result_dict["key_frame_confirmed_track_fnxl"] = search_result.key_frame_confirmed_track_fnxl
        return result_dict

    # WRITE RESULT

    def save_key_tracks(self, list_of_result_dicts):
        ft.create_directories_if_necessary(self.output_data_dir)
        # The FrameNameXyList track results for each key frame.
        for result_dict in list_of_result_dicts:
            key_frame_id_str = upf.frame_id_str_given_frame_id(result_dict["key_frame_id"], self.input_frame_id_format)
            # Projected.
            key_frame_projected_track_fnxl = result_dict["key_frame_projected_track_fnxl"]
            key_frame_projected_track_body_ext = (
                self.input_video_body + "_" + key_frame_id_str + "_projected_tracks_fnxl.csv"
            )
            key_frame_projected_track_dir_body_ext = os.path.join(
                self.key_frame_projected_tracks_dir, key_frame_projected_track_body_ext
            )
            print(
                "In KeyTracks.save_key_tracks(), writing found projected key frame track: ",
                key_frame_projected_track_dir_body_ext,
            )
            key_frame_projected_track_fnxl.save(key_frame_projected_track_dir_body_ext)
            # Confirmed.
            key_frame_confirmed_track_fnxl = result_dict["key_frame_confirmed_track_fnxl"]
            key_frame_confirmed_track_body_ext = (
                self.input_video_body + "_" + key_frame_id_str + "_confirmed_tracks_fnxl.csv"
            )
            key_frame_confirmed_track_dir_body_ext = os.path.join(
                self.key_frame_confirmed_tracks_dir, key_frame_confirmed_track_body_ext
            )
            print(
                "In KeyTracks.save_key_tracks(), writing found confirmed key frame track: ",
                key_frame_confirmed_track_dir_body_ext,
            )
            key_frame_confirmed_track_fnxl.save(key_frame_confirmed_track_dir_body_ext)

    def save_data(self, list_of_result_dicts):
        # Statistics.
        summary_dict = {}
        summary_dict["n_key_frame_tracks"] = len(list_of_result_dicts)
        print("In KeyTracks.save_data(), writing key frame summary statistics...")
        ft.write_dict_file("key frame tracks summary statistics", self.output_data_dir, self.dict_body, summary_dict)
        # Tracked frames per key frame.
        tracked_frames_per_key_frame_dict = {}
        for result_dict in list_of_result_dicts:
            key_frame_id = result_dict["key_frame_id"]
            key_frame_projected_track_fnxl = result_dict["key_frame_projected_track_fnxl"]
            tracked_frames_per_key_frame_dict[key_frame_id] = key_frame_projected_track_fnxl.number_of_frames()
        print(
            "In KeyTracks.save_data(), writing tracked_frames per key frame:",
            os.path.join(self.output_data_dir, self.tfpkf_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, self.tfpkf_body, tracked_frames_per_key_frame_dict)

    # LOAD RESULT

    def read_key_tracks(self):
        # Projected.
        print(
            "In KeyTracks.read_key_tracks(), reading found key frame projected tracks directory: ",
            self.key_frame_projected_tracks_dir,
        )
        key_frame_projected_track_body_ext_list = ft.files_in_directory(self.key_frame_projected_tracks_dir)
        self.list_of_key_frame_projected_track_dir_body_ext = []
        self.list_of_key_frame_projected_track_fnxls = []
        for key_frame_projected_track_body_ext in key_frame_projected_track_body_ext_list:
            key_frame_projected_track_dir_body_ext = os.path.join(
                self.key_frame_projected_tracks_dir, key_frame_projected_track_body_ext
            )
            self.list_of_key_frame_projected_track_dir_body_ext.append(key_frame_projected_track_dir_body_ext)
            print(
                "In KeyTracks.read_key_tracks(), reading found key frame projected track:        ",
                key_frame_projected_track_dir_body_ext,
            )
            key_frame_projected_track_fnxl = fnxl.FrameNameXyList()
            key_frame_projected_track_fnxl.load(key_frame_projected_track_dir_body_ext)
            self.list_of_key_frame_projected_track_fnxls.append(key_frame_projected_track_fnxl)
        # Confirmed.
        print(
            "In KeyTracks.read_key_tracks(), reading found key frame confirmed tracks directory: ",
            self.key_frame_confirmed_tracks_dir,
        )
        key_frame_confirmed_track_body_ext_list = ft.files_in_directory(self.key_frame_confirmed_tracks_dir)
        self.list_of_key_frame_confirmed_track_dir_body_ext = []
        self.list_of_key_frame_confirmed_track_fnxls = []
        for key_frame_confirmed_track_body_ext in key_frame_confirmed_track_body_ext_list:
            key_frame_confirmed_track_dir_body_ext = os.path.join(
                self.key_frame_confirmed_tracks_dir, key_frame_confirmed_track_body_ext
            )
            self.list_of_key_frame_confirmed_track_dir_body_ext.append(key_frame_confirmed_track_dir_body_ext)
            print(
                "In KeyTracks.read_key_tracks(), reading found key frame confirmed track:        ",
                key_frame_confirmed_track_dir_body_ext,
            )
            key_frame_confirmed_track_fnxl = fnxl.FrameNameXyList()
            key_frame_confirmed_track_fnxl.load(key_frame_confirmed_track_dir_body_ext)
            self.list_of_key_frame_confirmed_track_fnxls.append(key_frame_confirmed_track_fnxl)

    def read_data(self):
        # Statistics.
        print("In KeyTracks.read_data(), reading frame statistics: ", self.dict_dir_body_ext)
        self.frame_statistics_dict = ft.read_dict(self.dict_dir_body_ext)
        # Confirm what was read.
        print("In KeyTracks.read_data(), frame statistics read:")
        dt.print_dict(self.frame_statistics_dict, indent=4)
        # Tracked frames per key frame.
        print("In KeyTracks.read_data(), reading tracked frames per key frame: ", self.tfpkf_dir_body_ext)
        self.tfpkf_dict = ft.read_dict(self.tfpkf_dir_body_ext)
        # Confirm what was read.
        print("In KeyTracks.read_data(), tracked frames per key frame read:")
        dt.print_dict(self.tfpkf_dict, max_keys=7, max_value_length=200, indent=4)

    # RENDER RESULT

    def render(self):
        if self.render_control.draw_key_tracks and self.generated_key_tracks:  # Don't render unless we generated.
            print("In KeyTracks.render(), rendering key frame tracks...")
            print("WARNING: In KeyTracks.render(), track rendering not implemented yet.")


if __name__ == "__main__":
    # Execution control.
    force_construction = False
    # specific_frame_ids              = [3598]  # A big one, spanning two passes.  Output is more than three times larger than others, and run time would be commensurate.
    # specific_frame_ids              = [4569]  # Drifts off backward track.
    # specific_frame_ids              = [5417]  # Looks good both directions.
    # specific_frame_ids              = [6076]  # Drifts off forward track.
    # specific_frame_ids              = [12943]  # Looks good both directions.
    # specific_frame_ids              = [16259]  # Drifts off backward track.
    # specific_frame_ids              = [3896,4569,5417,6076]  # Heliostats 11W6, 11W7, and 11W8 are seen in multiple passes (Small cases).
    specific_frame_ids = None  # [6076]  #[12943] #[3896,4569,5417,6076] #[3598,12943,16259] #None #[1035,6076,12943,14639,16259] #None #[16768] #None #[14639] #[1035] #[1035,12943,14639,16259] #None #[777] #None #[777, 897, 1035] #None
    single_processor = False  # True #False
    # log_dir_body_ext                = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_150_KeyTracks/mavic_zoom/log/KeyTracks_log.txt'
    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/150_KeyTracks/mavic_zoom/log/KeyTracks_log.txt"
    )
    # Input/output sources.
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_key_projected_corners_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/140_KeyCorners/mavic_zoom/data/key_corners/"
    )  # ?? SCAFFOLDING RCB -- BUG: USING PROJECTED CORNERS FOR BOTH PROJECTED AND CONFIRMED.
    input_key_confirmed_corners_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/140_KeyCorners/mavic_zoom/data/key_corners/"
    )  # ?? SCAFFOLDING RCB -- BUG: USING PROJECTED CORNERS FOR BOTH PROJECTED AND CONFIRMED.
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    # output_data_dir                 = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_150_KeyTracks/mavic_zoom/data/'
    # output_render_dir               = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_150_KeyTracks/mavic_zoom/render/'
    # output_construction_dir         = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/Small_150c_KeyTracks/mavic_zoom/'
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/150_KeyTracks/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/150_KeyTracks/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/150c_KeyTracks/mavic_zoom/"
    )
    # Render control.
    render_control = rckt.default()

    key_tracks_object = KeyTracks(  # Execution control.
        force_construction,
        specific_frame_ids,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        input_video_dir_body_ext,
        input_key_projected_corners_dir,
        input_key_confirmed_corners_dir,
        input_frame_dir,
        input_frame_id_format,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        # Render control.
        render_control,
    )
