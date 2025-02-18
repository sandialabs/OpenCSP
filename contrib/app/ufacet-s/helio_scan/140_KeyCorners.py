"""
Finding heliostat facet corners in key frames of a UFACET scan video.



"""

import copy
from cv2 import cv2 as cv
import logging
from multiprocessing import Pool
import os
import sys

import lib.DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY

import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import lib.FrameNameXyList as fnxl
import lib.KeyFrameCornerSearch as kfcs
import opencsp.common.lib.tool.log_tools as logt
import opencsp.common.lib.render_control.RenderControlKeyCorners as rckc
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import lib.ufacet_pipeline_clear as upc
import lib.ufacet_pipeline_frame as upf


class KeyCorners:
    """
    Class controlling facet corner search given key frames of a UFACET scan video.

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
        input_key_frames_dir_body_ext,  # Where to find file with key frame numbers and associated [hel_name, polygon] pairs.
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
            raise ValueError("In KeyCorners.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In KeyCorners.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In KeyCorners.__init__(), null output_render_dir encountered.")

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
        self.input_key_frames_dir_body_ext = input_key_frames_dir_body_ext
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.render_control = render_control

        # Found key corners file name.
        self.all_key_frames_corners_body = self.input_video_body + "_all_frames_corners_fnxl"
        self.all_key_frames_corners_body_ext = self.all_key_frames_corners_body + ".csv"
        self.all_key_frames_corners_dir_body_ext = os.path.join(
            self.output_data_dir, self.all_key_frames_corners_body_ext
        )
        self.key_frame_corners_dir = os.path.join(self.output_data_dir, "key_corners")

        # Summary statistics file name.
        self.dict_body = self.input_video_body + "_key_frames_with_corners_statistics"
        self.dict_body_ext = self.dict_body + ".csv"
        self.dict_dir_body_ext = os.path.join(self.output_data_dir, self.dict_body_ext)

        # File listing key frames with mismatched heliostats.
        self.mismatched_ids_body = self.input_video_body + "_mismatched_key_frame_ids"
        self.mismatched_ids_body_ext = self.mismatched_ids_body + ".txt"
        self.mismatched_ids_dir_body_ext = os.path.join(self.output_data_dir, self.mismatched_ids_body_ext)

        # Heliostats per key frame file name.
        self.hpkf_body = self.input_video_body + "_heliostats_per_key_frame"
        self.hpkf_body_ext = self.hpkf_body + ".csv"
        self.hpkf_dir_body_ext = os.path.join(self.output_data_dir, self.hpkf_body_ext)

        # Points per key frame file name.
        self.ppkf_body = self.input_video_body + "_points_per_key_frame"
        self.ppkf_body_ext = self.ppkf_body + ".csv"
        self.ppkf_dir_body_ext = os.path.join(self.output_data_dir, self.ppkf_body_ext)

        # Load key frames file.
        print("In KeyCorners.__init__(), reading key frame specs: ", self.input_key_frames_dir_body_ext)
        self.key_frames_fnxl = fnxl.FrameNameXyList()
        self.key_frames_fnxl.load(self.input_key_frames_dir_body_ext)
        # Confirm what was read.
        print("In KeyCorners.__init__(), key frame specfication read:")
        self.key_frames_fnxl.print(max_keys=7, max_value_length=200, indent=4)

        # Fetch a list of all frame ids in the video (not just key frames).
        # The corresponding frame_ids are not necessarily in sequential order, because
        # we previously removed spurious duplicate frames.
        self.all_frame_file_list = ft.files_in_directory(self.input_frame_dir, sort=True)
        # # Confirm what was read.
        # print('In KeyCorners.__init__(), self.all_frame_file_list:')
        # for frame_file in self.all_frame_file_list:
        #     print('In KeyCorners.__init__()   ', frame_file)

        # Find facet corners in key frames, archiving the result.
        self.find_and_save_key_corners()

        # Load found corners.
        self.read_key_corners()

        # Load summary data.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT

    def find_and_save_key_corners(self):
        # Check if corners have already been found.
        if (
            self.force_construction
            or (not ft.directory_exists(self.output_data_dir))
            or ft.directory_is_empty(self.output_data_dir)
        ):
            print("In KeyCorners.find_and_save_key_corners(), searching for key corners...")

            # Determine which key frames to process.
            if self.specific_frame_ids == None:
                key_frame_ids_to_process = (
                    self.key_frames_fnxl.sorted_frame_id_list()
                )  # We sort only to better observe progress.  Not required.
            else:
                key_frame_ids_to_process = self.specific_frame_ids

            # Process each key frame_id.
            if self.single_processor:
                print("In KeyCorners.search_key_frames(), starting key frame corner search (single processor)...")
                list_of_result_dicts = []
                for key_frame_id in key_frame_ids_to_process:
                    list_of_result_dicts.append(self.search_key_frame(key_frame_id))
            else:
                print("In KeyCorners.search_key_frames(), starting key frame corner search (multi-processor)...")
                logger = logt.multiprocessing_logger(self.log_dir_body_ext, level=logging.INFO)
                logger.info("================================= Execution =================================")
                with Pool(36) as pool:
                    list_of_result_dicts = pool.map(self.search_key_frame, key_frame_ids_to_process)

            # Remove "None" entries.
            list_of_fnxl_or_None_results = [
                result_dict["pair_projected_fnxl_or_None"] for result_dict in list_of_result_dicts
            ]
            key_frame_fnxls = [x for x in list_of_fnxl_or_None_results if x is not None]
            print(
                "In KeyCorners.search_key_frames(), key frame corner search done.  len(key_frame_fnxls) =",
                len(key_frame_fnxls),
            )

            # Extract mismatched heliostat frame_ids.
            mismatched_key_frame_id_or_None_list = [
                result_dict["mismatched_key_frame_id_or_None"] for result_dict in list_of_result_dicts
            ]
            mismatched_key_frame_ids = [x for x in mismatched_key_frame_id_or_None_list if x is not None]
            print(
                "In KeyCorners.search_key_frames(), mismatched corner frames extracted.  len(mismatched_key_frame_ids) =",
                len(mismatched_key_frame_ids),
            )
            print(
                "In KeyCorners.search_key_frames(), mismatched corner frames extracted.  mismatched_key_frame_ids =",
                mismatched_key_frame_ids,
            )  # ?? SCAFFOLDING RCB -- TEMPORARY

            # Produce overall combined FrameNameXyList object.
            all_key_frames_corners_fnxl = fnxl.construct_merged_copy(key_frame_fnxls)

            # Summarize search results.
            print("In KeyCorners.find_and_save_key_corners(), len(key_frame_fnxls) =", len(key_frame_fnxls))
            for key_frame_fnxl in key_frame_fnxls:
                print("In KeyCorners.find_and_save_key_corners(),     key_frame_fnxl:")
                if key_frame_fnxl is None:
                    print("        None")
                else:
                    key_frame_fnxl.print(max_value_length=200, indent=8)
            print(
                "\nIn KeyCorners.find_and_save_key_corners(), all_key_frames_corners_fnxl.number_of_frames() =",
                all_key_frames_corners_fnxl.number_of_frames(),
            )
            print("In KeyCorners.find_and_save_key_corners(), all_key_frames_corners_fnxl:")
            all_key_frames_corners_fnxl.print(max_value_length=200, indent=8)

            # Write found corners files.
            self.save_key_corners(all_key_frames_corners_fnxl, key_frame_fnxls)

            # Write summary information.
            self.save_data(all_key_frames_corners_fnxl, mismatched_key_frame_ids)

            # Record that we generated the key corners.
            self.generated_key_corners = True

        else:
            # We skipped generation.
            self.generated_key_corners = False

    def search_key_frame(self, key_frame_id):
        # Notify start.
        print("In KeyCorners.search_key_frame(), fetching key frames for key_frame_id=" + str(key_frame_id) + "...")

        # Find the filenames of the key frame and the frame that follows it.
        key_frame_body_ext_1 = upf.frame_file_body_ext_given_frame_id(
            self.input_video_body, key_frame_id, self.input_frame_id_format
        )
        if key_frame_body_ext_1 not in self.all_frame_file_list:
            logt.log_and_raise_value_error(
                self.logger,
                "In KeyCorners.search_key_frames(), the full frame file directory:\n   ",
                self.input_frame_dir,
                "\ndoes not contain key frame file:\n   ",
                key_frame_body_ext_1,
            )
        key_frame_file_1_idx = self.all_frame_file_list.index(key_frame_body_ext_1)
        key_frame_file_2_idx = key_frame_file_1_idx + 1
        if key_frame_file_2_idx >= len(self.all_frame_file_list):
            logt.log_and_raise_value_error(
                self.logger,
                "In KeyCorners.search_key_frames(), key_frame_id="
                + str(key_frame_id)
                + " is the last file the full frame file directory:\n   ",
                self.input_frame_dir,
                "\nso there is no second frame file available.\nSelecting the final video frame as a key frame is not allowed.",
            )
        key_frame_body_ext_2 = self.all_frame_file_list[key_frame_file_2_idx]

        # Get the ids of the key frame pair.
        key_frame_id_1 = key_frame_id
        key_frame_id_2 = upf.frame_id_given_frame_file_body_ext(key_frame_body_ext_2)
        print("In KeyCorners.search_key_frames(), key_frame_id_1 =", key_frame_id_1)  # ?? SCAFFOLDING RCB -- TEMPORARY
        print("In KeyCorners.search_key_frames(), key_frame_id_2 =", key_frame_id_2)  # ?? SCAFFOLDING RCB -- TEMPORARY

        # Get the key frame name_polygon list.
        list_of_name_polygons = self.key_frames_fnxl.list_of_name_xy_lists(key_frame_id)  # Applies to both frames.

        # Assemble construction output directory paths.
        key_frame_id_str_1 = upf.frame_id_str_given_frame_file_body_ext(key_frame_body_ext_1)
        output_construction_dir_1 = os.path.join(self.output_construction_dir, key_frame_id_str_1, "frame1")
        output_construction_dir_2 = os.path.join(self.output_construction_dir, key_frame_id_str_1, "frame2")

        # Initialize logger.
        if not self.single_processor:
            # Don't make this a data member of self -- it will cause error: "can't pickle _thread.RLock objects"
            local_logger = logt.multiprocessing_logger(self.log_dir_body_ext, level=logging.INFO)
        else:
            local_logger = None

        # Search for key_frame 1 corners.
        try:
            search_result_1 = self.search_key_frame_aux(
                local_logger, key_frame_id_1, list_of_name_polygons, key_frame_body_ext_1, output_construction_dir_1
            )
        except:
            error_type, error_instance, traceback = sys.exc_info()
            logt.log_and_raise_value_error(
                local_logger,
                "In KeyCorners.search_key_frame(), Key Frame 1 Processing Exception for key_frame_id="
                + str(key_frame_id_1)
                + ", "
                + str(error_instance.args[0]),
            )

        # Search for key_frame 2 corners.
        try:
            search_result_2 = self.search_key_frame_aux(
                local_logger, key_frame_id_2, list_of_name_polygons, key_frame_body_ext_2, output_construction_dir_2
            )
        except:
            error_type, error_instance, traceback = sys.exc_info()
            logt.log_and_raise_value_error(
                local_logger,
                "In KeyCorners.search_key_frame(), Key Frame 2 Processing Exception for key_frame_id="
                + str(key_frame_id_2)
                + ", "
                + str(error_instance.args[0]),
            )

        # Construct a merged FrameNameXyList object representing both search results, synchronized.
        if (not search_result_1.successful()) or (not search_result_2.successful()):
            # Then one or both key frame searches failed for some reason.
            pair_projected_fnxl_or_None = None
        else:
            # Both searches succeeded.  Let's see if the results share any common heliostats.
            # Construct a new fnxl that combines both key frame search results.
            # Take care to ensure that both have the same set of heliostat names.
            # This can return None if there are no xy_lists with a common name.
            (pair_projected_fnxl_or_None, mismatched) = self.construct_merged_fnxl_synchronizing_heliostat_names(
                local_logger, search_result_1.projected_fnxl(), search_result_2.projected_fnxl()
            )

        # Determine whether any found heliostats were lost from one frame to the next.
        if mismatched:
            mismatched_key_frame_id_or_None = key_frame_id
        else:
            mismatched_key_frame_id_or_None = None

        # Assemble result dictionary.
        result_dict = {}
        result_dict["pair_projected_fnxl_or_None"] = pair_projected_fnxl_or_None
        result_dict["mismatched_key_frame_id_or_None"] = mismatched_key_frame_id_or_None
        return result_dict

    def search_key_frame_aux(
        self, local_logger, key_frame_id, list_of_name_polygons, key_frame_body_ext, output_construction_dir
    ):
        key_frame_id_str = upf.frame_id_str_given_frame_file_body_ext(key_frame_body_ext)
        # Read the key frame image.
        key_frame_dir_body_ext = os.path.join(self.input_frame_dir, key_frame_body_ext)
        key_frame_img = cv.imread(key_frame_dir_body_ext)
        if key_frame_img is None:
            logt.log_and_raise_value_error(
                self.logger, "In KeyCorners.search_key_frame_aux(), error reading image file:", key_frame_body_ext
            )
        # Initialize output directory.
        ft.create_directories_if_necessary(output_construction_dir)
        # Execute search.
        logt.info(
            local_logger,
            "In KeyCorners.search_key_frame_aux(), searching for corners in key_frame_id=" + str(key_frame_id) + "...",
        )
        search_result = kfcs.KeyFrameCornerSearch(
            key_frame_id=key_frame_id,
            key_frame_id_str=key_frame_id_str,
            key_frame_img=key_frame_img,
            list_of_name_polygons=list_of_name_polygons,
            specifications=Dspec.nsttf_specifications(),  # ?? SCAFFOLDING RCB -- TEMPORARY
            output_construction_dir=output_construction_dir,
            solvePnPtype="pnp",  # ?? SCAFFOLDING RCB -- solvePnPtype changed to 'pnp' from 'calib' since we are working with Mavic data.
            render_control=self.render_control,
        )
        logt.info(
            local_logger, "In KeyCorners.search_key_frame_aux(), corners done key_frame_id=" + str(key_frame_id) + "."
        )
        return search_result

    def construct_merged_fnxl_synchronizing_heliostat_names(self, local_logger, fnxl_1, fnxl_2):
        """
        This rouitne merges the key frame 1 and key frame 2 corner search results into a combined FrameNameXyList
        object containing the corners found for both frames.  However, it does not simply transcribe the corner
        lists for each, because they might not match for tracking.

        Here's how this can happen:  Suppose the key frame spec has two boxes, one for heliostat A, and one for
        heliostat B.  In frame 1, both are found, so there is a corner xy_list for A and a corner xy_list for B.
        But for some reason, in key frame 2 only heliostat B weas found.  In this case, this routine will return
        a combined FrameNameXyList object with two frames, each of which has an xy_list for only heliostat B.

        This allows subsequent corner tracking to occur with proper synchronization between frame_1 and frame_2.
        """
        # Check input.
        if fnxl_1.number_of_frames() != 1:
            logt.log_and_raise_value_error(
                local_logger,
                "In KeyCorners.construct_merged_fnxl_synchronizing_heliostat_names(), fnxl_1 does not have the expected singleton frame.  Its frames are:",
                fnxl_1.sorted_frame_id_list(),
            )
        if fnxl_2.number_of_frames() != 1:
            logt.log_and_raise_value_error(
                local_logger,
                "In KeyCorners.construct_merged_fnxl_synchronizing_heliostat_names(), fnxl_2 does not have the expected singleton frame.  Its frames are:",
                fnxl_2.sorted_frame_id_list(),
            )
        frame_id_1 = fnxl_1.sorted_frame_id_list()[
            0
        ]  # It's tempting to use frame_ids_unsorted() here, but the problem is that
        frame_id_2 = fnxl_2.sorted_frame_id_list()[
            0
        ]  # routine returns a dict_keys object, which does not support subscripts.
        list_of_name_xy_lists_1 = fnxl_1.list_of_name_xy_lists(frame_id_1)
        list_of_name_xy_lists_2 = fnxl_2.list_of_name_xy_lists(frame_id_2)

        # Find the names common to both.
        # (The xy_lists need to be synchronized to support tracking.)
        name_set_1 = set([name_xy_list[0] for name_xy_list in list_of_name_xy_lists_1])
        name_set_2 = set([name_xy_list[0] for name_xy_list in list_of_name_xy_lists_2])
        common_names = name_set_1.intersection(name_set_2)
        if len(common_names) == 0:
            return None

        # For each FrameNameXyList object, assemble a list of name_xy_lists, where the names are common to both.
        common_list_of_name_xy_lists_1 = []
        common_list_of_name_xy_lists_2 = []
        for name in common_names:
            name_xy_list_1 = fnxl_1.matching_name_xy_list(frame_id_1, name)
            name_xy_list_2 = fnxl_2.matching_name_xy_list(frame_id_2, name)
            common_list_of_name_xy_lists_1.append(copy.deepcopy(name_xy_list_1))
            common_list_of_name_xy_lists_2.append(copy.deepcopy(name_xy_list_2))

        # Createa a new FrameNameXyList and add the synchronized name_xy_lists to it.
        pair_projected_fnxl = fnxl.FrameNameXyList()
        pair_projected_fnxl.add_list_of_name_xy_lists(frame_id_1, common_list_of_name_xy_lists_1)
        pair_projected_fnxl.add_list_of_name_xy_lists(frame_id_2, common_list_of_name_xy_lists_2)

        # Determine whether heliostats are mismatched from key frame 1 to key frame 2.
        if (len(common_names) == len(name_set_1)) and (len(common_names) == len(name_set_2)):
            mismatched = False
        else:
            mismatched = True

        # Return.
        return pair_projected_fnxl, mismatched

    # WRITE RESULT

    def save_key_corners(self, all_key_frames_corners_fnxl, key_frame_fnxls):
        # The single FrameNameXyList object including all key frames.
        print(
            "In KeyCorners.save_key_corners(), writing all-frame summary of found key corners:  ",
            self.all_key_frames_corners_dir_body_ext,
        )
        all_key_frames_corners_fnxl.save(self.all_key_frames_corners_dir_body_ext)
        # The smaller FrameNameXyList objects for each key frame pair.
        ft.create_directories_if_necessary(self.key_frame_corners_dir)
        for key_frame_fnxl in key_frame_fnxls:
            if key_frame_fnxl is not None:
                frame_id_list = key_frame_fnxl.sorted_frame_id_list()
                if len(frame_id_list) == 0:
                    raise ValueError("In KeyCorners.save_key_corners(), empty key_frame_fnxl encountered.")
                if len(frame_id_list) != 2:
                    raise ValueError(
                        "In KeyCorners.save_key_corners(), encountered key_frame_fnxl with frames != 2.  len(frame_id_list) =",
                        len(frame_id_list),
                    )
                frame_id = frame_id_list[0]  # Sorted, so this is the key frame.
                frame_id_str = upf.frame_id_str_given_frame_id(frame_id, self.input_frame_id_format)
                key_frame_corners_body_ext = self.input_video_body + "_" + frame_id_str + "_corners_fnxl.csv"
                key_frame_corners_dir_body_ext = os.path.join(self.key_frame_corners_dir, key_frame_corners_body_ext)
                print(
                    "In KeyCorners.save_key_corners(), writing found frame_id key corners:              ",
                    key_frame_corners_dir_body_ext,
                )  # Pad spaces to line up output.
                key_frame_fnxl.save(key_frame_corners_dir_body_ext)

    def save_data(self, all_key_frames_corners_fnxl, mismatched_key_frame_ids):
        # Statistics.
        summary_dict = {}
        summary_dict["n_key_frames_with_corners"] = all_key_frames_corners_fnxl.number_of_frames()
        print("In KeyCorners.save_data(), writing key frame summary statistics...")
        ft.write_dict_file("key frame corners summary statistics", self.output_data_dir, self.dict_body, summary_dict)
        # Key frames with mismatched heliostats.
        ft.write_text_file(
            "mismatched key frame ids", self.output_data_dir, self.mismatched_ids_body, mismatched_key_frame_ids
        )
        # Heliostats per key frame.
        heliostats_per_key_frame_dict = all_key_frames_corners_fnxl.heliostats_per_frame()
        print(
            "In KeyCorners.save_data(), writing heliostats per key frame:",
            os.path.join(self.output_data_dir, self.hpkf_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, self.hpkf_body, heliostats_per_key_frame_dict)
        # Points per key frame.
        points_per_key_frame_dict = all_key_frames_corners_fnxl.points_per_frame()
        print(
            "In KeyCorners.save_data(), writing points per key frame:    ",
            os.path.join(self.output_data_dir, self.ppkf_body_ext),
        )
        ft.write_dict_file(None, self.output_data_dir, self.ppkf_body, points_per_key_frame_dict)

    # LOAD RESULT

    def read_key_corners(self):
        print(
            "In KeyCorners.read_key_corners(), reading all-frame summary of found key corners:  ",
            self.all_key_frames_corners_dir_body_ext,
        )
        self.all_key_frames_corners_fnxl = fnxl.FrameNameXyList()
        self.all_key_frames_corners_fnxl.load(self.all_key_frames_corners_dir_body_ext)
        # Confirm what was read.
        print("In KeyCorners.read_key_corners(), all-frame key corners read:")
        self.all_key_frames_corners_fnxl.print(max_keys=7, max_value_length=200, indent=4)
        print("In KeyCorners.read_key_corners(), reading found key corners directory: ", self.key_frame_corners_dir)
        key_frame_corners_body_ext_list = ft.files_in_directory(self.key_frame_corners_dir)
        self.list_of_key_frame_corners_dir_body_ext = []
        self.list_of_key_frame_corners_fnxls = []
        for key_frame_corners_body_ext in key_frame_corners_body_ext_list:
            key_frame_corners_dir_body_ext = os.path.join(self.key_frame_corners_dir, key_frame_corners_body_ext)
            self.list_of_key_frame_corners_dir_body_ext.append(key_frame_corners_dir_body_ext)
            print(
                "In KeyCorners.read_key_corners(), reading found key corners file:      ",
                key_frame_corners_dir_body_ext,
            )
            key_frames_corners_fnxl = fnxl.FrameNameXyList()
            key_frames_corners_fnxl.load(key_frame_corners_dir_body_ext)
            self.list_of_key_frame_corners_fnxls.append(key_frames_corners_fnxl)

    def read_data(self):
        # Statistics.
        print("In KeyCorners.read_data(), reading frame statistics: ", self.dict_dir_body_ext)
        self.frame_statistics_dict = ft.read_dict(self.dict_dir_body_ext)
        # Confirm what was read.
        print("In KeyCorners.read_data(), frame statistics read:")
        dt.print_dict(self.frame_statistics_dict, indent=4)
        # Key frames with mismatched heliostats.
        lines = ft.read_text_file(self.mismatched_ids_dir_body_ext)
        self.mismatched_key_frame_ids = [int(x) for x in lines]
        # Confirm what was read.
        print("In KeyCorners.read_data(), mismatched key frame ids read:")
        for key_frame_id in self.mismatched_key_frame_ids:
            print("    ", key_frame_id)
        # Heliostats per key frame.
        print("In KeyCorners.read_data(), reading heliostats per key frame: ", self.hpkf_dir_body_ext)
        self.hpkf_dict = ft.read_dict(self.hpkf_dir_body_ext)
        # Confirm what was read.
        print("In KeyCorners.read_data(), heliostats per key frame read:")
        dt.print_dict(self.hpkf_dict, max_keys=7, max_value_length=200, indent=4)
        # Points per key frame.
        print("In KeyCorners.read_data(), reading points per key frame: ", self.ppkf_dir_body_ext)
        self.ppkf_dict = ft.read_dict(self.ppkf_dir_body_ext)
        # Confirm what was read.
        print("In KeyCorners.read_data(), points per key frame read:")
        dt.print_dict(self.ppkf_dict, max_keys=7, max_value_length=200, indent=4)

    # RENDER RESULT

    def render(self):
        if self.render_control.draw_key_corners and self.generated_key_corners:  # Don't render unless we generated.
            print("In KeyCorners.render(), rendering frames with key corners...")
            # Descriptive strings.
            title_name = "Key Frame Corners"
            context_str = "KeyCorners.render()"
            # Required suffix strings.
            fig_suffix = "_key_corners_fig"
            delete_suffix = ".JPG" + fig_suffix + ".png"
            # Prepare directory.
            upc.prepare_render_directory(self.output_render_dir, delete_suffix, self.render_control)
            # Setup annotation styles.
            style_dict = {}
            style_dict["point_seq"] = rcps.marker(
                marker=self.render_control.key_corners_points_marker,
                markersize=self.render_control.key_corners_points_markersize,
                color=self.render_control.key_corners_points_color,
            )
            style_dict["text"] = rctxt.RenderControlText(
                horizontalalignment=self.render_control.key_corners_label_horizontalalignment,
                verticalalignment=self.render_control.key_corners_label_verticalalignment,
                fontsize=self.render_control.key_corners_label_fontsize,
                fontstyle=self.render_control.key_corners_label_fontstyle,
                fontweight=self.render_control.key_corners_label_fontweight,
                color=self.render_control.key_corners_label_color,
            )
            # Draw the frames.
            self.all_key_frames_corners_fnxl.draw_frames(
                self.single_processor,
                self.log_dir_body_ext,
                title_name,
                context_str,
                fig_suffix,
                self.input_video_body,
                self.input_frame_dir,
                self.input_frame_id_format,
                self.output_render_dir,
                dpi=self.render_control.key_corners_dpi,
                close_xy_list=True,
                style_dict=style_dict,
                crop=self.render_control.key_corners_crop,
            )


if __name__ == "__main__":
    # Execution control.
    force_construction = False
    specific_frame_ids = None  # [3598] #None #[1035,6076,12943,14639,16259] #None #[16768] #None #[14639] #[1035] #[1035,12943,14639,16259] #None #[777] #None #[777, 897, 1035] #None
    single_processor = False  # True #False
    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/140_KeyCorners/mavic_zoom/log/KeyCorners_log.txt"
    )
    # Input/output sources.
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_key_frames_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/data/DJI_427t_428_429_key_frames_fnxl.csv"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/140_KeyCorners/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/140_KeyCorners/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/140c_KeyCorners/mavic_zoom/"
    )
    # Render control.
    render_control = rckc.default()

    # Don't repeat expensive rendering if we do not generate.

    key_frames_object = KeyCorners(  # Execution control.
        force_construction,
        specific_frame_ids,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        input_video_dir_body_ext,
        input_key_frames_dir_body_ext,
        input_frame_dir,
        input_frame_id_format,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        # Render control.
        render_control,
    )
