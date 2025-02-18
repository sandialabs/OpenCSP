"""
Identifying key frames in a UFACET scan video, suitable for initial image processing.



"""

import csv
from cv2 import cv2 as cv
import os

import opencsp.common.lib.render.video_manipulation as vm
import opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual as rckfgm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import lib.FrameNameXyList as fnxl
import lib.ufacet_pipeline_clear as upc


class KeyFramesGivenManual:
    """
    Class for a list of key frames from a UFACET scan video, suitable for initial image processing.

    A suitable key frame will view one or more heliostats where at least the top row of facets is reflecting
    sky only, and the heliostat is not overlapping other sky-reflecting heliostat in the background.

    This version accepts a table of manually-selected key frames and the expected heliostat box.
    The input manual selection is in the original instance-based time model, which we replace
    with a frame_id time model.

    """

    def __init__(
        self,
        # Execution control.
        # force_construction,                   # Recompute even if results already exist.  (Disabled because we edited after conversion.)
        specific_frame_ids,  # List of frame_ids to process, e.g. [777, 1126, ...].  For all frames, set None.
        single_processor,  # Execute multi-processor steps with a single processor.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        # Input/output sources.
        input_video_dir_body_ext,  # Where to find the video file.
        input_original_keyinfo_dir_body_ext,  # Where to find the original keyinfo file produced by manual selection, in obsolete format.
        input_original_keyinfo_boxHD,  # Whether manual key frame selection used an HD video in lieu of full-resolution.  Boolean.
        input_original_keyinfo_hd_shape,  # Size of HD video used.  If not applicable, use "None."
        input_edited_key_frames_dir_body_ext,  # Best source of key frame information, after original was converted and then edited to correct errors.
        input_frame_dir,  # Where to read full frames, for rendering.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting sample frame plots.
        # Render control.
        render_control,
    ):  # Flags to control rendering on this run.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In KeyFramesGivenManual.__init__(), null input_video_dir_body_ext encountered.")
        if (input_original_keyinfo_dir_body_ext == None) or (len(input_original_keyinfo_dir_body_ext) == 0):
            raise ValueError(
                "In KeyFramesGivenManual.__init__(), null input_original_keyinfo_dir_body_ext encountered."
            )
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In KeyFramesGivenManual.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In KeyFramesGivenManual.__init__(), null output_render_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        # Execution control.
        # self.force_construction                   = force_construction
        self.specific_frame_ids = specific_frame_ids
        self.single_processor = single_processor
        self.log_dir_body_ext = log_dir_body_ext
        self.input_video_dir_body_ext = input_video_dir_body_ext
        # Input/output sources.
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_original_keyinfo_dir_body_ext = input_original_keyinfo_dir_body_ext
        self.input_original_keyinfo_boxHD = input_original_keyinfo_boxHD
        self.input_original_keyinfo_hd_shape = input_original_keyinfo_hd_shape
        self.input_edited_key_frames_dir_body_ext = input_edited_key_frames_dir_body_ext
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        # Render control.
        self.render_control = render_control

        # Lookup frames per second.
        video = cv.VideoCapture(input_video_dir_body_ext)
        self.fps = int(video.get(cv.CAP_PROP_FPS)) + 1  # +1 hard value  # RCB doesn't understand this.

        # Key frame file name.
        self.key_frames_body = self.input_video_body + "_key_frames_fnxl"
        self.key_frames_body_ext = self.key_frames_body + ".csv"
        self.key_frames_dir_body_ext = os.path.join(self.output_data_dir, self.key_frames_body_ext)

        # Summary statistics file name.
        self.dict_body = self.input_video_body + "_key_frame_statistics"
        self.dict_body_ext = self.dict_body + ".csv"
        self.dict_dir_body_ext = os.path.join(self.output_data_dir, self.dict_body_ext)

        # Heliostats per key frame file name.
        self.hpkf_body = self.input_video_body + "_heliostats_per_key_frame"
        self.hpkf_body_ext = self.hpkf_body + ".csv"
        self.hpkf_dir_body_ext = os.path.join(self.output_data_dir, self.hpkf_body_ext)

        # Convert original manual specifcation into new format, if not already.
        self.convert_and_save_original_keyinfo_file()

        # Load key frame definitions.
        self.read_key_frames()
        # for key_frame_id_key in self.key_frame_dict.keys():
        #     print('In KeyFramesGivenManual.__init__(),   ' + str(key_frame_id_key) + ':  ' + str(self.key_frame_dict[key_frame_id_key]))

        # Load summary statistics.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT

    def convert_and_save_original_keyinfo_file(self):
        # Check if keyinfo is already converted.
        if not ft.file_exists(self.key_frames_dir_body_ext):
            # RCB Note 8/28/2021:  We disable this code, because after converting the key frames,
            # rendering them, and then studying the tracking video, we determined there were errors
            # in the original key frame manual selection.  We determined corrections for these errors,
            # and since they were expressed in terms of frame id numnbers, we edited the output
            # "DJI_427t_428_429_key_frames.csv" file.  Thus to run the conversion again would clobber
            # these corrections.  We disable this code to prevent this from happening accidentally.
            if False:  # Set to True to force conversion from original data.
                # print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), converting keyinfo...')
                # Read keyinfo file.
                keyinfo_dict = self.read_keyinfo()
                # print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), keyinfo:')
                # for instance_key in keyinfo_dict.keys():
                #     print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(),   ' + str(instance_key) + ':  ' + str(keyinfo_dict[instance_key]))

                # Identify corresponding frames.
                # print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), instance_key --> frame_key correspondence:')
                instance_frame_correspondence_dict = {}
                for instance_key in keyinfo_dict.keys():
                    # Keyinfo is a tuple (frame_id_within_instance, number_of_heliosats, box_coords, heliostat_names)
                    keyinfo = keyinfo_dict[instance_key]
                    frame_id_within_instance = keyinfo[0]
                    starting_id, ending_id = self.instance_to_frameid(instance_key)
                    key_frame_id = starting_id + frame_id_within_instance
                    # print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(),   ' + str(instance_key) + ', ' + str(frame_id_within_instance) + ' ---> ' + str((starting_id, ending_id)) + ' ---> ' + str(key_frame_id))
                    instance_frame_correspondence_dict[instance_key] = key_frame_id

                # Construct key frames representation.
                key_frames_fnxl = fnxl.FrameNameXyList()
                for instance_key in keyinfo_dict.keys():
                    # Keyinfo is a tuple (frame_id_within_instance, number_of_heliosats, box_coords, heliostat_names)
                    keyinfo = keyinfo_dict[instance_key]
                    number_of_heliosats = keyinfo[1]
                    box_coords = keyinfo[2]
                    heliostat_names = keyinfo[3]
                    frame_id = instance_frame_correspondence_dict[instance_key]
                    # Construct key frame entry.
                    (key_frame_id, list_of_name_polygons) = self.construct_key_frame_entry(
                        instance_key, keyinfo_dict, instance_frame_correspondence_dict
                    )
                    key_frames_fnxl.add_list_of_name_xy_lists(key_frame_id, list_of_name_polygons)
                # for key_frame_id_key in key_frame_dict.keys():
                #     print('In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(),   ' + str(key_frame_id_key) + ':  ' + str(key_frame_dict[key_frame_id_key]))
            else:
                print(
                    "In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), reading already-converted keyframe specification..."
                )
                # Read keyframe file after editing.
                print(
                    "In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), reading key frames: ",
                    self.input_edited_key_frames_dir_body_ext,
                )
                key_frames_fnxl = fnxl.FrameNameXyList()
                key_frames_fnxl.load(self.input_edited_key_frames_dir_body_ext)
                # Confirm what was read.
                print("KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), key frame specfication read:")
                key_frames_fnxl.print(max_keys=7, max_value_length=200, indent=4)

            # Write key frame file.
            print(
                "In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), writing key frames: ",
                self.key_frames_dir_body_ext,
            )
            key_frames_fnxl.save(self.key_frames_dir_body_ext)

            # Write summary information.
            # Statistics.
            summary_dict = {}
            summary_dict["n_key_frames"] = key_frames_fnxl.number_of_frames()
            print(
                "In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), writing key frame summary statistics..."
            )
            ft.write_dict_file("key frame summary statistics", self.output_data_dir, self.dict_body, summary_dict)
            # Heliostats per key frame.
            heliostats_per_key_frame_dict = key_frames_fnxl.heliostats_per_frame()
            print(
                "In KeyFramesGivenManual.convert_and_save_original_keyinfo_file(), writing heliostats per key frame:",
                os.path.join(self.output_data_dir, self.hpkf_body_ext),
            )
            ft.write_dict_file(None, self.output_data_dir, self.hpkf_body, heliostats_per_key_frame_dict)

    def read_keyinfo(self):
        # Load original keyinfo file.
        print("In KeyFramesGivenManual.read_keyinfo(), reading file:", self.input_original_keyinfo_dir_body_ext)
        keyinfo_stream = open(self.input_original_keyinfo_dir_body_ext, "r")
        keyinfo_lines_with_newline = keyinfo_stream.readlines()
        keyinfo_lines = [line.rstrip("\n") for line in keyinfo_lines_with_newline]
        # Parse the lines, filling in the keyframes dictionary.
        keyinfo_dict = {}
        for line in keyinfo_lines:
            if line != "None" and line != "":
                # Parse the line.
                parts = line.split(" ")
                instance, frameid, nhel = parts[0], parts[1], parts[2]
                hel_names = [str(parts[x]) for x in range(3, 3 + int(nhel))]
                boxpoints = [float(parts[x]) for x in range(3 + int(nhel), len(parts))]
                # If the manual selection was done with an HD version of a higher-resolution
                # video, then we need tocorrect the box points.
                if self.input_original_keyinfo_boxHD:
                    if self.input_original_keyinfo_hd_shape == None:
                        print(
                            "ERROR: In KeyFramesGivenManual.read_keyinfo(), input_original_keyinfo_boxHD == True, but input_original_keyinfo_hd_shape == None."
                        )
                        assert False
                    scale_height = self.hd_shape[0] / self.height
                    scale_width = self.hd_shape[1] / self.width
                    for indx in range(0, len(boxpoints)):
                        if indx % 2 == 0:  # column
                            boxpoints[indx] *= 1 / scale_width
                        else:
                            boxpoints[indx] *= 1 / scale_height
                        boxpoints[indx] = int(boxpoints[indx])
                # Construct the keyinfo dictionary entry.
                keyinfo_dict[instance] = (int(frameid), int(nhel), boxpoints, hel_names)
        # Return.
        return keyinfo_dict

    def instance_to_frameid(self, instance):
        # Parse instance.
        secs = int(instance.split(":")[-1])
        mins = int(instance.split(":")[-2])
        # Compute frame id at beginning and end of instance (1 second).
        total_secs = mins * 60 + secs
        starting_id = total_secs * self.fps - 1
        ending_id = starting_id + self.fps
        # Return.
        return starting_id, ending_id

    def construct_key_frame_entry(self, instance_key, keyinfo_dict, instance_frame_correspondence_dict):
        # Fetch information.
        key_frame_id = instance_frame_correspondence_dict[instance_key]
        keyinfo = keyinfo_dict[
            instance_key
        ]  # Tuple (frame_id_within_instance, number_of_heliosats, box_coords, heliostat_names)
        number_of_heliostats = keyinfo[1]
        box_coords = keyinfo[2]
        heliostat_names = keyinfo[3]
        # Construct entry.
        entry = self.construct_key_frame_entry_aux(number_of_heliostats, heliostat_names, box_coords)
        # Return.
        return key_frame_id, entry

    def construct_key_frame_entry_aux(self, number_of_heliostats, heliostat_names, box_coords):
        """
        A key frame entry is a list:

            [ [hel_name_1, polygon_1], [hel_name_2, polygon_2], [hel_name_3, polygon_3], ... ]

        where each polygon_i is:

            [ [x1, x1], [x2, y2], [x3, y3], ... ]

        where the (x,y) coordinates are floats defined in image space.  The vertices are ordered
        counter-clockwise, if you view the heliostat looking at your reflection in the mirror.
        """
        # Check input.
        if number_of_heliostats != len(heliostat_names):
            print(
                "ERROR: In KeyFramesGivenManual.construct_key_frame_entry_aux(), number_of_heliostats="
                + str(number_of_heliostats)
                + " is not equal to len(heliostat_names)="
                + str(len(heliostat_names))
                + "."
            )
            assert False
        if (number_of_heliostats * 4) != len(box_coords):
            print(
                "ERROR: In KeyFramesGivenManual.construct_key_frame_entry_aux(), number_of_heliostats="
                + str(number_of_heliostats)
                + " is not equal to 1/4 len(box_coords)="
                + str(len(box_coords))
                + "."
            )
            assert False
        # Group box_coords into boxes.
        bboxes = []
        for idx in range(0, number_of_heliostats):
            bbox_idx = idx * 4
            bbox = [box_coords[bbox_idx], box_coords[bbox_idx + 1], box_coords[bbox_idx + 2], box_coords[bbox_idx + 3]]
            bboxes.append(bbox)
        # Assemble the entry.
        list_of_name_polygons = []
        for hel_name, bbox in zip(heliostat_names, bboxes):
            polygon = self.polygon_given_bbox(bbox)
            name_polygon = [hel_name, polygon]
            list_of_name_polygons.append(name_polygon)
        # Return.
        return list_of_name_polygons

    def polygon_given_bbox(self, bbox):
        """
        A bbox is a 4-tuple:
           [x1, y1, x2, y2]
        where (x1,y1) and (x2,y2) are corners of the bounding box.  They are top-left (tl) and bottom-right (br),
        but the order is not guaranteed.

        A polygon is a list of (x,y) points:
           [ [x_a, y_a], [x_b, y_b], ... ]

        In general, a polygon may have any number of vertices greater than or equal to 3, but of course the polygon
        resulting from a bounding box always has four vertices.

        The vertex order is chosen so that the vertices are ordered counter-clockwise, assuming a standard
        coordinate system where y is up.  In an image coordinate system where y is down, the standard math
        still works without revision, but when you view the vertex sequence overlaid on the image, they will
        appear clockwise.
        """
        # Extract coordinates.
        x_1 = bbox[0]
        y_1 = bbox[1]
        x_2 = bbox[2]
        y_2 = bbox[3]
        # Check validity.
        if x_1 == x_2:
            print("ERROR: In KeyFramesGivenManual.polygon_given_bbox(), corner x values are equal.  x_1 == x_2 ==", x_1)
            assert False
        if y_1 == y_2:
            print("ERROR: In KeyFramesGivenManual.polygon_given_bbox(), corner y values are equal.  y_1 == y_2 ==", y_1)
            assert False
        # Identify boundaries.
        x_min = min(x_1, x_2)
        x_max = max(x_1, x_2)
        y_min = min(y_1, y_2)
        y_max = max(y_1, y_2)
        # Construct vertices.
        ll = [x_min, y_min]  # lower-left
        lr = [x_max, y_min]  # lower-right
        ur = [x_max, y_max]  # upper-right
        ul = [x_min, y_max]  # upper-left
        # Return.
        return [ll, lr, ur, ul]

    # LOAD RESULT

    def read_key_frames(self):
        print("In KeyFramesGivenManual.read_key_frames(), reading key frames: ", self.key_frames_dir_body_ext)
        self.key_frames_fnxl = fnxl.FrameNameXyList()
        self.key_frames_fnxl.load(self.key_frames_dir_body_ext)
        # Confirm what was read.
        print("In KeyFramesGivenManual.read_key_corners(), key corners read:")
        self.key_frames_fnxl.print(max_keys=7, max_value_length=200, indent=4)

    def read_data(self):
        print("In KeyFramesGivenManual.read_data(), reading frame statistics: ", self.dict_dir_body_ext)
        self.frame_statistics_dict = ft.read_dict(self.dict_dir_body_ext)
        print("In KeyFramesGivenManual.read_data(), reading heliostats per key frame: ", self.hpkf_dir_body_ext)
        self.hpkf_dict = ft.read_dict(self.hpkf_dir_body_ext)
        # Confirm what was read.
        print("In KeyFramesGivenManual.read_data(), heliostats per key frame read:")
        dt.print_dict(self.hpkf_dict, max_keys=7, max_value_length=200, indent=4)

    # RENDER RESULT

    def render(self):
        if self.render_control.draw_key_frames:
            print("In KeyFramesGivenManual.render(), rendering frames with key corners...")
            # Descriptive strings.
            title_name = "Key Frame"
            context_str = "KeyFramesGivenManual.render()"
            # Required suffix strings.
            fig_suffix = "_key_frame_fig"
            delete_suffix = ".JPG" + fig_suffix + ".png"
            # Prepare directory.
            upc.prepare_render_directory(self.output_render_dir, delete_suffix, self.render_control)
            # Setup annotation styles.
            style_dict = {}
            style_dict["point_seq"] = rcps.outline(
                color=self.render_control.key_frame_polygon_color,
                linewidth=self.render_control.key_frame_polygon_linewidth,
            )
            style_dict["text"] = rctxt.RenderControlText(
                horizontalalignment=self.render_control.key_frame_label_horizontalalignment,
                verticalalignment=self.render_control.key_frame_label_verticalalignment,
                fontsize=self.render_control.key_frame_label_fontsize,
                fontstyle=self.render_control.key_frame_label_fontstyle,
                fontweight=self.render_control.key_frame_label_fontweight,
                color=self.render_control.key_frame_label_color,
            )
            # Draw the frames.
            self.key_frames_fnxl.draw_frames(
                self.single_processor,
                self.log_dir_body_ext,
                title_name,
                context_str,
                fig_suffix,
                self.input_video_body,
                self.input_frame_dir,
                self.input_frame_id_format,
                self.output_render_dir,
                dpi=self.render_control.key_frame_dpi,
                close_xy_list=True,
                style_dict=style_dict,
                crop=self.render_control.key_frame_crop,
            )


if __name__ == "__main__":
    # Execution control.
    # force_construction                   = True #False
    specific_frame_ids = None  # [777, 897, 1035] #None
    single_processor = False
    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/log/KeyFrames_log.txt"
    )
    # Input/output sources.
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_original_keyinfo_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/data/original/vflight_12_2020_keyinfo_with_instances_edited.txt"
    )
    input_original_keyinfo_boxHD = False
    input_original_keyinfo_hd_shape = None
    input_edited_key_frames_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/data/original/DJI_427t_428_429_key_frames_CONVERTED_FROM_ORIGINAL_CORRECTED.csv"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/130_KeyFrames/mavic_zoom/render/"
    )
    # Render control.
    render_control = rckfgm.default()

    key_frames_object = KeyFramesGivenManual(  # Execution control.
        # force_construction,
        specific_frame_ids,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        input_video_dir_body_ext,
        input_original_keyinfo_dir_body_ext,
        input_original_keyinfo_boxHD,
        input_original_keyinfo_hd_shape,
        input_edited_key_frames_dir_body_ext,
        input_frame_dir,
        input_frame_id_format,
        output_data_dir,
        output_render_dir,
        # Render control.
        render_control,
    )
