"""
A data structure for associating image frames with their contained heliostats and associated (x,y) points in image space.



"""

import copy
import csv
import io
from types import NoneType
from typing import Union, NewType
from cv2 import cv2 as cv
import logging
from multiprocessing import Pool
import os

import opencsp.common.lib.file.CsvInterface as csvi
import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.render.PlotAnnotation as pa
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as logt
import opencsp.app.ufacets.helio_scan.lib.ufacet_pipeline_frame as upf

XL = NewType("XyList", tuple[str, list[list[int]]])
# NXL = NewType("NameXyList", list[str, XL])
NXL = NewType("NameXyList", list[tuple[str, list[list[int]]]])
# FNXL = NewType("FNXL", dict[int, NXL)
FNXL = NewType("FrameNameXyList", dict[int, list[tuple[str, list[list[int]]]]])


class FrameNameXyList(csvi.CsvInterface):
    """
    Class for holding a set of frames from a UFACET scan video, with associated contained heliostats,
    and an (x,y) point list associeated with each heliostat.

    The result is a dictionary of the form:

          frame_id_a:  [ [hel_name_a1, xy_list_a1], [hel_name_a2, xy_list_a2], ...]
          frame_id_b:  [ [hel_name_b1, xy_list_b1], [hel_name_b2, xy_list_b2], [hel_name_b3, xy_list_b3], ...]
          frame_id_c:  [ [hel_name_c1, xy_list_c1], ...]
          frame_id_d:  [ [hel_name_d1, xy_list_d1], [hel_name_d2, xy_list_d2], ...]
          ...

    where the number of frames is arbritrary, and the number of heliostats per frame is arbitrary.

    Each xy_list_i is a list of vertices:

        xy_list_i = [ [x_i1, y_i1], [x_i2, y_i2], [x_i3, y_i3], ... ]

    where the (x,y) coordinates are in image space.

    The semantics of the xy_list can vary.  Here are example uses in the UFACET image
    processing pipeline:

        - Key frames.  The dictionary only contains key frames, and the associated heliostat/xy_list pairs
          denote the heliostats known to be in the key frame, and a closed polygon indicating its expected
          position in the image.  Only one instance.

        - Found corners.  The dictionary only contains key frames, and the associated heliostat/xy_list pairs
          denote the facet corners found by image processing, for each heliostat in the frame.  Only one instance.

        - Tracked corners.  The dictionary only contains frames preceding and following a particular key
          frame, and associated heliostat/xy_list pairs denote the facet corners found by forward/backward
          predict-and-confirm tracking, asscoiated with each heliostat.  An instance for each key frame.

        - All corners.  The dictionary contains all frames in the video, and associated heliostat/xy_list
          pairs corresponding to all heliostats found in that frame by any forward/backward tracking operation.
          Only one instance.

        - Heliostat corners.  The dictionary only contains frames in which a particular heliostat was seen,
          and the associated heliostat/xy_list pairs denote the facet corners found for that heliostat.
          This assembles input for heliostat 3-d reconstruction.  An instance for each heliostat.

    Here is an excerpt from a real example key frame dictionary:

        ...
        16824:  [['8E7', [[1113.6, 625.92], [2073.6, 625.92], [2073.6, 1639.6799999999998], [1113.6, 1639.6799999999998]]], ['8E8', [[2357.7599999999998, 384.0], [3110.4, 384.0], [3110.4, 1286.3999999999999], [2357.7599999999998, 1286.3999999999999]]]]
        16856:  [['9E7', [[660.48, 314.88], [1608.96, 314.88], [1608.96, 1221.12], [660.48, 1221.12]]], ['9E9', [[2837.7599999999998, -3.84], [3409.92, -3.84], [3409.92, 791.04], [2837.7599999999998, 791.04]]]]
        16918:  [['10E8', [[1386.24, 238.07999999999998], [2277.12, 238.07999999999998], [2277.12, 1128.96], [1386.24, 1128.96]]], ['10E9', [[2457.6, 46.08], [3121.92, 46.08], [3121.92, 879.36], [2457.6, 879.36]]]]
        17723:  [['10E8', [[1501.44, 153.6], [2269.44, 153.6], [2269.44, 1006.0799999999999], [1501.44, 1006.0799999999999]]], ['10E9', [[2461.44, 0.0], [3056.64, 0.0], [3056.64, 714.24], [2461.44, 714.24]]]]
        17786:  [['9E8', [[952.3199999999999, 284.15999999999997], [1777.9199999999998, 284.15999999999997], [1777.9199999999998, 1079.04], [952.3199999999999, 1079.04]]]]
        ...

    In computer memory this is represented as a dictionary with frame_id keys and values that are
    nested lists comprising [hel_name xy_list] pairs, where each xy_list is a list of [x,y] pairs.

    On disk this is stored in a csv format, with rigorous write/read procedures contained below.

    """

    def __init__(self):
        # Primary container.
        self.dictionary: FNXL = {}

    # ACCESS

    def sorted_frame_id_list(self):
        """
        Returns all frame_ids, in ascending order.  You can apply subscripts.
        """
        frame_id_list = list(self.dictionary.keys())
        frame_id_list.sort()
        return frame_id_list

    def frame_ids_unsorted(self):
        """
        Returns all frame_ids; order is undefined.  You cannot apply subscripts.
        """
        return self.dictionary.keys()

    def list_of_name_xy_lists(self, frame_id):
        return self.dictionary[frame_id]

    def matching_name_xy_list(self, frame_id, name) -> Union[NoneType, NXL]:
        list_of_name_xy_lists = self.dictionary[frame_id]
        # Implementing this as a nested dictionary would be nice.  I don't have time for that today.
        for name_xy_list in list_of_name_xy_lists:
            if name == name_xy_list[0]:
                return name_xy_list
        # Match not found.
        return None

    def number_of_frames(self):
        return len(self.dictionary.keys())

    def number_of_points(self):
        total_points = 0
        for frame_id in self.frame_ids_unsorted():
            list_of_name_xy_lists = self.list_of_name_xy_lists(frame_id)
            for name_xy_list in list_of_name_xy_lists:
                name = name_xy_list[0]
                xy_list = name_xy_list[1]
                for xy in xy_list:
                    total_points += 1
        return total_points

    def number_of_non_flag_points(self):
        """
        Doesn't count (-1,-1) points.
        """
        total_points = 0
        for frame_id in self.frame_ids_unsorted():
            list_of_name_xy_lists = self.list_of_name_xy_lists(frame_id)
            for name_xy_list in list_of_name_xy_lists:
                name = name_xy_list[0]
                xy_list = name_xy_list[1]
                for xy in xy_list:
                    if (xy[0] != -1) and (xy[1] != -1):
                        total_points += 1
        return total_points

    def number_of_points_in_bounds(self, bounding_box):
        """
        Bounding box is [xy_min, xy_max], where each xy is [x,y].
        """
        xy_min = bounding_box[0]
        xy_max = bounding_box[1]
        x_min = xy_min[0]
        y_min = xy_min[1]
        x_max = xy_max[0]
        y_max = xy_max[1]
        total_points = 0
        for frame_id in self.frame_ids_unsorted():
            list_of_name_xy_lists = self.list_of_name_xy_lists(frame_id)
            for name_xy_list in list_of_name_xy_lists:
                name = name_xy_list[0]
                xy_list = name_xy_list[1]
                for xy in xy_list:
                    x = xy[0]
                    y = xy[1]
                    if ((x_min <= x) and (x <= x_max)) and ((y_min <= y) and (y <= y_max)):
                        total_points += 1
        return total_points

    def heliostats_per_frame(self) -> dict[int, int]:
        """
        Returns a dict with same keys, but values are the number of heliostats associated with each frame.
        """
        result = {}
        for frame_id in self.dictionary.keys():
            list_of_name_xy_lists = self.dictionary[frame_id]
            n_heliostats = len(list_of_name_xy_lists)
            result[frame_id] = n_heliostats
        return result

    def names_in_frame(self) -> dict[int, list[str]]:
        """
        Returns a dict with same keys, but the value for each key is a list of the names in the associated name_xy_list.
        """
        result = {}
        for frame_id in self.dictionary.keys():
            list_of_name_xy_lists = self.dictionary[frame_id]
            name_list = [name_xy_list[0] for name_xy_list in list_of_name_xy_lists]
            result[frame_id] = name_list
        return result

    def points_per_frame(self) -> dict[int, int]:
        """
        Returns a dict with same keys, but values are the total number of (x,y) points associated with each frame.
        """
        result = {}
        for frame_id in self.dictionary.keys():
            list_of_name_xy_lists = self.dictionary[frame_id]
            n_points = 0
            for name_xy_list in list_of_name_xy_lists:
                xy_list = name_xy_list[1]
                n_points += len(xy_list)
            result[frame_id] = n_points
        return result

    # MODIFICATION

    def add_list_of_name_xy_lists(self, frame_id: int, input_list_of_name_xy_lists: NXL):
        """
        Add a list of [name, xy_list] pairs to the dictionary, under the given frame_id key.
        Assumes the frame_id is not already there.
        """
        if frame_id in self.dictionary:
            print(
                "ERROR: In FrameNameXyList.add_list_of_name_xy_lists(), attempt to add frame_id="
                + str(frame_id)
                + ", which is already present."
            )
            assert False
        self.dictionary[frame_id] = input_list_of_name_xy_lists

        return self

    def merge_list_of_name_xy_lists(
        self,
        frame_id: int,
        input_list_of_name_xy_lists: NXL,
        warn_if_common_name=True,
        skip_if_common_name=True,
        error_if_common_name=False,
    ):
        """
        Add a list of [name, xy_list] pairs to the dictionary, under the given frame_id key.
        If frame_id is not already there, it simply add it.
        If the frame is already there, then merge the list of [name, xy_list] entries:
           - If the name is not present, then add the [name, xy_list] to the existing list.
           - If the name is present, then optionally throw an error, because we don't know which xy_list is valid, or if it simply extends the xy_list.
        """
        if frame_id not in self.dictionary:
            self.dictionary[frame_id] = copy.deepcopy(input_list_of_name_xy_lists)
        else:
            existing_list_of_name_xy_lists = self.dictionary[frame_id]
            existing_name_list = [name_xy_list[0] for name_xy_list in existing_list_of_name_xy_lists]
            for input_name_xy_list in input_list_of_name_xy_lists:
                input_name_xy_list_copy = copy.deepcopy(input_name_xy_list)
                input_name_copy = input_name_xy_list_copy[0]
                input_xy_list_copy = input_name_xy_list_copy[1]
                if input_name_copy not in existing_name_list:
                    existing_list_of_name_xy_lists.append(input_name_xy_list_copy)
                else:
                    if error_if_common_name:
                        print(
                            "ERROR: In FrameNameXyList.merge_list_of_name_xy_lists(), attempt to add xy_list for frame_id="
                            + str(frame_id)
                            + ' and name="'
                            + str(input_name_copy)
                            + '", both of which are already present.'
                        )
                        assert False
                    if warn_if_common_name:
                        if skip_if_common_name:
                            suffix = "  Skipping."
                        else:
                            suffix = ""
                        print(
                            "WARNING: In FrameNameXyList.merge_list_of_name_xy_lists(), attempt to add xy_list for frame_id="
                            + str(frame_id)
                            + ' and name="'
                            + str(input_name_copy)
                            + '", both of which are already present.'
                            + suffix
                        )
                    # Proceed.
                    if not skip_if_common_name:
                        if warn_if_common_name:
                            print(
                                "WARNING: In FrameNameXyList.merge_list_of_name_xy_lists(), adding to already-existing (frame_id="
                                + str(frame_id)
                                + ', name="'
                                + str(input_name_copy)
                                + '") pair.'
                            )
                        for existing_name_xy_list in existing_list_of_name_xy_lists:
                            existing_name = existing_name_xy_list[0]
                            existing_xy_list = existing_name_xy_list[1]
                            if existing_name == input_name_copy:
                                existing_xy_list += input_xy_list_copy

    # READ

    def load(self, input_dir_body_ext: str):  # "fnxl" abbreviates "FrameNameXyList"
        """
        Reads the stored FrameNameXyList file, and adds it to the current dictionary.
        If data is already already present in this FrameNameXyList, extends the current
        content as follows:

          - If a frame is not already present, adds the frame, and the associated hel_name/xy_list pair.

          - If the loaded frame is present, but the heliostat of the loaded hel_name/xy_list pair is not
            already associated with the frame, then add the hel_name/xy_list pair to the frame entry.

          - If the loaded frame is present, and it already contains the heliostat with some associated
            (x,y) points, extend the existing xy_list for the heliostat by adding the newly loaded (x,y)
            points.  Do not check for duplicate points, simply append the loaded points to the points
            that are already there.  Also do not group the points -- thus if the points are supposed
            to represent a polygon, they are not combined geometrically, but rather by simply
            concatenating the point lists.
        """
        # print('In FrameNameXyList.load(), loading input file: ', input_dir_body_ext)
        # Check if the input file exists.
        if not ft.file_exists(input_dir_body_ext):
            raise OSError("In FrameNameXyList.load(), file does not exist: " + str(input_dir_body_ext))
        # Open and read the file.
        with open(input_dir_body_ext, newline="") as input_stream:
            reader = csv.reader(input_stream, delimiter=",")
            for input_row in reader:
                self.add_row_to_dictionary(input_row)

    def add_row_to_dictionary(self, input_row: list[str]):
        frame_id = int(input_row[0])
        list_of_name_xylists = self.parse_row_list_of_name_xylists(input_row[1:])
        # Add the entry.
        self.dictionary[frame_id] = list_of_name_xylists

    def parse_row_list_of_name_xylists(self, input_row_remainder: list[str]) -> NXL:
        n_hel = int(input_row_remainder[0])
        if n_hel == 0:
            return []
        else:
            return self.parse_row_list_of_name_xylists_aux(input_row_remainder[1:], n_hel, return_remainder=False)

    def from_csv_line(self, data: list[str]) -> tuple[NXL, list[str]]:
        n_hel = int(data[0])
        if n_hel == 0:
            return [], data[1:]
        else:
            return self.parse_row_list_of_name_xylists_aux(data[1:], n_hel, return_remainder=True)

    def parse_row_list_of_name_xylists_aux(
        self, input_row_list_of_name_xylists: list[str], n_hel: int, return_remainder=False
    ) -> NXL | tuple[NXL, list[str]]:
        # Fetch this name_xylist's heliostat name and the number of points in its xylist.
        hel_name = input_row_list_of_name_xylists[0]
        n_points = int(input_row_list_of_name_xylists[1])
        # Parse the xylist.
        xylist = []
        for idx in range(0, n_points):
            idx_x = 2 + (2 * idx)
            idx_y = idx_x + 1
            vertex_str = [input_row_list_of_name_xylists[idx_x], input_row_list_of_name_xylists[idx_y]]
            x_str = vertex_str[0]
            y_str = vertex_str[1]
            x = float(x_str)
            y = float(y_str)
            xylist.append([x, y])
            last_idx = idx_y
        name_xylist = [hel_name, xylist]
        # If there are more heliostats, then recurse.
        retlist = input_row_list_of_name_xylists[last_idx + 2 :]
        if n_hel > 1:
            further_name_xylists = self.parse_row_list_of_name_xylists_aux(
                input_row_list_of_name_xylists[last_idx + 1 :], n_hel - 1
            )
        else:
            further_name_xylists = []
            retlist = input_row_list_of_name_xylists[last_idx + 1 :]
        # Assemble the result, including the name_xylist we assembled here.
        list_of_name_xylists = [name_xylist] + further_name_xylists
        # Return.
        if return_remainder:
            return list_of_name_xylists, retlist
        else:
            return list_of_name_xylists

    # WRITE

    def save(self, output_dir_body_ext: str):
        # Extract path components.
        output_dir, output_body, output_ext = ft.path_components(output_dir_body_ext)
        # Create output directory if necessary.
        ft.create_directories_if_necessary(output_dir)

        # Write the frame dictionary in a structured format.
        # print('In FrameNameXyList.save(), saving file:', output_dir_body_ext)
        with open(output_dir_body_ext, "w") as output_stream:
            # Assemble the row.
            for frame_id in dt.sorted_keys(self.dictionary):
                output_stream.write(self.to_csv_line(frame_id=frame_id))

    @staticmethod
    def csv_header(delimeter=",") -> str:
        return delimeter.join(["frame_id", "n_heliostats", "hel_name", "n_vertices", "vert_1_x", "vert_1_y"])

    def to_csv_line(self, delimeter=",", frame_id=None):
        """Converts a single frame id to a string."""
        if frame_id == None:
            if self.number_of_frames() != 1:
                raise RuntimeError(
                    f"Expected 1 frame in this {self.__class__.__name__} but found {self.number_of_frames()}!"
                )
            frame_id = self.dictionary.keys()[0]

        with io.StringIO() as output_stream:
            with csv.writer(output_stream, delimiter=delimeter) as writer:
                # Assemble the row.
                row_items = [frame_id]
                name_polygon_list = self.dictionary[frame_id]
                n_heliostats = len(name_polygon_list)
                row_items.append(n_heliostats)
                for name_polygon_list in name_polygon_list:
                    hel_name = name_polygon_list[0]
                    polygon = name_polygon_list[1]
                    n_vertices = len(polygon)
                    row_items.append(hel_name)
                    row_items.append(n_vertices)
                    for vertex in polygon:
                        if vertex is None:
                            # This can occur for FrameNameXyList objects holding confirmed points.
                            # Convention is to represent missing points with (-1,-1).
                            # I believe this is expected by OpenCV's reconstruction routines.
                            row_items.append(-1)  # x
                            row_items.append(-1)  # y
                        else:
                            row_items.append(vertex[0])  # x
                            row_items.append(vertex[1])  # y
                # Write the row.
                writer.writerow(row_items)
            return output_stream.getvalue()

    # RENDER

    def print(
        self,
        max_keys=10,  # Maximum number of keys to print.  Elipsis after that.
        max_value_length=70,  # Maximum value length to print.  Elipsis after that.
        indent=None,
    ):  # Number of blankss to print at the beginning of each line.
        # Print.
        dt.print_dict(self.dictionary, max_keys=max_keys, max_value_length=max_value_length, indent=indent)

    def draw_frames(
        self,
        single_processor,  # Boolean flag indicating whether to execute single- or multi-processor.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        title_name,  # String describing this topic.  Video name and frame automatically added.
        context_str,  # String describing calling context, for status messages.
        fig_suffix,  # Added to video name to produce figure filename body.
        input_video_body,  # Base name of the flight video, used for title and output filename.
        input_frame_dir,  # Directory containing the full list of full-size frames.
        input_frame_id_format,  # Format used to include frame_id numbers in frame filenames.
        output_render_dir,  # Directory to place the output figures.
        dpi,  # Dots per inch to write figures.
        close_xy_list,  # Boolean.  Set to true if drawing pt+list as a closed polygon.
        style_dict,  # Dictionary of render control objects.  Currently expects 'point_seq'
        # and 'text' entries.  Future may include per-heliostat styles.
        crop,
    ):  # Boolean indicating whether to suppress annotations outside image frame.
        """
        Generates annotated figure for each frame, writing to the designated output directory.
        """
        # Set class member variables so we can have a single-argument draw_frame() function for multi-processing.
        self.draw_frame_title_name = title_name
        self.draw_frame_context_str = context_str
        self.draw_frame_fig_suffix = fig_suffix
        self.draw_frame_input_video_body = input_video_body
        self.draw_frame_input_frame_dir = input_frame_dir
        self.draw_frame_input_frame_id_format = input_frame_id_format
        self.draw_frame_output_render_dir = output_render_dir
        self.draw_frame_dpi = dpi
        self.draw_frame_close_xy_list = close_xy_list
        self.draw_frame_style_dict = style_dict
        self.draw_frame_crop = crop

        # Call draw_frame(), using appropriate execution mode.
        if single_processor == True:
            print("In In FrameNameXyList.draw_frames(), starting frame rendering (single processor)...")
            for frame_id in self.dictionary.keys():
                self.draw_frame(frame_id)
            print("In In FrameNameXyList.draw_frames(), frame rendering done.")

        elif single_processor == False:
            print("In In FrameNameXyList.draw_frames(), starting frame rendering (multi-processor)...")
            logger = logt.multiprocessing_logger(log_dir_body_ext, level=logging.INFO)
            logger.info("================================= Execution =================================")
            with Pool(25) as pool:
                pool.map(self.draw_frame, self.dictionary.keys())
            print("In In FrameNameXyList.draw_frames(), frame rendering done.")

        else:
            print("ERROR: In FrameNameXyList.draw_frames(), unexpected value single_processor =", str(single_processor))
            assert False

    def draw_frame(self, frame_id: int):
        # Fetch name_xy_list from dictionary.
        name_xy_list = self.dictionary[frame_id]
        # Fetch parameters held as class member variables.
        title_name = self.draw_frame_title_name
        context_str = self.draw_frame_context_str
        fig_suffix = self.draw_frame_fig_suffix
        input_video_body = self.draw_frame_input_video_body
        input_frame_dir = self.draw_frame_input_frame_dir
        input_frame_id_format = self.draw_frame_input_frame_id_format
        output_render_dir = self.draw_frame_output_render_dir
        dpi = self.draw_frame_dpi
        close_xy_list = self.draw_frame_close_xy_list
        style_dict = self.draw_frame_style_dict
        crop = self.draw_frame_crop
        # Construct frame file name.
        key_frame_body_ext = upf.frame_file_body_ext_given_frame_id(input_video_body, frame_id, input_frame_id_format)
        # Load frame.
        input_dir_body_ext = os.path.join(input_frame_dir, key_frame_body_ext)
        frame_img = cv.imread(input_dir_body_ext)
        # Construct annotations.
        annotation_list = []
        for hel_name_xy_list in name_xy_list:
            hel_name = hel_name_xy_list[0]
            xy_list = hel_name_xy_list[1]
            active_xy_list = self.remove_flag_points(xy_list)
            label_xy = g2d.label_point(active_xy_list)
            active_xy_list_2 = active_xy_list.copy()
            if close_xy_list:
                if len(active_xy_list_2) < 3:
                    print(
                        "ERROR: In FrameNameXyList.draw_frame(), attempting to close an xy_list with too few vertices; len(xy_list) =",
                        len(active_xy_list_2),
                    )
                    assert False
                first_pt = active_xy_list_2[0]
                active_xy_list_2.append(first_pt)
            annotation_list.append(pa.PlotAnnotation("point_seq", active_xy_list_2, None, style_dict["point_seq"]))
            annotation_list.append(pa.PlotAnnotation("text", [label_xy], hel_name, style_dict["text"]))
        # Prepare crop_box.
        # Crop box is [[x_min, y_min], [x_max, y_max]] or None.
        if crop:
            max_row = frame_img.shape[0] - 1
            max_col = frame_img.shape[1] - 1
            crop_box = [[0, 0], [max_col, max_row]]
        else:
            crop_box = None

        # Draw.
        ip.plot_image_figure(
            frame_img,
            rgb=False,
            title=(str(input_video_body) + " Frame " + str(frame_id) + ", " + title_name),
            annotation_list=annotation_list,
            crop_box=crop_box,
            context_str=context_str,
            save=True,
            output_dir=output_render_dir,
            output_body=(
                key_frame_body_ext + fig_suffix
            ),  # Add the suffix on top of the existing extension.  plot_image_figure() will add ".png"
            dpi=dpi,
            include_figure_idx_in_filename=False,
        )

    def remove_flag_points(self, input_xy_list: XL) -> XL:
        """
        Removes all (-1, -1) points, which are used to indicate "None" confirmed points.
        """
        active_xy_list = []
        for xy in input_xy_list:
            if (xy[0] != -1) or (xy[1] != -1):
                active_xy_list.append(xy)
        return active_xy_list


# HELPER FUNCTIONS


def construct_merged_copy(input_fnxl_list: FNXL) -> FNXL:  # A list of FrameNameXyList objects.
    """
    Constructs a new FrameNameXyList object, combining the entries of the input FrameNameXyList objects, without modifying them.
    """
    new_fnxl = FrameNameXyList()
    for input_fnxl in input_fnxl_list:
        for frame_id in input_fnxl.frame_ids_unsorted():
            input_name_xy_list = input_fnxl.list_of_name_xy_lists(frame_id)
            new_name_xy_list = copy.deepcopy(input_name_xy_list)
            new_fnxl.add_list_of_name_xy_lists(frame_id, new_name_xy_list)
    return new_fnxl
