"""
A data structure for associating heliostat names with all image frames in which they were found, and associated (x,y) points in image space.



"""

import copy
import csv
from cv2 import cv2 as cv
import logging
from multiprocessing import Pool
import os

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.render.PlotAnnotation as pa
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import ufacet_pipeline_frame as upf


class NameFrameXyList:
    """
    Class for holding a set of heliostats found in a UFACET scan video, with associated contained frames in which they were found,
    and an (x,y) point list associeated with each frame.

    The result is a dictionary of the form:

          hel_name_a:  [ [frame_id_a1, xy_list_a1], [frame_id_a2, xy_list_a2], ...]
          hel_name_b:  [ [frame_id_b1, xy_list_b1], [frame_id_b2, xy_list_b2], [frame_id_b3, xy_list_b3], ...]
          hel_name_c:  [ [frame_id_c1, xy_list_c1], ...]
          hel_name_d:  [ [frame_id_d1, xy_list_d1], [frame_id_d2, xy_list_d2], ...]
          ...

    where the number of heliostat names is arbritrary, and the number of frames per heliostat is arbitrary.

    Each xy_list_i is a list of vertices:

        xy_list_i = [ [x_i1, y_i1], [x_i2, y_i2], [x_i3, y_i3], ... ]

    where the (x,y) coordinates are in image space.

    The semantics of the xy_list can vary.  Here are example uses in the UFACET image
    processing pipeline:

        - Heliostat Tracks.  The dictionary contains all heliostats found in the video, and associated
          frame_id/xy_list pairs corresponding to all frames where that heliostat was found by any
          forward/backward tracking operation.
          Only one instance.

    In computer memory this is represented as a dictionary with hel_name keys and values that are
    nested lists comprising [frame_id xy_list] pairs, where each xy_list is a list of [x,y] pairs.

    On disk this is stored in a csv format, with rigorous write/read procedures contained below.

    """

    def __init__(self):
        # Primary container.
        self.dictionary = {}

    # ACCESS

    def sorted_hel_name_list(self):
        """
        Returns all hel_names, in ascending order.  You can apply subscripts.
        """
        hel_name_list = list(self.dictionary.keys())
        hel_name_list.sort()
        return hel_name_list

    def hel_names_unsorted(self):
        """
        Returns all hel_names; order is undefined.  You cannot apply subscripts.
        """
        return self.dictionary.keys()

    def list_of_frame_xy_lists(self, hel_name):
        return self.dictionary[hel_name]

    def matching_frame_xy_list(self, hel_name, frame_id):
        list_of_frame_xy_lists = self.dictionary[hel_name]
        # Implementing this as a nested dictionary would be nice.  I don't have time for that today.
        for frame_xy_list in list_of_frame_xy_lists:
            if frame_id == frame_xy_list[0]:
                return frame_xy_list
        # Match not found.
        return None

    def number_of_frames(self):
        return len(self.dictionary.keys())

    def frames_per_heliostat(self):
        """
        Returns a dict with same keys, but values are the number of frames associated with each heliostat.
        """
        result = {}
        for hel_name in self.dictionary.keys():
            list_of_frame_xy_lists = self.dictionary[hel_name]
            n_frames = len(list_of_frame_xy_lists)
            result[hel_name] = n_frames
        return result

    def frames_in_each_heliostat_name(self):
        """
        Returns a dict with same keys, but the value for each key is a list of the frames in the associated frame_xy_list.
        """
        result = {}
        for hel_name in self.dictionary.keys():
            list_of_frame_xy_lists = self.dictionary[hel_name]
            name_list = [frame_xy_list[0] for frame_xy_list in list_of_frame_xy_lists]
            result[hel_name] = name_list
        return result

    def points_per_heliostat_name(self):
        """
        Returns a dict with same keys, but values are the total number of (x,y) points associated with each heliostat name.
        """
        result = {}
        for hel_name in self.dictionary.keys():
            list_of_frame_xy_lists = self.dictionary[hel_name]
            n_points = 0
            for frame_xy_list in list_of_frame_xy_lists:
                xy_list = frame_xy_list[1]
                n_points += len(xy_list)
            result[hel_name] = n_points
        return result

    # MODIFICATION

    def add_list_of_frame_xy_lists(self, hel_name, input_list_of_frame_xy_lists):
        """
        Add a list of [frame, xy_list] pairs to the dictionary, under the given hel_name key.
        Assumes the hel_name is not already there.
        """
        if hel_name in self.dictionary:
            print(
                "ERROR: In NameFrameXyList.add_list_of_frame_xy_lists(), attempt to add hel_name="
                + str(hel_name)
                + ", which is already present."
            )
            assert False
        self.dictionary[hel_name] = input_list_of_frame_xy_lists

    def merge_list_of_frame_xy_lists(
        self,
        hel_name,
        input_list_of_frame_xy_lists,
        warn_if_common_frame=True,
        skip_if_common_frame=True,
        error_if_common_frame=False,
    ):
        """
        Add a list of [frame, xy_list] pairs to the dictionary, under the given hel_name key.
        If hel_name is not already there, it simply add it.
        If hel_name is already there, then merge the list of [frame, xy_list] entries.
        If the frame is not present, then add the [frame, xy_list] to the existing list.
        If it is present, then optionally throw an error, because we don't know which xy_list is valid, or simply extends the xy_list.
        """
        if hel_name not in self.dictionary:
            self.dictionary[hel_name] = copy.deepcopy(input_list_of_frame_xy_lists)
        else:
            existing_list_of_frame_xy_lists = self.dictionary[hel_name]
            existing_frame_list = [frame_xy_list[0] for frame_xy_list in existing_list_of_frame_xy_lists]
            for input_frame_xy_list in input_list_of_frame_xy_lists:
                input_frame_xy_list_copy = copy.deepcopy(input_frame_xy_list)
                input_frame_copy = input_frame_xy_list_copy[0]
                input_xy_list_copy = input_frame_xy_list_copy[1]
                if input_frame_copy not in existing_frame_list:
                    existing_list_of_frame_xy_lists.append(input_frame_xy_list_copy)
                else:
                    if error_if_common_frame:
                        print(
                            "ERROR: In NameFrameXyList.merge_list_of_frame_xy_lists(), attempt to add xy_list for hel_name="
                            + str(hel_name)
                            + ' and frame="'
                            + str(input_frame_copy)
                            + '", both of which are already present.'
                        )
                        assert False
                    if warn_if_common_frame:
                        if skip_if_common_frame:
                            suffix = "  Skipping."
                        else:
                            suffix = ""
                        print(
                            "WARNING: In NameFrameXyList.merge_list_of_frame_xy_lists(), attempt to add xy_list for hel_name="
                            + str(hel_name)
                            + ' and frame="'
                            + str(input_frame_copy)
                            + '", both of which are already present.'
                            + suffix
                        )
                    # Proceed.
                    if not skip_if_common_frame:
                        if warn_if_common_frame:
                            print(
                                "WARNING: In NameFrameXyList.merge_list_of_frame_xy_lists(), adding to already-existing (hel_name="
                                + str(hel_name)
                                + ', frame="'
                                + str(input_frame_copy)
                                + '") pair.'
                            )
                        for existing_frame_xy_list in existing_list_of_frame_xy_lists:
                            existing_frame = existing_frame_xy_list[0]
                            existing_xy_list = existing_frame_xy_list[1]
                            if existing_frame == input_frame_copy:
                                existing_xy_list += input_xy_list_copy

    def merge_frame_xy_list(
        self,
        hel_name,
        input_frame_xy_list,
        warn_if_common_frame=True,
        skip_if_common_frame=True,
        error_if_common_frame=False,
    ):
        """
        Add a [frame, xy_list] pair to the dictionary, under the given hel_name key.
        If hel_name is not already there, it simply create a new surrounding list and add it.
        If hel_name is already there, then merge the new [frame_xy_list into the existing list of [frame, xy_list] entries:
            - If the frame is not present, then add the [frame, xy_list] to the existing list.
            - If the frame is present, then optionally throw an error, because we don't know which xy_list is valid, or if it simply extends the xy_list.
        """
        if hel_name not in self.dictionary:
            self.dictionary[hel_name] = [copy.deepcopy(input_frame_xy_list)]
        else:
            existing_list_of_frame_xy_lists = self.dictionary[hel_name]
            existing_frame_list = [frame_xy_list[0] for frame_xy_list in existing_list_of_frame_xy_lists]
            input_frame_xy_list_copy = copy.deepcopy(input_frame_xy_list)
            input_frame_copy = input_frame_xy_list_copy[0]
            input_xy_list_copy = input_frame_xy_list_copy[1]
            if input_frame_copy not in existing_frame_list:
                existing_list_of_frame_xy_lists.append(input_frame_xy_list_copy)
            else:
                if error_if_common_frame:
                    print(
                        "ERROR: In NameFrameXyList.merge_frame_xy_lists(), attempt to add xy_list for hel_name="
                        + str(hel_name)
                        + ' and frame="'
                        + str(input_frame_copy)
                        + '", both of which are already present.'
                    )
                    assert False
                if warn_if_common_frame:
                    if skip_if_common_frame:
                        suffix = "  Skipping."
                    else:
                        suffix = ""
                    print(
                        "WARNING: In NameFrameXyList.merge_frame_xy_lists(), attempt to add xy_list for hel_name="
                        + str(hel_name)
                        + ' and frame="'
                        + str(input_frame_copy)
                        + '", both of which are already present.'
                        + suffix
                    )
                # Proceed.
                if not skip_if_common_frame:
                    if warn_if_common_frame:
                        print(
                            "WARNING: In NameFrameXyList.merge_frame_xy_lists(), adding to already-existing (hel_name="
                            + str(hel_name)
                            + ', frame="'
                            + str(input_frame_copy)
                            + '") pair.'
                        )
                    for existing_frame_xy_list in existing_list_of_frame_xy_lists:
                        existing_frame = existing_frame_xy_list[0]
                        existing_xy_list = existing_frame_xy_list[1]
                        if existing_frame == input_frame_copy:
                            existing_xy_list += input_xy_list_copy

    def add_FrameNameXyList(
        self, input_fnxl, warn_if_common_frame=False, skip_if_common_frame=False, error_if_common_frame=True
    ):
        # Walk the input FranemNameXyList, adding data to the current NameFrameXyList.
        for frame_id in input_fnxl.sorted_frame_id_list():
            list_of_name_xy_lists = input_fnxl.list_of_name_xy_lists(frame_id)
            for name_xy_list in list_of_name_xy_lists:
                hel_name = name_xy_list[0]
                xy_list = name_xy_list[1]
                new_frame_xy_list = [frame_id, xy_list]
                self.merge_frame_xy_list(
                    hel_name, new_frame_xy_list, warn_if_common_frame, skip_if_common_frame, error_if_common_frame
                )

    # READ

    def load(self, input_dir_body_ext):  # "nfxl" abbreviates "NameFrameXyList"
        """
        Reads the stored NameFrameXyList file, and adds it to the current dictionary.
        If data is already already present in this NameFrameXyList, extends the current
        content as follows:

          - If a heliostat is not already present, adds the heliostat, and the associated frame_id/xy_list pair.

          - If the loaded heliostat is present, but the frame_id of the loaded frame_id/xy_list pair is not
            already associated with the heliostat, then add the frame_id/xy_list pair to the heliostat entry.

          - If the loaded heliostat is present, and it already contains the frame_id with some associated
            (x,y) points, extend the existing xy_list for the frame_id by adding the newly loaded (x,y)
            points.  Do not check for duplicate points, simply append the loaded points to the points
            that are already there.  Also do not group the points -- thus if the points are supposed
            to represent a polygon, they are not combined geometrically, but rather by simply
            concatenating the point lists.
        """
        # print('In NameFrameXyList.load(), loading input file: ', input_dir_body_ext)
        # Check if the input file exists.
        if not ft.file_exists(input_dir_body_ext):
            raise OSError("In NameFrameXyList.load(), file does not exist: " + str(input_dir_body_ext))
        # Open and read the file.
        with open(input_dir_body_ext, newline="") as input_stream:
            reader = csv.reader(input_stream, delimiter=",")
            for input_row in reader:
                self.add_row_to_dictionary(input_row)

    def add_row_to_dictionary(self, input_row):
        hel_name = input_row[0]
        list_of_frame_xylists = self.parse_row_list_of_frame_xylists(input_row[1:])
        # Add the entry.
        self.dictionary[hel_name] = list_of_frame_xylists

    def parse_row_list_of_frame_xylists(self, input_row_remainder):
        n_frames = int(input_row_remainder[0])
        if n_frames == 0:
            return []
        else:
            return self.parse_row_list_of_frame_xylists_aux(input_row_remainder[1:], n_frames)

    def parse_row_list_of_frame_xylists_aux(self, input_row_list_of_frame_xylists, n_frames):
        # Fetch this frame_xylist's frame_id and the number of points in its xylist.
        frame_id = int(input_row_list_of_frame_xylists[0])
        n_points = int(input_row_list_of_frame_xylists[1])
        # Parse the xylist.
        xylist = []
        for idx in range(0, n_points):
            idx_x = 2 + (2 * idx)
            idx_y = idx_x + 1
            vertex_str = [input_row_list_of_frame_xylists[idx_x], input_row_list_of_frame_xylists[idx_y]]
            x_str = vertex_str[0]
            y_str = vertex_str[1]
            x = float(x_str)
            y = float(y_str)
            xylist.append([x, y])
            last_idx = idx_y
        frame_xylist = [frame_id, xylist]
        # If there are more frame_ids, then recurse.
        if n_frames > 1:
            further_frame_xylists = self.parse_row_list_of_frame_xylists_aux(
                input_row_list_of_frame_xylists[last_idx + 1 :], n_frames - 1
            )
        else:
            further_frame_xylists = []
        # Assemble the result, including the frame_xylist we assembled here.
        list_of_frame_xylists = [frame_xylist] + further_frame_xylists
        # Return.
        return list_of_frame_xylists

    # WRITE

    def save(self, output_dir_body_ext):
        # Extract path components.
        output_dir, output_body, output_ext = ft.path_components(output_dir_body_ext)
        # Create output directory if necessary.
        ft.create_directories_if_necessary(output_dir)

        # Write the heliostat dictionary in a structured format.
        # print('In NameFrameXyList.save(), saving file:', output_dir_body_ext)
        with open(output_dir_body_ext, "w") as output_stream:
            writer = csv.writer(output_stream, delimiter=",")
            # Assemble the row.
            for hel_name in dt.sorted_keys(self.dictionary):
                row_items = [hel_name]
                frame_polygon_list = self.dictionary[hel_name]
                n_frames = len(frame_polygon_list)
                row_items.append(n_frames)
                for frame_polygon_list in frame_polygon_list:
                    frame_id = frame_polygon_list[0]
                    polygon = frame_polygon_list[1]
                    n_vertices = len(polygon)
                    row_items.append(frame_id)
                    row_items.append(n_vertices)
                    for vertex in polygon:
                        row_items.append(vertex[0])  # x
                        row_items.append(vertex[1])  # y
                # Write the row.
                writer.writerow(row_items)

    # RENDER

    def print(
        self,
        max_keys=10,  # Maximum number of keys to print.  Elipsis after that.
        max_value_length=70,  # Maximum value length to print.  Elipsis after that.
        indent=None,
    ):  # Number of blankss to print at the beginning of each line.
        # Print.
        dt.print_dict(self.dictionary, max_keys=max_keys, max_value_length=max_value_length, indent=indent)


# HELPER FUNCTIONS


def construct_merged_copy(input_nfxl_list):  # A list of NameFrameXyList objects.
    """
    Constructs a new NameFrameXyList object, combining the entries of the input NameFrameXyList objects, without modifying them.
    """
    new_nfxl = NameFrameXyList()
    for input_nfxl in input_nfxl_list:
        for hel_name in input_nfxl.hel_names_unsorted():
            input_frame_xy_list = input_nfxl.list_of_frame_xy_lists(hel_name)
            new_frame_xy_list = copy.deepcopy(input_frame_xy_list)
            new_nfxl.add_list_of_frame_xy_lists(hel_name, new_frame_xy_list)
    return new_nfxl
