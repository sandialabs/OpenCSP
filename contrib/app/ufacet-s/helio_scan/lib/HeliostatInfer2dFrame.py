"""
A data structure for estimating a heliostat's 2d shape given a single frame observation.



"""

import copy
import csv
from cv2 import cv2 as cv
import numpy as np
import os

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.render.PlotAnnotation as pa
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.list_tools as lt
import opencsp.common.lib.tool.log_tools as logt
import ufacet_pipeline_frame as upf


class HeliostatInfer2dFrame:
    """
    Class for estimating a heliostat 2d-shape from a single frame.

    Input is a nominal heliostat model, and a frame with associated found corners.

    This class contains several quality metrics.

    """

    def __init__(
        self,
        # Data.
        hel_name,  # Heliostat name for printing, reference.
        frame,  # Frame for printing, reference.
        observed_corner_xy_list,  # Already undistorted if distorted_or_undistorted_str == 'undistorted'
        flat_corner_xyz_list,
    ):  # A "flat heliostat" model, oriented with z parallel to the nominal heliostat optical axis.
        # For this calculation, z values should be coplanar, for coplanar sections of the heliostat.
        # For most heliostats this means z will be a single common value.  But for heliostats such
        # as NSTTF which have a step in the middle, there will be two z values, with one z value
        # corresponding to rows 1, 2, 4, and 5, with a second z value corresponding to row 3.

        # Data.
        self.hel_name = hel_name
        self.frame = frame

        # Check input.
        if len(observed_corner_xy_list) != len(flat_corner_xyz_list):
            msg = (
                "In HeliostatInfer2dFrame.__init__(), input len(observed_corner_xy_list)="
                + str(len(observed_corner_xy_list))
                + " is not equal to input len(flat_corner_xyz_list)="
                + str(len(flat_corner_xyz_list))
            )
            print("ERROR: " + msg)
            raise ValueError(msg)

        # Derived values.
        self.n_corners = len(flat_corner_xyz_list)
        # Group observed and ideal corners, and associate their list position.  We will break up the list according to various criteria,
        # and this will let us put them back together in the correct order.
        self.corner_dicts = []
        for idx in range(0, self.n_corners):
            corner_dict = {}
            corner_dict["idx"] = idx
            corner_dict["observed_xy"] = observed_corner_xy_list[idx]
            corner_dict["flat_xyz"] = flat_corner_xyz_list[idx]
            self.corner_dicts.append(corner_dict)

        # Initialize homography dictionaries.
        # We need a separate homography for each common plane in the flat heliostat.  For most heliostats, this will be
        # a single plane.  But for the NSTTF heliostat, a second homography is needed for its offset center row.
        self.homography_dicts = self.setup_homography_dicts()

        # Construct homographies (one for each common plane).
        self.setup_homography_dicts()

        # Compute homographies.
        self.contruct_homographies()

        # Map the observed points onto the flat model plane.
        self.map_observed_points_onto_flat_model()

    # print('\nIn HeliostatInfer2dFrame.__init__(), self.homography_dicts:')  # ?? SCAFFOLDING RCB -- TEMPORARY
    # for homography_dict in self.homography_dicts:  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     dt.print_dict(homography_dict, indent=4)  # ?? SCAFFOLDING RCB -- TEMPORARY

    # print('\nIn HeliostatInfer2dFrame.__init__(), self.corner_dicts:')  # ?? SCAFFOLDING RCB -- TEMPORARY
    # for corner_dict in self.corner_dicts:  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     dt.print_dict(corner_dict, indent=4)  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     print()  # ?? SCAFFOLDING RCB -- TEMPORARY

    def setup_homography_dicts(self):
        # Partition points into common z planes, since these will need separate homographies.
        # Find unique z values.
        unique_z_values = set()
        for corner_dict in self.corner_dicts:
            unique_z_values.add(corner_dict["flat_xyz"][2])
        # Set up a list of homography data containers, one for each z value.
        homography_dicts = []
        for z in unique_z_values:
            homography_dict = {"z": z}
            homography_dicts.append(homography_dict)
        # Partition the corners, and find common corners.
        for homography_dict in homography_dicts:
            z = homography_dict["z"]
            coplanar_corner_dicts = []  # All points within homography plane.
            common_corner_dicts = []  # Points within homography plane, excluding missing points.
            for corner_dict in self.corner_dicts:
                if corner_dict["flat_xyz"][2] == z:
                    corner_dict["homography_z"] = z
                    coplanar_corner_dicts.append(corner_dict)
                    if (corner_dict["observed_xy"][0] != -1) or (corner_dict["observed_xy"][1] != -1):
                        common_corner_dicts.append(corner_dict)
            homography_dict["coplanar_corner_dicts"] = coplanar_corner_dicts
            homography_dict["common_corner_dicts"] = common_corner_dicts
        # Return.
        return homography_dicts

    def contruct_homographies(self):
        for homography_dict in self.homography_dicts:
            self.contruct_homography(homography_dict)  # Adds to homography_dict as a side effect.

    def contruct_homography(self, homography_dict):
        common_corner_dicts = homography_dict["common_corner_dicts"]
        common_observed_xy_list = [d["observed_xy"] for d in common_corner_dicts]
        common_flat_xy_list = [d["flat_xyz"][0:2] for d in common_corner_dicts]  # Note drop z.
        source_points = np.array(common_observed_xy_list)
        destination_points = np.array(common_flat_xy_list)
        H, retval = cv.findHomography(source_points, destination_points)
        homography_dict["H"] = H
        # homography_dict['retval'] = retval  # Ignore until we need it, for example by using RANSAC homography method.

    def map_observed_points_onto_flat_model(self):
        for homography_dict in self.homography_dicts:
            self.map_observed_points_onto_flat_model_aux(homography_dict)

    def map_observed_points_onto_flat_model_aux(self, homography_dict):
        H = homography_dict["H"]
        for corner_dict in homography_dict["common_corner_dicts"]:
            observed_xy = corner_dict["observed_xy"]
            observed_xy1 = np.array(observed_xy + [1])  # Homogeneous form.
            mapped_onto_flat_xy1 = H.dot(observed_xy1)
            # I observed that the product was not homogeneous.  Remembering that the homography may include multiplication
            # by a scale factor, I normalize to achieve a proper homogeneous form where the third coordinate equals 1.0.
            # Comparing ordinary and normalized output showed that this led to a clear improvement in output quality.
            normalized_onto_flat_xy1 = mapped_onto_flat_xy1 / mapped_onto_flat_xy1[2]
            # Construct a 3-d point for convenience in rendering up to this point in the computation.
            normalized_onto_flat_xy = normalized_onto_flat_xy1[0:2]
            normalized_onto_flat_xyz = list(normalized_onto_flat_xy) + [corner_dict["homography_z"]]
            # Store mapped points.
            corner_dict["observed_xy1"] = observed_xy1
            corner_dict["mapped_onto_flat_xy1"] = mapped_onto_flat_xy1
            corner_dict["normalized_onto_flat_xy1"] = normalized_onto_flat_xy1
            corner_dict["normalized_onto_flat_xyz"] = normalized_onto_flat_xyz


# print('In HeliostatInfer2dFrame.map_observed_points_onto_flat_model_aux()')  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.map_observed_points_onto_flat_model_aux(), observed_xy1                      = ', corner_dict['observed_xy1'])  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.map_observed_points_onto_flat_model_aux(), flat_xyz                          = ', np.array(corner_dict['flat_xyz']))  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.map_observed_points_onto_flat_model_aux(), observed_normalized_onto_flat_xy1 = ', corner_dict['observed_normalized_onto_flat_xy1'])  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.map_observed_points_onto_flat_model_aux(), observed_mapped_onto_flat_xy1     = ', corner_dict['observed_mapped_onto_flat_xy1'])  # ?? SCAFFOLDING RCB -- TEMPORARY


# common_observed_corner_xy_list = []
# unique_z_values = set()
# common_flat_corner_xyz_list = []  # Keep z for now
# for observed_corner_xy, flat_corner_xyz in zip(self.observed_corner_xy_list, self.flat_corner_xyz_list):
#     if (observed_corner_xy[0] != -1) and (observed_corner_xy[1] != -1):
#         common_observed_corner_xy_list.append(observed_corner_xy)
#         common_flat_corner_xyz_list.append(flat_corner_xyz)
#         unique_z_values.add(flat_corner_xyz[2])

# print('In HeliostatInfer2dFrame.__init__(), len(common_observed_corner_xy_list) = ', len(common_observed_corner_xy_list))  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), common_observed_corner_xy_list:')  # ?? SCAFFOLDING RCB -- TEMPORARY
# lt.print_list(common_observed_corner_xy_list)  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), len(common_flat_corner_xyz_list) = ', len(common_flat_corner_xyz_list))  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), common_flat_corner_xyz_list:')  # ?? SCAFFOLDING RCB -- TEMPORARY
# lt.print_list(common_flat_corner_xyz_list)  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), unique_z_values = ', unique_z_values)  # ?? SCAFFOLDING RCB -- TEMPORARY

# # Construct point lists, skipping missing values.  Also collect flat z values.
# common_observed_corner_xy_list = []
# unique_z_values = set()
# common_flat_corner_xyz_list = []  # Keep z for now
# for observed_corner_xy, flat_corner_xyz in zip(self.observed_corner_xy_list, self.flat_corner_xyz_list):
#     if (observed_corner_xy[0] != -1) and (observed_corner_xy[1] != -1):
#         common_observed_corner_xy_list.append(observed_corner_xy)
#         common_flat_corner_xyz_list.append(flat_corner_xyz)
#         unique_z_values.add(flat_corner_xyz[2])

# print('In HeliostatInfer2dFrame.__init__(), len(common_observed_corner_xy_list) = ', len(common_observed_corner_xy_list))  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), common_observed_corner_xy_list:')  # ?? SCAFFOLDING RCB -- TEMPORARY
# lt.print_list(common_observed_corner_xy_list)  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), len(common_flat_corner_xyz_list) = ', len(common_flat_corner_xyz_list))  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), common_flat_corner_xyz_list:')  # ?? SCAFFOLDING RCB -- TEMPORARY
# lt.print_list(common_flat_corner_xyz_list)  # ?? SCAFFOLDING RCB -- TEMPORARY
# print('In HeliostatInfer2dFrame.__init__(), unique_z_values = ', unique_z_values)  # ?? SCAFFOLDING RCB -- TEMPORARY

# # Construct common xy lists, skipping missing values.
# common_observed_corner_xy_list = []
# common_flat_corner_xy_list = []  # Note dropping z
# for observed_corner_xy, flat_corner_xyz in zip(self.observed_corner_xy_list, self.flat_corner_xyz_list):
#     if (observed_corner_xy[0] != -1) and (observed_corner_xy[1] != -1):
#         common_observed_corner_xy_list.append(observed_corner_xy)
#         common_flat_corner_xy_list.append(flat_corner_xyz[0:2])  # Drop z coordinate

# Four points minimum are required to compute a homography.

# ACCESS

# def sorted_hel_name_list(self):
#     """
#     Returns all hel_names, in ascending order.  You can apply subscripts.
#     """
#     hel_name_list = list( self.dictionary.keys())
#     hel_name_list.sort()
#     return hel_name_list


# MODIFICATION

# def add_list_of_frame_xy_lists(self, hel_name, input_list_of_frame_xy_lists):
#     """
#     Add a list of [frame, xy_list] pairs to the dictionary, under the given hel_name key.
#     Assumes the hel_name is not already there.
#     """
#     if hel_name in self.dictionary:
#         print('ERROR: In HeliostatInfer2dFrame.add_list_of_frame_xy_lists(), attempt to add hel_name='+str(hel_name)+', which is already present.')
#         assert False
#     self.dictionary[hel_name] = input_list_of_frame_xy_lists


# READ

# def load(self, input_dir_body_ext):   # "nfxl" abbreviates "HeliostatInfer2dFrame"
#     """
#     Reads the stored HeliostatInfer2dFrame file, and adds it to the current dictionary.
#     If data is already already present in this HeliostatInfer2dFrame, extends the current
#     content as follows:

#       - If a heliostat is not already present, adds the heliostat, and the associated frame_id/xy_list pair.

#       - If the loaded heliostat is present, but the frame_id of the loaded frame_id/xy_list pair is not
#         already associated with the heliostat, then add the frame_id/xy_list pair to the heliostat entry.

#       - If the loaded heliostat is present, and it already contains the frame_id with some associated
#         (x,y) points, extend the existing xy_list for the frame_id by adding the newly loaded (x,y)
#         points.  Do not check for duplicate points, simply append the loaded points to the points
#         that are already there.  Also do not group the points -- thus if the points are supposed
#         to represent a polygon, they are not combined geometrically, but rather by simply
#         concatenating the point lists.
#     """
#     #print('In HeliostatInfer2dFrame.load(), loading input file: ', input_dir_body_ext)
#     # Check if the input file exists.
#     if not ft.file_exists(input_dir_body_ext):
#         raise OSError('In HeliostatInfer2dFrame.load(), file does not exist: ' + str(input_dir_body_ext))
#     # Open and read the file.
#     with open(input_dir_body_ext, newline='') as input_stream:
#         reader = csv.reader(input_stream, delimiter=',')
#         for input_row in reader:
#             self.add_row_to_dictionary(input_row)


# WRITE

# def save(self, output_dir_body_ext):
#     # Extract path components.
#     output_dir, output_body, output_ext = ft.path_components(output_dir_body_ext)
#     # Create output directory if necessary.
#     ft.create_directories_if_necessary(output_dir)

#     # Write the heliostat dictionary in a structured format.
#     #print('In HeliostatInfer2dFrame.save(), saving file:', output_dir_body_ext)
#     with open(output_dir_body_ext, "w") as output_stream:
#         writer = csv.writer(output_stream, delimiter=',')
#         # Assemble the row.
#         for hel_name in dt.sorted_keys(self.dictionary):
#             row_items = [hel_name]
#             frame_polygon_list = self.dictionary[hel_name]
#             n_frames = len(frame_polygon_list)
#             row_items.append(n_frames)
#             for frame_polygon_list in frame_polygon_list:
#                 frame_id = frame_polygon_list[0]
#                 polygon  = frame_polygon_list[1]
#                 n_vertices = len(polygon)
#                 row_items.append(frame_id)
#                 row_items.append(n_vertices)
#                 for vertex in polygon:
#                     row_items.append(vertex[0])  # x
#                     row_items.append(vertex[1])  # y
#             # Write the row.
#             writer.writerow(row_items)


# RENDER

# def print(self,
#           max_keys=10,          # Maximum number of keys to print.  Elipsis after that.
#           max_value_length=70,  # Maximum value length to print.  Elipsis after that.
#           indent=None):         # Number of blankss to print at the beginning of each line.
#     # Print.
#     dt.print_dict(self.dictionary,
#                   max_keys=max_keys,
#                   max_value_length=max_value_length,
#                   indent=indent)


# HELPER FUNCTIONS

# def construct_merged_copy(input_nfxl_list):  # A list of HeliostatInfer2dFrame objects.
#     """
#     Constructs a new HeliostatInfer2dFrame object, combining the entries of the input HeliostatInfer2dFrame objects, without modifying them.
#     """
#     new_nfxl = HeliostatInfer2dFrame()
#     for input_nfxl in input_nfxl_list:
#         for hel_name in input_nfxl.hel_names_unsorted():
#             input_frame_xy_list = input_nfxl.list_of_frame_xy_lists(hel_name)
#             new_frame_xy_list = copy.deepcopy(input_frame_xy_list)
#             new_nfxl.add_list_of_frame_xy_lists(hel_name, new_frame_xy_list)
#     return new_nfxl
