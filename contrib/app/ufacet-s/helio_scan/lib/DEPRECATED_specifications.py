"""
Class with solar field parameters.

# ?? SCAFFOLDING RCB -- REWORK THIS, INTEGRATE WITH PLANNER SOLAR FIELD MODEL.



"""

import copy
import csv
import numpy as np

import opencsp.common.lib.geometry.transform_3d as t3d
import opencsp.common.lib.tool.dict_tools as dt
from . import DEPRECATED_save_read as sr
from . import ufacet_heliostat_3d_analysis as uh3a


class Specifications:
    def __init__(
        self,
        name=None,
        heliostat_design_name=None,
        corners_per_facet=0,
        corners_per_heliostat=0,
        facets_per_row=0,
        facets_per_heliostat=0,
        facet_height=0,
        facet_width=0,
        z_offset=0,
        centered_facet=0,
        top_left_facet_indx=0,
        top_right_facet_indx=0,
        bottom_right_facet_indx=0,
        bottom_left_facet_indx=0,
        facets_centroids_file=None,
        heliostat_locations_file=None,
        # Aim point assumed when computing an ideal focal length for a heliostat.
        design_aim_point_x=0,  # m
        design_aim_point_y=0,  # m
        design_aim_point_z=0,  # m
        # Key heliostats.
        nearest_heliostat=None,  # Heliostat name (e.g. '5W1').
        farthest_heliostat=None,  # Heliostat name (e.g. '13W14').
    ):
        super(Specifications, self).__init__()
        self.name = name
        self.heliostat_design_name = heliostat_design_name
        self.corners_per_facet = corners_per_facet
        self.corners_per_heliostat = corners_per_heliostat
        self.facets_per_row = facets_per_row
        self.facets_per_heliostat = facets_per_heliostat
        self.facet_height = facet_height
        self.facet_width = facet_width
        self.z_offset = z_offset
        self.centered_facet = centered_facet
        self.top_left_facet_indx = top_left_facet_indx
        self.top_right_facet_indx = top_right_facet_indx
        self.bottom_right_facet_indx = bottom_right_facet_indx
        self.bottom_left_facet_indx = bottom_left_facet_indx
        self.facets_centroids_file = facets_centroids_file
        self.facets_centroids = sr.read_centers3d(facets_centroids_file)
        self.facets_corners = sr.centers3d_to_corners3d(self.facets_centroids, self.facet_width, self.facet_height)
        self.flat_corner_xyz_list = self.facets_corners
        self.heliostat_locations_file = heliostat_locations_file
        self.heliostat_locations_dict = self.read_heliostat_locations_dict(self.heliostat_locations_file)
        # Design aim point.
        self.design_aim_point_x = design_aim_point_x
        self.design_aim_point_y = design_aim_point_y
        self.design_aim_point_z = design_aim_point_z
        # Key heliostats.
        self.nearest_heliostat = nearest_heliostat
        self.farthest_heliostat = farthest_heliostat

    def read_heliostat_locations_dict(
        self, heliostat_locations_file
    ):  # ?? SCAFFOLDING RCB -- ADAPTED FROM ROUTINE "heliostats_read_file" IN SolarField.py; INTEGRATE WITH THAT
        with open(heliostat_locations_file) as input_stream:
            readCSV = csv.reader(input_stream, delimiter=",")
            id_row = 0
            id_heliostat = 0
            heliostat_dict = {}
            # heliostats = []
            for row in readCSV:
                if not id_row:
                    # get rid of the header row in csv
                    id_row += 1
                    continue
                # row info
                name, x, y, z = str(row[0]), float(row[1]), float(row[2]), float(row[3])
                num_facets, num_rows, num_cols = int(row[4]), int(row[5]), int(row[6])
                pivot_height, pivot_offset = float(row[7]), float(row[8])
                facet_width, facet_height = float(row[9]), float(row[10])

                # # creating heliostat
                # heliostat = Heliostat.Heliostat(name=name, origin=[x, y, z],
                #                                 num_facets=num_facets, num_rows=num_rows, num_cols=num_cols,
                #                                 file_centroids_offsets=file_centroids_offsets,
                #                                 pivot_height=pivot_height, pivot_offset=pivot_offset,
                #                                 facet_width=facet_width, facet_height=facet_height)
                # storing
                # heliostats.append(heliostat)
                # heliostat_dict[name] = id_heliostat  # ?? SCAFFOLDING RCB -- ORIGINAL CODE
                heliostat_dict[name] = [x, y, z]
                id_heliostat += 1

        # return heliostats, heliostat_dict
        return heliostat_dict  # ?? SCAFFOLDING RCB -- ORIGINAL CODE

    # TJL: This is what I want to use to get a focal length, talk with
    # Randy about where on the heliostat he wants this aimed at
    def slant_range_distance(self, hel_xyz):
        """
        Distance from heliostat center to design aim point.
        """
        dx = hel_xyz[0] - self.design_aim_point_x
        dy = hel_xyz[1] - self.design_aim_point_y
        dz = hel_xyz[2] - self.design_aim_point_z
        slant_distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Return.
        return slant_distance

    def ideal_focal_length(self, hel_xyz):
        """
        Focal length of an ideal heliostat located at the given position.

        This routine assumes the design goal that the focal length should be symmetric
        and based on a hypothetical axis-aligned situation where the sun, heliostat, and
        aim point lie on a common line.

        Other assumptions are plausible, such as maximizing the reflection onto the
        receiver at the spring equinox.  Other design goals may lead to asymmetric
        paraboloids, with different focal lengths in the x and y directions.
        """
        return self.slant_range_distance(hel_xyz)

    def design_focal_length(self, hel_xyz):
        """
        Focal length designed into the implementation of an real heliostat at the given position.

        The design focal length may vary from the ideal focal length for various reasons.
        For example, consider a heliostat assembly line where canting angles are set by
        precisely controlling the construction of the heliostat frame.  Such a system would
        avoid the cost of canting adjustment.  Further suppose that the frame was produced
        by precise assembly on a jig, or by selection of preciely formed stamped parts.
        Both the jig and the stamping tool are discrete, so the produced heliostats will
        be of a set of fixed focal lengths.  A similar discretization might apply to mirror
        facets, produced by a fixed set of molds.  For such a system, the design focal
        length will be selection among discrete values, instead of a continuous ideal
        focal length.

        This implementation assumes the simple continuous focal length model.
        """
        return self.ideal_focal_length(hel_xyz)

    def smooth_heliostat_corner_xyz_list_given_focal_length(self, focal_length):
        return heliostat_corner_xyz_list_given_focal_length_smooth(self.flat_corner_xyz_list, focal_length)

    def faceted_heliostat_corner_xyz_list_given_focal_length(self, focal_length):
        return heliostat_corner_xyz_list_given_focal_length_faceted(self.flat_corner_xyz_list, focal_length)

    def design_heliostat_corner_xyz_list(self, hel_xyz):
        """
        This routine returns an xyz list of the facet corners, for the nominal design of a heliostat located in the given position.
        As noted above, the issues determining the design vary depending on solar field design and manufacturing decisions.
        """
        return self.lift_flat_corner_xyz_list(hel_xyz, self.flat_corner_xyz_list)

    def lift_flat_corner_xyz_list(self, hel_xyz, input_flat_corner_xyz_list):
        """
        Given an input flat corner xyz list, returns a copy of the list with the z values set according to the design canting.

        The current implementation only supports the simplistic model that heliostats should have symmetric focal legnths,
        set to the continuously-varying slat range distance to a predefined aim point.

        Future implementations are envisioned to support a more realistic suite of design decisions.
        """
        focal_length = self.design_focal_length(hel_xyz)
        return heliostat_corner_xyz_list_given_focal_length_faceted(input_flat_corner_xyz_list, focal_length)

    def heliostat_xyz(self, hel_name):
        """
        Returns the (x,y,z) location of the given heliostat.
        """
        return self.heliostat_locations_dict[hel_name]

    def nearest_heliostat_xyz(self):
        """
        Returns the (x,y,z) location of the heliostat closest to the design aim point.
        """
        return self.heliostat_locations_dict[self.nearest_heliostat]

    def farthest_heliostat_xyz(self):
        """
        Returns the (x,y,z) location of the heliostat closest to the design aim point.
        """
        return self.heliostat_locations_dict[self.farthest_heliostat]

    def heliostat_corner_xyz_list_given_heliostat_spec(self, heliostat_spec):
        w = self.facet_width
        h = self.facet_height
        half_w = w / 2.0
        half_h = h / 2.0
        heliostat_corner_xyz_list = []
        for facet_spec in heliostat_spec.values():
            # Construct the facet centered at the origin, pointing straight up.
            centered_flat_ul_xyz = [-half_w, half_h, 0]
            centered_flat_ur_xyz = [half_w, half_h, 0]
            centered_flat_lr_xyz = [half_w, -half_h, 0]
            centered_flat_ll_xyz = [-half_w, -half_h, 0]
            # Rotate the facet.
            R_x = t3d.axisrotation([1, 0, 0], facet_spec["rot_x"])
            R_y = t3d.axisrotation([0, 1, 0], facet_spec["rot_y"])
            R_z = t3d.axisrotation([0, 0, 1], facet_spec["rot_z"])
            R = R_y.dot(
                R_x.dot(R_z)
            )  # First Rz, then Rx, then Ry  # ?? SCAFFOLDING RCB -- USE FASTER AND MORE PRECISE ROTx, ROTy, ROTz FUNCTIONS.  See https://learnopencv.com/rotation-matrix-to-euler-angles/
            centered_rotated_ul_xyz = R.dot(centered_flat_ul_xyz)
            centered_rotated_ur_xyz = R.dot(centered_flat_ur_xyz)
            centered_rotated_lr_xyz = R.dot(centered_flat_lr_xyz)
            centered_rotated_ll_xyz = R.dot(centered_flat_ll_xyz)
            # Translate the facet.
            cx = facet_spec["center_x"]
            cy = facet_spec["center_y"]
            cz = facet_spec["center_z"]
            center = np.array([cx, cy, cz])
            final_ul_xyz = centered_rotated_ul_xyz + center
            final_ur_xyz = centered_rotated_ur_xyz + center
            final_lr_xyz = centered_rotated_lr_xyz + center
            final_ll_xyz = centered_rotated_ll_xyz + center
            # Construct and save the facet.
            heliostat_corner_xyz_list += [
                list(final_ul_xyz),
                list(final_ur_xyz),
                list(final_lr_xyz),
                list(final_ll_xyz),
            ]
        # Return.
        return heliostat_corner_xyz_list

    def construct_flat_heliostat_spec(self):
        heliostat_spec = {}
        idx = 0
        for facet_centroid_xyz in self.facets_centroids:
            facet_spec = {}
            facet_spec["center_x"] = facet_centroid_xyz[0]
            facet_spec["center_y"] = facet_centroid_xyz[1]
            facet_spec["center_z"] = facet_centroid_xyz[2]
            facet_spec["rot_x"] = 0.0
            facet_spec["rot_y"] = 0.0
            facet_spec["rot_z"] = 0.0
            heliostat_spec[idx] = facet_spec
            idx += 1
        # Return.
        return heliostat_spec

    def construct_focal_length_heliostat_spec(self, focal_length):
        heliostat_spec = {}
        idx = 0
        for facet_centroid_xyz in self.facets_centroids:
            # Center.
            cx = facet_centroid_xyz[0]
            cy = facet_centroid_xyz[1]
            cz = facet_centroid_xyz[2]
            # Surface normal.
            dx = 0.01  # m
            dy = 0.01  # m
            ul_x = cx - dx
            ul_y = cy + dy
            ul_z = smooth_z(ul_x, ul_y, focal_length)
            ur_x = cx + dx
            ur_y = cy + dy
            ur_z = smooth_z(ur_x, ur_y, focal_length)
            lr_x = cx + dx
            lr_y = cy - dy
            lr_z = smooth_z(lr_x, lr_y, focal_length)
            ll_x = cx - dx
            ll_y = cy - dy
            ll_z = smooth_z(ll_x, ll_y, focal_length)
            ll_to_ur = np.array([(ur_x - ll_x), (ur_y - ll_y), (ur_z - ll_z)])
            lr_to_ul = np.array([(ul_x - lr_x), (ul_y - lr_y), (ul_z - lr_z)])
            long_normal = np.cross(ll_to_ur, lr_to_ul)
            normal = long_normal / np.linalg.norm(long_normal)
            # Rotation angles.
            # Compute the rotation which will rotate the z axis onto the desired surface normal.
            R_z_to_normal = uh3a.rotate_align_vectors(np.array([0, 0, 1]), normal)
            rot_x, rot_y, rot_z = t3d.rotation_matrix_to_euler_angles(R_z_to_normal)
            # Assemble facet spec.
            facet_spec = {}
            facet_spec["center_x"] = cx
            facet_spec["center_y"] = cy
            facet_spec["center_z"] = cz
            facet_spec["rot_x"] = rot_x
            facet_spec["rot_y"] = rot_y
            facet_spec["rot_z"] = rot_z
            # Add to heliostat_spec.
            heliostat_spec[idx] = facet_spec
            idx += 1
        # Return.
        return heliostat_spec

    def construct_design_heliostat_spec(self, hel_xyz):
        """
        This routine returns a heliostat_spec, for the nominal design of a heliostat located in the given position.
        As noted above, the issues determining the design vary depending on solar field design and manufacturing decisions.
        """
        return self.construct_focal_length_heliostat_spec(self.design_focal_length(hel_xyz))


# HELPER FUNCTIONS

# ?? SCAFFOLDING RCB -- BEGIN ORIGINAL LIFTED APPROACH -- RETIRE?


def smooth_z(x, y, focal_length):
    k = 1 / (4 * focal_length)
    r = np.sqrt((x * x) + (y * y))
    return k * (r * r)


def heliostat_corner_xyz_list_given_focal_length_smooth(flat_corner_xyz_list, focal_length):
    return [[xyz[0], xyz[1], xyz[2] + smooth_z(xyz[0], xyz[1], focal_length)] for xyz in flat_corner_xyz_list]


def heliostat_corner_xyz_list_given_focal_length_faceted(flat_corner_xyz_list, focal_length):
    # Construct initial smooth design.
    smooth_xyz_list = [
        [xyz[0], xyz[1], xyz[2] + smooth_z(xyz[0], xyz[1], focal_length)] for xyz in flat_corner_xyz_list
    ]
    return vertically_move_facets_to_flat_z_heights(flat_corner_xyz_list, smooth_xyz_list)


def vertically_move_facets_to_flat_z_heights(flat_corner_xyz_list, smooth_xyz_list):
    def shift_xyz(xyz, dz):
        return [xyz[0], xyz[1], xyz[2] + dz]

    # Copy input.
    flat_corner_xyz_list_2 = copy.deepcopy(flat_corner_xyz_list)
    smooth_xyz_list_2 = copy.deepcopy(smooth_xyz_list)

    # Build shifted xyz list.
    shifted_xyz_list = []
    while len(flat_corner_xyz_list_2) > 0:
        # Flat facet.
        flat_xyz_A = flat_corner_xyz_list_2.pop(0)
        flat_xyz_B = flat_corner_xyz_list_2.pop(0)
        flat_xyz_C = flat_corner_xyz_list_2.pop(0)
        flat_xyz_D = flat_corner_xyz_list_2.pop(0)
        flat_z_mean = (flat_xyz_A[2] + flat_xyz_B[2] + flat_xyz_C[2] + flat_xyz_D[2]) / 4
        # Smooth facet.
        smooth_xyz_A = smooth_xyz_list_2.pop(0)
        smooth_xyz_B = smooth_xyz_list_2.pop(0)
        smooth_xyz_C = smooth_xyz_list_2.pop(0)
        smooth_xyz_D = smooth_xyz_list_2.pop(0)
        smooth_z_mean = (smooth_xyz_A[2] + smooth_xyz_B[2] + smooth_xyz_C[2] + smooth_xyz_D[2]) / 4
        # Shift.
        dz = flat_z_mean - smooth_z_mean
        shifted_xyz_list.append(shift_xyz(smooth_xyz_A, dz))
        shifted_xyz_list.append(shift_xyz(smooth_xyz_B, dz))
        shifted_xyz_list.append(shift_xyz(smooth_xyz_C, dz))
        shifted_xyz_list.append(shift_xyz(smooth_xyz_D, dz))
    # Return.
    return shifted_xyz_list


# ?? SCAFFOLDING RCB -- END ORIGINAL LIFTED APPROACH -- RETIRE?


# COMMON CASES


def nsttf_specifications():
    return Specifications(
        name="nsttf",
        heliostat_design_name="Nsttf",
        corners_per_facet=4,
        corners_per_heliostat=100,
        facets_per_row=5,
        facets_per_heliostat=25,
        facet_height=1.2192,
        facet_width=1.2192,
        z_offset=0.0115824,
        centered_facet=13,
        top_left_facet_indx=1,
        top_right_facet_indx=5,
        bottom_right_facet_indx=25,
        bottom_left_facet_indx=21,
        #        facets_centroids_file    = '<home_dir>/temporary_ufacet_input/csv_files/nsttf_centroids.csv',
        facets_centroids_file=experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Centroids.csv",  # ?? SCAFFOLDING RCB -- RENAME TO "facet_centroids_file"  ALSO, THIS SHOULD BE PASSED IN.
        heliostat_locations_file=experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/020_FieldModel/data/NSTTF_Heliostats_origin_at_torque_tube.csv",  # ?? SCAFFOLDING RCB -- THIS SHOULD BE PASSED IN.
        # Aim point assumed when computing an ideal focal length for a heliostat.
        design_aim_point_x=0.00,  # m. BCS radiometer.
        design_aim_point_y=8.00,  # m.
        design_aim_point_z=30.00,  # m.
        # Key heliostats.
        nearest_heliostat="5W1",
        farthest_heliostat="13E14",
    )
