"""
Heliostat Class



"""

import csv
import matplotlib.pyplot as plt
import math
import numpy as np

from opencsp.common.lib.csp.Facet import Facet
import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geometry.transform_3d as t3d


class Heliostat:
    """
    Heliostat representation.
    """

    def __init__(
        self,
        name,
        origin,
        num_facets,
        num_rows=0,
        num_cols=0,
        file_centroids_offsets=None,
        pivot_height=0,
        pivot_offset=0,
        facet_width=0,
        facet_height=0,
    ):
        super(Heliostat, self).__init__()
        self.name = name
        self.origin = origin
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_facets = num_facets
        self.ncenterfacet = (self.num_facets + 1) / 2
        self.pivot_height = pivot_height
        self.pivot_offset = pivot_offset
        self.facet_width = facet_width
        self.facet_height = facet_height

        # Facets
        self.facets, self.facet_dict = self.facets_read_file(file_centroids_offsets)

        # Heliostat Edge Facets
        self.top_left_facet = self.facets[self.facet_dict[str(1)]]
        self.top_right_facet = self.facets[self.facet_dict[str(num_cols)]]
        self.bottom_right_facet = self.facets[self.facet_dict[str(num_facets)]]
        self.bottom_left_facet = self.facets[self.facet_dict[str(num_facets - num_cols + 1)]]

        # Heliostat Corners [offsets in terms of heliostat's centroid]
        self.top_left_corner_offset = [
            x + y for x, y in zip(self.top_left_facet.centroid_offset, self.top_left_facet.top_left_corner_offset)
        ]
        self.top_right_corner_offset = [
            x + y for x, y in zip(self.top_right_facet.centroid_offset, self.top_right_facet.top_right_corner_offset)
        ]
        self.bottom_right_corner_offset = [
            x + y
            for x, y in zip(self.bottom_right_facet.centroid_offset, self.bottom_right_facet.bottom_right_corner_offset)
        ]
        self.bottom_left_corner_offset = [
            x + y
            for x, y in zip(self.bottom_left_facet.centroid_offset, self.bottom_left_facet.bottom_left_corner_offset)
        ]

        # Centroid
        self.centroid = [origin[0], origin[1], origin[2] + pivot_offset]  # Origin is at torque tube center.

        self.az = np.deg2rad(180)  # (az,el) = (180,90) degrees corresponds to pointing straight up,
        self.el = np.deg2rad(90)  # as if transitioned by tilting up from face south orientation.
        self.surface_normal = [0, 0, 1]  #
        self.rx_rotation = np.identity(3)
        self.rz_rotation = np.identity(3)
        self.set_corner_positions_in_space()

        # Tracking
        self._aimpoint_xyz = (
            None  # (x,y,y) in m. Do not access this member externally; use aimpoint_xyz() function instead.
        )
        self._when_ymdhmsz = (
            None  # (y,m,d,h,m,s,z). Do not access this member externally; use when_ymdhmsz() function instead.
        )

    def facets_read_file(self, file):
        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            id_row = 0
            id_facet = 0
            facets = []
            facet_dict = {}
            for row in readCSV:
                if not id_row:
                    # get rid of the header in csv
                    id_row += 1
                    continue
                name, x, y, z = str(row[0]), float(row[1]), float(row[2]), float(row[3])

                # creating facet
                facet = Facet(name=name, centroid_offset=[x, y, z], width=self.facet_width, height=self.facet_height)

                # storing
                facets.append(facet)
                facet_dict[name] = id_facet
                id_facet += 1

        return facets, facet_dict

    # ACCESS

    def aimpoint_xyz(self):
        if self._aimpoint_xyz == None:
            print("ERROR: In Heliostat.aimpoint_xyz(), attempt to fetch unset _aimpoint_xyz.")
            assert False
        return self._aimpoint_xyz

    def when_ymdhmsz(self):
        if self._when_ymdhmsz == None:
            print("ERROR: In Heliostat.when_ymdhmsz(), attempt to fetch unset _when_ymdhmsz.")
            assert False
        return self._when_ymdhmsz

    def surface_normal_ray(self, base, length):
        # Constructs the head and tail of a vector of given length, placed at the base
        # position (computed after applying the heliostat configuration).
        #
        # This assumes that set_configuration() has already been called for
        # the current heliostat configuration.  This is required to set the internal
        # surface_normal.
        tail = base
        head = tail + (length * np.array(self.surface_normal))
        ray = [tail, head]
        return ray

    def compute_tracking_configuration(self, aimpoint_xyz, location_lon_lat, when_ymdhmsz):
        # Heliostat centroid coordinates.
        # Coordinates are (x,z) center, z=0 is at torque tube height.
        h_tube = np.array(self.origin)
        h = h_tube  # Later, add correction for center facet offset.

        # Compute heliostat surface normal which tracks the sun to the aimpoint.
        n_xyz = sun_track.tracking_surface_normal_xyz(h, aimpoint_xyz, location_lon_lat, when_ymdhmsz)

        # Compute heliostat configuration.
        return hc.heliostat_configuration_given_surface_normal_xyz(n_xyz)

    def compute_stow_configuration(self):
        # ?? SCAFFOLDING RCB -- FOR NOW, ASSUME RADIAL STOW.  MAKE CONTROLLABLE.
        origin_x = self.origin[0]
        origin_y = self.origin[1]
        theta = math.atan2(-origin_y, -origin_x)
        azimuth = ((5.0 / 2.0) * math.pi) - theta
        if azimuth > (2.0 * math.pi):
            azimuth -= 2.0 * math.pi
        return hc.HeliostatConfiguration(az=azimuth, el=np.deg2rad(90.0))

    def corners(self):
        # Assumes that heliostat coordinates have been set, and the corners have been set.
        # Later we can add a more meaningful check for this.
        return [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]

    # MODIFICATION

    def set_tracking(self, aimpoint_xyz, location_lon_lat, when_ymdhmsz):
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        self._when_ymdhmsz = when_ymdhmsz
        # Set tracking configuration.
        h_config = self.compute_tracking_configuration(aimpoint_xyz, location_lon_lat, when_ymdhmsz)
        self.set_configuration(h_config, clear_tracking=False)

    def set_stow(self):
        h_config = self.compute_stow_configuration()
        self.set_configuration(h_config, clear_tracking=True)

    def set_configuration(self, h_config, clear_tracking=True):
        # Clear tracking command.
        if clear_tracking:
            self._aimpoint_xyz = None
            self._when_ymdhmsz = None
        # Fetch azimuth and elevation parameters.
        el = h_config.el
        az = h_config.az

        # The heliostat begins face up, with its final link z axis pointing straight up.
        # To rotate it ot teh desired az,el) configuration, we first rotate it about the
        # x axis, and then about the z axis.  These angles are right-hand-rule rotations
        # about these axes.
        #
        #   1. For example, to make the heliostat point to an elevation of 30 degrees, we rotate
        #      60 degrees about the x axis.
        #
        #   2. As another example, to achieve a final configuration where the heliostat is vertical,
        #      facing south, we rotate 90 degrees about the x axis, and do not rotate about the z axis.
        #
        #   3. As a third example, to achieve a final configuration where the heliostat is vertical,
        #      facing east, we rotate 90 degrees about the x axis, and then rotate 90 degrees about the z axis.
        #
        # Meanwhile, azimuth and elevation are defined differently.  Elevation is measured up from horizontal,
        # and azimuth is measured clockwise from north (compass headings).  These two lines convert these (az,el)
        # conventions into the proper rotation angles about the x and z axes.
        #
        rotation_about_x = (np.pi / 2) - el
        rotation_about_z = np.pi - az

        Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
        Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)

        vector = np.array([0, 0, self.pivot_offset])
        vector_offset = Rz_rotation.dot(Rx_rotation).dot(vector)
        centroid = np.array(self.origin) + vector_offset  # Origin is at torque tube center.

        hel_rotation = Rz_rotation.dot(Rx_rotation)
        surface_normal = hel_rotation.dot([0, 0, 1])  # Before rotation, heliostat is face up.

        self.el = el
        self.az = az
        self.centroid = list(centroid)
        self.surface_normal = list(surface_normal)
        self.rx_rotation = Rx_rotation
        self.rz_rotation = Rz_rotation
        self.set_corner_positions_in_space()

    def set_corner_positions_in_space(self):
        # Sets corner positions given heliostat configuration.
        hel_centroid = np.array(self.centroid)
        hel_rotation = self.rz_rotation.dot(self.rx_rotation)
        self.top_left_corner = hel_centroid + hel_rotation.dot(np.array(self.top_left_corner_offset))
        self.top_right_corner = hel_centroid + hel_rotation.dot(np.array(self.top_right_corner_offset))
        self.bottom_right_corner = hel_centroid + hel_rotation.dot(np.array(self.bottom_right_corner_offset))
        self.bottom_left_corner = hel_centroid + hel_rotation.dot(np.array(self.bottom_left_corner_offset))

    # RENDERING

    def draw(self, view, heliostat_styles):
        # Assumes that heliostat configuration has already been set.

        # Fetch draw style control.
        heliostat_style = heliostat_styles.style(self.name)

        # Centroid.
        if heliostat_style.draw_centroid:
            view.draw_xyz(self.centroid, style=heliostat_style.centroid_style)

        # Outline.
        if heliostat_style.draw_outline:
            corners = [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]
            view.draw_xyz_list(corners, close=True, style=heliostat_style.outline_style)

        # Facets.
        if heliostat_style.draw_facets:
            hel_rotation = self.rz_rotation.dot(self.rx_rotation)
            hel_centroid = self.centroid
            for facet in self.facets:
                facet.draw(view, heliostat_style.facet_styles, hel_centroid, hel_rotation)

        # Surface normal.
        if heliostat_style.draw_surface_normal:
            # Construct ray.
            surface_normal_ray = self.surface_normal_ray(self.centroid, heliostat_style.surface_normal_length)
            # Draw ray and its base.
            view.draw_xyz(self.centroid, style=heliostat_style.surface_normal_base_style)
            view.draw_xyz_list(surface_normal_ray, style=heliostat_style.surface_normal_style)

        # Surface normal drawn at corners.
        # (Not the surface normal at the corner.  Facet curvature is not shown.)
        if heliostat_style.draw_surface_normal_at_corners:
            # Construct rays.
            top_left_ray = self.surface_normal_ray(self.top_left_corner, heliostat_style.corner_normal_length)
            top_right_ray = self.surface_normal_ray(self.top_right_corner, heliostat_style.corner_normal_length)
            bottom_left_ray = self.surface_normal_ray(self.bottom_left_corner, heliostat_style.corner_normal_length)
            bottom_right_ray = self.surface_normal_ray(self.bottom_right_corner, heliostat_style.corner_normal_length)
            rays = [top_left_ray, top_right_ray, bottom_left_ray, bottom_right_ray]
            # Draw each ray and its base.
            for base, ray in zip(corners, rays):
                view.draw_xyz(base, style=heliostat_style.corner_normal_base_style)
                view.draw_xyz_list(ray, style=heliostat_style.corner_normal_style)

        # Name.
        if heliostat_style.draw_name:
            view.draw_xyz_text(self.centroid, self.name, style=heliostat_style.name_style)
