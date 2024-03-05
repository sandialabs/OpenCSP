"""


"""

import csv
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

import opencsp.common.lib.csp.ufacet.Heliostat as Heliostat
import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.geometry.geometry_3d as g3d
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.list_tools as listt
import opencsp.common.lib.tool.log_tools as logt
import opencsp.common.lib.uas.Scan as Scan
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.csp.ufacet.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlSolarField import \
    RenderControlSolarField
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr


class SolarField(RayTraceable):
    """
    Representation of a heliostat solar field.
    """

    # CONSTRUCTION

    def __init__(self,
                 name: str,
                 short_name: str,
                 origin_lon_lat: list[float] | tuple[float, float],
                 heliostats: list[Heliostat.Heliostat]) -> None:
        # super(SolarField, self).__init__()

        # Input parameters.
        self.name = name
        self.short_name = short_name
        self.origin = np.array([0, 0, 0])
        self.rotation = Rotation.identity()
        self.origin_lon_lat = origin_lon_lat
        # self.heliostat_file = heliostat_file
        # self.facet_centroids_file = facet_centroids_file

        # Constructed members.
        # self.heliostats, self.heliostat_dict = self.heliostats_read_file(heliostat_file, facet_centroids_file)
        self.heliostats = heliostats
        self.heliostat_dict = {heliostat.name: i for i, heliostat in enumerate(heliostats)}
        self.num_heliostats = len(self.heliostats)
        self.heliostat_origin_xyz_list = [h.origin for h in self.heliostats]
        self.heliostat_origin_fit_plane = g3d.best_fit_plane(self.heliostat_origin_xyz_list)  # 3-d plane fit through heliostat origins.
        self._aimpoint_xyz = None   # (x,y,y) in m. Do not access this member externally; use aimpoint_xyz() function instead.
        self._when_ymdhmsz = None   # (y,m,d,h,m,s,z). Do not access this member externally; use when_ymdhmsz() function instead.

        self.set_position_in_space(self.origin, self.rotation)

    # required for RayTracable object but currently has no use
    def set_position_in_space(self, translation: np.ndarray, rotation: Rotation) -> None:
        for h in self.heliostats:
            h.set_position_in_space(translation + h.origin, rotation)

    # ACCESS

    def heliostat_name_list(self):
        name_list = [h.name for h in self.heliostats]
        name_list.sort()
        return name_list

    def aimpoint_xyz(self):
        if self._aimpoint_xyz == None:
            logt.error('ERROR: In SolarField.aimpoint_xyz(), attempt to fetch unset _aimpoint_xyz.')
            assert False
        return self._aimpoint_xyz

    def when_ymdhmsz(self):
        if self._when_ymdhmsz == None:
            logt.error('ERROR: In SolarField.when_ymdhmsz(), attempt to fetch unset _when_ymdhmsz.')
            assert False
        return self._when_ymdhmsz

    def lookup_heliostat(self, heliostat_name: str):
        # Check input.
        if heliostat_name not in self.heliostat_dict.keys():
            raise RuntimeError(f"ERROR: In lookup_heliostat.lookup_heliostat(), heliostat_name=\"{heliostat_name}\" not found in SolarField dictionary.")
        # Lookup the heliostat.
        heliostat_id = self.heliostat_dict[heliostat_name]
        heliostat = self.heliostats[heliostat_id]
        # Return.
        return heliostat

    def heliostat_plane_z(self, x: float, y: float) -> float:
        A = self.heliostat_origin_fit_plane[0]
        B = self.heliostat_origin_fit_plane[1]
        C = self.heliostat_origin_fit_plane[2]
        return (A * x) + (B * y) + C

    def heliostat_bounding_box_xyz(self):
        heliostat_xyz_list = self.heliostat_origin_xyz_list
        x_min = min([xyz[0] for xyz in heliostat_xyz_list])
        x_max = max([xyz[0] for xyz in heliostat_xyz_list])
        y_min = min([xyz[1] for xyz in heliostat_xyz_list])
        y_max = max([xyz[1] for xyz in heliostat_xyz_list])
        z_min = min([xyz[2] for xyz in heliostat_xyz_list])
        z_max = max([xyz[2] for xyz in heliostat_xyz_list])
        xyz_min: list[float] = [x_min, y_min, z_min]
        xyz_max: list[float] = [x_max, y_max, z_max]
        return [xyz_min, xyz_max]

    def heliostat_bounding_box_xy(self):
        heliostat_xyz_list = self.heliostat_origin_xyz_list
        x_min = min([xyz[0] for xyz in heliostat_xyz_list])
        x_max = max([xyz[0] for xyz in heliostat_xyz_list])
        y_min = min([xyz[1] for xyz in heliostat_xyz_list])
        y_max = max([xyz[1] for xyz in heliostat_xyz_list])
        xy_min: list[float] = [x_min, y_min]
        xy_max: list[float] = [x_max, y_max]
        return [xy_min, xy_max]

    def heliostat_field_regular_grid_xy(self, n_x: int, n_y: int):
        bbox_xy = self.heliostat_bounding_box_xy()
        xy_min = bbox_xy[0]
        x_min = xy_min[0]
        y_min = xy_min[1]
        xy_max = bbox_xy[1]
        x_max = xy_max[0]
        y_max = xy_max[1]
        grid_xy: list[list[float]] = []
        for x in np.linspace(x_min, x_max, n_x):
            for y in np.linspace(y_min, y_max, n_y):
                grid_xy.append([x, y])
        return grid_xy

    def situation_abbrev(self):
        year = self.when_ymdhmsz()[0]
        month = self.when_ymdhmsz()[1]
        day = self.when_ymdhmsz()[2]
        hour = self.when_ymdhmsz()[3]
        minute = self.when_ymdhmsz()[4]
        date_time = '{0:d}-{1:02d}-{2:02d}-{3:02d}{4:02d}'.format(year, month, day, hour, minute)
        aim_Z = 'aimZ=' + str(self.aimpoint_xyz()[2])
# ?? SCAFFOLDING RCB -- MAKE THIS CONTROLLABLE: ON FOR NSTTF, OFF OTHERWISE.
#        return self.short_name + '_' + date_time + '_' + aim_Z
#        return self.short_name + '_' + date_time
# ?? SCAFFOLDING RCB -- MAKE THIS CONTROLLABLE, DEPENDING ON USER PREFERENCE.
        return (date_time + '_' + self.short_name)

    def situation_str(self):
        year = self.when_ymdhmsz()[0]
        month = self.when_ymdhmsz()[1]
        day = self.when_ymdhmsz()[2]
        hour = self.when_ymdhmsz()[3]
        minute = self.when_ymdhmsz()[4]
        date_time = '{0:d}-{1:d}-{2:d} at {3:02d}{4:02d}'.format(year, month, day, hour, minute)
        aim_pt = 'Aim=({0:.1f}, {1:.1f}, {2:.1f})'.format(self.aimpoint_xyz()[0], self.aimpoint_xyz()[1], self.aimpoint_xyz()[2])
        # ?? SCAFFOLDING RCB -- MAKE THIS CONTROLLABLE: ON FOR NSTTF, OFF OTHERWISE.
#        return self.name + ', ' + date_time + ', ' + aim_pt
        return self.name + ', ' + date_time

    # MODIFICATION

    def set_full_field_tracking(self, aimpoint_xyz: np.ndarray, when_ymdhmsz: tuple):
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        self._when_ymdhmsz = when_ymdhmsz
        # Set each heliostat tracking.
        for heliostat in self.heliostats:
            heliostat.set_tracking(aimpoint_xyz, self.origin_lon_lat, when_ymdhmsz)

    def set_full_field_stow(self):
        for heliostat in self.heliostats:
            heliostat.set_stow()

    def set_full_field_face_up(self):
        for heliostat in self.heliostats:
            heliostat.set_face_up()

    def set_heliostats_configuration(self, heliostat_name_list_to_set: list[str], h_config: HeliostatConfiguration) -> None:
        # If all heliostats are reset, then clear tracking command.
        all_heliostats = set(list(self.heliostat_dict.keys()))
        set_heliostats = set(heliostat_name_list_to_set)
        if len(all_heliostats - set_heliostats) == 0:
            self._aimpoint_xyz = None
            self._when_ymdhmsz = None
        for heliostat_name in heliostat_name_list_to_set:
            heliostat = self.lookup_heliostat(heliostat_name)
            heliostat.set_configuration(h_config)

    # RENDERING

    def draw(self, view: View3d, solar_field_style: RenderControlSolarField = RenderControlSolarField()) -> None:
        # Heliostats.
        if solar_field_style.draw_heliostats:
            for heliostat in self.heliostats:
                heliostat.draw(view, solar_field_style.heliostat_styles)

    def draw_figure(self,
                    figure_control,
                    axis_control_m,
                    view_spec,
                    title,
                    solar_field_style,
                    grid=True):
        # Setup view
        fig_record: rcfr.RenderControlFigureRecord = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, grid=grid, title=title)
        view = fig_record.view
        # Draw
        self.draw(view, solar_field_style)
        # Return
        return view

    def survey_of_points(self, resolution, random_dist: bool = False) -> tuple[Pxyz, Vxyz]:
        """
        Returns a grid of equispaced points and the normal vectors at those points.

        Parameters
        ----------
        resolution:
            the rectangular resolution of the points gathered (add other forms later, like triangular or polar survey).

        Returns
        -------
            a tuple of the points (np.ndarray) and normals at the respective points (np.ndarray).

        """
        points = Pxyz([[], [], []])
        normals = Vxyz([[], [], []])
        for heliostat in self.heliostats:
            additional_points, additional_normals = heliostat.survey_of_points(resolution, random_dist)
            points = points.concatenate(additional_points)
            normals = normals.concatenate(additional_normals)

        return (points, normals)

# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#


def setup_solar_field(solar_field_spec,
                      aimpoint_xyz,
                      when_ymdhmsz,
                      synch_azelhnames,
                      up_azelhnames) -> SolarField:
    # Notify progress.
    logt.info('Setting up solar field...')

    # Load solar field data.
    solar_field = sf_from_csv_files(solar_field_spec['name'],
                                    solar_field_spec['short_name'],
                                    solar_field_spec['field_origin_lon_lat'],
                                    solar_field_spec['field_heliostat_file'],
                                    solar_field_spec['field_facet_centroids_file'])

    # Set heliostat configurations.
    solar_field.set_full_field_tracking(aimpoint_xyz, when_ymdhmsz)

    # Set synchronized heliostats.
    if synch_azelhnames != None:
        synch_az = synch_azelhnames[0]
        synch_el = synch_azelhnames[1]
        synch_h_name_list = synch_azelhnames[2]
        synch_h_config = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        solar_field.set_heliostats_configuration(synch_h_name_list, synch_h_config)

    # Set up heliostats.
    if up_azelhnames != None:
        up_az = up_azelhnames[0]
        up_el = up_azelhnames[1]
        up_h_name_list = up_azelhnames[2]
        up_h_config = hc.HeliostatConfiguration(az=up_az, el=up_el)
        solar_field.set_heliostats_configuration(up_h_name_list, up_h_config)

    # Notify result.
    logt.info('Solar field situation: ' + solar_field.situation_str())

    # Return.
    return solar_field


# -------------------------------------------------------------------------------------------------------
# TOP-LEVEL RENDERING ROUTINES
#

def draw_solar_field(figure_control,
                     solar_field,
                     solar_field_style,
                     view_spec,
                     name_suffix='',
                     axis_equal=True):
    # Assumes that solar field is already set up with heliostat configurations, etc.
    # Select name and title.
    if (solar_field.short_name == None) or (len(solar_field.short_name) == 0):
        figure_name = 'Solar Field'
    else:
        figure_name = solar_field.short_name
    if (solar_field.name == None) or (len(solar_field.name) == 0):
        figure_title = 'Solar Field'
    else:
        figure_title = solar_field.name
    if (name_suffix != None) and (len(name_suffix) > 0):
        figure_name += '_' + name_suffix
        figure_title += ' (' + name_suffix + ')'
    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(figure_control, rca.meters(), view_spec,
                                             name=figure_name,
                                             title=figure_title,
                                             equal=axis_equal)
    view = fig_record.view
    # Comment.
    fig_record.comment.append("Solar field.")
    # Draw.
    solar_field.draw(view, solar_field_style)
    # Finish.
    view.show()
    # Return.
    return view


# -------------------------------------------------------------------------------------------------------
# RASTER SURVEY SCANS
#

def construct_solar_field_heliostat_survey_scan(solar_field, raster_scan_parameters):
    # Fetch control parameters.
    n_horizontal = raster_scan_parameters['n_horizontal']
    n_vertical = raster_scan_parameters['n_vertical']

    # Construct segments spanning the region of interest.
    box_xy = solar_field.heliostat_bounding_box_xy()
    xy_min = box_xy[0]
    x_min = xy_min[0]
    y_min = xy_min[1]
    xy_max = box_xy[1]
    x_max = xy_max[0]
    y_max = xy_max[1]

    # North-South passes.
    vertical_segments = []
    if n_vertical > 0:
        for x in np.linspace(x_min, x_max, n_vertical):
            x0 = x
            y0 = y_min
            z0 = solar_field.heliostat_plane_z(x0, y0)
            x1 = x
            y1 = y_max
            z1 = solar_field.heliostat_plane_z(x1, y1)
            vertical_segments.append([[x0, y0, z0], [x1, y1, z1]])
        # If applicable, reverse so that we scan further distance first, and then move closer to base.
        if raster_scan_parameters['reverse_vertical']:
            vertical_segments.reverse()
    # East-West passes.
    horizontal_segments = []
    if n_horizontal > 0:
        for y in np.linspace(y_min, y_max, n_horizontal):
            x0 = x_min
            y0 = y
            z0 = solar_field.heliostat_plane_z(x0, y0)
            x1 = x_max
            y1 = y
            z1 = solar_field.heliostat_plane_z(x1, y1)
            horizontal_segments.append([[x0, y0, z0], [x1, y1, z1]])
        # If applicable, reverse so that we scan further distance first, and then move closer to base.
        if raster_scan_parameters['reverse_horizontal']:
            horizontal_segments.reverse()
    # All passes.
    list_of_xyz_segments = vertical_segments + horizontal_segments

    # Construct the scan.
    scan = Scan.construct_scan_given_segments_of_interest(list_of_xyz_segments, raster_scan_parameters)

    # Return.
    return scan


# -------------------------------------------------------------------------------------------------------
# VANITY SCANS
#

def construct_solar_field_vanity_scan(solar_field, vanity_scan_parameters):
    # Fetch control parameters.
    vanity_heliostat_name = vanity_scan_parameters['vanity_heliostat_name']
    n_horizontal = vanity_scan_parameters['n_horizontal']
    n_vertical = vanity_scan_parameters['n_vertical']
    facet_array_width = vanity_scan_parameters['facet_array_width']
    facet_array_height = vanity_scan_parameters['facet_array_height']

# # Construct segments spanning the region of interest.
# box_xy = solar_field.heliostat_bounding_box_xy()
# xy_min = box_xy[0]
# x_min = xy_min[0]
# y_min = xy_min[1]
# xy_max = box_xy[1]
# x_max = xy_max[0]
# y_max = xy_max[1]

    # Fetch the heliostat of interest.
    vanity_heliostat = solar_field.lookup_heliostat(vanity_heliostat_name)

    # Construct segments spanning the region of interest.
# x_origin = -200  # m.  # ?? SCAFFOLDING RCB -- TEMPORARY
# y_origin =  700  # m.
    x_origin = vanity_heliostat.origin[0]
    y_origin = vanity_heliostat.origin[1]
    half_width = facet_array_width / 2.0
    half_height = facet_array_height / 2.0
    x_min = x_origin - half_width
    x_max = x_origin + half_width
    y_min = y_origin - half_height
    y_max = y_origin + half_height

    # North-South passes.
    vertical_segments = []
    if n_vertical > 0:
        for x in np.linspace(x_min, x_max, n_vertical):
            x0 = x
            y0 = y_min
            z0 = solar_field.heliostat_plane_z(x0, y0)
            x1 = x
            y1 = y_max
            z1 = solar_field.heliostat_plane_z(x1, y1)
            vertical_segments.append([[x0, y0, z0], [x1, y1, z1]])
    # East-West passes.
    horizontal_segments = []
    if n_horizontal > 0:
        for y in np.linspace(y_min, y_max, n_horizontal):
            x0 = x_min
            y0 = y
            z0 = solar_field.heliostat_plane_z(x0, y0)
            x1 = x_max
            y1 = y
            z1 = solar_field.heliostat_plane_z(x1, y1)
            horizontal_segments.append([[x0, y0, z0], [x1, y1, z1]])
    # All passes.
    # Shuffle to avoid passes that are too close to each other for the UAS to distinguish.
    list_of_xyz_segments = listt.zamboni_shuffle(vertical_segments) + listt.zamboni_shuffle(horizontal_segments)

    logt.info('In construct_solar_field_vanity_scan(), number of segments = ', len(list_of_xyz_segments))

    # Rotate to stow azmiuth.
    rotated_segments = []
    for xyz_segment in list_of_xyz_segments:
        # Fetch points.
        xyz0 = xyz_segment[0]
        xyz1 = xyz_segment[1]
        # Fetch rotation angle.
        dtheta = np.pi - vanity_heliostat.az
        # Fetch rotation center.
        center_xy = [x_origin, y_origin]
        # Rotate points.
        xyz0r = g2d.rotate_xyz_about_center_xy(xyz0, dtheta, center_xy)
        xyz1r = g2d.rotate_xyz_about_center_xy(xyz1, dtheta, center_xy)
        # Assemble and save rotated segment.
        rotated_segments.append([xyz0r, xyz1r])

    # Construct the scan.
    scan = Scan.construct_scan_given_segments_of_interest(rotated_segments, vanity_scan_parameters)

    # Return.
    return scan

# GENERATORS


def sf_from_csv_files(name: str,
                      short_name: str,
                      origin_lon_lat: list[float],
                      heliostat_file: str,
                      facet_centroids_file: str,
                      autoset_canting_and_curvature: np.ndarray = None) -> SolarField:
    """ Reads in a list of heliostats from heliostat_file with heliostats_read_file,
    and a list of facets to populate the heliostats with from Heliostat.facets_read_file """

    # Constructed members.
    heliostats: list[Heliostat.Heliostat]
    heliostats, _ = heliostats_read_file(heliostat_file, facet_centroids_file, autoset_canting_and_curvature)
    return SolarField(name, short_name, origin_lon_lat, heliostats)


def heliostats_read_file(file_field: str, file_centroids_offsets: str, autoset_canting_and_curvature: np.ndarray = None) -> tuple[list[Heliostat.Heliostat], dict[str, int]]:
    """ Reads in a list of heliostats from the given file_field.

    Arguments
    ---------
        file_field: one heliostat per row, with the format:
            name, x, y, z, num_facets, num_rows, num_cols, pivot_height, pivot_offset, facet_width, facet_height

    Returns
    -------
        A list of heliostats, and a dict[heliostat_name:heliostat_id]

    See Also
    --------
        Heliostat.facets_read_file """
    with open(file_field) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        id_row = 0
        id_heliostat = 0
        heliostat_dict = {}
        heliostats = []
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

            # creating heliostat
            if autoset_canting_and_curvature is None:
                curvature_func: Callable[[float, float], float] = lambda x, y: x * 0
            else:
                foc_len = np.linalg.norm(autoset_canting_and_curvature - np.array([x, y, z]))
                a = 1.0 / (4 * foc_len)
                def curvature_func(x, y): return a * (x**2 + y**2)

            heliostat = Heliostat.h_from_facet_centroids(name=name, origin=[x, y, z],
                                                         num_facets=num_facets, num_rows=num_rows, num_cols=num_cols,
                                                         file_centroids_offsets=file_centroids_offsets,
                                                         pivot_height=pivot_height, pivot_offset=pivot_offset,
                                                         facet_width=facet_width, facet_height=facet_height, default_mirror_shape=curvature_func)
            heliostat.set_canting_from_equation(curvature_func)

            # storing
            heliostats.append(heliostat)
            heliostat_dict[name] = id_heliostat
            id_heliostat += 1

    return heliostats, heliostat_dict
