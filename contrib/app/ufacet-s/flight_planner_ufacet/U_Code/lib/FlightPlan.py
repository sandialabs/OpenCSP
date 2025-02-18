"""


"""

import copy
import os

import opencsp.common.lib.tool.file_tools as ft


class FlightPlan:
    """
    A plan for a flight, comprised of a sequence of scans.
    """

    def __init__(
        self,
        name,
        short_name,
        locale,  # String used to specify template file location and name.
        launch_name,  # String describing name of launch point within locale, used to specify template file name.
    ):
        super(FlightPlan, self).__init__()

        # Input parameters.
        self.name = name
        self.short_name = short_name
        self.locale = locale
        self.launch_name = launch_name
        self.scans = None  # Only set if this flight plan is constructed from scans.
        self._waypoint_list = None  # Do not access this member externally; use waypoints() function instead.

    def set_waypoints_from_scans(self, scans):
        # Set data member.
        self.scans = scans
        # Set scan pass index values.
        next_scan_index = 0
        for scan in self.scans:
            next_scan_index = scan.set_scan_pass_numbers(next_scan_index)
        # Construct waypoints.
        waypoint_list = []
        for scan in self.scans:
            waypoint_list += scan.waypoints()
        # ?? SCAFFOLDING RCB -- MAKE WHETHER TO DO THIS CONTROLLABLE.
        # Add preliminary waypoint, to ensure gimbal is established for first scan pass.
        if len(waypoint_list) == 0:
            print("ERROR: In FlightPlan.set_waypoints_from_scans(), unexpected empty waypoint list encountered.")
            assert False
        first_scan_waypt = waypoint_list[0]
        first_scan_x = first_scan_waypt.xyz[0]
        pre_scan_waypt = copy.deepcopy(first_scan_waypt)
        pre_scan_xyz = pre_scan_waypt.xyz
        # ?? SCAFFOLDING RCB -- MAKE THIS SHIFT MAGNITUDE CONTROLLABLE.
        pre_scan_dx = -5.0  # m.
        pre_scan_x = first_scan_x + pre_scan_dx
        pre_scan_xyz[0] = pre_scan_x
        pre_scan_waypt.xyz = pre_scan_xyz
        pre_scan_waypt.set_longitude_latitude()
        waypoint_list = [pre_scan_waypt] + waypoint_list
        # Set data member.
        self._waypoint_list = waypoint_list
        # Set waypoint index values.
        self.set_waypoint_numbers()

    def set_waypoint_numbers(self):
        idx = 0
        for waypoint in self.waypoints():
            waypoint.idx = idx
            idx += 1

    def waypoints(self):
        if self._waypoint_list == None:
            print("ERROR: In FlightPlan.waypoints(), attempt to fetch unset _waypoint_list.")
            assert False
        return self._waypoint_list

    #     def save_to_csv(self,
    #                     output_path,        # Directory to write to.
    #                     elevation_offset):  # m.
    #         output_file_body = ft.convert_string_to_file_body(self.name)  # ?? SCAFFOLDING RCB -- ADD LOCATION, DATE AND TIME, VARIATIONS, ETC.
    #         output_file = output_file_body + '.csv'
    #         output_path_file = os.path.join(output_path, output_file)
    #         print('Saving flight plan file: ', output_path_file)
    #         output_stream = open(output_path_file, 'w')
    #         # Write heading.
    #         output_stream.write('Waypoint, ')
    #         output_stream.write('x (m), ')
    #         output_stream.write('y (m), ')
    #         output_stream.write('z (m), ')
    #         output_stream.write('theta (rad), ')
    #         output_stream.write('eta (rad), ')
    #         output_stream.write('longitude (deg), ')
    #         output_stream.write('latitude (deg), ')
    #         output_stream.write('altitude wrt origin (m), ')
    #         output_stream.write('heading (deg), ')
    #         output_stream.write('gimbal pitch (deg), ')
    #         output_stream.write('\n')
    #         # Write data rows.
    #         i = 1
    #         for waypoint in self.waypoint_list():
    #             # Write the data row.
    #             output_stream.write('{0:d}, '.format(i))
    #             output_stream.write('{0:.3f}, '.format(waypoint.xyz[0]))
    #             output_stream.write('{0:.3f}, '.format(waypoint.xyz[1]))
    #             output_stream.write('{0:.3f}, '.format(waypoint.xyz[2]))
    #             output_stream.write('{0:.7f}, '.format(waypoint.theta))
    #             output_stream.write('{0:.7f}, '.format(waypoint.eta))
    #             output_stream.write('{0:.8f}, '.format(waypoint.lon))
    #             output_stream.write('{0:.8f}, '.format(waypoint.lat))
    #             output_stream.write('{0:.3f}, '.format(waypoint.xyz[2]))
    #             output_stream.write('{0:.6f}, '.format(waypoint.heading_deg()))
    #             output_stream.write('{0:.6f}'.format(waypoint.gimbal_pitch_deg()))
    #             output_stream.write('\n')
    #             i += 1
    #         output_stream.close()

    def save_to_litchi_csv(self, output_path, elevation_offset):  # Directory to write to.  # m.
        # Construct input template file path.
        template_dir_path = os.path.join("..", "U_Code_data", self.locale)
        template_base_name = "Litchi_Template"
        locale_file_str = ft.convert_string_to_file_body(self.locale)
        launch_file_str = ft.convert_string_to_file_body(self.launch_name)
        template_file_name = template_base_name + "_" + locale_file_str + "_" + launch_file_str + ".csv"
        template_path_file = os.path.join(template_dir_path, template_file_name)
        # Read input template.
        template_lines = []
        input_stream = open(template_path_file, "r")
        for line in input_stream:
            template_lines.append(line)
        input_stream.close()

        # Check input.
        if len(template_lines) < 3:
            print("ERROR: In FlightPlan.save_to_litchi_csv(), fewer than three lines in template.")
            assert False

        # Find indices of key column headings.
        heading_line = template_lines[0]
        heading_list = heading_line.split(",")
        longitude_idx = heading_list.index("longitude")
        latitude_idx = heading_list.index("latitude")
        altitude_idx = heading_list.index("altitude(m)")
        heading_idx = heading_list.index("heading(deg)")
        gimbal_pitch_idx = heading_list.index("gimbalpitchangle")

        # Select data template line.
        data_template_line = template_lines[-1]
        data_template_list = data_template_line.split(",")

        # Write output file.
        output_file_body = ft.convert_string_to_file_body(self.short_name)
        # ?? SCAFFOLDING RCB -- MAKE THIS CONTROLLABLE, DEPENDING ON USER PREFERENCE.
        #        output_file = self.launch_name + '_' + output_file_body + '.csv'
        output_file_name = output_file_body
        if elevation_offset != 0:
            output_file_name += "_dz={0:.1f}m".format(elevation_offset)
        output_file_name = output_file_name + "_" + self.launch_name + ".csv"
        output_path_file = os.path.join(output_path, output_file_name)
        print("Saving flight plan file: ", output_path_file)
        output_stream = open(output_path_file, "w")
        # Write heading lines.
        for line in template_lines[0:-1]:
            output_stream.write(line)
        # Write data rows.
        for waypoint in self.waypoints():
            # Prepare the data row.
            data_list = data_template_list.copy()
            data_list[longitude_idx] = "{0:.8f}".format(waypoint.lon)
            data_list[latitude_idx] = "{0:.8f}".format(waypoint.lat)
            data_list[altitude_idx] = "{0:.3f}".format(waypoint.xyz[2] + elevation_offset)
            data_list[heading_idx] = "{0:.6f}".format(waypoint.heading_deg())
            data_list[gimbal_pitch_idx] = "{0:.6f}".format(waypoint.gimbal_pitch_deg())
            data_line = ",".join(data_list)
            output_stream.write(data_line)
        # Add return-to-start waypoint.
        if len(template_lines) != 3:
            print(
                "ERROR: In FlightPlan.save_to_litchi_csv(), unexpected len(template_lines) = "
                + str(len(template_lines))
                + " encountered."
            )
            assert False
        output_stream.write(template_lines[-2])
        output_stream.close()

    def draw_outline(self, view, flight_plan_style):
        xyz_list = [w.xyz for w in self.waypoints()]
        view.draw_xyz_list(xyz_list)

    def draw(self, view, flight_plan_style):
        # Waypoints.
        if flight_plan_style.draw_waypoints:
            for waypoint in self.waypoints():
                waypoint.draw(view, flight_plan_style.waypoint_styles)
        # Scan.
        if flight_plan_style.draw_scan and (self.scans != None):
            for scan in self.scans:
                scan.draw(view, flight_plan_style.scan_pass_styles)
        # Outline (draw last).
        if flight_plan_style.draw_outline:
            self.draw_outline(view, flight_plan_style)


# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#


def construct_flight_plan_from_scan(name, short_name, launch_name, scan):  # Scan object.
    return construct_flight_plan_from_scans(name, short_name, launch_name, [scan])


def construct_flight_plan_from_scans(name, short_name, launch_name, scans):  # List of scan opbjects.
    # Notify progress.
    print("Constructing flight plan...")

    # Check input.
    if len(scans) == 0:
        print("In construct_flight_plan_from_scans(), empty list of scans encountered.")
        assert False

    # Fetch locale.
    locale = scans[0].locale()

    # Construct flight plan.
    flight_plan = FlightPlan(name, short_name, locale, launch_name)
    flight_plan.set_waypoints_from_scans(scans)
    return flight_plan
