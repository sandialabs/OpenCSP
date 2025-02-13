"""
Define scan configuration -- NSTTF.

These are shorthand routines for setting up some relevant test NSTTF configurations.

This file was created to move cruft out of the toplevel experiment file.  It is not intended to be final code.

Later these should be eliminated by defining input configuration files, and also a 
test suite for checking various standard cases.



"""

import os
import numpy as np

import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.opencsp_root_path as root_path
import opencsp.common.lib.tool.file_tools as ft


def define_scan_NSTTF_half_and_half(solar_field_short_name):
    """
    Generates flight plan for half-and-half NSTTF field configuraiton.
    However, the soalr field model has full field set to common configuration.
    Do not set the actual full field to this  configuration, because you'll flash the tower.
    This half-and-hlaf configuration is only safe for time past solar noon (roughly).

    This setup achieves this by hacking the aimpoitn and time to fool the UFACET scan path
    generation to produce something reasonable.
    These hack parameters were selected manually by trial and error.  They are approximate.
    """
    # Solar field spec.
    solar_field_spec = {}
    solar_field_spec["name"] = "Sandia NSTTF"
    solar_field_spec["short_name"] = "NSTTF"
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_heliostat_file"] = "../U_Code_data/NSTTF/NSTTF_Heliostats_origin_at_torque_tube.csv"
    solar_field_spec["field_facet_centroids_file"] = "../U_Code_data/NSTTF/NSTTF_Facet_Centroids.csv"

    # Define UFACET control flags.
    ufacet_control_parameters = {}
    # Define key points for gaze curve construction.
    # Seed points.
    ufacet_curve_keys_x = np.linspace(-131.7, 131.7, 28)
    ufacet_curve_keys_y = [136.9] * len(ufacet_curve_keys_x)
    ufacet_curve_key_xy_list = [[key_x, key_y] for key_x, key_y in zip(ufacet_curve_keys_x, ufacet_curve_keys_y)]
    ufacet_control_parameters["curve_key_xy_list"] = ufacet_curve_key_xy_list
    # Maximum altitude.
    # Half-and-Half
    ufacet_control_parameters["maximum_altitude"] = 18.0  # m.  Maximum altitude, roughly AGL, including slope effects.
    # Gaze control.
    ufacet_control_parameters["gaze_type"] = "constant"  # 'constant' or 'linear'
    ufacet_control_parameters["delta_eta"] = np.deg2rad(
        0.0
    )  # deg.  Offset to add to gaze angle eta.  Set to zero for no offset.

    # Define tracking parameters.
    # Aim point.
    # Half-and-Half
    aimpoint_xyz = [
        10000 * np.cos(np.deg2rad(450 - 145)),
        10000 * np.sin(np.deg2rad(450 - 145)),
        1000.0,
    ]  # For half-and-half

    # Time.
    # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    #
    # Zone: Mountain Daylight Time (MDT) = -6,  Mountain Standard Time (MST) = -7
    #
    #               year, month, day, hour, minute, second, zone]
    # Half-and-Half
    # when_ymdhmsz = [2021,   5,   13,    11,    0,       0,    -6]  # For half-and-half.  Was this one used on May 13, 2021?
    when_ymdhmsz = [2021, 5, 13, 12, 30, 0, -6]  # For half-and-half

    # Define fixed heliostat orientation.
    # Half-and-Half
    synch_az = np.deg2rad(145)
    synch_el = np.deg2rad(35)
    synch_azelhnames = [
        synch_az,
        synch_el,
        [
            "14E1",
            "14E2",
            "14E3",
            "14E4",
            "14E5",
            "14E6",
            "13E1",
            "13E2",
            "13E3",
            "13E4",
            "13E5",
            "13E6",
            "13E7",
            "13E8",
            "13E9",
            "13E10",
            "13E11",
            "13E12",
            "13E13",
            "13E14",
            "12E1",
            "12E2",
            "12E3",
            "12E4",
            "12E5",
            "12E6",
            "12E7",
            "12E8",
            "12E9",
            "12E10",
            "12E11",
            "12E12",
            "12E13",
            "12E14",
            "11E1",
            "11E2",
            "11E3",
            "11E4",
            "11E5",
            "11E6",
            "11E7",
            "11E8",
            "11E9",
            "11E10",
            "11E11",
            "11E12",
            "11E13",
            "11E14",
            "10E1",
            "10E2",
            "10E3",
            "10E4",
            "10E5",
            "10E6",
            "10E7",
            "10E8",
            "10E9",
            "10E10",
            "10E11",
            "10E12",
            "9E1",
            "9E2",
            "9E3",
            "9E4",
            "9E5",
            "9E6",
            "9E7",
            "9E8",
            "9E9",
            "9E10",
            "9E11",
            "8E1",
            "8E2",
            "8E3",
            "8E4",
            "8E5",
            "8E6",
            "8E7",
            "8E8",
            "8E9",
            "8E10",
            "7E1",
            "7E2",
            "7E3",
            "7E4",
            "7E5",
            "7E6",
            "7E7",
            "7E8",
            "7E9",
            "6E1",
            "6E2",
            "6E3",
            "6E4",
            "6E5",
            "6E6",
            "6E7",
            "6E8",
            "6E9",
            "5E1",
            "5E2",
            "5E3",
            "5E4",
            "5E5",
            "5E6",
            "5E7",
            "5E8",
            "5E9",
            "5E10",
            "14W1",
            "14W2",
            "14W3",
            "14W4",
            "14W5",
            "14W6",
            "13W1",
            "13W2",
            "13W3",
            "13W4",
            "13W5",
            "13W6",
            "13W7",
            "13W8",
            "13W9",
            "13W10",
            "13W11",
            "13W12",
            "13W13",
            "13W14",
            "12W1",
            "12W2",
            "12W3",
            "12W4",
            "12W5",
            "12W6",
            "12W7",
            "12W8",
            "12W9",
            "12W10",
            "12W11",
            "12W12",
            "12W13",
            "12W14",
            "11W1",
            "11W2",
            "11W3",
            "11W4",
            "11W5",
            "11W6",
            "11W7",
            "11W8",
            "11W9",
            "11W10",
            "11W11",
            "11W12",
            "11W13",
            "11W14",
            "10W1",
            "10W2",
            "10W3",
            "10W4",
            "10W5",
            "10W6",
            "10W7",
            "10W8",
            "10W9",
            "10W10",
            "10W11",
            "10W12",
            "9W1",
            "9W2",
            "9W3",
            "9W4",
            "9W5",
            "9W6",
            "9W7",
            "9W8",
            "9W9",
            "9W10",
            "9W11",
            "8W1",
            "8W2",
            "8W3",
            "8W4",
            "8W5",
            "8W6",
            "8W7",
            "8W8",
            "8W9",
            "8W10",
            "7W1",
            "7W2",
            "7W3",
            "7W4",
            "7W5",
            "7W6",
            "7W7",
            "7W8",
            "7W9",
            "6W1",
            "6W2",
            "6W3",
            "6W4",
            "6W5",
            "6W6",
            "6W7",
            "6W8",
            "6W9",
            "5W1",
            "5W2",
            "5W3",
            "5W4",
            "5W5",
            "5W6",
            "5W7",
            "5W8",
            "5W9",  #'5W10',
        ],
    ]

    # Define upward-facing heliostat orientation.
    up_azelhnames = None

    # Return.
    return (solar_field_spec, ufacet_control_parameters, aimpoint_xyz, when_ymdhmsz, synch_azelhnames, up_azelhnames)


def define_scan_NSTTF_demo(solar_field_short_name):
    """
    Generates flight plan with field rendering showing some heliostsa either face up or in a common fixed configuration.
    The path planner doesn't know abouthtese out-of-the-ordinary heliostats; it plans as if the full field is up.
    But the field representation models these differences, which influences rendering and could be exploited in later analysis.
    """
    # Solar field spec.
    solar_field_spec = {}
    solar_field_spec["name"] = "Sandia NSTTF"
    solar_field_spec["short_name"] = "NSTTF"
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_heliostat_file"] = "../U_Code_data/NSTTF/NSTTF_Heliostats_origin_at_torque_tube.csv"
    solar_field_spec["field_facet_centroids_file"] = "../U_Code_data/NSTTF/NSTTF_Facet_Centroids.csv"

    # Define UFACET control flags.
    ufacet_control_parameters = {}
    # Define key points for gaze curve construction.
    # Seed points.
    ufacet_curve_keys_x = np.linspace(-131.7, 131.7, 28)
    ufacet_curve_keys_y = [136.9] * len(ufacet_curve_keys_x)
    ufacet_curve_key_xy_list = [[key_x, key_y] for key_x, key_y in zip(ufacet_curve_keys_x, ufacet_curve_keys_y)]
    ufacet_control_parameters["curve_key_xy_list"] = ufacet_curve_key_xy_list
    # Maximum altitude.
    ufacet_control_parameters["maximum_altitude"] = 25.0  # m.  Maximum altitude, roughly AGL, including slope effects.
    # Gaze control.
    ufacet_control_parameters["gaze_type"] = "constant"  # 'constant' or 'linear'
    ufacet_control_parameters["delta_eta"] = np.deg2rad(
        0.0
    )  # deg.  Offset to add to gaze angle eta.  Set to zero for no offset.

    # Define tracking parameters.
    # Aim point.
    aimpoint_xyz = [60.0, 8.8, 28.9]  # NSTTF BCS standby - low

    # Time.
    # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    #
    # Zone: Mountain Daylight Time (MDT) = -6,  Mountain Standard Time (MST) = -7
    #
    #               year, month, day, hour, minute, second, zone]
    when_ymdhmsz = [2021, 5, 13, 13, 0, 0, -6]

    # Define fixed heliostat orientation.
    # Demo.
    synch_az = np.deg2rad(205)
    synch_el = np.deg2rad(30)
    synch_azelhnames = [synch_az, synch_el, ["8W3", "8W4", "8W5", "7W3", "7W4", "6W5", "6W3", "6W4", "6W5"]]

    # Define upward-facing heliostat orientation.
    # Demo.
    up_az = np.deg2rad(180)
    up_el = np.deg2rad(90)
    up_azelhnames = [up_az, up_el, ["7E6", "12W7"]]

    # Return.
    return (solar_field_spec, ufacet_control_parameters, aimpoint_xyz, when_ymdhmsz, synch_azelhnames, up_azelhnames)


def define_scan_NSTTF_full_field(solar_field_short_name):
    """
    Simple case where the full field is set to tracking a single aim point.
    """
    basedir = os.path.join(root_path.opencsp.dir(), INSERT_CORRECT_DIRECTORY_PATH_HERE)  # TODO: Fill-in correct path.

    # Solar field spec.
    solar_field_spec = {}
    solar_field_spec["name"] = "Sandia NSTTF"
    solar_field_spec["short_name"] = "NSTTF"
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_origin_lon_lat"] = (lln.LON_NSTTF_ORIGIN, lln.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_heliostat_file"] = os.path.join(basedir, "Solar_Field.csv")
    solar_field_spec["field_facet_centroids_file"] = os.path.join(basedir, "Facets_Centroids.csv")

    # Define UFACET control flags.
    ufacet_control_parameters = {}
    # Define key points for gaze curve construction.
    # Seed points.
    ufacet_curve_keys_x = np.linspace(-131.7, 131.7, 28)
    ufacet_curve_keys_y = [136.9] * len(ufacet_curve_keys_x)
    ufacet_curve_key_xy_list = [[key_x, key_y] for key_x, key_y in zip(ufacet_curve_keys_x, ufacet_curve_keys_y)]
    ufacet_control_parameters["curve_key_xy_list"] = ufacet_curve_key_xy_list
    # Maximum altitude.
    ufacet_control_parameters["maximum_altitude"] = 25.0  # m.  Maximum altitude, roughly AGL, including slope effects.
    # Gaze control.
    ufacet_control_parameters["gaze_type"] = "constant"  # 'constant' or 'linear'
    ufacet_control_parameters["delta_eta"] = np.deg2rad(
        0.0
    )  # deg.  Offset to add to gaze angle eta.  Set to zero for no offset.

    # Define tracking parameters.
    # Aim point.
    #    aimpoint_xyz = [60.0, 8.8, 7]     # For debugging
    aimpoint_xyz = [60.0, 8.8, 28.9]  # NSTTF BCS standby - low
    #    aimpoint_xyz = [60.0, 8.8, 60]    # NSTTF BCS standby - high
    #    aimpoint_xyz = [60.0, 8.8, 100]   # For debugging

    # Time.
    # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    #
    # Zone: Mountain Daylight Time (MDT) = -6,  Mountain Standard Time (MST) = -7
    #
    #               year, month, day, hour, minute, second, zone]
    #    when_ymdhmsz = [2021,   5,   13,    9,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   11,    0,       0,    -6]
    when_ymdhmsz = [2021, 5, 13, 13, 0, 0, -6]
    #    when_ymdhmsz = [2021,   5,   13,   15,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   17,    0,       0,    -6]

    # Define fixed heliostat orientation.
    synch_azelhnames = None

    # Define upward-facing heliostat orientation.
    up_azelhnames = None

    # Return.
    return (solar_field_spec, ufacet_control_parameters, aimpoint_xyz, when_ymdhmsz, synch_azelhnames, up_azelhnames)


def define_scan_NSTTF(solar_field_short_name, nsttf_configuration):
    if nsttf_configuration == "Half-and-Half":
        return define_scan_NSTTF_half_and_half(solar_field_short_name)
    elif nsttf_configuration == "Demo":
        return define_scan_NSTTF_demo(solar_field_short_name)
    elif nsttf_configuration == "Full Field":
        return define_scan_NSTTF_full_field(solar_field_short_name)
    else:
        print(
            'ERROR: In define_scan_NSTTF(), unexpected nsttf_configuration = "'
            + str(nsttf_configuration)
            + '" encountered.'
        )
        assert False
