"""
Demonstrate Solar Field Plotting Routines



"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.camera.UCamera as cam
import opencsp.common.lib.render.figure_management as fm
import lib.FlightOverSolarField as fosf
import lib.FlightPlan as fp
import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.geo.lon_lat_nsttf as nll
import lib.plan_and_render_scan as pars
import lib.plan_scan_raster as psr
import lib.plan_scan_ufacet as psu
import lib.plan_scan_ufacet_parameters as psup
import lib.plan_scan_ufacet_section_analysis_render as psusar
import lib.plan_scan_ufacet_section_construction_render as psuscr
import lib.plan_scan_ufacet_xy_analysis_render as psuxyar
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import lib.RenderControlFlightOverSolarField as rcfosf
import lib.RenderControlFlightPlan as rcfp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import lib.RenderControlScanSectionAnalysis as rcssa
import lib.RenderControlScanSectionSetup as rcsss
import lib.RenderControlScanXyAnalysis as rcsxa
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import lib.RenderControlTopLevel as rctl
import opencsp.common.lib.uas.ScanPass as sp
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.render.view_spec as vs


# -------------------------------------------------------------------------------------------------------
# RASTER SCANS
#


# -------------------------------------------------------------------------------------------------------
# UFACET SCANS
#

# RENDERING


def setup_render_control_scan_xy_analysis():
    # Setup render control.
    render_control_scan_xy_analysis = rcsxa.RenderControlScanXyAnalysis()
    # Modify default flags.
    # render_control_scan_xy_analysis.draw_xy_segment_analysis = False
    # render_control_scan_xy_analysis.draw_xy_segment_result = False
    return render_control_scan_xy_analysis


def setup_render_control_scan_section_setup():
    # Setup render control.
    render_control_scan_section_setup = rcsss.RenderControlScanSectionSetup()
    # Modify default flags.
    # render_control_scan_section_setup.draw_section_setup = False
    # render_control_scan_section_setup.highlight_candidate_heliostats = True
    # render_control_scan_section_setup.highlight_selected_heliostats = False
    # render_control_scan_section_setup.highlight_rejected_heliostats = True
    return render_control_scan_section_setup


def setup_render_control_scan_section_analysis():
    # Setup render control.
    analysis_render_control = rcssa.RenderControlScanSectionAnalysis()
    # Modify default flags.
    analysis_render_control.draw_context = False
    # analysis_render_control.draw_context_mnsa_ray = False
    # analysis_render_control.draw_context_mxsa_ray = False
    # analysis_render_control.draw_single_heliostat_analysis = False
    # analysis_render_control.draw_single_heliostat_analysis_list = ['5E6', '6E6', '7E6', '8E6', '13E6', '14E6']
    # analysis_render_control.draw_single_heliostat_analysis_list = ['10E6']
    # analysis_render_control.draw_single_heliostat_analysis_list = ['12E6', '13E6']
    # single_heliostat_render_pass = scan.passes[0].ufacet_scan_pass()
    # analysis_render_control.draw_single_heliostat_analysis_list = single_heliostat_render_pass.heliostat_name_list
    # analysis_render_control.draw_single_heliostat_analysis_list = ufacet_pass_E06.heliostat_name_list
    analysis_render_control.draw_single_heliostat_constraints = False  # KEY DETAIL
    # analysis_render_control.draw_single_heliostat_constraints_heliostats = False
    # analysis_render_control.draw_single_heliostat_constraints_mnsa_ray = False
    # analysis_render_control.draw_single_heliostat_constraints_mxsa_ray = False
    # analysis_render_control.draw_single_heliostat_constraints_key_points = False
    # analysis_render_control.draw_single_heliostat_constraints_assessed_normals = False
    # analysis_render_control.draw_single_heliostat_constraints_detail = False
    # analysis_render_control.draw_single_heliostat_constraints_all_targets = False
    analysis_render_control.draw_single_heliostat_constraints_summary = False  # KEY SUMMARY
    # analysis_render_control.draw_single_heliostat_constraints_gaze_example = False
    # analysis_render_control.draw_single_heliostat_constraints_gaze_example_C = C_draw
    analysis_render_control.draw_single_heliostat_constraints_legend = False
    analysis_render_control.draw_single_heliostat_gaze_angle = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_example = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_fill = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_legend = False
    analysis_render_control.draw_single_heliostat_select_gaze = False  # KEY DETAIL
    # analysis_render_control.draw_single_heliostat_select_gaze_per_heliostat = False
    # analysis_render_control.draw_single_heliostat_select_gaze_shifted = False
    # analysis_render_control.draw_single_heliostat_select_gaze_envelope = False
    # analysis_render_control.draw_single_heliostat_select_gaze_shrunk = False
    # analysis_render_control.draw_single_heliostat_select_gaze_clipped = False
    # analysis_render_control.draw_single_heliostat_select_gaze_fill = False
    # analysis_render_control.draw_single_heliostat_select_gaze_legend = False
    analysis_render_control.draw_multi_heliostat_gaze_angle = False  # KEY SUMMARY
    # analysis_render_control.draw_multi_heliostat_gaze_angle_per_heliostat = False
    # analysis_render_control.draw_multi_heliostat_gaze_angle_envelope = False
    analysis_render_control.draw_multi_heliostat_gaze_angle_example = False
    analysis_render_control.draw_multi_heliostat_gaze_angle_fill = False
    analysis_render_control.draw_multi_heliostat_gaze_angle_legend = False
    analysis_render_control.draw_multi_heliostat_vertical_fov_required = False  # KEY SUMMARY
    analysis_render_control.draw_multi_heliostat_vertical_fov_required_legend = False
    # analysis_render_control.draw_multi_heliostat_select_gaze = False  # KEY SUMMARY
    # analysis_render_control.draw_multi_heliostat_select_gaze_per_heliostat = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shifted = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_envelope = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shrunk = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_clipped = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_fill = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_legend = False
    # analysis_render_control.draw_multi_heliostat_result = False  # KEY SUMMARY **
    # analysis_render_control.draw_multi_heliostat_result_heliostats = False
    # analysis_render_control.draw_multi_heliostat_result_mnsa_ray = False
    # analysis_render_control.draw_multi_heliostat_result_mxsa_ray = False
    # analysis_render_control.draw_multi_heliostat_result_selected_line = False
    # analysis_render_control.draw_multi_heliostat_result_length_margin = 15 # m.
    # analysis_render_control.draw_multi_heliostat_result_selected_segment = False
    # analysis_render_control.draw_multi_heliostat_result_start_end_loci = False
    analysis_render_control.draw_multi_heliostat_result_legend = False
    # analysis_render_control.draw_single_heliostat_etaC_dict = False  # KEY SUMMARY
    # Return.
    return analysis_render_control


# -------------------------------------------------------------------------------------------------------
# < FILL-IN >
#


# -------------------------------------------------------------------------------------------------------
# MAIN PROGRAM
#

if __name__ == "__main__":
    # plt.close('all')
    # fm.reset_figure_management()
    # figure_control = rcfg.RenderControlFigure(tile_array=(2,2), tile_square=False)

    # Figure control.
    tile_array = (2, 2)
    #    tile_array=(1,1)
    #    tile_array=(2,1)

    render_control_top_level = rctl.RenderControlTopLevel()
    #    render_control_top_level.draw_ufacet_xy_analysis = False
    render_control_top_level.draw_ufacet_section_construction = False
    # render_control_top_level.draw_ufacet_scan = False
    # render_control_top_level.draw_flight_plan = False
    # render_control_top_level.save_flight_plan = False
    # render_control_top_level.summarize_figures = True
    # render_control_top_level.save_figures = False
    render_control_scan_xy_analysis = setup_render_control_scan_xy_analysis()
    render_control_scan_section_setup = setup_render_control_scan_section_setup()
    render_control_scan_section_analysis = setup_render_control_scan_section_analysis()

    # Per-run input parameters.
    elevation_offset = 0.0  # m.

    # Scan control parameters.
    # Raster.
    #    scan_type = 'Raster'
    raster_scan_parameter_file = "DUMMY FILL IN LATER"  # ?? SCAFFOLDING RCB -- TEMPORARY
    # UFACET.
    scan_type = "UFACET"
    ufacet_scan_parameter_file = "DUMMY FILL IN LATER"  # ?? SCAFFOLDING RCB -- TEMPORARY
    # Define UFACET control flags.
    ufacet_control_parameters = {}
    # Seed points.
    # Define key points for gaze curve construction.
    ufacet_curve_keys_x = np.linspace(-131.7, 131.7, 28)
    #    ufacet_curve_keys_x = [-43.9] # ?? SCAFFOLDING RCB -- TEMPORARY
    #    ufacet_curve_keys_x = [73.2] # ?? SCAFFOLDING RCB -- TEMPORARY
    #    ufacet_curve_keys_x = [92.7] # ?? SCAFFOLDING RCB -- TEMPORARY
    #    ufacet_curve_keys_x = [112.2] # ?? SCAFFOLDING RCB -- TEMPORARY
    ufacet_curve_keys_y = [136.9] * len(ufacet_curve_keys_x)
    ufacet_curve_key_xy_list = [[key_x, key_y] for key_x, key_y in zip(ufacet_curve_keys_x, ufacet_curve_keys_y)]
    ufacet_control_parameters["curve_key_xy_list"] = ufacet_curve_key_xy_list
    # Maximum altitude.
    # ufacet_control_parameters['maximum_altitude'] = 25.0  # m.  Maximum altitude, roughly AGL, including slope effects.
    ufacet_control_parameters["maximum_altitude"] = 18.0  # m.  Maximum altitude, roughly AGL, including slope effects.
    # Gaze control.
    ufacet_control_parameters["gaze_type"] = "constant"  # 'constant' or 'linear'
    ufacet_control_parameters["delta_eta"] = np.deg2rad(
        0.0
    )  # deg.  Offset to add to gaze angle eta.  Set to zero for no offset.

    # Define solar field.
    solar_field_spec = {}
    solar_field_spec["name"] = "Sandia NSTTF"
    solar_field_spec["short_name"] = "NSTTF"
    solar_field_spec["field_origin_lon_lat"] = (nll.LON_NSTTF_ORIGIN, nll.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_origin_lon_lat"] = (nll.LON_NSTTF_ORIGIN, nll.LAT_NSTTF_ORIGIN)
    solar_field_spec["field_heliostat_file"] = "./data/NSTTF_Heliostats.csv"
    solar_field_spec["field_facet_centroids_file"] = "./data/NSTTF_Facet_Centroids.csv"

    # Define tracking parameters.
    # Aim point.
    #    aimpoint_xyz = [60.0, 8.8, 7]     # For debugging
    #    aimpoint_xyz = [60.0, 8.8, 28.9]  # NSTTF BCS standby - low
    #    aimpoint_xyz = [60.0, 8.8, 60]    # NSTTF BCS standby - high
    #    aimpoint_xyz = [60.0, 8.8, 100]   # For debugging
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
    #    when_ymdhmsz = [2021,   5,   13,    9,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   11,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   13,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   15,    0,       0,    -6]
    #    when_ymdhmsz = [2021,   5,   13,   17,    0,       0,    -6]
    when_ymdhmsz = [2021, 5, 13, 11, 0, 0, -6]  # For half-and-half

    # Define fixed heliostat orientation.
    # Test
    # synch_az = np.deg2rad(205)
    # synch_el = np.deg2rad(30)
    # synch_azelhnames = [synch_az, synch_el, ['8W3', '8W4', '8W5', '7W3', '7W4', '6W5', '6W3', '6W4', '6W5',]]
    # Half-and-Half
    synch_az = np.deg2rad(145)
    synch_el = np.deg2rad(35)
    # synch_azelhnames = [synch_az, synch_el, ['14W1', '14E1', '14E2', '14E3', '14E4', '14E5', '14E6',
    #                                          '13W1', '13E1', '13E2', '13E3', '13E4', '13E5', '13E6', '13E7', '13E8', '13E9', '13E10', '13E11', '13E12', '13E13', '13E14',
    #                                          '12W1', '12E1', '12E2', '12E3', '12E4', '12E5', '12E6', '12E7', '12E8', '12E9', '12E10', '12E11', '12E12', '12E13', '12E14',
    #                                          '11W1', '11E1', '11E2', '11E3', '11E4', '11E5', '11E6', '11E7', '11E8', '11E9', '11E10', '11E11', '11E12', '11E13', '11E14',
    #                                          '10W1', '10E1', '10E2', '10E3', '10E4', '10E5', '10E6', '10E7', '10E8', '10E9', '10E10', '10E11', '10E12',
    #                                           '9W1',  '9E1',  '9E2',  '9E3',  '9E4',  '9E5',  '9E6',  '9E7',  '9E8',  '9E9',  '9E10',  '9E11',
    #                                           '8W1',  '8E1',  '8E2',  '8E3',  '8E4',  '8E5',  '8E6',  '8E7',  '8E8',  '8E9',  '8E10',
    #                                           '7W1',  '7E1',  '7E2',  '7E3',  '7E4',  '7E5',  '7E6',  '7E7',  '7E8',  '7E9',
    #                                           '6W1',  '6E1',  '6E2',  '6E3',  '6E4',  '6E5',  '6E6',  '6E7',  '6E8',  '6E9',
    #                                           '5W1',  '5E1',  '5E2',  '5E3',  '5E4',  '5E5',  '5E6',  '5E7',  '5E8',  '5E9',  '5E10',
    #                                          ]]
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
    # synch_azelhnames = None

    # Define upward-facing heliostat orientation.
    # up_az = np.deg2rad(180)
    # up_el = np.deg2rad(90)
    # up_azelhnames = [up_az, up_el, ['7E6', '12W7']]
    up_azelhnames = None

    # Single trial.
    pars.scan_plan_trial(
        tile_array,
        solar_field_spec,
        aimpoint_xyz,
        when_ymdhmsz,
        synch_azelhnames,
        up_azelhnames,
        elevation_offset,
        scan_type,
        raster_scan_parameter_file,
        ufacet_scan_parameter_file,
        ufacet_control_parameters,
        render_control_top_level,
        render_control_scan_xy_analysis,
        render_control_scan_section_setup,
        render_control_scan_section_analysis,
    )

#     # Multiple trials.
#     trial_spec_z_aim_idx  = 0
#     trial_spec_z_max_idx  = 1
#     trial_spec_hour_idx   = 2
#     trial_spec_minute_idx = 3
#     trial_spec_list = [ [ 28.9,  19.0,  10,  0 ],
#                         [ 28.9,  19.0,  10, 30 ],
#                         [ 28.9,  19.0,  11,  0 ],
#                         [ 28.9,  19.0,  11, 30 ],
#                         [ 28.9,  19.0,  12,  0 ],
#                         [ 28.9,  19.0,  12, 30 ],
#                         [ 28.9,  19.0,  13,  0 ],
#                         [ 28.9,  19.0,  13, 30 ],
#                         [ 28.9,  19.0,  14,  0 ],
#                         [ 28.9,  19.0,  14, 30 ],
#                         [ 28.9,  19.0,  15,  0 ],
#                         [ 28.9,  19.0,  15, 30 ],
#                         [ 28.9,  19.0,  16,  0 ],
#                         [ 28.9,  19.0,  16, 30 ],
#                         [ 28.9,  19.0,  17,  0 ],
#                         [ 28.9,  19.0,  17, 30 ],
#
#                         [ 45.0, 25.0,  10,  0 ],
#                         [ 45.0, 25.0,  10, 30 ],
#                         [ 45.0, 25.0,  11,  0 ],
#                         [ 45.0, 25.0,  11, 30 ],
#                         [ 45.0, 25.0,  12,  0 ],
#                         [ 45.0, 25.0,  12, 30 ],
#                         [ 45.0, 25.0,  13,  0 ],
#                         [ 45.0, 25.0,  13, 30 ],
#                         [ 45.0, 25.0,  14,  0 ],
#                         [ 45.0, 25.0,  14, 30 ],
#                         [ 45.0, 25.0,  15,  0 ],
#                         [ 45.0, 25.0,  15, 30 ],
#                         [ 45.0, 25.0,  16,  0 ],
#                         [ 45.0, 25.0,  16, 30 ],
#                         [ 45.0, 25.0,  17,  0 ],
#                         [ 45.0, 25.0,  17, 30 ],
#
#                         [ 60.0,  25.0,  10,  0 ],
#                         [ 60.0,  25.0,  10, 30 ],
#                         [ 60.0,  25.0,  11,  0 ],
#                         [ 60.0,  25.0,  11, 30 ],
#                         [ 60.0,  25.0,  12,  0 ],
#                         [ 60.0,  25.0,  12, 30 ],
#                         [ 60.0,  25.0,  13,  0 ],
#                         [ 60.0,  25.0,  13, 30 ],
#                         [ 60.0,  25.0,  14,  0 ],
#                         [ 60.0,  25.0,  14, 30 ],
#                         [ 60.0,  25.0,  15,  0 ],
#                         [ 60.0,  25.0,  15, 30 ],
#                         [ 60.0,  25.0,  16,  0 ],
#                         [ 60.0,  25.0,  16, 30 ],
#                         [ 60.0,  25.0,  17,  0 ],
#                         [ 60.0,  25.0,  17, 30 ],
#                       ]
#
#
#     for trial_spec in trial_spec_list:
#         aimpoint_xyz_2  = aimpoint_xyz
#         when_ymdhmsz_2 = when_ymdhmsz.copy()
#         when_hour_idx   = 3
#         when_minute_idx = 4
#         when_ymdhmsz_2[when_hour_idx]                 = trial_spec[trial_spec_hour_idx]
#         when_ymdhmsz_2[when_minute_idx]               = trial_spec[trial_spec_minute_idx]
#         aimpoint_xyz_2[2]                              = trial_spec[trial_spec_z_aim_idx]
#         ufacet_control_parameters['maximum_altitude'] = trial_spec[trial_spec_z_max_idx]
#         pars.scan_plan_trial(tile_array,
#                             solar_field_spec,
#                             aimpoint_xyz_2,
#                             when_ymdhmsz_2,
#                             synch_azelhnames,
#                             up_azelhnames,
#                             elevation_offset,
#                             scan_type,
#                             raster_scan_parameter_file,
#                             ufacet_scan_parameter_file,
#                             ufacet_control_parameters,
#                             render_control_top_level,
#                             render_control_scan_xy_analysis,
#                             render_control_scan_section_setup,
#                             render_control_scan_section_analysis)
