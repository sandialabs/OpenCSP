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
    analysis_render_control.draw_single_heliostat_constraints = False
    # analysis_render_control.draw_single_heliostat_constraints_heliostats = False
    # analysis_render_control.draw_single_heliostat_constraints_mnsa_ray = False
    # analysis_render_control.draw_single_heliostat_constraints_mxsa_ray = False
    # analysis_render_control.draw_single_heliostat_constraints_key_points = False
    # analysis_render_control.draw_single_heliostat_constraints_assessed_normals = False
    # analysis_render_control.draw_single_heliostat_constraints_detail = False
    # analysis_render_control.draw_single_heliostat_constraints_all_targets = False
    # analysis_render_control.draw_single_heliostat_constraints_summary = False
    # analysis_render_control.draw_single_heliostat_constraints_gaze_example = False
    # analysis_render_control.draw_single_heliostat_constraints_gaze_example_C = C_draw
    analysis_render_control.draw_single_heliostat_constraints_legend = False
    analysis_render_control.draw_single_heliostat_gaze_angle = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_example = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_fill = False
    # analysis_render_control.draw_single_heliostat_gaze_angle_legend = False
    analysis_render_control.draw_single_heliostat_select_gaze = False
    # analysis_render_control.draw_single_heliostat_select_gaze_per_heliostat = False
    # analysis_render_control.draw_single_heliostat_select_gaze_shifted = False
    # analysis_render_control.draw_single_heliostat_select_gaze_envelope = False
    # analysis_render_control.draw_single_heliostat_select_gaze_shrunk = False
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
    analysis_render_control.draw_multi_heliostat_select_gaze = False  # KEY SUMMARY
    # analysis_render_control.draw_multi_heliostat_select_gaze_per_heliostat = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shifted = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_envelope = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shrunk = False
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
    analysis_render_control.draw_single_heliostat_etaC_dict = False
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
    #    render_control_top_level.draw_ufacet_section_construction = False
    #    render_control_top_level.draw_ufacet_scan = False
    #    render_control_top_level.draw_flight_plan = False
    #    render_control_top_level.save_flight_plan = False
    #    render_control_top_level.summarize_figures = True
    #    render_control_top_level.save_figures = False
    render_control_scan_xy_analysis = setup_render_control_scan_xy_analysis()
    render_control_scan_section_setup = setup_render_control_scan_section_setup()
    render_control_scan_section_analysis = setup_render_control_scan_section_analysis()

    # Per-run input parameters.
    #    scan_type = 'Raster'
    raster_scan_parameter_file = "DUMMY FILL IN LATER"  # ?? SCAFFOLDING RCB -- TEMPORARY
    scan_type = "UFACET"
    ufacet_scan_parameter_file = "DUMMY FILL IN LATER"  # ?? SCAFFOLDING RCB -- TEMPORARY
    elevation_offset = 0.0  # m.

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
    aimpoint_xyz = [60.0, 8.8, 60]  # NSTTF BCS standby - high
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
    # synch_az = np.deg2rad(205)
    # synch_el = np.deg2rad(30)
    # synch_azelhnames = [synch_az, synch_el, ['8W3', '8W4', '8W5', '7W3', '7W4', '6W5', '6W3', '6W4', '6W5',]]
    synch_azelhnames = None

    # Define upward-facing heliostat orientation.
    # up_az = np.deg2rad(180)
    # up_el = np.deg2rad(90)
    # up_azelhnames = [up_az, up_el, ['7E6', '12W7']]
    up_azelhnames = None

    pars.scan_plan_trial(
        tile_array,
        solar_field_spec,
        aimpoint_xyz,
        when_ymdhmsz,
        synch_azelhnames,
        up_azelhnames,
        scan_type,
        raster_scan_parameter_file,
        ufacet_scan_parameter_file,
        elevation_offset,
        render_control_top_level,
        render_control_scan_xy_analysis,
        render_control_scan_section_setup,
        render_control_scan_section_analysis,
    )
