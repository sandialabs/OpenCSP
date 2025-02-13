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
import lib.plan_scan_ufacet_section_analysis_render as pussar
import lib.plan_scan_ufacet_section_construction as pussc
import lib.plan_scan_ufacet_section_construction_render as pusscr
import lib.plan_scan_ufacet_xy_analysis as pusxya
import lib.plan_scan_ufacet_xy_analysis_render as pusxyar
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import lib.RenderControlFlightOverSolarField as rcfosf
import lib.RenderControlFlightPlan as rcfp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import lib.RenderControlScanSectionAnalysis as rcssa
import lib.RenderControlScanSectionSetup as rcsss
import lib.RenderControlScanXyAnalysis as rcsxa
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.uas.Scan as Scan
import opencsp.common.lib.uas.ScanPass as sp
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import lib.UfacetScanPass as usp
import opencsp.common.lib.render.view_spec as vs


# -------------------------------------------------------------------------------------------------------
# < FILL-IN NAME >
#


# -------------------------------------------------------------------------------------------------------
# UFACET SCANS
#


def construct_ufacet_scan_pass(solar_field, lead_in, run_past):
    # Control parameters.
    scan_parameters = {}
    scan_parameters["locale"] = (
        "Sandia NSTTF"  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
    )
    scan_parameters["camera"] = cam.sony_alpha_20mm_landscape()  # Camera model.
    # scan_parameters['camera'] = cam.sony_alpha_20mm_portrait()  # Camera model.
    scan_parameters["camera"] = cam.ultra_wide_angle()  # Camera model.
    # scan_parameters['camera'] = cam.mavic_zoom()  # Camera model.
    scan_parameters["section_plane_tolerance"] = 3  # m.  Lateral distance to include heliostats in section.
    scan_parameters["p_margin"] = 0  # 2 # m.  Lateral distance to add to constraints to allow UAS postiion error.
    scan_parameters["altitude_margin"] = 2.5  # m.  Clearance of highest possible heliostat point.
    scan_parameters["maximum_safe_altitude"] = (
        90.0  # meters.  # ?? SCAFFOLDING -- BASE THIS ON TECHNICAL FACTORS:  SOLAR FLUX, ETC
    )
    scan_parameters["maximum_target_lookback"] = 3  # Number of heliostats to look back for reflection targets.
    scan_parameters["gaze_tolerance"] = np.deg2rad(
        1
    )  # Uncertainty in gaze angle.  True angle is +/- tolerance from nominal.

    # Construct scan segments.
    R05_y = 57.9  # m.
    R14_y = 194.8  # m.
    E04_x = 34.1  # m.
    E05_x = 43.9  # m.
    E06_x = 53.5  # m.
    E07_x = 63.4  # m.
    E08_x = 73.2  # m.
    segment_xy_E04 = [[E04_x, R05_y], [E04_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E05 = [[E05_x, R05_y], [E05_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E06 = [[E06_x, R05_y], [E06_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E07 = [[E07_x, R05_y], [E07_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E08 = [[E08_x, R05_y], [E08_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN

    # Construct individual UFACET scan passes.
    # # ufacet_pass_E04 = usp.UfacetScanPass(solar_field, segment_xy_E04, scan_parameters)  # ?? SCAFFOLDING RCB -- CRASHES
    ufacet_pass_E05 = usp.UfacetScanPass(solar_field, segment_xy_E05, scan_parameters)
    ufacet_pass_E06 = usp.UfacetScanPass(solar_field, segment_xy_E06, scan_parameters)
    ufacet_pass_E07 = usp.UfacetScanPass(solar_field, segment_xy_E07, scan_parameters)
    ufacet_pass_E08 = usp.UfacetScanPass(solar_field, segment_xy_E08, scan_parameters)

    ufacet_scan_pass_list = [ufacet_pass_E05, ufacet_pass_E06, ufacet_pass_E07, ufacet_pass_E08]

    # Construct the scan.
    scan = Scan.construct_scan_given_UFACET_scan_passes(ufacet_scan_pass_list, lead_in, run_past)
    # Return.
    return scan


def construct_ufacet_scan_passes(solar_field, lead_in, run_past):
    # Control parameters.
    scan_parameters = {}
    scan_parameters["locale"] = (
        "Sandia NSTTF"  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
    )
    scan_parameters["camera"] = cam.sony_alpha_20mm_landscape()  # Camera model.
    # scan_parameters['camera'] = cam.sony_alpha_20mm_portrait()  # Camera model.
    scan_parameters["camera"] = cam.ultra_wide_angle()  # Camera model.
    # scan_parameters['camera'] = cam.mavic_zoom()  # Camera model.
    scan_parameters["section_plane_tolerance"] = 3  # m.  Lateral distance to include heliostats in section.
    scan_parameters["p_margin"] = 0  # 2 # m.  Lateral distance to add to constraints to allow UAS postiion error.
    scan_parameters["altitude_margin"] = 2.5  # m.  Clearance of highest possible heliostat point.
    scan_parameters["maximum_safe_altitude"] = (
        90.0  # meters.  # ?? SCAFFOLDING -- BASE THIS ON TECHNICAL FACTORS:  SOLAR FLUX, ETC
    )
    scan_parameters["maximum_target_lookback"] = 3  # Number of heliostats to look back for reflection targets.
    scan_parameters["gaze_tolerance"] = np.deg2rad(
        1
    )  # Uncertainty in gaze angle.  True angle is +/- tolerance from nominal.

    # Construct scan segments.
    R05_y = 57.9  # m.
    R14_y = 194.8  # m.
    E04_x = 34.1  # m.
    E05_x = 43.9  # m.
    E06_x = 53.5  # m.
    E07_x = 63.4  # m.
    E08_x = 73.2  # m.
    segment_xy_E04 = [[E04_x, R05_y], [E04_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E05 = [[E05_x, R05_y], [E05_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E06 = [[E06_x, R05_y], [E06_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E07 = [[E07_x, R05_y], [E07_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E08 = [[E08_x, R05_y], [E08_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN

    # Construct individual UFACET scan passes.
    # # ufacet_pass_E04 = usp.UfacetScanPass(solar_field, segment_xy_E04, scan_parameters)  # ?? SCAFFOLDING RCB -- CRASHES
    ufacet_pass_E05 = usp.UfacetScanPass(solar_field, segment_xy_E05, scan_parameters)
    ufacet_pass_E06 = usp.UfacetScanPass(solar_field, segment_xy_E06, scan_parameters)
    ufacet_pass_E07 = usp.UfacetScanPass(solar_field, segment_xy_E07, scan_parameters)
    ufacet_pass_E08 = usp.UfacetScanPass(solar_field, segment_xy_E08, scan_parameters)

    ufacet_scan_pass_list = [ufacet_pass_E05, ufacet_pass_E06, ufacet_pass_E07, ufacet_pass_E08]

    # Construct the scan.
    scan = Scan.construct_scan_given_UFACET_scan_passes(ufacet_scan_pass_list, lead_in, run_past)
    # Return.
    return scan


def construct_ufacet_scan(solar_field, lead_in, run_past):
    # Control parameters.
    scan_parameters = {}
    scan_parameters["locale"] = (
        "Sandia NSTTF"  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
    )
    scan_parameters["camera"] = cam.sony_alpha_20mm_landscape()  # Camera model.
    # scan_parameters['camera'] = cam.sony_alpha_20mm_portrait()  # Camera model.
    scan_parameters["camera"] = cam.ultra_wide_angle()  # Camera model.
    # scan_parameters['camera'] = cam.mavic_zoom()  # Camera model.
    scan_parameters["section_plane_tolerance"] = 3  # m.  Lateral distance to include heliostats in section.
    scan_parameters["p_margin"] = 0  # 2 # m.  Lateral distance to add to constraints to allow UAS postiion error.
    scan_parameters["altitude_margin"] = 2.5  # m.  Clearance of highest possible heliostat point.
    scan_parameters["maximum_safe_altitude"] = (
        90.0  # meters.  # ?? SCAFFOLDING -- BASE THIS ON TECHNICAL FACTORS:  SOLAR FLUX, ETC
    )
    scan_parameters["maximum_target_lookback"] = 3  # Number of heliostats to look back for reflection targets.
    scan_parameters["gaze_tolerance"] = np.deg2rad(
        1
    )  # Uncertainty in gaze angle.  True angle is +/- tolerance from nominal.

    # Construct scan segments.
    R05_y = 57.9  # m.
    R14_y = 194.8  # m.
    E04_x = 34.1  # m.
    E05_x = 43.9  # m.
    E06_x = 53.5  # m.
    E07_x = 63.4  # m.
    E08_x = 73.2  # m.
    segment_xy_E04 = [[E04_x, R05_y], [E04_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E05 = [[E05_x, R05_y], [E05_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E06 = [[E06_x, R05_y], [E06_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E07 = [[E07_x, R05_y], [E07_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E08 = [[E08_x, R05_y], [E08_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN

    # Construct individual UFACET scan passes.
    # # ufacet_pass_E04 = usp.UfacetScanPass(solar_field, segment_xy_E04, scan_parameters)  # ?? SCAFFOLDING RCB -- CRASHES
    ufacet_pass_E05 = usp.UfacetScanPass(solar_field, segment_xy_E05, scan_parameters)
    ufacet_pass_E06 = usp.UfacetScanPass(solar_field, segment_xy_E06, scan_parameters)
    ufacet_pass_E07 = usp.UfacetScanPass(solar_field, segment_xy_E07, scan_parameters)
    ufacet_pass_E08 = usp.UfacetScanPass(solar_field, segment_xy_E08, scan_parameters)

    ufacet_scan_pass_list = [ufacet_pass_E05, ufacet_pass_E06, ufacet_pass_E07, ufacet_pass_E08]

    # Construct the scan.
    scan = Scan.construct_scan_given_UFACET_scan_passes(ufacet_scan_pass_list, lead_in, run_past)
    # Return.
    return scan


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


def render_ufacet_scan(figure_control, scan):
    analysis_render_control = setup_render_control_scan_section_analysis()

    # Render the analysis.
    for scan_pass in scan.passes:
        scan_pass.ufacet_scan_pass().draw_section_analysis(figure_control, analysis_render_control)


# -------------------------------------------------------------------------------------------------------
# MAIN PROGRAM
#

if __name__ == "__main__":
    # Figure control.
    plt.close("all")
    fm.reset_figure_management()
    figure_control = rcfg.RenderControlFigure(tile_array=(2, 2), tile_square=False)
    #    figure_control = rcfg.RenderControlFigure(tile_array=(1,1), tile_square=False)
    #    figure_control = rcfg.RenderControlFigure(tile_array=(2,1), tile_square=False)
    save_figures = True

    # Per-run input parameters.
    elevation_offset = 0.0  # m.

    # Load solar field data.
    field_origin_lon_lat = (nll.LON_NSTTF_ORIGIN, nll.LAT_NSTTF_ORIGIN)
    field_heliostat_file = "./data/NSTTF_Heliostats.csv"
    field_facet_centroids_file = "./data/NSTTF_Facet_Centroids.csv"
    solar_field = sf.SolarField(
        origin_lon_lat=field_origin_lon_lat,
        heliostat_file=field_heliostat_file,
        facet_centroids_file=field_facet_centroids_file,
    )

    # Define tracking parameters.
    # Aim point.
    #    aimpoint_xyz = [60.0, 8.8, 7]     # For debugging
    #    aimpoint_xyz = [60.0, 8.8, 28.9]  # NSTTF BCS standby - low
    aimpoint_xyz = [60.0, 8.8, 60]  # NSTTF BCS standby - high
    #    aimpoint_xyz = [60.0, 8.8, 100]   # For debugging
    # TIme.
    # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    year = 2021
    month = 5
    day = 13
    #    hour   =  9
    #    hour   = 11
    hour = 13
    #    hour   = 15
    #    hour   = 17
    minute = 3
    second = 0
    zone = -6  # Mountain Daylight Time (MDT) = -6,  Mountain Standard Time (MST) = -7
    when_ymdhmsz = (year, month, day, hour, minute, second, zone)

    # Define fixed heliostat orientation.
    synch_az = np.deg2rad(205)
    synch_el = np.deg2rad(30)

    # Define upward-facing heliostat orientation.
    up_az = np.deg2rad(180)
    up_el = np.deg2rad(90)

    # Set heliostat configurations.
    solar_field.set_full_field_tracking(aimpoint_xyz, when_ymdhmsz)

    # Raster scan parameters.
    n_horizontal = 10  # Number of horizontal passes.
    n_vertical = 6  # Number of vertical passes.

    # UFACET scan parameters
    candidate_margin_w = 10.00  # m.  Margin on either side of section plane to bring in heliostats.
    #     Should be larger than side-to-side heliostat distance.
    discard_threshold_p = (
        9.00  # m.  Threshold to discarb heliostats that are close togethre ona section, presumably abreast.
    )
    #     Should be smaller than minimum heliostat row spacing.

    # Scan flight parameters (both raster and UFACET).
    lead_in = 30  # m.
    run_past = 15  # m.
    fly_forward_backward = False  # True

    # Define key points for gaze curve construction.
    curve_keys_x = np.linspace(-131.7, 131.7, 28)
    #    curve_keys_x = [-43.9] # ?? SCAFFOLDING RCB -- TEMPORARY
    #    curve_keys_x = [92.7] # ?? SCAFFOLDING RCB -- TEMPORARY
    curve_keys_y = [136.9] * len(curve_keys_x)
    curve_key_xy_list = [[key_x, key_y] for key_x, key_y in zip(curve_keys_x, curve_keys_y)]

    # UFACET (x,y) analysis.
    list_of_ideal_xy_lists, list_of_best_fit_segment_xys = pusxya.ufacet_xy_analysis(
        solar_field, aimpoint_xyz, when_ymdhmsz, curve_key_xy_list
    )

    # Draw UFACET (x,y) analysis.
    render_control_scan_xy_analysis = setup_render_control_scan_xy_analysis()
    pusxyar.draw_ufacet_xy_analysis(
        figure_control,
        solar_field,
        aimpoint_xyz,
        field_origin_lon_lat,
        when_ymdhmsz,
        curve_key_xy_list,
        list_of_ideal_xy_lists,
        list_of_best_fit_segment_xys,
        render_control_scan_xy_analysis,
    )

    # UFACET section analysis.
    section_list = pussc.construct_ufacet_sections(
        solar_field, list_of_best_fit_segment_xys, candidate_margin_w, discard_threshold_p
    )

    # Draw UFACET section analysis.
    render_control_scan_section_setup = setup_render_control_scan_section_setup()
    pusscr.draw_construct_ufacet_sections(
        figure_control, solar_field, section_list, vs.view_spec_3d(), render_control_scan_section_setup
    )
    pusscr.draw_construct_ufacet_sections(
        figure_control, solar_field, section_list, vs.view_spec_xy(), render_control_scan_section_setup
    )
    pusscr.draw_construct_ufacet_sections(
        figure_control, solar_field, section_list, None, render_control_scan_section_setup
    )  # Use section view.

    #     # Construct the scan.
    #     scan_type = 'UFACET'
    #     # scan_type = 'survey'
    #     if scan_type == 'UFACET':
    #         # Construct UFACET scan.
    #         scan = construct_ufacet_scan(solar_field, lead_in, run_past)
    # #        render_ufacet_scan(figure_control, scan)
    #     elif scan_type == 'survey':
    #         # Construct raster survey scan.
    #         # Construct the segment scanning specification.
    #         locale = 'Sandia NSTTF' # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
    #         eta = np.deg2rad(-35.0)  # Degrees, converted to radians here.
    #         relative_z = 20          # m.
    #         speed = 10               # m/sec.
    #         scan_segment_spec = {}
    #         scan_segment_spec['locale']      = locale
    #         scan_segment_spec['eta']         = eta
    #         scan_segment_spec['relative_z']  = relative_z
    #         scan_segment_spec['speed']       = speed
    #         # Construct the scan.
    #         scan = sf.construct_solar_field_heliostat_survey_scan(solar_field,
    #                                                               scan_segment_spec,
    #                                                               n_horizontal,
    #                                                               n_vertical,
    #                                                               lead_in,
    #                                                               run_past,
    #                                                               fly_forward_backward)
    #
    #     # Construct the flight plan.
    #     flight_plan = fp.construct_flight_plan_from_scan('N-S Columns', scan)
    #
    # #    # Write the flight plan file.
    # #    # Output directory.
    # #    output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
    # #    if not(os.path.exists(output_path)):
    # #        os.makedirs(output_path)
    # #    flight_plan.save_to_lichi_csv(output_path, elevation_offset)
    #
    #     # Construct object representing the flight over the solar field.
    #     flight_over_solar_field = fosf.FlightOverSolarField(solar_field, flight_plan)
    #
    #     # Flight over solar field draw style.
    #     rcfosf_default = rcfosf.default()
    #     rcfosf_vfield  = rcfosf.RenderControlFlightOverSolarField(solar_field_style = rcsf.heliostat_vector_field_outlines(color='grey'))
    #
    #     # Draw the flight plan.
    #     # view_3d = fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_3d())
    #     view_xy = fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_vfield,  vs.view_spec_xy())
    #     # view_xz = fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_xz())
    #     # view_yz = fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_yz())

    # Summarize.
    print("\n\nFigure Summary:")
    fm.print_figure_summary()

    # Save figures.
    if save_figures:
        print("\n\nSaving figures...")
        # Output directory.
        output_path = os.path.join("..", ("output_" + datetime.now().strftime("%Y_%m_%d_%H%M")))
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        fm.save_all_figures(output_path)
