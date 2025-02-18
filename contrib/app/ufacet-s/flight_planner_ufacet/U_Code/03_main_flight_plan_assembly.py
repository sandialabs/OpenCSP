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
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import lib.RenderControlFlightOverSolarField as rcfosf
import lib.RenderControlFlightPlan as rcfp
import lib.RenderControlScanSectionAnalysis as rcsa
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.uas.Scan as Scan
import opencsp.common.lib.uas.ScanPass as sp
from opencsp.common.lib.csp.SolarField import SolarField
import lib.UfacetScanPass as usp
import opencsp.common.lib.render.view_spec as vs


# -------------------------------------------------------------------------------------------------------
# UFACET SCANS
#


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


def render_ufacet_scan(figure_control, scan):
    # Setup render control.
    analysis_render_control = rcsa.RenderControlScanSectionAnalysis()
    analysis_render_control.draw_context = False
    # analysis_render_control.draw_context_mnsa_ray = False
    # analysis_render_control.draw_context_mxsa_ray = False
    # analysis_render_control.draw_single_heliostat_analysis = False
    # analysis_render_control.draw_single_heliostat_analysis_list = ['5E6', '6E6', '7E6', '8E6', '13E6', '14E6']
    # analysis_render_control.draw_single_heliostat_analysis_list = ['10E6']
    # analysis_render_control.draw_single_heliostat_analysis_list = ['12E6', '13E6']
    single_heliostat_render_pass = scan.passes[0].ufacet_scan_pass()
    analysis_render_control.draw_single_heliostat_analysis_list = single_heliostat_render_pass.heliostat_name_list
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

    # Render the analysis.
    for scan_pass in scan.passes:
        scan_pass.ufacet_scan_pass().draw_section_analysis(figure_control, analysis_render_control)


# -------------------------------------------------------------------------------------------------------
# RASTER SURVEY SCANS
#


def construct_raster_survey_scan(
    solar_field, scan_segment_spec, n_horizontal, n_vertical, lead_in, run_past, fly_forward_backward
):
    # Construct segments spanning the region of interest.
    heliostat_xyz_list = solar_field.heliostat_origin_xyz_list

    x_min = min([xyz[0] for xyz in heliostat_xyz_list])
    x_max = max([xyz[0] for xyz in heliostat_xyz_list])
    y_min = min([xyz[1] for xyz in heliostat_xyz_list])
    y_max = max([xyz[1] for xyz in heliostat_xyz_list])

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
    list_of_xyz_segments = vertical_segments + horizontal_segments

    # Construct the scan.
    scan = Scan.construct_scan_given_segments_of_interest(
        scan_segment_spec, list_of_xyz_segments, lead_in, run_past, fly_forward_backward
    )

    # Return.
    return scan


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
    save_figures = False  # True

    # Per-run input parameters.
    elevation_offset = 0.0  # m.

    # Load solar field data.
    file_field = "./data/Solar_Field.csv"
    file_centroids_offsets = "./data/Facets_Centroids.csv"
    solar_field = SolarField(file_field=file_field, file_centroids_offsets=file_centroids_offsets)

    # Define tracking time.
    #    aimpoint = [60.0, 8.8, 28.9]
    aimpoint = [60.0, 8.8, 60]
    #    aimpoint = [60.0, 8.8, 100]
    #    aimpoint = [60.0, 8.8, 7]  # For debugging
    day = 126  # 4/26/2021 minus 12/21/2020
    time = 12  # Solar noon

    # Define fixed heliostat orientation.
    synch_az = np.deg2rad(205)
    synch_el = np.deg2rad(30)

    # Define upward-facing heliostat orientation.
    up_az = np.deg2rad(180)
    up_el = np.deg2rad(90)

    # Set heliostat configurations.
    configure_heliostat_name_list = [
        "5E4",
        "6E4",
        "7E4",
        "8E4",
        "9E4",
        "10E4",
        "11E4",
        "12E6",
        "13E6",
        "14E6",
        "5E5",
        "6E5",
        "7E5",
        "8E5",
        "9E5",
        "10E5",
        "11E5",
        "12E5",
        "13E5",
        "14E5",
        "5E6",
        "6E6",
        "7E6",
        "8E6",
        "9E6",
        "10E6",
        "11E6",
        "12E6",
        "13E6",
        "14E6",
        "5E7",
        "6E7",
        "7E7",
        "8E7",
        "9E7",
        "10E7",
        "11E7",
        "12E7",
        "13E7",
        "5E8",
        "6E8",
        "7E8",
        "8E8",
        "9E8",
        "10E8",
        "11E8",
        "12E8",
        "13E8",
    ]
    for heliostat_name in configure_heliostat_name_list:
        heliostat = solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_tracking(aimpoint, day, time)

    # Scan margin parameters.
    n_horizontal = 10  # Number of horizontal passes.
    n_vertical = 6  # Number of vertical passes.
    lead_in = 30  # m.
    run_past = 15  # m.
    fly_forward_backward = False  # True

    # Construct the scan.
    scan_type = "UFACET"
    # scan_type = 'survey'
    if scan_type == "UFACET":
        # Construct UFACET scan.
        scan = construct_ufacet_scan(solar_field, lead_in, run_past)
        render_ufacet_scan(figure_control, scan)
    elif scan_type == "survey":
        # Construct raster survey scan.
        # Construct the segment scanning specification.
        locale = "Sandia NSTTF"  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
        eta = np.deg2rad(-35.0)  # Degrees, converted to radians here.
        relative_z = 20  # m.
        speed = 10  # m/sec.
        scan_segment_spec = {}
        scan_segment_spec["locale"] = locale
        scan_segment_spec["eta"] = eta
        scan_segment_spec["relative_z"] = relative_z
        scan_segment_spec["speed"] = speed
        # Construct the scan.
        scan = construct_raster_survey_scan(
            solar_field, scan_segment_spec, n_horizontal, n_vertical, lead_in, run_past, fly_forward_backward
        )

    # Construct the flight plan.
    flight_plan = fp.construct_flight_plan_from_scan("N-S Columns", scan)

    #    # Write the flight plan file.
    #    # Output directory.
    #    output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
    #    if not(os.path.exists(output_path)):
    #        os.makedirs(output_path)
    #    flight_plan.save_to_lichi_csv(output_path, elevation_offset)

    # Construct object representing the flight over the solar field.
    flight_over_solar_field = fosf.FlightOverSolarField(solar_field, flight_plan)

    # Flight over solar field draw style.
    rcfosf_default = rcfosf.default()
    rcfosf_vfield = rcfosf.RenderControlFlightOverSolarField(solar_field_style=rcsf.heliostat_vector_field_outlines())

    # Draw the flight plan.
    fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_3d())
    fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_vfield, vs.view_spec_xy())
    fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_xz())
    fosf.draw_flight_over_solar_field(figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_yz())

#     # Summarize.
#     print('\n\nFigure Summary:')
#     fm.print_figure_summary()
#
#     # Save figures.
#     if save_figures:
#         print('\n\nSaving figures...')
#         # Output directory.
#         output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
#         if not(os.path.exists(output_path)):
#             os.makedirs(output_path)
#         fm.save_all_figures(output_path)
