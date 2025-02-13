"""
Demonstrate Solar Field Plotting Routines



"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.camera.UCamera as cam
import opencsp.common.lib.render.figure_management as fm
import lib.FlightPlan as fp
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import lib.RenderControlScanSectionAnalysis as rcsa
import opencsp.common.lib.uas.Scan as Scan
import opencsp.common.lib.uas.ScanPass as sp
from opencsp.common.lib.csp.SolarField import SolarField


# -------------------------------------------------------------------------------------------------------
# SECTION BREAK
#


if __name__ == "__main__":
    plt.close("all")
    fm.reset_figure_management()

    # Per-run input parameters.
    elevation_offset = 0.0  # m.

    # Load solar field data.
    file_field = "./data/Solar_Field.csv"
    file_centroids_offsets = "./data/Facets_Centroids.csv"
    solar_field = SolarField(file_field=file_field, file_centroids_offsets=file_centroids_offsets)

    # Define tracking time.
    aimpoint = [60.0, 8.8, 28.9]
    aimpoint = [60.0, 8.8, 60]
    # aimpoint = [60.0, 8.8, 100]
    # aimpoint = [60.0, 8.8, 7]  # ?? SCAFFOLDING RCB -- TEMPORARY
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
    ]
    for heliostat_name in configure_heliostat_name_list:
        heliostat = solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_tracking(aimpoint, day, time)

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
    segment_xy_E04 = [[E04_x, R05_y], [E04_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E05 = [[E05_x, R05_y], [E05_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E06 = [[E06_x, R05_y], [E06_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN
    segment_xy_E07 = [[E07_x, R05_y], [E07_x, R14_y]]  # ?? SCAFFOLDING RCB -- REPLACE WITH FLIGHT PLAN

    # Construct individual scan passes.
    # # pass_E04 = sp.ScanPass(solar_field, segment_xy_E04, scan_parameters)  # ?? SCAFFOLDING RCB -- CRASHES
    pass_E05 = sp.ScanPass(solar_field, segment_xy_E05, scan_parameters)
    pass_E06 = sp.ScanPass(solar_field, segment_xy_E06, scan_parameters)
    pass_E07 = sp.ScanPass(solar_field, segment_xy_E07, scan_parameters)

    # Construct the ensemble of scan passes.
    scan_passes = [pass_E05, pass_E06, pass_E07]

    # Construct the scan.
    scan = Scan.Scan(scan_passes)

    # Construct the flight plan.
    flight_plan = fp.FlightPlan("N-S Columns", [scan])

    # Write the flight plan file.
    # Output directory.
    output_path = os.path.join("..", ("output_" + datetime.now().strftime("%Y_%m_%d_%H%M")))
    if not (os.path.exists(output_path)):
        os.makedirs(output_path)
    flight_plan.save_to_lichi_csv(output_path, elevation_offset)

#     # Setup render control.
#     figure_control = rcfg.RenderControlFigure(tile_array=(2,2), tile_square=False)
#     save_figures = False#True
#     analysis_render_control = rcsa.RenderControlScanSectionAnalysis()
#     analysis_render_control.draw_context = False
#     # analysis_render_control.draw_context_mnsa_ray = False
#     # analysis_render_control.draw_context_mxsa_ray = False
#     # analysis_render_control.draw_single_heliostat_analysis = False
# #analysis_render_control.draw_single_heliostat_analysis_list = ['5E6', '6E6', '7E6', '8E6', '13E6', '14E6']
# #analysis_render_control.draw_single_heliostat_analysis_list = ['10E6']
# #analysis_render_control.draw_single_heliostat_analysis_list = ['12E6', '13E6']
#     analysis_render_control.draw_single_heliostat_analysis_list = pass_E06.heliostat_name_list
#     analysis_render_control.draw_single_heliostat_constraints = False
#     # analysis_render_control.draw_single_heliostat_constraints_heliostats = False
#     # analysis_render_control.draw_single_heliostat_constraints_mnsa_ray = False
#     # analysis_render_control.draw_single_heliostat_constraints_mxsa_ray = False
#     # analysis_render_control.draw_single_heliostat_constraints_key_points = False
#     # analysis_render_control.draw_single_heliostat_constraints_assessed_normals = False
#     # analysis_render_control.draw_single_heliostat_constraints_detail = False
#     # analysis_render_control.draw_single_heliostat_constraints_all_targets = False
#     # analysis_render_control.draw_single_heliostat_constraints_summary = False
#     # analysis_render_control.draw_single_heliostat_constraints_gaze_example = False
#     # analysis_render_control.draw_single_heliostat_constraints_gaze_example_C = C_draw
#     analysis_render_control.draw_single_heliostat_constraints_legend = False
#     analysis_render_control.draw_single_heliostat_gaze_angle = False
#     # analysis_render_control.draw_single_heliostat_gaze_angle_example = False
#     # analysis_render_control.draw_single_heliostat_gaze_angle_fill = False
#     # analysis_render_control.draw_single_heliostat_gaze_angle_legend = False
#     analysis_render_control.draw_single_heliostat_select_gaze = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_per_heliostat = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_shifted = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_envelope = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_shrunk = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_fill = False
#     # analysis_render_control.draw_single_heliostat_select_gaze_legend = False
#     analysis_render_control.draw_multi_heliostat_gaze_angle = False  # KEY SUMMARY
#     # analysis_render_control.draw_multi_heliostat_gaze_angle_per_heliostat = False
#     # analysis_render_control.draw_multi_heliostat_gaze_angle_envelope = False
#     analysis_render_control.draw_multi_heliostat_gaze_angle_example = False
#     analysis_render_control.draw_multi_heliostat_gaze_angle_fill = False
#     analysis_render_control.draw_multi_heliostat_gaze_angle_legend = False
#     analysis_render_control.draw_multi_heliostat_vertical_fov_required = False  # KEY SUMMARY
#     analysis_render_control.draw_multi_heliostat_vertical_fov_required_legend = False
#     analysis_render_control.draw_multi_heliostat_select_gaze = False  # KEY SUMMARY
#     # analysis_render_control.draw_multi_heliostat_select_gaze_per_heliostat = False
#     # analysis_render_control.draw_multi_heliostat_select_gaze_shifted = False
#     # analysis_render_control.draw_multi_heliostat_select_gaze_envelope = False
#     # analysis_render_control.draw_multi_heliostat_select_gaze_shrunk = False
#     # analysis_render_control.draw_multi_heliostat_select_gaze_fill = False
#     # analysis_render_control.draw_multi_heliostat_select_gaze_legend = False
#     # analysis_render_control.draw_multi_heliostat_result = False  # KEY SUMMARY **
#     # analysis_render_control.draw_multi_heliostat_result_heliostats = False
#     # analysis_render_control.draw_multi_heliostat_result_mnsa_ray = False
#     # analysis_render_control.draw_multi_heliostat_result_mxsa_ray = False
#     # analysis_render_control.draw_multi_heliostat_result_selected_line = False
#     # analysis_render_control.draw_multi_heliostat_result_length_margin = 15 # m.
#     # analysis_render_control.draw_multi_heliostat_result_selected_segment = False
#     # analysis_render_control.draw_multi_heliostat_result_start_end_loci = False
#     analysis_render_control.draw_multi_heliostat_result_legend = False
#     analysis_render_control.draw_single_heliostat_etaC_dict = False
#
#     # Render the analysis.
# # # pass_E04.draw_section_analysis(figure_control, analysis_render_control)  # ?? SCAFFOLDING RCB -- CRASHES
#     pass_E05.draw_section_analysis(figure_control, analysis_render_control)
#     pass_E06.draw_section_analysis(figure_control, analysis_render_control)
#     pass_E07.draw_section_analysis(figure_control, analysis_render_control)
#
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
