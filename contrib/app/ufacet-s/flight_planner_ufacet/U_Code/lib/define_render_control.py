"""
Shortcuts for defining UFACET scan render control.

This file was created to move cruft out of the toplevel experiment file.  It is not intended to be final code.

These need to be cleaned up.  Perhaps this file should be eliminated.



"""

import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlFlightOverSolarField as rcfosf
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlFlightPlan as rcfp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlScanSectionAnalysis as rcssa
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlScanSectionSetup as rcsss
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlScanXyAnalysis as rcsxa
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf


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
    analysis_render_control.draw_multi_heliostat_select_gaze = False  # KEY SUMMARY
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
    analysis_render_control.draw_single_heliostat_etaC_dict = False  # KEY SUMMARY
    # Return.
    return analysis_render_control


def setup_render_control_scan_section_analysis_section_plus_flight_4view():
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
    # analysis_render_control.draw_single_heliostat_constraints_summary = False  # KEY SUMMARY
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
    # analysis_render_control.draw_multi_heliostat_gaze_angle = False  # KEY SUMMARY
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


def setup_render_control_scan_section_analysis_flight_4view():
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
    analysis_render_control.draw_multi_heliostat_select_gaze = False  # KEY SUMMARY
    # analysis_render_control.draw_multi_heliostat_select_gaze_per_heliostat = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shifted = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_envelope = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_shrunk = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_clipped = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_fill = False
    # analysis_render_control.draw_multi_heliostat_select_gaze_legend = False
    analysis_render_control.draw_multi_heliostat_result = False  # KEY SUMMARY **
    # analysis_render_control.draw_multi_heliostat_result_heliostats = False
    # analysis_render_control.draw_multi_heliostat_result_mnsa_ray = False
    # analysis_render_control.draw_multi_heliostat_result_mxsa_ray = False
    # analysis_render_control.draw_multi_heliostat_result_selected_line = False
    # analysis_render_control.draw_multi_heliostat_result_length_margin = 15 # m.
    # analysis_render_control.draw_multi_heliostat_result_selected_segment = False
    # analysis_render_control.draw_multi_heliostat_result_start_end_loci = False
    analysis_render_control.draw_multi_heliostat_result_legend = False
    analysis_render_control.draw_single_heliostat_etaC_dict = False  # KEY SUMMARY
    # Return.
    return analysis_render_control
