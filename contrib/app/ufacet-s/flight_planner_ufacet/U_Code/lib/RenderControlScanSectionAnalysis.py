"""


"""


class RenderControlScanSectionAnalysis:
    """
    Render control for plot axes.
    """

    def __init__(
        self,
        draw_context=True,
        draw_context_mnsa_ray=True,
        draw_context_mxsa_ray=True,
        draw_single_heliostat_analysis=True,
        draw_single_heliostat_analysis_list=None,  # List of specific heliostats to render.  If None, render all.  If [], render none.
        draw_single_heliostat_constraints=True,
        draw_single_heliostat_constraints_heliostats=True,
        draw_single_heliostat_constraints_mnsa_ray=True,
        draw_single_heliostat_constraints_mxsa_ray=True,
        draw_single_heliostat_constraints_key_points=True,
        draw_single_heliostat_constraints_assessed_normals=True,
        draw_single_heliostat_constraints_detail=True,
        draw_single_heliostat_constraints_all_targets=True,
        draw_single_heliostat_constraints_summary=True,
        draw_single_heliostat_constraints_gaze_example=True,
        draw_single_heliostat_constraints_gaze_example_C=None,  # m, or None
        draw_single_heliostat_constraints_legend=True,
        draw_single_heliostat_gaze_angle=True,
        draw_single_heliostat_gaze_angle_mnsa=True,
        draw_single_heliostat_gaze_angle_mxsa=True,
        draw_single_heliostat_gaze_angle_critical=True,
        draw_single_heliostat_gaze_angle_example=True,
        draw_single_heliostat_gaze_angle_fill=True,
        draw_single_heliostat_gaze_angle_legend=True,
        draw_single_heliostat_select_gaze=True,
        draw_single_heliostat_select_gaze_shifted=True,
        draw_single_heliostat_select_gaze_envelope=True,
        draw_single_heliostat_select_gaze_shrunk=True,
        draw_single_heliostat_select_gaze_clipped=True,
        draw_single_heliostat_select_gaze_selected=True,
        draw_single_heliostat_select_gaze_mnsa=True,
        draw_single_heliostat_select_gaze_mxsa=True,
        draw_single_heliostat_select_gaze_critical=True,
        draw_single_heliostat_select_gaze_fill=True,
        draw_single_heliostat_select_gaze_legend=True,
        draw_multi_heliostat_gaze_angle=True,
        draw_multi_heliostat_gaze_angle_per_heliostat=True,
        draw_multi_heliostat_gaze_angle_envelope=True,
        draw_multi_heliostat_gaze_angle_mnsa=True,
        draw_multi_heliostat_gaze_angle_mxsa=True,
        draw_multi_heliostat_gaze_angle_critical=True,
        draw_multi_heliostat_gaze_angle_example=True,
        draw_multi_heliostat_gaze_angle_fill=True,
        draw_multi_heliostat_gaze_angle_legend=True,
        draw_multi_heliostat_vertical_fov_required=True,
        draw_multi_heliostat_vertical_fov_required_mnsa=True,
        draw_multi_heliostat_vertical_fov_required_mxsa=True,
        draw_multi_heliostat_vertical_fov_required_critical=True,
        draw_multi_heliostat_vertical_fov_required_camera=True,
        draw_multi_heliostat_vertical_fov_required_legend=True,
        draw_multi_heliostat_select_gaze=True,
        draw_multi_heliostat_select_gaze_shifted=True,
        draw_multi_heliostat_select_gaze_envelope=True,
        draw_multi_heliostat_select_gaze_shrunk=True,
        draw_multi_heliostat_select_gaze_clipped=True,
        draw_multi_heliostat_select_gaze_selected=True,
        draw_multi_heliostat_select_gaze_mnsa=True,
        draw_multi_heliostat_select_gaze_mxsa=True,
        draw_multi_heliostat_select_gaze_critical=True,
        draw_multi_heliostat_select_gaze_fill=True,
        draw_multi_heliostat_select_gaze_legend=True,
        draw_multi_heliostat_result=True,
        draw_multi_heliostat_result_heliostats=True,
        draw_multi_heliostat_result_mnsa_ray=True,
        draw_multi_heliostat_result_mxsa_ray=True,
        draw_multi_heliostat_result_selected_cacg_line=True,  # "cacg" == "constant altitude, constant gaze"
        draw_multi_heliostat_result_length_margin=10,  # m.
        draw_multi_heliostat_result_selected_cacg_segment=True,  # "cacg" == "constant altitude, constant gaze"
        draw_multi_heliostat_result_start_end_loci=True,
        draw_multi_heliostat_result_legend=True,
        draw_single_heliostat_etaC_dict=True,
    ):
        super(RenderControlScanSectionAnalysis, self).__init__()

        self.draw_context = draw_context
        self.draw_context_mnsa_ray = draw_context_mnsa_ray
        self.draw_context_mxsa_ray = draw_context_mxsa_ray
        self.draw_single_heliostat_analysis = draw_single_heliostat_analysis
        self.draw_single_heliostat_analysis_list = draw_single_heliostat_analysis_list
        self.draw_single_heliostat_constraints = draw_single_heliostat_constraints
        self.draw_single_heliostat_constraints_heliostats = draw_single_heliostat_constraints_heliostats
        self.draw_single_heliostat_constraints_mnsa_ray = draw_single_heliostat_constraints_mnsa_ray
        self.draw_single_heliostat_constraints_mxsa_ray = draw_single_heliostat_constraints_mxsa_ray
        self.draw_single_heliostat_constraints_key_points = draw_single_heliostat_constraints_key_points
        self.draw_single_heliostat_constraints_assessed_normals = draw_single_heliostat_constraints_assessed_normals
        self.draw_single_heliostat_constraints_detail = draw_single_heliostat_constraints_detail
        self.draw_single_heliostat_constraints_all_targets = draw_single_heliostat_constraints_all_targets
        self.draw_single_heliostat_constraints_summary = draw_single_heliostat_constraints_summary
        self.draw_single_heliostat_constraints_gaze_example = draw_single_heliostat_constraints_gaze_example
        self.draw_single_heliostat_constraints_gaze_example_C = draw_single_heliostat_constraints_gaze_example_C
        self.draw_single_heliostat_constraints_legend = draw_single_heliostat_constraints_legend
        self.draw_single_heliostat_gaze_angle = draw_single_heliostat_gaze_angle
        self.draw_single_heliostat_gaze_angle_mnsa = draw_single_heliostat_gaze_angle_mnsa
        self.draw_single_heliostat_gaze_angle_mxsa = draw_single_heliostat_gaze_angle_mxsa
        self.draw_single_heliostat_gaze_angle_critical = draw_single_heliostat_gaze_angle_critical
        self.draw_single_heliostat_gaze_angle_example = draw_single_heliostat_gaze_angle_example
        self.draw_single_heliostat_gaze_angle_fill = draw_single_heliostat_gaze_angle_fill
        self.draw_single_heliostat_gaze_angle_legend = draw_single_heliostat_gaze_angle_legend
        self.draw_single_heliostat_select_gaze = draw_single_heliostat_select_gaze
        self.draw_single_heliostat_select_gaze_shifted = draw_single_heliostat_select_gaze_shifted
        self.draw_single_heliostat_select_gaze_envelope = draw_single_heliostat_select_gaze_envelope
        self.draw_single_heliostat_select_gaze_shrunk = draw_single_heliostat_select_gaze_shrunk
        self.draw_single_heliostat_select_gaze_clipped = draw_single_heliostat_select_gaze_clipped
        self.draw_single_heliostat_select_gaze_selected = draw_single_heliostat_select_gaze_selected
        self.draw_single_heliostat_select_gaze_mnsa = draw_single_heliostat_select_gaze_mnsa
        self.draw_single_heliostat_select_gaze_mxsa = draw_single_heliostat_select_gaze_mxsa
        self.draw_single_heliostat_select_gaze_critical = draw_single_heliostat_select_gaze_critical
        self.draw_single_heliostat_select_gaze_fill = draw_single_heliostat_select_gaze_fill
        self.draw_single_heliostat_select_gaze_legend = draw_single_heliostat_select_gaze_legend
        self.draw_multi_heliostat_gaze_angle = draw_multi_heliostat_gaze_angle
        self.draw_multi_heliostat_gaze_angle_per_heliostat = draw_multi_heliostat_gaze_angle_per_heliostat
        self.draw_multi_heliostat_gaze_angle_envelope = draw_multi_heliostat_gaze_angle_envelope
        self.draw_multi_heliostat_gaze_angle_mnsa = draw_multi_heliostat_gaze_angle_mnsa
        self.draw_multi_heliostat_gaze_angle_mxsa = draw_multi_heliostat_gaze_angle_mxsa
        self.draw_multi_heliostat_gaze_angle_critical = draw_multi_heliostat_gaze_angle_critical
        self.draw_multi_heliostat_gaze_angle_example = draw_multi_heliostat_gaze_angle_example
        self.draw_multi_heliostat_gaze_angle_fill = draw_multi_heliostat_gaze_angle_fill
        self.draw_multi_heliostat_gaze_angle_legend = draw_multi_heliostat_gaze_angle_legend
        self.draw_multi_heliostat_vertical_fov_required = draw_multi_heliostat_vertical_fov_required
        self.draw_multi_heliostat_vertical_fov_required_mnsa = draw_multi_heliostat_vertical_fov_required_mnsa
        self.draw_multi_heliostat_vertical_fov_required_mxsa = draw_multi_heliostat_vertical_fov_required_mxsa
        self.draw_multi_heliostat_vertical_fov_required_critical = draw_multi_heliostat_vertical_fov_required_critical
        self.draw_multi_heliostat_vertical_fov_required_camera = draw_multi_heliostat_vertical_fov_required_camera
        self.draw_multi_heliostat_vertical_fov_required_legend = draw_multi_heliostat_vertical_fov_required_legend
        self.draw_multi_heliostat_select_gaze = draw_multi_heliostat_select_gaze
        self.draw_multi_heliostat_select_gaze_shifted = draw_multi_heliostat_select_gaze_shifted
        self.draw_multi_heliostat_select_gaze_envelope = draw_multi_heliostat_select_gaze_envelope
        self.draw_multi_heliostat_select_gaze_shrunk = draw_multi_heliostat_select_gaze_shrunk
        self.draw_multi_heliostat_select_gaze_clipped = draw_multi_heliostat_select_gaze_clipped
        self.draw_multi_heliostat_select_gaze_selected = draw_multi_heliostat_select_gaze_selected
        self.draw_multi_heliostat_select_gaze_mnsa = draw_multi_heliostat_select_gaze_mnsa
        self.draw_multi_heliostat_select_gaze_mxsa = draw_multi_heliostat_select_gaze_mxsa
        self.draw_multi_heliostat_select_gaze_critical = draw_multi_heliostat_select_gaze_critical
        self.draw_multi_heliostat_select_gaze_fill = draw_multi_heliostat_select_gaze_fill
        self.draw_multi_heliostat_select_gaze_legend = draw_multi_heliostat_select_gaze_legend
        self.draw_multi_heliostat_result = draw_multi_heliostat_result
        self.draw_multi_heliostat_result_heliostats = draw_multi_heliostat_result_heliostats
        self.draw_multi_heliostat_result_mnsa_ray = draw_multi_heliostat_result_mnsa_ray
        self.draw_multi_heliostat_result_mxsa_ray = draw_multi_heliostat_result_mxsa_ray
        self.draw_multi_heliostat_result_selected_cacg_line = draw_multi_heliostat_result_selected_cacg_line
        self.draw_multi_heliostat_result_length_margin = draw_multi_heliostat_result_length_margin
        self.draw_multi_heliostat_result_selected_cacg_segment = draw_multi_heliostat_result_selected_cacg_segment
        self.draw_multi_heliostat_result_start_end_loci = draw_multi_heliostat_result_start_end_loci
        self.draw_multi_heliostat_result_legend = draw_multi_heliostat_result_legend
        self.draw_single_heliostat_etaC_dict = draw_single_heliostat_etaC_dict

    def gaze_example_C(self, section_context):
        if self.draw_single_heliostat_constraints_gaze_example_C:
            return self.draw_single_heliostat_constraints_gaze_example_C
        else:
            C_min = section_context["path_family_C_min"]
            C_max = section_context["path_family_C_max"]
            return C_min + (0.25 * (C_max - C_min))


# def meters(grid=True,):
#     """
#     Labels indicating units of meters.
#     """
#     return RenderControlScanSectionAnalysis(x_label='x (m)',
#                               y_label='y (m)',
#                               z_label='z (m)',
#                               grid=True,)
