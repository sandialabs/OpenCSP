"""
Render UFACET scan section constraint analysis.



"""

import matplotlib.pyplot as plt
import numpy as np

import opencsp.common.lib.geometry.angle as a
import opencsp.common.lib.render.color as c
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.tool.math_tools as mt
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rct
import opencsp.common.lib.render.view_spec as vs


# -------------------------------------------------------------------------------------------------------
# RENDERING CONSTRAINT ANALYSIS
#


def draw_key_points(view, at_pq, ab_pq, t_pq_list, bb_pq):
    color_at = "tab:orange"
    color_ab = "c"
    color_t = "r"
    color_bb = "b"

    view.draw_pq(at_pq, style=rcps.marker(color=color_at), label="at: Assessed top")
    view.draw_pq(ab_pq, style=rcps.marker(color=color_ab), label="ab: Assessed bottom")
    view.draw_pq_text(
        at_pq, "at", style=rct.RenderControlText(color=color_at, horizontalalignment="left", verticalalignment="top")
    )
    view.draw_pq_text(
        ab_pq, "ab", style=rct.RenderControlText(color=color_ab, horizontalalignment="left", verticalalignment="top")
    )

    t_idx = 1
    for t_pq in t_pq_list:
        view.draw_pq(t_pq, style=rcps.marker(color=color_t), label="t{0:d}: Target {0:d} top".format(t_idx))
        view.draw_pq_text(
            t_pq,
            "t{0:d}".format(t_idx),
            style=rct.RenderControlText(color=color_t, horizontalalignment="left", verticalalignment="top"),
        )
        t_idx += 1

    if bb_pq:
        view.draw_pq_text(
            bb_pq,
            "bb",
            style=rct.RenderControlText(color=color_bb, horizontalalignment="left", verticalalignment="top"),
        )
        view.draw_pq(bb_pq, style=rcps.marker(color=color_bb), label="bb: Background bottom")


def draw_sca_point(view, sca_pq, color):
    view.draw_pq(sca_pq, style=rcps.marker(color=color), label="sca: Start critical altitude")
    view.draw_pq_text(
        sca_pq, "sca", style=rct.RenderControlText(color=color, horizontalalignment="left", verticalalignment="top")
    )


def draw_eca_point(view, eca_pq, color):
    view.draw_pq(eca_pq, style=rcps.marker(color=color), label="eca: End critical altitude")
    view.draw_pq_text(
        eca_pq, "eca", style=rct.RenderControlText(color=color, horizontalalignment="left", verticalalignment="top")
    )


def draw_surface_normal(view, ab_pq, nu, color):
    length = 3
    head_p = ab_pq[0] + (length * np.cos(nu))
    head_q = ab_pq[1] + (length * np.sin(nu))
    head_pq = [head_p, head_q]
    view.draw_pq_list([ab_pq, head_pq], style=rcps.outline(color=color))


def draw_constraint_lower_bound(section_context, view, ray, color, label):
    extended_ray = g2d.extend_ray(ray, section_context["clip_pq_box"], fail_if_null_result=False)
    # This is rendering code, so if the ray is outside the bounding box, we want to do the best we can and keep going.
    if extended_ray == None:
        extended_ray = ray
    view.draw_pq_list(extended_ray, style=rcps.outline(color=color), label=label)
    head_pq = extended_ray[1]
    dir_pq = head_pq.copy()
    dir_pq[0] += 5
    view.draw_pq(head_pq, style=rcps.marker(color=color))
    view.draw_pq_list([head_pq, dir_pq], style=rcps.outline(color=color))


def draw_constraint_upper_bound(section_context, view, ray, color, label):
    extended_ray = g2d.extend_ray(ray, section_context["clip_pq_box"], fail_if_null_result=False)
    # This is rendering code, so if the ray is outside the bounding box, we want to do the best we can and keep going.
    if extended_ray == None:
        extended_ray = ray
    view.draw_pq_list(extended_ray, style=rcps.outline(color=color), label=label)
    head_pq = extended_ray[1]
    dir_pq = head_pq.copy()
    dir_pq[0] -= 5
    view.draw_pq(head_pq, style=rcps.marker(color=color))
    view.draw_pq_list([head_pq, dir_pq], style=rcps.outline(color=color))


def draw_heliostat_section(figure_control, section_context, heliostat_name_list, analysis_render_control):
    # Setup view.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control, rca.meters(), section_context["view_spec"], title="N-S Pass Section"
    )
    view = fig_record.view
    fig_record.comment.append("Path segment analysis aection.")

    # Draw heliostats.
    for heliostat_name in heliostat_name_list:
        heliostat = section_context["solar_field"].lookup_heliostat(heliostat_name)
        # Style setup
        heliostat_style = rch.outline()
        heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
        # Draw
        heliostat.draw(view, heliostat_styles)

    # Draw safe altitude lines.
    if analysis_render_control.draw_context_mnsa_ray:
        view.draw_pq_list(
            section_context["mnsa_ray"], style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None)
        )
    if analysis_render_control.draw_context_mxsa_ray:
        view.draw_pq_list(
            section_context["mxsa_ray"], style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None)
        )

    # Finish the figure.
    view.show()

    # Return.
    return view


def draw_single_heliostat_constraint_analysis(
    figure_control, section_context, heliostat_name_list, assess_heliostat_name, constraints, analysis_render_control
):
    # Setup view.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.meters(),
        section_context["view_spec"],
        title=(
            assess_heliostat_name + " in " + heliostat_name_list[0] + "-" + heliostat_name_list[-1] + " Pass Section"
        ),
    )
    view = fig_record.view
    fig_record.comment.append("Scan pass section analysis for " + assess_heliostat_name)

    # Draw heliostats.
    if analysis_render_control.draw_single_heliostat_constraints_heliostats:
        for heliostat_name in heliostat_name_list:
            heliostat = section_context["solar_field"].lookup_heliostat(heliostat_name)
            # Style setup
            if heliostat_name == assess_heliostat_name:
                heliostat_style = rch.name_outline(color="m", horizontalalignment="left", verticalalignment="top")
            else:
                heliostat_style = rch.outline()
            heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
            # Draw
            heliostat.draw(view, heliostat_styles)

    # Fetch constraints.
    mnsa_ray = section_context["mnsa_ray"]
    mxsa_ray = section_context["mxsa_ray"]
    at_pq = constraints["at_pq"]
    ab_pq = constraints["ab_pq"]
    t_pq_list = constraints["t_pq_list"]
    bb_pq = constraints["bb_pq"]
    nu = constraints["nu"]
    abv_lb = constraints["abv_lb"]  # Assessed bottom visibility, p lower bound.
    abvm_lb = constraints["abvm_lb"]  # Assessed bottom visibility margin, p lower bound.
    atv_lb = constraints["atv_lb"]  # Assessed top visibility, p lower bound.
    atvm_lb = constraints["atvm_lb"]  # Assessed top visibility margin, p lower bound.
    ts_ub_list = constraints["ts_ub_list"]  # Target reflection start list, p upper bound.
    ts_ub = constraints["ts_ub"]  # Target reflection start, p upper bound.
    tsm_ub = constraints["tsm_ub"]  # Target reflection margin, p upper bound.
    sca_pq = constraints["sca_pq"]  # Path start critical altitude point.
    s_locus = constraints["s_locus"]  # Valid pass start points.
    te_lb_list = constraints["te_lb_list"]  # Target reflection end list, p lower bound.
    te_lb = constraints["te_lb"]  # Target reflection end, p lower bound.
    tem_lb = constraints["tem_lb"]  # Target reflection end margin, p lower bound.
    pln_ub = constraints["pln_ub"]  # Mirror plane, p upper bound.
    eca_pq = constraints["eca_pq"]  # Path end critical altitude point.
    e_locus = constraints["e_locus"]  # Valid pass end points.

    # Draw safe altitude lines.
    if analysis_render_control.draw_single_heliostat_constraints_mnsa_ray:
        view.draw_pq_list(mnsa_ray, style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None))
    if analysis_render_control.draw_single_heliostat_constraints_mxsa_ray:
        view.draw_pq_list(mxsa_ray, style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None))

    # Draw key points.
    if analysis_render_control.draw_single_heliostat_constraints_key_points:
        draw_key_points(view, at_pq, ab_pq, t_pq_list, bb_pq)

    # Draw assessed heliostat normal at top and bottom points.
    if analysis_render_control.draw_single_heliostat_constraints_assessed_normals:
        draw_surface_normal(view, ab_pq, nu, color="c")
        draw_surface_normal(view, at_pq, nu, color="tab:orange")

    # Draw constraints.
    if analysis_render_control.draw_single_heliostat_constraints_detail:
        # Start pass (at or above, at or before).
        if abv_lb:
            draw_constraint_lower_bound(
                section_context, view, abv_lb, color="r", label="Assessed bottom visibility, p lower bound"
            )
        if abvm_lb:
            draw_constraint_lower_bound(
                section_context, view, abvm_lb, color="pink", label="Assessed bottom visibility, p lower bound"
            )
        if atv_lb:
            draw_constraint_lower_bound(
                section_context, view, atv_lb, color="b", label="Assessed top visibility, p lower bound"
            )
        if atvm_lb:
            draw_constraint_lower_bound(
                section_context, view, atvm_lb, color="skyblue", label="Top visibility margin, p lower bound"
            )
        if len(ts_ub_list) > 0:
            # The first target dominates.
            draw_constraint_upper_bound(
                section_context, view, ts_ub, color="g", label="Target reflection start, p upper bound"
            )
            if len(ts_ub_list) > 1:
                if analysis_render_control.draw_single_heliostat_constraints_all_targets:
                    for ts_ub2 in ts_ub_list[1:]:  # Differentiate from ts_ub.
                        draw_constraint_upper_bound(
                            section_context, view, ts_ub2, color="g", label="Target reflection start, p upper bound"
                        )
        if tsm_ub:
            draw_constraint_upper_bound(
                section_context, view, tsm_ub, color="c", label="Target reflection margin, p upper bound"
            )
        if len(te_lb_list) > 0:
            # The first target dominates.
            draw_constraint_lower_bound(
                section_context, view, te_lb, color="g", label="Target reflection end, p lower bound"
            )
            if len(te_lb_list) > 1:
                if analysis_render_control.draw_single_heliostat_constraints_all_targets:
                    for te_lb2 in te_lb_list[1:]:  # Differentiate from te_lb.
                        # These are nearly superimposed.
                        draw_constraint_lower_bound(
                            section_context, view, te_lb2, color="g", label="Target reflection end, p lower bound"
                        )
        if tem_lb:
            draw_constraint_upper_bound(
                section_context, view, tem_lb, color="olive", label="Target reflection margin, p upper bound"
            )
        draw_constraint_upper_bound(
            section_context, view, pln_ub, color="c", label="Mirror plane, p upper bound"
        )  # Reflection end dominates.
    if analysis_render_control.draw_single_heliostat_constraints_summary:
        draw_sca_point(view, sca_pq, "r")
        draw_eca_point(view, eca_pq, "b")
        draw_constraint_upper_bound(section_context, view, s_locus, color="r", label="Valid pass start points")
        draw_constraint_lower_bound(section_context, view, e_locus, color="b", label="Valid pass end points")

    # Draw example gaze constraints.
    if analysis_render_control.draw_single_heliostat_constraints_gaze_example:
        C_example = analysis_render_control.gaze_example_C(section_context)
        C_example = mt.clamp(C_example, section_context["path_family_C_min"], section_context["path_family_C_max"])
        for path_s_pq, path_e_pq, ray_min_eta, ray_max_eta, min_etaC, max_etaC in zip(
            constraints["path_s_pq_list"],
            constraints["path_e_pq_list"],
            constraints["ray_min_eta_list"],
            constraints["ray_max_eta_list"],
            constraints["min_etaC_list"],
            constraints["max_etaC_list"],
        ):
            if min_etaC[1] > C_example:  # Inexact match.
                view.draw_pq_list(
                    [path_s_pq, path_e_pq], style=rcps.RenderControlPointSeq(linestyle="--", color="g", marker=None)
                )
                view.draw_pq_list(ray_min_eta, style=rcps.RenderControlPointSeq(linestyle=":", color="b", marker=None))
                view.draw_pq_list(ray_max_eta, style=rcps.RenderControlPointSeq(linestyle=":", color="r", marker=None))
                break

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_single_heliostat_constraints_legend)

    # Return.
    return view


def draw_single_heliostat_gaze_angle_analysis(
    figure_control, section_context, h_a_name, constraints, analysis_render_control
):
    # Fetch constraints.
    min_etaC_list = constraints["min_etaC_list"]
    max_etaC_list = constraints["max_etaC_list"]

    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.RenderControlAxis(x_label="eta (deg)", y_label="C (m)", z_label="", grid=True),
        vs.view_spec_xy(),
        title=(h_a_name + " Gaze Angle Analysis"),
    )
    view = fig_record.view

    # Draw bounding curves.
    view.draw_pq_list(a.p2deg(min_etaC_list), style=rcps.data_curve(color="b"), label="{0:s} eta_min".format(h_a_name))
    view.draw_pq_list(a.p2deg(max_etaC_list), style=rcps.data_curve(color="r"), label="{0:s} eta_max".format(h_a_name))

    # Start and end points for altitude lines.
    draw_eta_min = min([pq[0] for pq in min_etaC_list]) - np.deg2rad(5.0)
    draw_eta_max = max([pq[0] for pq in max_etaC_list]) + np.deg2rad(5.0)

    # Safe altitudes.
    C_mnsa = section_context["path_family_C_mnsa"]
    C_mxsa = section_context["path_family_C_mxsa"]
    if analysis_render_control.draw_single_heliostat_gaze_angle_mxsa:
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mxsa], [draw_eta_max, C_mxsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None),
            label="Maximum Safe",
        )
    if analysis_render_control.draw_single_heliostat_gaze_angle_mnsa:
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mnsa], [draw_eta_max, C_mnsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None),
            label="Minimum Safe",
        )

    # Critical altitude.
    if analysis_render_control.draw_single_heliostat_gaze_angle_critical:
        C_critical = constraints["C_critical"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_critical], [draw_eta_max, C_critical]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="orange", marker=None),
            label="Critical",
        )

    # Example C value.
    if analysis_render_control.draw_single_heliostat_gaze_angle_example:
        C_example = analysis_render_control.gaze_example_C(section_context)
        C_example = mt.clamp(C_example, section_context["path_family_C_min"], section_context["path_family_C_max"])
        for min_etaC, max_etaC in zip(min_etaC_list, max_etaC_list):
            if min_etaC[1] > C_example:  # Inexact match.
                view.draw_pq_list(
                    a.p2deg([min_etaC, max_etaC]),
                    style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None),
                )
                break

    # Fill region between curves.
    if analysis_render_control.draw_single_heliostat_gaze_angle_fill:
        for min_etaC, max_etaC in zip(min_etaC_list, max_etaC_list):
            view.draw_pq_list(
                a.p2deg([min_etaC, max_etaC]), style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None)
            )

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_single_heliostat_gaze_angle_legend)

    # Return.
    return view


def draw_single_heliostat_select_gaze(figure_control, section_context, h_a_name, constraints, analysis_render_control):
    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.RenderControlAxis(x_label="eta (deg)", y_label="C (m)", z_label="", grid=True),
        vs.view_spec_xy(),
        title=(h_a_name + " Gaze Angle and Altitude Selection"),
    )
    view = fig_record.view

    # Draw shifted worst-case limits.
    if analysis_render_control.draw_single_heliostat_select_gaze_shifted:
        shifted_min_etaC_list = constraints["shifted_min_etaC_list"]
        shifted_max_etaC_list = constraints["shifted_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(shifted_min_etaC_list),
            style=rcps.data_curve(color="b", linewidth=0.5),
            label="Shifted Pass eta_min",
        )
        view.draw_pq_list(
            a.p2deg(shifted_max_etaC_list),
            style=rcps.data_curve(color="r", linewidth=0.5),
            label="Shifted Pass eta_max",
        )

    # Draw gaze angle ideal envelope.
    if analysis_render_control.draw_single_heliostat_select_gaze_envelope:
        envelope_min_etaC_list = constraints["envelope_min_etaC_list"]
        envelope_max_etaC_list = constraints["envelope_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(envelope_min_etaC_list), style=rcps.data_curve(color="b"), label="Gaze Envelope eta_min"
        )
        view.draw_pq_list(
            a.p2deg(envelope_max_etaC_list), style=rcps.data_curve(color="r"), label="Gaze Envelope eta_max"
        )

    # Draw gaze angle envelope after shrinking for uncertainty.
    if analysis_render_control.draw_single_heliostat_select_gaze_shrunk:
        shrunk_min_etaC_list = constraints["shrunk_min_etaC_list"]
        shrunk_max_etaC_list = constraints["shrunk_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(shrunk_min_etaC_list), style=rcps.data_curve(color="b", linewidth=1.5), label="Shrunk eta_min"
        )
        view.draw_pq_list(
            a.p2deg(shrunk_max_etaC_list), style=rcps.data_curve(color="r", linewidth=1.5), label="Shrunk eta_max"
        )

    # Draw gaze angle envelope after clipping for gaze angle limits.
    if analysis_render_control.draw_single_heliostat_select_gaze_clipped:
        clipped_min_etaC_list = constraints["clipped_min_etaC_list"]
        clipped_max_etaC_list = constraints["clipped_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(clipped_min_etaC_list), style=rcps.data_curve(color="b", linewidth=1.5), label="clipped eta_min"
        )
        view.draw_pq_list(
            a.p2deg(clipped_max_etaC_list), style=rcps.data_curve(color="r", linewidth=1.5), label="clipped eta_max"
        )

    # Draw selected gaze angle and altitude.
    if analysis_render_control.draw_single_heliostat_select_gaze_selected:
        selected_cacg_etaC = constraints["selected_cacg_etaC"]
        view.draw_pq(
            a.p2deg(selected_cacg_etaC),
            style=rcps.marker(color="g", marker="D", markersize=6),
            label="Selected (eta,C)",
        )

    # Start and end points for altitude lines.
    draw_eta_min = min([pq[0] for pq in constraints["shifted_min_etaC_list"]]) - np.deg2rad(5.0)
    draw_eta_max = max([pq[0] for pq in constraints["shifted_max_etaC_list"]]) + np.deg2rad(5.0)

    # Safe altitudes.
    if analysis_render_control.draw_single_heliostat_select_gaze_mxsa:
        C_mxsa = section_context["path_family_C_mxsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mxsa], [draw_eta_max, C_mxsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None),
            label="Maximum Safe",
        )
    if analysis_render_control.draw_single_heliostat_select_gaze_mnsa:
        C_mnsa = section_context["path_family_C_mnsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mnsa], [draw_eta_max, C_mnsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None),
            label="Minimum Safe",
        )

    # Critical altitude.
    if analysis_render_control.draw_single_heliostat_select_gaze_critical:
        C_critical = constraints["C_critical"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_critical], [draw_eta_max, C_critical]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="orange", linewidth=0.5, marker=None),
            label="C_critical",
        )

    # Fill region between curves.
    if analysis_render_control.draw_single_heliostat_select_gaze_fill:
        clipped_min_etaC_list = constraints["clipped_min_etaC_list"]
        clipped_max_etaC_list = constraints["clipped_max_etaC_list"]
        for min_etaC, max_etaC in zip(clipped_min_etaC_list, clipped_max_etaC_list):
            view.draw_pq_list(
                a.p2deg([min_etaC, max_etaC]), style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None)
            )

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_single_heliostat_select_gaze_legend)

    # Return.
    return view


def draw_multi_heliostat_gaze_angle_analysis(
    figure_control, section_context, pass_constraints, analysis_render_control
):
    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.RenderControlAxis(x_label="eta (deg)", y_label="C (m)", z_label="", grid=True),
        vs.view_spec_xy(),
        title="Full-Pass Gaze Angle Analysis",
    )
    view = fig_record.view

    # Draw per-heliostat required curves.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_per_heliostat:
        # Draw min required gaze angle.
        color_idx = 0
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            min_etaC_list = constraints["min_etaC_list"]
            view.draw_pq_list(
                a.p2deg(min_etaC_list),
                style=rcps.outline(color=c.color(color_idx), linewidth=0.5),
                label="{0:s} eta_min".format(assess_heliostat_name),
            )
            color_idx += 1

        # Draw max required gaze angle.
        color_idx = 0
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            max_etaC_list = constraints["max_etaC_list"]
            view.draw_pq_list(
                a.p2deg(max_etaC_list),
                style=rcps.outline(color=c.color(color_idx), linewidth=0.5),
                label="{0:s} eta_max".format(assess_heliostat_name),
            )
            color_idx += 1

    # Draw worst-case limits.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_envelope:
        pass_min_etaC_list = pass_constraints["pass_min_etaC_list"]
        pass_max_etaC_list = pass_constraints["pass_max_etaC_list"]
        view.draw_pq_list(a.p2deg(pass_min_etaC_list), style=rcps.data_curve(color="b"), label="Pass eta_min")
        view.draw_pq_list(a.p2deg(pass_max_etaC_list), style=rcps.data_curve(color="r"), label="Pass eta_max")

    # Start and end points for altitude lines.
    draw_eta_min = min([pq[0] for pq in pass_min_etaC_list]) - np.deg2rad(5.0)
    draw_eta_max = max([pq[0] for pq in pass_max_etaC_list]) + np.deg2rad(5.0)

    # Safe altitudes.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_mxsa:
        C_mxsa = section_context["path_family_C_mxsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mxsa], [draw_eta_max, C_mxsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None),
            label="Maximum Safe",
        )
    if analysis_render_control.draw_multi_heliostat_gaze_angle_mnsa:
        C_mnsa = section_context["path_family_C_mnsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mnsa], [draw_eta_max, C_mnsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None),
            label="Minimum Safe",
        )

    # Critical altitudes.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_critical:
        color_idx = 0
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            C_critical = constraints["C_critical"]
            view.draw_pq_list(
                a.p2deg([[draw_eta_min, C_critical], [draw_eta_max, C_critical]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color=c.color(color_idx), linewidth=0.5, marker=None),
                label="{0:s} C_critical".format(assess_heliostat_name),
            )
            color_idx += 1

    # Draw eta range for an example C value.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_example:
        C_example = analysis_render_control.gaze_example_C(section_context)
        C_example = mt.clamp(C_example, section_context["path_family_C_min"], section_context["path_family_C_max"])
        pass_min_etaC_list = pass_constraints["pass_min_etaC_list"]
        pass_max_etaC_list = pass_constraints["pass_max_etaC_list"]
        for min_etaC, max_etaC in zip(pass_min_etaC_list, pass_max_etaC_list):
            if min_etaC[1] > C_example:  # Inexact match.
                view.draw_pq_list(
                    a.p2deg([min_etaC, max_etaC]),
                    style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None),
                )
                break

    # Fill region between curves.
    if analysis_render_control.draw_multi_heliostat_gaze_angle_fill:
        pass_min_etaC_list = pass_constraints["pass_min_etaC_list"]
        pass_max_etaC_list = pass_constraints["pass_max_etaC_list"]
        for min_etaC, max_etaC in zip(pass_min_etaC_list, pass_max_etaC_list):
            view.draw_pq_list(
                a.p2deg([min_etaC, max_etaC]), style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None)
            )

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_multi_heliostat_gaze_angle_legend)

    # Return.
    return view


def draw_required_vertical_field_of_view(figure_control, section_context, pass_constraints, analysis_render_control):
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.RenderControlAxis(x_label="Minimum Required Vertical FOV (deg)", y_label="C (m)", z_label="", grid=True),
        vs.view_spec_xy(),
        title="Vertical Field of View Analysis",
    )
    view = fig_record.view

    # Plot the curve.
    vertical_fovC_list = pass_constraints["vertical_fovC_list"]
    view.draw_pq_list(
        a.p2deg(vertical_fovC_list),
        style=rcps.RenderControlPointSeq(linestyle="-", color="b", marker="."),
        label="From constraints",
    )

    # Start and end points for altitude lines.
    draw_fov_min = min([pq[0] for pq in vertical_fovC_list]) - np.deg2rad(5.0)
    draw_fov_max = max([pq[0] for pq in vertical_fovC_list]) + np.deg2rad(5.0)

    # Safe altitudes.
    if analysis_render_control.draw_multi_heliostat_vertical_fov_required_mxsa:
        C_mxsa = section_context["path_family_C_mxsa"]
        view.draw_pq_list(
            a.p2deg([[draw_fov_min, C_mxsa], [draw_fov_max, C_mxsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None),
            label="Maximum Safe",
        )
    if analysis_render_control.draw_multi_heliostat_vertical_fov_required_mnsa:
        C_mnsa = section_context["path_family_C_mnsa"]
        view.draw_pq_list(
            a.p2deg([[draw_fov_min, C_mnsa], [draw_fov_max, C_mnsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None),
            label="Minimum Safe",
        )

    # Critical altitudes.
    if analysis_render_control.draw_multi_heliostat_vertical_fov_required_critical:
        color_idx = 0
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            C_critical = constraints["C_critical"]
            view.draw_pq_list(
                a.p2deg([[draw_fov_min, C_critical], [draw_fov_max, C_critical]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color=c.color(color_idx), linewidth=0.5, marker=None),
                label="{0:s} C_critical".format(assess_heliostat_name),
            )
            color_idx += 1

    # Camera field of view.
    if analysis_render_control.draw_multi_heliostat_vertical_fov_required_camera:
        C_mnsa = section_context["path_family_C_mnsa"]
        C_mxsa = section_context["path_family_C_mxsa"]
        camera = section_context["camera"]
        vertical_fov_min = camera.fov_vertical_min
        vertical_fov_max = camera.fov_vertical_max
        draw_C_min = C_mnsa - 2.0  # m
        draw_C_max = C_mxsa + 2.0  # m
        if vertical_fov_min == vertical_fov_max:
            view.draw_pq_list(
                a.p2deg([[vertical_fov_min, draw_C_min], [vertical_fov_min, draw_C_max]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color="g", linewidth=1.5, marker=None),
                label="{0:s} FOV".format(camera.name),
            )
        else:
            view.draw_pq_list(
                a.p2deg([[vertical_fov_min, draw_C_min], [vertical_fov_min, draw_C_max]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color="g", linewidth=1.5, marker=None),
                label="{0:s} min FOV".format(camera.name),
            )
            view.draw_pq_list(
                a.p2deg([[vertical_fov_max, draw_C_min], [vertical_fov_max, draw_C_max]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color="g", linewidth=1.5, marker=None),
                label="{0:s} max FOV".format(camera.name),
            )

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_multi_heliostat_vertical_fov_required_legend)

    # Return.
    return view


def draw_multi_heliostat_select_gaze(figure_control, section_context, pass_constraints, analysis_render_control):
    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.RenderControlAxis(x_label="eta (deg)", y_label="C (m)", z_label="", grid=True),
        vs.view_spec_xy(),
        title="Gaze Angle and Altitude Selection",
    )
    view = fig_record.view

    # Draw shifted worst-case limits.
    if analysis_render_control.draw_multi_heliostat_select_gaze_shifted:
        shifted_min_etaC_list = pass_constraints["shifted_min_etaC_list"]
        shifted_max_etaC_list = pass_constraints["shifted_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(shifted_min_etaC_list),
            style=rcps.data_curve(color="b", linewidth=0.5),
            label="Shifted Pass eta_min",
        )
        view.draw_pq_list(
            a.p2deg(shifted_max_etaC_list),
            style=rcps.data_curve(color="r", linewidth=0.5),
            label="Shifted Pass eta_max",
        )

    # Draw gaze angle ideal envelope.
    if analysis_render_control.draw_multi_heliostat_select_gaze_envelope:
        envelope_min_etaC_list = pass_constraints["envelope_min_etaC_list"]
        envelope_max_etaC_list = pass_constraints["envelope_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(envelope_min_etaC_list), style=rcps.data_curve(color="b"), label="Gaze Envelope eta_min"
        )
        view.draw_pq_list(
            a.p2deg(envelope_max_etaC_list), style=rcps.data_curve(color="r"), label="Gaze Envelope eta_max"
        )

    # Draw gaze angle envelope after shrinking for uncertainty.
    if analysis_render_control.draw_multi_heliostat_select_gaze_shrunk:
        shrunk_min_etaC_list = pass_constraints["shrunk_min_etaC_list"]
        shrunk_max_etaC_list = pass_constraints["shrunk_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(shrunk_min_etaC_list), style=rcps.data_curve(color="b", linewidth=1.5), label="Shrunk eta_min"
        )
        view.draw_pq_list(
            a.p2deg(shrunk_max_etaC_list), style=rcps.data_curve(color="r", linewidth=1.5), label="Shrunk eta_max"
        )

    # Draw gaze angle envelope after clipping for gaze angle limits.
    if analysis_render_control.draw_multi_heliostat_select_gaze_clipped:
        clipped_min_etaC_list = pass_constraints["clipped_min_etaC_list"]
        clipped_max_etaC_list = pass_constraints["clipped_max_etaC_list"]
        view.draw_pq_list(
            a.p2deg(clipped_min_etaC_list), style=rcps.data_curve(color="b", linewidth=1.5), label="clipped eta_min"
        )
        view.draw_pq_list(
            a.p2deg(clipped_max_etaC_list), style=rcps.data_curve(color="r", linewidth=1.5), label="clipped eta_max"
        )

    # Draw selected gaze angle and altitude.
    if analysis_render_control.draw_multi_heliostat_select_gaze_selected:
        selected_cacg_etaC = pass_constraints["selected_cacg_etaC"]
        view.draw_pq(
            a.p2deg(selected_cacg_etaC),
            style=rcps.marker(color="g", marker="D", markersize=6),
            label="Selected (eta,C)",
        )

    # Start and end points for altitude lines.
    draw_eta_min = min([pq[0] for pq in pass_constraints["shifted_min_etaC_list"]]) - np.deg2rad(5.0)
    draw_eta_max = max([pq[0] for pq in pass_constraints["shifted_max_etaC_list"]]) + np.deg2rad(5.0)

    # Safe altitudes.
    if analysis_render_control.draw_multi_heliostat_select_gaze_mxsa:
        C_mxsa = section_context["path_family_C_mxsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mxsa], [draw_eta_max, C_mxsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None),
            label="Maximum Safe",
        )
    if analysis_render_control.draw_multi_heliostat_select_gaze_mnsa:
        C_mnsa = section_context["path_family_C_mnsa"]
        view.draw_pq_list(
            a.p2deg([[draw_eta_min, C_mnsa], [draw_eta_max, C_mnsa]]),
            style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None),
            label="Minimum Safe",
        )

    # Critical altitudes.
    if analysis_render_control.draw_multi_heliostat_select_gaze_critical:
        color_idx = 0
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            C_critical = constraints["C_critical"]
            view.draw_pq_list(
                a.p2deg([[draw_eta_min, C_critical], [draw_eta_max, C_critical]]),
                style=rcps.RenderControlPointSeq(linestyle="--", color=c.color(color_idx), linewidth=0.5, marker=None),
            )
            #                              label='{0:s} C_critical'.format(assess_heliostat_name))
            color_idx += 1

    # Fill region between curves.
    if analysis_render_control.draw_multi_heliostat_select_gaze_fill:
        clipped_min_etaC_list = pass_constraints["clipped_min_etaC_list"]
        clipped_max_etaC_list = pass_constraints["clipped_max_etaC_list"]
        for min_etaC, max_etaC in zip(clipped_min_etaC_list, clipped_max_etaC_list):
            view.draw_pq_list(
                a.p2deg([min_etaC, max_etaC]), style=rcps.RenderControlPointSeq(linestyle="--", color="c", marker=None)
            )

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_multi_heliostat_select_gaze_legend)

    # Return.
    return view


def draw_multi_heliostat_result(
    figure_control, section_context, heliostat_name_list, pass_constraints, analysis_render_control
):
    # Setup view.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control,
        rca.meters(),
        section_context["view_spec"],
        title=(heliostat_name_list[0] + "-" + heliostat_name_list[-1] + " Pass Section Result"),
    )
    view = fig_record.view
    fig_record.comment.append("Scan pass section analysis result,")

    # Draw heliostats.
    if analysis_render_control.draw_multi_heliostat_result_heliostats:
        for heliostat_name in heliostat_name_list:
            heliostat = section_context["solar_field"].lookup_heliostat(heliostat_name)
            # Setup styles.
            heliostat_style = rch.outline()
            heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
            # Draw.
            heliostat.draw(view, heliostat_styles)

    # Draw safe altitude lines.
    if analysis_render_control.draw_multi_heliostat_result_mnsa_ray:
        mnsa_ray = section_context["mnsa_ray"]
        view.draw_pq_list(mnsa_ray, style=rcps.RenderControlPointSeq(linestyle="--", color="brown", marker=None))
    if analysis_render_control.draw_multi_heliostat_result_mxsa_ray:
        mxsa_ray = section_context["mxsa_ray"]
        view.draw_pq_list(mxsa_ray, style=rcps.RenderControlPointSeq(linestyle="--", color="r", marker=None))

    # Draw selected flight path line.
    if analysis_render_control.draw_multi_heliostat_result_selected_cacg_line:
        selected_cacg_line = pass_constraints["selected_cacg_line"]
        # Copy start and end points from altitude ray.
        mnsa_ray = section_context["mnsa_ray"]
        mnsa_p0 = mnsa_ray[0][0]
        mnsa_p1 = mnsa_ray[1][0]
        # Copy start and end points from selected path segment.
        selected_cacg_segment = pass_constraints["selected_cacg_segment"]
        segment_p0 = selected_cacg_segment[0][0]
        segment_p1 = selected_cacg_segment[1][0]
        # Construct interval spanning both.
        length_margin = analysis_render_control.draw_multi_heliostat_result_length_margin
        p0 = min(mnsa_p0, (segment_p0 - length_margin))
        p1 = max(mnsa_p1, (segment_p1 + length_margin))
        # Construct embedding line.
        selected_cacg_line_q0 = g2d.homogeneous_line_y_given_x(p0, selected_cacg_line)
        selected_cacg_line_q1 = g2d.homogeneous_line_y_given_x(p1, selected_cacg_line)
        selected_cacg_line_ray = [[p0, selected_cacg_line_q0], [p1, selected_cacg_line_q1]]
        view.draw_pq_list(
            selected_cacg_line_ray, style=rcps.RenderControlPointSeq(linestyle="--", color="g", marker=None)
        )

    # Draw selected flight path segment.
    if analysis_render_control.draw_multi_heliostat_result_selected_cacg_segment:
        selected_cacg_segment = pass_constraints["selected_cacg_segment"]
        view.draw_pq_list(selected_cacg_segment, style=rcps.outline(color="g", linewidth=4))

    # Draw start and end loci.
    if analysis_render_control.draw_multi_heliostat_result_start_end_loci:
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            constraints = per_heliostat_constraints[assess_heliostat_name]
            sca_pq = constraints["sca_pq"]  # Path start critical altitude point.
            eca_pq = constraints["eca_pq"]  # Path end critical altitude point.
            s_locus = constraints["s_locus"]  # Valid pass start points.
            e_locus = constraints["e_locus"]  # Valid pass end points.
            draw_sca_point(view, sca_pq, "r")
            draw_eca_point(view, eca_pq, "b")
            draw_constraint_upper_bound(section_context, view, s_locus, color="r", label="Valid pass start points")
            draw_constraint_lower_bound(section_context, view, e_locus, color="b", label="Valid pass end points")

    # Finish the figure.
    view.show(legend=analysis_render_control.draw_multi_heliostat_result_legend)

    # Return.
    return view


def draw_single_heliostat_etaC_dict(figure_control, pass_constraints):
    # Fetch individual heliostat selected (eta,C) results.
    selected_cacg_etaC_dict = pass_constraints["selected_cacg_etaC_dict"]

    # Assemble lists of individual parameters.
    h_a_name_list = []
    selected_cacg_eta_deg_list = []
    selected_cacg_C_list = []
    for h_a_name in selected_cacg_etaC_dict.keys():
        selected_cacg_etaC = selected_cacg_etaC_dict[h_a_name]
        h_a_name_list.append(h_a_name)
        selected_cacg_eta_deg_list.append(np.rad2deg(selected_cacg_etaC[0]))
        selected_cacg_C_list.append(selected_cacg_etaC[1])

    # Plot selected gaze values.
    fig_record_1 = fm.setup_figure(figure_control, title="Gaze Angle Selected for Individual Heliostats")
    plt.plot(selected_cacg_eta_deg_list, ".-", color="b")
    plt.xlabel("Heliostat Index")
    plt.ylabel("eta (deg)")
    ax_1 = plt.gca()
    ax_1.set_ylim([-90, -60])
    plt.grid()
    plt.show()

    # Plot selected C values.
    fig_record_2 = fm.setup_figure(figure_control, title="Altitude Selected for Individual Heliostats")
    plt.plot(selected_cacg_C_list, ".-", color="g")
    plt.xlabel("Heliostat Index")
    plt.ylabel("C (m)")
    ax_2 = plt.gca()
    ax_2.set_ylim([10, 40])
    plt.grid()
    plt.show()

    # Return.
    return fig_record_1, fig_record_2


def draw_section_analysis(
    figure_control, section_context, heliostat_name_list, pass_constraints, analysis_render_control
):
    # Notify progress.
    print("Drawing section " + heliostat_name_list[0] + "-" + heliostat_name_list[-1] + " analysis...")

    # Draw the section context.
    if analysis_render_control.draw_context:
        draw_heliostat_section(figure_control, section_context, heliostat_name_list, analysis_render_control)

    # Draw the constraint analysis results.
    if analysis_render_control.draw_single_heliostat_analysis:
        per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
        for assess_heliostat_name in per_heliostat_constraints.keys():
            if (analysis_render_control.draw_single_heliostat_analysis_list == None) or (
                assess_heliostat_name in analysis_render_control.draw_single_heliostat_analysis_list
            ):
                # Lookup constraints.
                constraints = per_heliostat_constraints[assess_heliostat_name]
                # Draw constraint analysis.
                if analysis_render_control.draw_single_heliostat_constraints:
                    draw_single_heliostat_constraint_analysis(
                        figure_control,
                        section_context,
                        heliostat_name_list,
                        assess_heliostat_name,
                        constraints,
                        analysis_render_control,
                    )
                # Draw gaze angle analysis.
                if analysis_render_control.draw_single_heliostat_gaze_angle:
                    draw_single_heliostat_gaze_angle_analysis(
                        figure_control, section_context, assess_heliostat_name, constraints, analysis_render_control
                    )
                # Draw gaze angle selection.
                if analysis_render_control.draw_single_heliostat_select_gaze:
                    draw_single_heliostat_select_gaze(
                        figure_control, section_context, assess_heliostat_name, constraints, analysis_render_control
                    )

    # Draw summary gaze angle analysis.
    if analysis_render_control.draw_multi_heliostat_gaze_angle:
        draw_multi_heliostat_gaze_angle_analysis(
            figure_control, section_context, pass_constraints, analysis_render_control
        )

    # Draw vertical field of view requirement.
    if analysis_render_control.draw_multi_heliostat_vertical_fov_required:
        draw_required_vertical_field_of_view(figure_control, section_context, pass_constraints, analysis_render_control)

    # Draw gaze angle and altitude selection.
    if analysis_render_control.draw_multi_heliostat_select_gaze:
        draw_multi_heliostat_select_gaze(figure_control, section_context, pass_constraints, analysis_render_control)

    # Draw the selected flight path in the heliostat context.
    if analysis_render_control.draw_multi_heliostat_result:
        draw_multi_heliostat_result(
            figure_control, section_context, heliostat_name_list, pass_constraints, analysis_render_control
        )

    # Draw the individual heliostat selected (eta,C) results.
    if analysis_render_control.draw_single_heliostat_etaC_dict:
        draw_single_heliostat_etaC_dict(figure_control, pass_constraints)
