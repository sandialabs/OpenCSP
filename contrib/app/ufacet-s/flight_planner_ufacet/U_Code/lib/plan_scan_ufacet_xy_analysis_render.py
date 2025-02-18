"""
Rendring the analysis in (x,y) space to find UFACET scan segments.



"""

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.render.view_spec as vs


def draw_ufacet_xy_analysis(
    figure_control,
    solar_field,
    aimpoint_xyz,
    field_origin_lon_lat,
    when_ymdhmsz,
    curve_key_xy_list,
    list_of_ideal_xy_lists,
    list_of_best_fit_segment_xys,
    render_control_scan_xy_analysis,
):
    # Draw analysis to find segments.
    if render_control_scan_xy_analysis.draw_xy_segment_analysis:
        # Solar field.
        rcsf_vfield = rcsf.heliostat_vector_field_outlines(color="grey")
        view_xy = sf.draw_solar_field(figure_control, solar_field, rcsf_vfield, vs.view_spec_xy())
        # Dense vector field.
        grid_xy = solar_field.heliostat_field_regular_grid_xy(30, 15)
        grid_xydxy = [
            [p, sun_track.tracking_surface_normal_xy(p + [0], aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz)]
            for p in grid_xy
        ]
        view_xy.draw_pqdpq_list(grid_xydxy, style=rcps.vector_field(color="k", vector_scale=5.0))
        # Key points.
        view_xy.draw_pq_list(curve_key_xy_list, style=rcps.marker(markersize=5, color="r"))
        # Ideal gaze curves.
        for ideal_xy_list in list_of_ideal_xy_lists:
            view_xy.draw_pq_list(ideal_xy_list, style=rcps.outline(color="r"))
        # Best fit segments.
        for segment_xy in list_of_best_fit_segment_xys:
            view_xy.draw_pq_list(segment_xy, style=rcps.outline(color="g", linewidth=2))

    # Draw best-fit segment result.
    if render_control_scan_xy_analysis.draw_xy_segment_result:
        # Solar field.
        rcsf_vfield = rcsf.heliostat_vector_field_outlines(color="grey")
        view_xy = sf.draw_solar_field(figure_control, solar_field, rcsf_vfield, vs.view_spec_xy())
        # Best fit segments.
        for segment_xy in list_of_best_fit_segment_xys:
            view_xy.draw_pq_list(segment_xy, style=rcps.outline(color="g", linewidth=2))
