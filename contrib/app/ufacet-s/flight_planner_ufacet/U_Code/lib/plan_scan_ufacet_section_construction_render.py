"""
Rendering the Construction of UFACET scan sections.



"""

import numpy as np

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf


def draw_construct_ufacet_section(
    figure_control, solar_field, section, render_view_spec, render_control_scan_section_setup
):
    # Draw setup of the section.
    if render_control_scan_section_setup.draw_section_setup:
        # Setup figure.
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, rca.meters(), render_view_spec, title="UFACET Section Construction"
        )
        view = fig_record.view
        # Comment.
        fig_record.comment.append("UFACET section construction.")
        # Solar field.
        # Solar field style setup.
        # Assumes that solar field is already set up with heliostat configurations, etc.
        if render_view_spec["type"] == "xy":
            solar_field_style = rcsf.heliostat_outlines(color="lightgrey")
        elif render_view_spec["type"] == "vplane":
            solar_field_style = rcsf.heliostat_blanks(color="lightgrey")
        else:
            solar_field_style = rcsf.heliostat_outlines(color="lightgrey")
        # Highlight the heliostats.
        if render_control_scan_section_setup.highlight_candidate_heliostats:
            solar_field_style.heliostat_styles.add_special_names(
                section["candidate_heliostat_name_list"], rch.normal_outline(color="c")
            )
        else:
            if render_control_scan_section_setup.highlight_selected_heliostats:
                solar_field_style.heliostat_styles.add_special_names(
                    section["selected_heliostat_name_list"], rch.normal_outline(color="g")
                )
            if render_control_scan_section_setup.highlight_rejected_heliostats:
                solar_field_style.heliostat_styles.add_special_names(
                    section["rejected_heliostat_name_list"], rch.normal_outline(color="r")
                )
        # Draw the solar field.
        solar_field.draw(view, solar_field_style)
        # Section.
        section_view_spec = section["view_spec"]
        # Fetch view spec projection information.
        segment_xy = section_view_spec["defining_segment_xy"]
        line_xy = section_view_spec["line_intersecting_xy_plane"]
        origin_xyz = np.array(section_view_spec["origin_xyz"])  # Make arrays so we can do simple vactor math.
        p_uxyz = np.array(section_view_spec["p_uxyz"])  #
        q_uxyz = np.array(section_view_spec["q_uxyz"])  #
        w_uxyz = np.array(section_view_spec["w_uxyz"])  #
        # Defining segment.
        segment_xyz = [p + [0] for p in segment_xy]
        view.draw_xyz_list(segment_xyz, style=rcps.outline(color="brown", linewidth=2.5))
        # Section plane.
        box_xyz = solar_field.heliostat_bounding_box_xyz()
        box_min_xyz = box_xyz[0]
        box_max_xyz = box_xyz[1]
        x_margin = 30.0  # meters
        y_margin = x_margin
        z_margin = 60.0  # meters
        x_min = min(box_min_xyz[0], origin_xyz[0]) - x_margin
        x_max = max(box_max_xyz[0], origin_xyz[0]) + x_margin
        y_min = min(box_min_xyz[1], origin_xyz[1]) - y_margin
        y_max = max(box_max_xyz[1], origin_xyz[1]) + y_margin
        z_min = min(box_min_xyz[2], origin_xyz[2], 0.0)
        z_max = max(box_max_xyz[2], origin_xyz[2]) + z_margin
        clip_xy_box = [[x_min, y_min], [x_max, y_max]]
        line_segment_xy = g2d.clip_line_to_xy_box(line_xy, clip_xy_box)
        line_segment_xy0 = line_segment_xy[0]
        line_segment_xy1 = line_segment_xy[1]
        view.draw_xyz_list(
            [line_segment_xy0 + [0], line_segment_xy1 + [0]], style=rcps.outline(color="c", linewidth=0.5)
        )
        view.draw_xyz_list(
            [line_segment_xy0 + [z_min], line_segment_xy1 + [z_min]], style=rcps.outline(color="c", linewidth=0.5)
        )
        view.draw_xyz_list(
            [line_segment_xy0 + [z_max], line_segment_xy1 + [z_max]], style=rcps.outline(color="c", linewidth=0.5)
        )
        view.draw_xyz_list(
            [line_segment_xy0 + [z_min], line_segment_xy0 + [z_max]], style=rcps.outline(color="c", linewidth=0.5)
        )
        view.draw_xyz_list(
            [line_segment_xy1 + [z_min], line_segment_xy1 + [z_max]], style=rcps.outline(color="c", linewidth=0.5)
        )
        # Origin
        view.draw_xyz(origin_xyz, style=rcps.marker(marker="o", color="r"))
        # Coordinate system.
        # Consruct rays for the coordinate system axes.
        length = 20
        p_ray = [origin_xyz, origin_xyz + (length * p_uxyz)]
        q_ray = [origin_xyz, origin_xyz + (length * q_uxyz)]
        w_ray = [origin_xyz, origin_xyz + (length * w_uxyz)]
        # Plot the coordinate system rays.
        view.draw_xyz_list(p_ray, style=rcps.outline(color="r", linewidth=2))
        view.draw_xyz_list(q_ray, style=rcps.outline(color="g", linewidth=2))
        view.draw_xyz_list(w_ray, style=rcps.outline(color="b", linewidth=2))
        # # Example projected points.
        # example_xyz_list = [ [130,50,20], [100,90,50], [120,150,40] ]
        # print('In draw_construct_ufacet_section(), example_xyz_list = ', example_xyz_list)
        # view.draw_xyz_list(example_xyz_list, style=rcps.RenderControlPointSeq(color='b', markersize=3))
        # example_pqw_list = [vs.xyz2pqw(xyz, section_view_spec) for xyz in example_xyz_list]
        # print('In draw_construct_ufacet_section(), example_pqw_list = ', example_pqw_list)
        # example_pq_list = [[pqw[0], pqw[1]] for pqw in example_pqw_list]
        # print('In draw_construct_ufacet_section(), example_pq_list = ', example_pq_list)
        # projected_example_xyz_list = [vs.pq2xyz(pq, section_view_spec) for pq in example_pq_list]
        # print('In draw_construct_ufacet_section(), projected_example_xyz_list = ', projected_example_xyz_list)
        # view.draw_xyz_list(projected_example_xyz_list, style=rcps.RenderControlPointSeq(color='g', markersize=3))
        # for example_xyz, projected_example_xyz in zip(example_xyz_list, projected_example_xyz_list):
        #     view.draw_xyz_list([example_xyz, projected_example_xyz], style=rcps.outline(color='orange'))
        # Finish.
        view.show()
        # Return.
        return view


def draw_construct_ufacet_sections(
    figure_control, solar_field, section_list, input_view_spec, render_control_scan_section_setup
):
    for section in section_list:
        if input_view_spec == None:
            render_view_spec = section["view_spec"]
        else:
            render_view_spec = input_view_spec
        draw_construct_ufacet_section(
            figure_control, solar_field, section, render_view_spec, render_control_scan_section_setup
        )
