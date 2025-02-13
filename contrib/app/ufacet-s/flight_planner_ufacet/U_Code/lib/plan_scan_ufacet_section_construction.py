"""
Constructing UFACET scan sections, given best-fit segments.



"""

import opencsp.common.lib.render.view_spec as vs


def p(name_pqw):
    return name_pqw[1][0]


def abs_w(name_pqw):
    return abs(name_pqw[1][2])


def heliostat_name_pqw_list_within_margin(solar_field, section_view_spec, ufacet_scan_parameters):
    # Fetch required control parameters.
    margin_w = ufacet_scan_parameters["candidate_margin_w"]
    # Find heliostats within margin.
    heliostat_name_pqw_list = []
    for heliostat in solar_field.heliostats:
        origin_xyz = heliostat.origin
        origin_pqw = vs.xyz2pqw(origin_xyz, section_view_spec)
        if abs(origin_pqw[2]) <= margin_w:
            heliostat_name_pqw_list.append([heliostat.name, origin_pqw])
    # Return.
    return heliostat_name_pqw_list


def sort_heliostat_name_pqw_list_by_p(heliostat_name_pqw_list):
    heliostat_name_pqw_list.sort(key=p)
    return heliostat_name_pqw_list


def sort_heliostat_name_pqw_list_by_w(heliostat_name_pqw_list):
    heliostat_name_pqw_list.sort(key=abs_w)
    return heliostat_name_pqw_list


def select_min_w_reject_nearby_p_aux(
    selected_heliostat_name_pqw_list,
    rejected_heliostat_name_pqw_list,
    remaining_heliostat_name_pqw_list,
    discard_threshold_p,
):
    # This routine assumes tha thte input selected_heliostat_name_pqw_list has been sorted in order of increasing w.
    if len(remaining_heliostat_name_pqw_list) == 0:
        # There are no more heliostats to consider, so return.
        return (selected_heliostat_name_pqw_list, rejected_heliostat_name_pqw_list, remaining_heliostat_name_pqw_list)
    else:
        # Select the heliostat closest to the section plane.
        selected_heliostat_name_pqw = remaining_heliostat_name_pqw_list[0]
        selected_heliostat_name_pqw_list.append(selected_heliostat_name_pqw)
        remaining_heliostat_name_pqw_list = remaining_heliostat_name_pqw_list[1:]
        selected_p = p(selected_heliostat_name_pqw)
        new_remaining_heliostat_name_pqw_list = []
        for name_pqw in remaining_heliostat_name_pqw_list:
            this_p = p(name_pqw)
        return select_min_w_reject_nearby_p_aux(
            selected_heliostat_name_pqw_list,
            rejected_heliostat_name_pqw_list,
            new_remaining_heliostat_name_pqw_list,
            discard_threshold_p,
        )


def select_min_w_reject_nearby_p(candidate_heliostat_name_pqw_list, ufacet_scan_parameters):
    # Fetch required control parameters.
    discard_threshold_p = ufacet_scan_parameters["discard_threshold_p"]
    # Prepare recursion variables.
    selected_heliostat_name_pqw_list = []
    rejected_heliostat_name_pqw_list = []
    remaining_heliostat_name_pqw_list = candidate_heliostat_name_pqw_list
    # Recursive calculation to select the heliostats near the selection plane,
    # eliminate close-p clusters, and return both selected and rejected heliostats.
    (selected_heliostat_name_pqw_list, rejected_heliostat_name_pqw_list, remaining_heliostat_name_pqw_list) = (
        select_min_w_reject_nearby_p_aux(
            selected_heliostat_name_pqw_list,
            rejected_heliostat_name_pqw_list,
            remaining_heliostat_name_pqw_list,
            discard_threshold_p,
        )
    )
    # Return.
    # Don't return the recursion variables.
    return selected_heliostat_name_pqw_list, rejected_heliostat_name_pqw_list


def construct_ufacet_section(solar_field, best_fit_segment_xy, ufacet_scan_parameters):
    # Fetch section plane.
    section_view_spec = vs.view_spec_vplane(best_fit_segment_xy)
    # Identify candidate heliostats for section.
    candidate_heliostat_name_pqw_list = heliostat_name_pqw_list_within_margin(
        solar_field, section_view_spec, ufacet_scan_parameters
    )
    # Sort by w.
    sort_heliostat_name_pqw_list_by_w(candidate_heliostat_name_pqw_list)
    # Select heliostats close to section plane, and discard close neighbors.
    (selected_heliostat_name_pqw_list, rejected_heliostat_name_pqw_list) = select_min_w_reject_nearby_p(
        candidate_heliostat_name_pqw_list, ufacet_scan_parameters
    )
    # Sort in order of ascending p.
    sort_heliostat_name_pqw_list_by_p(selected_heliostat_name_pqw_list)
    # Extract heliostat names.
    candidate_heliostat_name_list = [name_pqw[0] for name_pqw in candidate_heliostat_name_pqw_list]
    selected_heliostat_name_list = [name_pqw[0] for name_pqw in selected_heliostat_name_pqw_list]
    rejected_heliostat_name_list = [name_pqw[0] for name_pqw in rejected_heliostat_name_pqw_list]
    # Store results.
    section = {}
    section["view_spec"] = section_view_spec
    section["candidate_heliostat_name_list"] = candidate_heliostat_name_list
    section["selected_heliostat_name_list"] = selected_heliostat_name_list
    section["rejected_heliostat_name_list"] = rejected_heliostat_name_list
    # Return.
    return section


def construct_ufacet_sections(solar_field, list_of_best_fit_segment_xys, ufacet_scan_parameters):
    # Notify progress.
    print("Constructing UFACET scan sections...")

    # Analyze each segment.
    section_list = []
    for best_fit_segment_xy in list_of_best_fit_segment_xys:
        section = construct_ufacet_section(solar_field, best_fit_segment_xy, ufacet_scan_parameters)
        section_list.append(section)
    return section_list
