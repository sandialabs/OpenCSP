"""
Construct a UFACET-s scan.



"""

import numpy as np

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_parameters as psup
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_section_construction as psusc
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_xy_analysis as psuxya
import opencsp.common.lib.uas.Scan as Scan
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.UfacetScanPass as usp


def construct_ufacet_scan(
    solar_field, aimpoint_xyz, when_ymdhmsz, ufacet_scan_parameter_file, ufacet_control_parameters
):
    # Notify progress.
    print("Constructing UFACET scan...")

    # Fetch scan parameters.
    # Add-in control parameters, so there is only one container.
    ufacet_scan_parameters = psup.construct_ufacet_scan_parameters(
        ufacet_scan_parameter_file, ufacet_control_parameters
    )

    # Fetch seed curve points.
    curve_key_xy_list = ufacet_control_parameters["curve_key_xy_list"]

    # UFACET (x,y) analysis.
    list_of_ideal_xy_lists, list_of_best_fit_segment_xys = psuxya.ufacet_xy_analysis(
        solar_field, aimpoint_xyz, when_ymdhmsz, curve_key_xy_list
    )

    # UFACET section analysis.
    section_list = psusc.construct_ufacet_sections(solar_field, list_of_best_fit_segment_xys, ufacet_scan_parameters)

    # Construct individual UFACET scan passes.
    scan_pass_list = usp.construct_ufacet_passes(solar_field, section_list, ufacet_scan_parameters)

    # Construct the scan.
    scan = Scan.construct_scan_given_UFACET_scan_passes(scan_pass_list, ufacet_scan_parameters)

    # Store results.
    ufacet_scan_construction = {}
    ufacet_scan_construction["curve_key_xy_list"] = ufacet_control_parameters["curve_key_xy_list"]
    ufacet_scan_construction["list_of_ideal_xy_lists"] = list_of_ideal_xy_lists
    ufacet_scan_construction["list_of_best_fit_segment_xys"] = list_of_best_fit_segment_xys
    ufacet_scan_construction["section_list"] = section_list
    ufacet_scan_construction["scan_pass_list"] = scan_pass_list

    # Return.
    # Return the scan parameters, because they include information for converting the scan into a flight.
    return scan, ufacet_scan_parameters, ufacet_scan_construction
