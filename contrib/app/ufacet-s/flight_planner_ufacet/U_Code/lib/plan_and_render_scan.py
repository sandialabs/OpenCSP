"""
Construct and render a scan, common across scan types.



"""

import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.render.figure_management as fm
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.FlightOverSolarField as fosf
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.FlightPlan as fp
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_raster as psr
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet as psu
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_vanity as psv
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_render as psur
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_section_construction_render as psuscr
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_xy_analysis_render as psuxyar
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlFlightOverSolarField as rcfosf
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.render.view_spec as vs


def plan_and_render_scan(
    solar_field,
    aimpoint_xyz,
    when_ymdhmsz,
    launch_name,
    elevation_offset,
    scan_type,
    raster_scan_parameter_file,  # Only used for raster scan types.
    ufacet_scan_parameter_file,  # Only used for UFACET scan types.
    ufacet_control_parameters,  #
    vanity_scan_parameter_file,  # Only used for vanity scan types.
    vanity_heliostat_name,  # Only used for vanity scan types.
    vanity_heliostat_azimuth,  # Only used for vanity scan types.
    figure_control,
    render_control_top_level,
    render_control_scan_xy_analysis,
    render_control_scan_section_setup,
    render_control_scan_section_analysis,
):
    # Construct scan name and output directory.
    scan_name = scan_type + " Scan Over " + solar_field.situation_str()
    if scan_type == "UFACET":
        scan_short_name = solar_field.situation_abbrev() + "_" + scan_type[0]
    elif scan_type == "Raster":
        scan_short_name = solar_field.short_name + "_" + scan_type[0]
    elif scan_type == "Vanity":
        scan_short_name = solar_field.short_name + "_" + scan_type[0]
    else:
        print('ERROR: In plan_and_render_scan(), unexpected scan_type="' + str(scan_type) + '" encountered.')
        assert False
    # We're not using this control flag, so eliminate it to streamline file names.
    # if scan_type == 'UFACET':
    #     scan_name += ', Zmax={0:.0f}m'.format(ufacet_control_parameters['maximum_altitude'])
    #     scan_short_name += '_Zmax={0:.0f}'.format(ufacet_control_parameters['maximum_altitude'])
    #     if ufacet_control_parameters['delta_eta'] != 0:
    #         scan_name += ', Deta={0:.1f}deg'.format(np.rad2deg(ufacet_control_parameters['delta_eta']))
    #         scan_short_name += '_Deta={0:.1f}'.format(np.rad2deg(ufacet_control_parameters['delta_eta']))
    output_path = os.path.join(render_control_top_level.figure_output_path, scan_short_name)

    # Construct the scan.
    print("Constructing scan...")
    if scan_type == "UFACET":
        # Construct UFACET scan.
        scan, scan_parameters, ufacet_scan_construction = psu.construct_ufacet_scan(
            solar_field, aimpoint_xyz, when_ymdhmsz, ufacet_scan_parameter_file, ufacet_control_parameters
        )
    elif scan_type == "Raster":
        # Construct raster survey scan.
        scan, scan_parameters = psr.construct_raster_scan(solar_field, raster_scan_parameter_file)

    elif scan_type == "Vanity":
        # Construct raster survey scan.
        scan, scan_parameters = psv.construct_vanity_scan(
            solar_field, vanity_scan_parameter_file, vanity_heliostat_name, vanity_heliostat_azimuth
        )
    else:
        print("ERROR: In plan_and_render_scan, unexpected scan.type = " + str(scan.type) + " encountered.")
        assert False

    # Construct the flight plan.
    flight_plan = fp.construct_flight_plan_from_scan(scan_name, scan_short_name, launch_name, scan)

    # Construct object representing the flight over the solar field.
    flight_over_solar_field = fosf.FlightOverSolarField(solar_field, flight_plan)

    # Write the flight plan file.
    if render_control_top_level.save_flight_plan:
        print("Writing flight plan file...")
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        flight_plan.save_to_litchi_csv(output_path, elevation_offset)

    # Draw UFACET construction steps.
    if scan_type == "UFACET":
        if render_control_top_level.draw_ufacet_xy_analysis:
            # Draw UFACET (x,y) analysis.
            print("Drawing UFACET (x,y) analysis...")
            curve_key_xy_list = ufacet_scan_construction["curve_key_xy_list"]
            list_of_ideal_xy_lists = ufacet_scan_construction["list_of_ideal_xy_lists"]
            list_of_best_fit_segment_xys = ufacet_scan_construction["list_of_best_fit_segment_xys"]
            psuxyar.draw_ufacet_xy_analysis(
                figure_control,
                solar_field,
                aimpoint_xyz,
                solar_field.origin_lon_lat,
                when_ymdhmsz,
                curve_key_xy_list,
                list_of_ideal_xy_lists,
                list_of_best_fit_segment_xys,
                render_control_scan_xy_analysis,
            )

        if render_control_top_level.draw_ufacet_section_construction:
            # Draw UFACET section construction.
            print("Drawing UFACET section construction...")
            section_list = ufacet_scan_construction["section_list"]
            #            psuscr.draw_construct_ufacet_sections(figure_control, solar_field, section_list, vs.view_spec_3d(), render_control_scan_section_setup)
            psuscr.draw_construct_ufacet_sections(
                figure_control, solar_field, section_list, vs.view_spec_xy(), render_control_scan_section_setup
            )
        #            psuscr.draw_construct_ufacet_sections(figure_control, solar_field, section_list, None, render_control_scan_section_setup)  # Use section view.

        if render_control_top_level.draw_ufacet_scan:
            # Draw the scan.
            print("Drawing UFACET scan...")
            psur.draw_ufacet_scan(figure_control, scan, render_control_scan_section_analysis)

    # Draw the flight plan.
    if render_control_top_level.draw_flight_plan:
        print("Drawing flight plan over solar field...")
        # Style setup.
        xy_solar_field_style = rcsf.heliostat_vector_field_outlines(color="grey")
        if render_control_top_level.xy_solar_field_style != None:
            xy_solar_field_style = render_control_top_level.xy_solar_field_style
        rcfosf_default = rcfosf.default()
        rcfosf_vfield = rcfosf.RenderControlFlightOverSolarField(solar_field_style=xy_solar_field_style)
        # Draw.
        view_3d = fosf.draw_flight_over_solar_field(
            figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_3d()
        )
        view_xy = fosf.draw_flight_over_solar_field(
            figure_control, flight_over_solar_field, rcfosf_vfield, vs.view_spec_xy()
        )
        view_xz = fosf.draw_flight_over_solar_field(
            figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_xz()
        )
        view_yz = fosf.draw_flight_over_solar_field(
            figure_control, flight_over_solar_field, rcfosf_default, vs.view_spec_yz()
        )

    # Summarize.
    if render_control_top_level.summarize_figures:
        print("\nSummarizing figures...")
        fm.print_figure_summary()

    # Save figures.
    if render_control_top_level.save_figures:
        print("\nSaving figures...")
        figure_output_path = os.path.join(output_path, "plots")
        if not (os.path.exists(figure_output_path)):
            os.makedirs(figure_output_path)
        fm.save_all_figures(figure_output_path)


# -------------------------------------------------------------------------------------------------------
#
# EXECUTING A SCAN PLANNING TRIAL
#


def scan_plan_trial(
    tile_array,
    solar_field_spec,
    aimpoint_xyz,
    when_ymdhmsz,
    synch_azelhnames,
    up_azelhnames,
    launch_name,
    elevation_offset,
    scan_type,
    raster_scan_parameter_file,
    ufacet_scan_parameter_file,
    ufacet_control_parameters,
    vanity_scan_parameter_file,
    vanity_heliostat_name,
    vanity_heliostat_azimuth,
    render_control_top_level,
    render_control_scan_xy_analysis,
    render_control_scan_section_setup,
    render_control_scan_section_analysis,
):
    # Notify progress.
    print("\n\n\nStarting trial:")
    print("    solar field:        ", solar_field_spec["name"])
    print("    solar field abbrev: ", solar_field_spec["short_name"])
    print("    aim_point:          ", aimpoint_xyz)
    print("    when_ymdhmsz:       ", when_ymdhmsz)
    print("    launch_name:        ", launch_name)
    print("    synch_azelhnames:   ", synch_azelhnames)
    print("    up_azelhnames:      ", up_azelhnames)
    print("    elevation_offset:   ", elevation_offset)
    print("    scan_type:          ", scan_type)
    print()

    # Figure control.
    plt.close("all")
    fm.reset_figure_management()
    figure_control = rcfg.RenderControlFigure(tile_array=tile_array, tile_square=False)

    # Initialize solar field.
    solar_field = sf.setup_solar_field(solar_field_spec, aimpoint_xyz, when_ymdhmsz, synch_azelhnames, up_azelhnames)
    if scan_type == "Vanity":
        solar_field.set_full_field_stow()  # ?? SCAFFOLDING RCB -- MAKE GENERAL

    # Construct and draw scan.
    plan_and_render_scan(
        solar_field,
        aimpoint_xyz,
        when_ymdhmsz,
        launch_name,
        elevation_offset,
        scan_type,
        raster_scan_parameter_file,
        ufacet_scan_parameter_file,
        ufacet_control_parameters,
        vanity_scan_parameter_file,
        vanity_heliostat_name,
        vanity_heliostat_azimuth,
        figure_control,
        render_control_top_level,
        render_control_scan_xy_analysis,
        render_control_scan_section_setup,
        render_control_scan_section_analysis,
    )
