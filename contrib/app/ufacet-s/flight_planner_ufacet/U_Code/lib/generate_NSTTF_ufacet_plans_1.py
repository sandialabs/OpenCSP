"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.define_render_control as drc
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.define_scan_nsttf as dsn
import opencsp.common.lib.render.figure_management as fm
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_and_render_scan as pars
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlTopLevel as rctl


# -------------------------------------------------------------------------------------------------------
#  GENERATE NSTTF PLANS
#


def generate_NSTTF_ufacet_plans(
    solar_field_short_name, nsttf_configuration, elevation_offset, save_flight_plan, save_figures
):
    # Figure layout.
    #    tile_array=(2,2)
    tile_array = (1, 1)
    #    tile_array=(2,1)

    render_control_top_level = rctl.RenderControlTopLevel()
    # render_control_top_level.draw_ufacet_xy_analysis = False
    # render_control_top_level.draw_ufacet_section_construction = False
    # render_control_top_level.draw_ufacet_scan = False
    # render_control_top_level.draw_flight_plan = False
    render_control_top_level.save_flight_plan = save_flight_plan
    # render_control_top_level.summarize_figures = True
    render_control_top_level.save_figures = save_figures
    render_control_scan_xy_analysis = drc.setup_render_control_scan_xy_analysis()
    render_control_scan_section_setup = drc.setup_render_control_scan_section_setup()
    #    render_control_scan_section_analysis = setup_render_control_scan_section_analysis()
    render_control_scan_section_analysis = drc.setup_render_control_scan_section_analysis_section_plus_flight_4view()
    #    render_control_scan_section_analysis = drc.setup_render_control_scan_section_analysis_flight_4view()

    # Figure control.
    plt.close("all")
    fm.reset_figure_management()
    figure_control = rcfg.RenderControlFigure(tile_array=tile_array, tile_square=False)

    # Scan control parameters.
    scan_type = "UFACET"
    ufacet_scan_parameter_file = solar_field_short_name  # ?? SCAFFOLDING RCB -- TEMPORARY
    raster_scan_parameter_file = None

    # Define scan.
    if solar_field_short_name == "NSTTF":
        (
            solar_field_spec,
            ufacet_control_parameters,
            aimpoint_xyz,
            defined_when_ymdhmsz,
            synch_azelhnames,
            up_azelhnames,
        ) = dsn.define_scan_NSTTF(solar_field_short_name, nsttf_configuration)
        # Favored launch point.
        launch_name = "A"  # Any string.  Examples are 'A', 'B', 'C', etc.
    else:
        # print("ERROR: In generate_NSTTF_ufacet_plans(1), unexpected solar_field_short_name = '"  + solar_field_short_name + "' encountered.")
        assert False

    # Highlight unusual configurations.
    if nsttf_configuration == "Full Field":
        pass
    elif nsttf_configuration == "Demo":
        solar_field_spec["short_name"] += "demo"
        solar_field_spec["name"] += " (Demo Configuration)"
    elif nsttf_configuration == "Half-and-Half":
        solar_field_spec["short_name"] += "hh"
        solar_field_spec["name"] += " (Half-and-Half)"
    else:
        print(
            'ERROR: In generate_NSTTF_ufacet_plans(), unexpected nsttf_configuration="'
            + str(nsttf_configuration)
            + '" encountered.'
        )
        assert False

    # Time.
    # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    #
    # Zone: Mountain Daylight Time (MDT) = -6,  Mountain Standard Time (MST) = -7
    #
    if defined_when_ymdhmsz != None:
        # Don't iterate over case list
        when_ymdhmsz = defined_when_ymdhmsz
        # Single trial.
        pars.scan_plan_trial(
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
            render_control_top_level,
            render_control_scan_xy_analysis,
            render_control_scan_section_setup,
            render_control_scan_section_analysis,
        )
    else:
        # Iterate over case list.
        #               year, month, day, hour, minute, second, zone]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon

        # Multiple trials (NSTTF).
        trial_spec_z_aim_idx = 0
        trial_spec_z_max_idx = 1
        trial_spec_hour_idx = 2
        trial_spec_minute_idx = 3
        trial_spec_list = [
            [28.9, 19.0, 10, 0],
            [28.9, 19.0, 10, 30],
            [28.9, 19.0, 11, 0],
            [28.9, 19.0, 11, 30],
            [28.9, 19.0, 12, 0],
            [28.9, 19.0, 12, 30],
            [28.9, 19.0, 13, 0],
            [28.9, 19.0, 13, 30],
            [28.9, 19.0, 14, 0],
            [28.9, 19.0, 14, 30],
            [28.9, 19.0, 15, 0],
            [28.9, 19.0, 15, 30],
            [28.9, 19.0, 16, 0],
            [28.9, 19.0, 16, 30],
            [28.9, 19.0, 17, 0],
            [28.9, 19.0, 17, 30],
            [45.0, 25.0, 10, 0],
            [45.0, 25.0, 10, 30],
            [45.0, 25.0, 11, 0],
            [45.0, 25.0, 11, 30],
            [45.0, 25.0, 12, 0],
            [45.0, 25.0, 12, 30],
            [45.0, 25.0, 13, 0],
            [45.0, 25.0, 13, 30],
            [45.0, 25.0, 14, 0],
            [45.0, 25.0, 14, 30],
            [45.0, 25.0, 15, 0],
            [45.0, 25.0, 15, 30],
            [45.0, 25.0, 16, 0],
            [45.0, 25.0, 16, 30],
            [45.0, 25.0, 17, 0],
            [45.0, 25.0, 17, 30],
            [60.0, 25.0, 10, 0],
            [60.0, 25.0, 10, 30],
            [60.0, 25.0, 11, 0],
            [60.0, 25.0, 11, 30],
            [60.0, 25.0, 12, 0],
            [60.0, 25.0, 12, 30],
            [60.0, 25.0, 13, 0],
            [60.0, 25.0, 13, 30],
            [60.0, 25.0, 14, 0],
            [60.0, 25.0, 14, 30],
            [60.0, 25.0, 15, 0],
            [60.0, 25.0, 15, 30],
            [60.0, 25.0, 16, 0],
            [60.0, 25.0, 16, 30],
            [60.0, 25.0, 17, 0],
            [60.0, 25.0, 17, 30],
        ]

        for trial_spec in trial_spec_list:
            aimpoint_xyz_2 = aimpoint_xyz
            when_ymdhmsz_2 = when_ymdhmsz.copy()
            when_hour_idx = 3
            when_minute_idx = 4
            when_ymdhmsz_2[when_hour_idx] = trial_spec[trial_spec_hour_idx]
            when_ymdhmsz_2[when_minute_idx] = trial_spec[trial_spec_minute_idx]
            aimpoint_xyz_2[2] = trial_spec[trial_spec_z_aim_idx]
            ufacet_control_parameters["maximum_altitude"] = trial_spec[trial_spec_z_max_idx]
            pars.scan_plan_trial(
                tile_array,
                solar_field_spec,
                aimpoint_xyz_2,
                when_ymdhmsz_2,
                synch_azelhnames,
                up_azelhnames,
                launch_name,
                elevation_offset,
                scan_type,
                raster_scan_parameter_file,
                ufacet_scan_parameter_file,
                ufacet_control_parameters,
                render_control_top_level,
                render_control_scan_xy_analysis,
                render_control_scan_section_setup,
                render_control_scan_section_analysis,
            )


def generate_NSTTF_ufacet_plans_1(save_flight_plan, save_figures):
    elevation_offset = 0
    generate_NSTTF_ufacet_plans("NSTTF", "Full Field", elevation_offset, save_flight_plan, save_figures)
    # generate_NSTTF_ufacet_plans('NSTTF', 'Demo',          elevation_offset, save_flight_plan, save_figures)
    generate_NSTTF_ufacet_plans("NSTTF", "Half-and-Half", elevation_offset, save_flight_plan, save_figures)
