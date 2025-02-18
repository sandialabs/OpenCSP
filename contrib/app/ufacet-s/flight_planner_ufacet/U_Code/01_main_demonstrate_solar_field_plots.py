"""
Demonstrate Solar Field Plotting Routines



"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import lib.RenderControlFlightOverSolarField as rcfosf
import lib.RenderControlFlightPlan as rcfp
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import lib.RenderControlScanSectionAnalysis as rcssa
import lib.RenderControlScanSectionSetup as rcsss
import lib.RenderControlScanXyAnalysis as rcsxa
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import lib.RenderControlTopLevel as rctl
import opencsp.common.lib.uas.Scan as Scan
import opencsp.common.lib.uas.ScanPass as sp
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.render.view_spec as vs


def draw_demonstration_figures(
    figure_control: rcfg.RenderControlFigure,
    axis_control_m: rca.RenderControlAxis,
    view_spec: dict,
    solar_field: sf.SolarField,
    aimpoint_xyz,
    when_ymdhmsz,
    synch_az,
    synch_el,
    up_az,
    up_el,
):
    # Figure selection. ?? Should this be added to the method header?
    draw_single_heliostat = True
    draw_annotated_heliostat = True
    draw_multi_heliostat = True
    draw_solar_field_h_names = True
    draw_solar_field_h_centroids = True
    draw_solar_field_h_centroids_names = True
    draw_solar_field_h_outlines = True
    draw_annotated_solar_field = True
    draw_solar_field_subset = True
    draw_heliostat_vector_field = True
    draw_dense_vector_field = True

    # One heliostat.
    if draw_single_heliostat:
        # Heliostat selection
        heliostat_name = "5E10"
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title=("Heliostat " + heliostat_name)
        )
        view = fig_record.view
        # Configuration setup
        heliostat = solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_configuration(hc.face_south())
        # Style setup
        heliostat_styles = rce.RenderControlEnsemble(rch.normal_facet_outlines_names(color="b"))
        # Comment
        fig_record.comment.append("Demonstration of heliostat drawing.")
        fig_record.comment.append("Facet outlines shown, with facet names and overall heliostat surface normal.")
        # Draw
        heliostat.draw(view, heliostat_styles)
        view.show()

    # Annotated heliostat.
    if draw_annotated_heliostat:
        # Heliostat selection
        heliostat_name = "5E10"
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title=("Heliostat " + heliostat_name + ", with Highlighting")
        )
        view = fig_record.view
        # Tracking setup
        heliostat = solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_tracking(aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)
        # Style setup
        default_heliostat_style = rch.normal_facet_outlines()
        default_heliostat_style.facet_styles.add_special_name(16, rcf.corner_normals_outline_name(color="c"))
        default_heliostat_style.facet_styles.add_special_names([1, 4, 7, 24, 25], rcf.normal_outline(color="r"))
        heliostat_styles = rce.RenderControlEnsemble(default_heliostat_style)
        # Comment
        fig_record.comment.append("Demonstration of example heliostat annotations.")
        fig_record.comment.append("Black:   Facet outlines.")
        fig_record.comment.append("Black:   Overall heliostat surface normal.")
        fig_record.comment.append("Red:     Highlighted facets and their surface normals.")
        fig_record.comment.append(
            "Cyan:    Highlighted facet with facet name and facet surface normal drawn at corners."
        )
        # Draw
        heliostat.draw(view, heliostat_styles)
        view.show()

    # Multiple heliostats.
    if draw_multi_heliostat:
        # Heliostat selection
        heliostat_spec_list = [
            ["11W1", hc.face_up(), rch.name()],
            ["11E1", hc.face_up(), rch.centroid(color="r")],
            ["11E2", hc.face_up(), rch.centroid_name(color="g")],
            ["12W1", hc.face_north(), rch.facet_outlines(color="b")],
            ["12E1", hc.face_south(), rch.normal_outline(color="c")],
            ["12E2", hc.face_east(), rch.corner_normals_outline(color="m")],
            ["13W1", hc.face_west(), rch.normal_facet_outlines(color="g")],
            ["13E1", hc.face_up(), rch.facet_outlines_normals(color="c")],
            ["13E2", hc.NSTTF_stow(), rch.facet_outlines_corner_normals()],
        ]
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title="Example Poses and Styles"
        )
        view = fig_record.view
        # Setup and draw
        for heliostat_spec in heliostat_spec_list:
            heliostat_name = heliostat_spec[0]
            heliostat_config = heliostat_spec[1]
            heliostat_style = heliostat_spec[2]
            # Configuration setup
            heliostat = solar_field.lookup_heliostat(heliostat_name)
            heliostat.set_configuration(heliostat_config)
            # Style setup
            heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
            # Draw
            heliostat.draw(view, heliostat_styles)
        # Comment
        fig_record.comment.append("Demonstration of various example heliostat drawing modes.")
        fig_record.comment.append("Black:   Name only.")
        fig_record.comment.append("Red:     Centroid only.")
        fig_record.comment.append("Green:   Centroid and name.")
        fig_record.comment.append("Blue:    Facet outlines.")
        fig_record.comment.append("Cyan:    Overall outline and overall surface normal.")
        fig_record.comment.append("Magneta: Overall outline and overall surface normal, drawn at corners.")
        fig_record.comment.append("Green:   Facet outlines and overall surface normal.")
        fig_record.comment.append("Cyan:    Facet outlines and facet surface normals.")
        fig_record.comment.append("Black:   Facet outlines and facet surface normals drawn at facet corners.")
        view.show()

    # Solar field heliostat names.
    if draw_solar_field_h_names:
        # View setup
        fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title="Heliostat Names")
        view = fig_record.view
        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field_style = rcsf.heliostat_names(color="m")
        # Comment
        fig_record.comment.append("Heliostat names, drawn at each heliostat's centroid.")
        fig_record.comment.append("At NSTTF, centroids appear to be at the midpoint of the torque tube.")
        # Draw
        solar_field.draw(view, solar_field_style)
        view.show()

    # Solar field heliostat centroids.
    if draw_solar_field_h_centroids:
        # View setup
        fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title="Heliostat Centroids")
        view = fig_record.view
        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field_style = rcsf.heliostat_centroids(color="b")
        # Comment
        fig_record.comment.append(
            "Heliostat centroids, which at NSTTF appear to be at the midpoint of the torque tube."
        )
        # Draw
        solar_field.draw(view, solar_field_style)
        view.show()

    # Solar field heliostat centroids and names.
    if draw_solar_field_h_centroids_names:
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title="Heliostat Labelled Centroids"
        )
        view = fig_record.view
        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field_style = rcsf.heliostat_centroids_names(color="g")
        # Comment
        fig_record.comment.append("Heliostat names and labels drawn together.")
        # Draw
        solar_field.draw(view, solar_field_style)
        view.show()

    # Solar field heliostat outlines.
    # (Plus aim point and legend.)
    if draw_solar_field_h_outlines:
        # View setup
        fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title="Heliostat Outlines")
        view = fig_record.view
        # Tracking setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color="b")
        # Comment
        fig_record.comment.append("A simple way of rendering a solar field.")
        # Draw
        solar_field.draw(view, solar_field_style)
        view.draw_xyz(aimpoint_xyz, style=rcps.marker(color="tab:orange"), label="aimpoint_xyz")
        view.show()

    # Annotated solar field.
    if draw_annotated_solar_field:
        # Heliostat selection
        up_heliostats = ["6E3", "8E3"]
        stowed_heliostats = ["9W1", "12E14"]
        synched_heliostats = [
            "5E1",
            "5E2",
            "5E3",
            "5E4",
            "5E5",
            "5E6",
            "5E7",
            "6E1",
            "6E2",
            "6E4",
            "6E5",
            "6E6",
            "6E7",
            "7E1",
            "7E2",
            "7E3",
            "7E4",
            "7E5",
            "7E6",
            "7E7",
        ]
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title="Solar Field Situation"
        )
        view = fig_record.view
        # Configuration setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
        synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        solar_field.set_heliostats_configuration(up_heliostats, up_configuration)
        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color="b")
        solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.normal_outline(color="c"))
        solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color="r"))
        solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color="g"))
        # Comment
        fig_record.comment.append("A solar field situation with heliostats in varying status.")
        fig_record.comment.append("Blue heliostats are tracking.")
        fig_record.comment.append("Cyan heliostats are face up.")
        fig_record.comment.append("Red heliostats are in stow (out of service).")
        fig_record.comment.append(
            "Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(
                np.rad2deg(synch_az), np.rad2deg(synch_el)
            )
        )
        # Draw
        solar_field.draw(view, solar_field_style)
        view.show()

    # Solar field subset.
    if draw_solar_field_subset:
        # Heliostat selection
        up_heliostats = ["6E3", "8E3"]
        stowed_heliostats = ["6E2", "8E5"]
        synched_heliostats = [
            "5E1",
            "5E2",
            "5E3",
            "5E4",
            "5E5",
            "5E6",
            "5E7",
            "6E1",
            "6E4",
            "6E5",
            "6E6",
            "6E7",
            "7E1",
            "7E2",
            "7E3",
            "7E4",
            "7E5",
            "7E6",
            "7E7",
        ]
        tracking_heliostats = ["8E1", "8E2", "8E4", "8E6", "8E7", "9E1", "9E2", "9E3", "9E4", "9E5", "9E6", "9E7"]
        # View setup
        fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title="Selected Heliostats")
        view = fig_record.view
        # Configuration setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
        synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        solar_field.set_heliostats_configuration(up_heliostats, up_configuration)
        # Style setup
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.normal_outline(color="c"))
        solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color="r"))
        solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color="g"))
        solar_field_style.heliostat_styles.add_special_names(tracking_heliostats, rch.normal_outline(color="b"))
        # Comment
        fig_record.comment.append("A subset of heliostats selected, so that plot is effectively zoomed in.")
        fig_record.comment.append("Blue heliostats are tracking.")
        fig_record.comment.append("Cyan heliostats are face up.")
        fig_record.comment.append("Red heliostats are in stow (out of service).")
        fig_record.comment.append(
            "Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(
                np.rad2deg(synch_az), np.rad2deg(synch_el)
            )
        )
        # Draw
        solar_field.draw(view, solar_field_style)
        view.show()

    # Heliostat vector field.
    if draw_heliostat_vector_field:
        # View setup
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, view_spec, title="Heliostat Vector Field"
        )
        view = fig_record.view
        # Tracking setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        # Style setup
        solar_field_style = rcsf.heliostat_vector_field(color="b")
        # Comment
        fig_record.comment.append("Each heliostat's surface normal, which can be viewed as a vector field.")
        # Draw
        solar_field.draw(view, solar_field_style)
        view.draw_xyz(aimpoint_xyz, style=rcps.marker(color="tab:orange"), label="aimpoint_xyz")
        view.show()

    # Dense vector field.
    if draw_dense_vector_field:
        # View setup (xy only)
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control_m, vs.view_spec_xy(), title="Dense Tracking Vector Field"
        )
        view_xy = fig_record.view
        # Tracking setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        # Style setup
        solar_field_style = rcsf.heliostat_vector_field_outlines(color="grey")
        # Comment
        fig_record.comment.append("Dense vector field of tracking surface normals.")
        # Draw solar field and aim point.
        solar_field.draw(view_xy, solar_field_style)
        view_xy.draw_xyz(aimpoint_xyz, style=rcps.marker(color="tab:orange"), label="aimpoint_xyz")
        # Draw dense vector field.
        grid_xy = solar_field.heliostat_field_regular_grid_xy(40, 20)
        grid_xydxy = [
            [p, sun_track.tracking_surface_normal_xy(p + [0], aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)]
            for p in grid_xy
        ]
        view_xy.draw_pqdpq_list(grid_xydxy, style=rcps.vector_field(color="b", vector_scale=5.0))
        # Finish.
        view.show()


if __name__ == "__main__":
    plt.close("all")

    fm.reset_figure_management()

    figure_control = rcfg.RenderControlFigure(tile_array=(2, 1), tile_square=True)
    axis_control_m = rca.meters()
    # Control flags
    # (Also see draw_demonstration_figures() above.)
    save_figures = True

    # Load solar field data.
    solar_field = None  # TODO
    # sf.heliostats_read_file(...)
    # solar_field = sf.SolarField(...)
    # solar_field = sf.SolarField(name='Sandia NSTTF',
    #                             short_name='NSTTF',
    #                             origin_lon_lat=lln.NSTTF_ORIGIN,
    #                             heliostat_file='../U_Code_data/NSTTF/NSTTF_Heliostats_origin_at_torque_tube.csv',
    #                             facet_centroids_file='../U_Code_data/NSTTF/NSTTF_Facet_Centroids.csv')

    # Define tracking time.
    aimpoint_xyz = [60.0, 8.8, 28.9]
    #               year, month, day, hour, minute, second, zone]
    when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # Aproximately NSTTF solar noon

    # Define fixed heliostat orientation.
    synch_az = np.deg2rad(205)
    synch_el = np.deg2rad(30)

    # Define upward-facing heliostat orientation.
    up_az = np.deg2rad(180)
    up_el = np.deg2rad(90)

    # Draw figure suite.
    draw_demonstration_figures(
        figure_control,
        axis_control_m,
        vs.view_spec_3d(),
        solar_field,
        aimpoint_xyz,
        when_ymdhmsz,
        synch_az=synch_az,
        synch_el=synch_el,
        up_az=up_az,
        up_el=up_el,
    )
    draw_demonstration_figures(
        figure_control,
        axis_control_m,
        vs.view_spec_xy(),
        solar_field,
        aimpoint_xyz,
        when_ymdhmsz,
        synch_az=synch_az,
        synch_el=synch_el,
        up_az=up_az,
        up_el=up_el,
    )
    # draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_xz(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)
    # draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_yz(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)

    # Summarize.
    print("\n\nFigure Summary:")
    fm.print_figure_summary()

    # Save figures.
    if save_figures:
        print("\n\nSaving figures...")
        # Output directory.
        output_path = os.path.join("..", ("output_" + datetime.now().strftime("%Y_%m_%d_%H%M")))
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        fm.save_all_figures(output_path)
