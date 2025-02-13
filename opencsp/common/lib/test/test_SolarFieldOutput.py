"""Demonstrate Solar Field Plotting Routines"""

import numpy as np

import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz

UP = Vxyz([0, 0, 1])
NORTH = Vxyz([0, 1, 0])
SOUTH = -NORTH
EAST = Vxyz([1, 0, 0])
WEST = -EAST
STOW = Vxyz([-1, 0, -11.4]).normalize()


class TestSolarFieldOutput(to.TestOutput):

    @classmethod
    def setUpClass(
        self,
        source_file_body: str = "TestSolarFieldOutput",  # Set these here, because pytest calls
        figure_prefix_root: str = "tsfo",  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):

        # Save input.
        # Interactive mode flag.
        # This has two effects:
        #    1. If interactive mode, plots stay displayed.
        #    2. If interactive mode, after several plots have been displayed,
        #       plot contents might change due to Matplotlib memory issues.
        #       This may in turn cause misleading failures in comparing actual
        #       output to expected output.
        # Thus:
        #   - Run pytest in non-interactive mode.
        #   - If viewing run output interactively, be advised that figure content might
        #     change (e.g., garbled plot titles, etc), and actual-vs-expected comparison
        #     might erroneously fail (meaning they might report an error which is not
        #     actually a code error).
        #
        # Functions names `old_...` used to be tests and are just kept
        #   as possible examples.

        super(TestSolarFieldOutput, self).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
        )

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

    def setUp(self):
        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.
        # Load solar field data.
        self.solar_field: SolarField = sf.SolarField.from_csv_files(
            lln.NSTTF_ORIGIN,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            "Sandia NSTTF",
        )

    def old_single_heliostat(self) -> None:
        """
        Draws one heliostat.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_name = "5E10"

        # View setup
        title = "Heliostat " + heliostat_name + ", Face West"
        caption = "A single Sandia NSTTF heliostat " + heliostat_name + "."
        comments = []

        # Configuration setup
        heliostat = self.solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_orientation_from_pointing_vector(WEST)

        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            draw_outline=True,
            outline_style=rcps.outline(color="g"),
            draw_name=False,
            draw_centroid=False,
            draw_surface_normal=False,
        )
        fe_control = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control,
            draw_normal_vector=True,
            normal_vector_style=rcps.outline(color="g"),
            draw_centroid=True,
        )
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=fe_control)
        # heliostat_style = (rch.normal_facet_outlines(color='g'))

        # Setup render control.
        # Style setup
        # heliostat_control = rce.RenderControlEnsemble(rch.mirror_surfaces(color='b'))
        # heliostat_styles = rce.RenderControlEnsemble(heliostat_style)

        # comments\
        comments.append("Demonstration of heliostat drawing.")
        comments.append("Facet outlines shown, with facet names and overall heliostat surface normal.")
        comments.append("Render mirror surfaces only.")
        comments.append("Green:   Facet outlines and overall surface normal.")

        # Draw.
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                1
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        heliostat.draw(fig_record.view, heliostat_style)
        # heliostat.draw(fig_record.view, heliostat_style)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_multi_heliostat(self) -> None:
        """
        Draws multiple heliostats.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_spec_list: list[tuple[str, Vxyz, rch.RenderControlHeliostat]]
        heliostat_spec_list = [
            ["11W1", UP, rch.name()],
            ["11E1", UP, rch.centroid(color="r")],
            ["11E2", UP, rch.centroid_name(color="g")],
            #    ['12W1', NORTH, rch.facet_outlines(color='b')],
            ["12W1", NORTH, rch.facet_outlines(color="b")],
            ["12E1", SOUTH, rch.normal_outline(color="c")],
            #    ['12E2', EAST, rch.corner_normals_outline(color='m')],
            ["12E2", EAST, rch.normal_outline(color="m")],
            ["13W1", WEST, rch.normal_facet_outlines(color="g")],
            ["13E1", UP, rch.facet_outlines_normals(color="c")],
            #    ['13E2', STOW, rch.facet_outlines_corner_normals()]]
            ["13E2", STOW, rch.facet_outlines_normals()],
        ]

        # View setup
        title = "Example Poses and Styles"
        caption = title
        comments = []

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                3
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )

        # Style setup and draw
        for heliostat_spec in heliostat_spec_list:
            heliostat_name = heliostat_spec[0]
            heliostat_config = heliostat_spec[1]
            heliostat_style = heliostat_spec[2]
            # Configuration setup
            heliostat = self.solar_field.lookup_heliostat(heliostat_name)
            heliostat.set_orientation_from_pointing_vector(heliostat_config)
            heliostat.draw(fig_record.view, heliostat_style)

        # Comment
        comments.append("Demonstration of various example heliostat drawing modes.")
        comments.append("Black:   Name only.")
        comments.append("Red:     Centroid only.")
        comments.append("Green:   Centroid and name.")
        comments.append("Blue:    Facet outlines.")
        comments.append("Cyan:    Overall outline and overall surface normal.")
        comments.append("Magneta: Overall outline and overall surface normal, drawn at corners.")
        comments.append("Green:   Facet outlines and overall surface normal.")
        comments.append("Cyan:    Facet outlines and facet surface normals.")
        comments.append("Black:   Facet outlines and facet surface normals drawn at facet corners.")

        # Output.
        self.show_save_and_check_figure(fig_record)

    def old_solar_field_h_names(self) -> None:
        """
        Draws solar field heliostats, with names.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = "Heliostat Names"
        caption = title
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_names(color="m")

        # Comment
        comments.append("Heliostat names, drawn at each heliostat's centroid.")
        comments.append("At NSTTF, centroids appear to be at the midpoint of the torque tube.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control_large,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                4
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def old_solar_field_h_centroids(self) -> None:
        """
        Draws solar field heliostats, with centroids.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = "Heliostat Centroids"
        caption = "Position of heliostat centroid, in heliostat face-up configuration."
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_centroids(color="b")

        # Comment
        comments.append("Heliostat centroids, which at NSTTF appear to be at the midpoint of the torque tube.")

        # Draw and output in 3d
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                5
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # Draw and output in xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                6
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # Draw and output in xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                7
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # Draw and output in yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                8
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

    def test_solar_field_h_centroids_names(self) -> None:
        """
        Draws solar field heliostats, with centroids and names.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = "Heliostat Labelled Centroids"
        caption = "Heliostat centroids with adjacent heliostat name labels."
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_centroids_names(color="g")

        # Comment
        comments.append("Heliostat names and labels drawn together.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control_large,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                9
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def old_solar_field_h_outlines(self) -> None:
        """
        Draws solar field heliostat outlines.  (Plus aim point and legend.)
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = "Heliostat Outlines"
        caption = "Overall outlines of each heliostat mirror array."
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color="b")

        # Comment
        comments.append("A simple way of rendering a solar field.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                10
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        aimpoint_xyz.draw_line(fig_record.view, style=rcps.marker(color="tab:orange"), label="aimpoint_xyz")

        # Output.
        self.show_save_and_check_figure(fig_record)

    def old_annotated_solar_field(self) -> None:
        """
        Draws annotated solar field.
        """
        # Initialize test.
        self.start_test()

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
        title = "Solar Field Situation"
        caption = (
            "Rendering of a complex solar field situation, with some heliostats tracking, some face up, and some in stow.  "
            + "Demonstrates rendering options applied to each."
        )
        comments = []

        # Define tracking time
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        # Define fixed heliostat orientation.
        synch_az = np.deg2rad(205)
        synch_el = np.deg2rad(30)

        # Configuration setup
        solar_field = self.solar_field
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        for h_name in stowed_heliostats:
            solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(STOW)
        for h_name in synched_heliostats:
            h: HeliostatAzEl = solar_field.lookup_heliostat(h_name)
            h.set_orientation_from_az_el(synch_az, synch_el)  # Az El heliostat orientation defined by tuple (az, el)
        for h_name in up_heliostats:
            solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(UP)

        # synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        # solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        # up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        # solar_field.set_heliostats_configuration(up_heliostats, up_configuration)

        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color="b")
        solar_field_style.add_special_names(up_heliostats, rch.normal_outline(color="c"))
        solar_field_style.add_special_names(stowed_heliostats, rch.normal_outline(color="r"))
        solar_field_style.add_special_names(synched_heliostats, rch.normal_outline(color="g"))

        # Comment
        comments.append("A solar field situation with heliostats in varying status.")
        comments.append("Blue heliostats are tracking.")
        comments.append("Cyan heliostats are face up.")
        comments.append("Red heliostats are in stow (out of service).")
        comments.append(
            "Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(
                np.rad2deg(synch_az), np.rad2deg(synch_el)
            )
        )

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                11
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_solar_field_subset(self) -> None:
        """
        Draws solar field subset.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        mirrored_heliostats = ["7E4"]
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
            "7E5",
            "7E6",
            "7E7",
        ]
        tracking_heliostats = ["8E1", "8E2", "8E4", "8E6", "8E7", "9E1", "9E2", "9E3", "9E4", "9E5", "9E6", "9E7"]

        # View setup
        title = "Selected Heliostats"
        caption = "View of a selected range of heliostats, demonstrating different rendering for tracking, face up, and stowed heliostats."
        comments = []

        # Define tracking time
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        # Define fixed heliostat orientation.
        synch_az = np.deg2rad(205)
        synch_el = np.deg2rad(30)

        # Configuration setup
        solar_field = self.solar_field
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        for h_name in up_heliostats:
            solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(UP)
        for h_name in stowed_heliostats:
            solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(STOW)
        for h_name in synched_heliostats + mirrored_heliostats:
            config = HeliostatConfiguration("az-el", az=synch_az, el=synch_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
        # for h_name in tracking_heliostats:
        #     solar_field.lookup_heliostat(h_name).set_tracking_configuration(aimpoint_xyz,
        #                                                                     solar_field.origin_lon_lat,
        #                                                                     when_ymdhmsz,)

        # solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
        # synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        # solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        # solar_field.set_heliostats_configuration(mirrored_heliostats, synch_configuration)
        # up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        # solar_field.set_heliostats_configuration(up_heliostats, up_configuration)

        # Style setup
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(mirrored_heliostats, rch.mirror_surfaces())
        solar_field_style.add_special_names(up_heliostats, rch.facet_outlines(color="c"))
        solar_field_style.add_special_names(stowed_heliostats, rch.normal_outline(color="r"))
        solar_field_style.add_special_names(synched_heliostats, rch.normal_outline(color="g"))
        solar_field_style.add_special_names(tracking_heliostats, rch.facet_outlines(color="b"))

        # Comment
        comments.append("A subset of heliostats selected, so that plot is effectively zoomed in.")
        comments.append("Grey heliostat shows mirrored surfaces.")
        comments.append("Blue heliostats are tracking.")
        comments.append("Cyan heliostats are face up.")
        comments.append("Red heliostats are in stow (out of service).")
        comments.append(
            "Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(
                np.rad2deg(synch_az), np.rad2deg(synch_el)
            )
        )

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                12
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_heliostat_vector_field(self) -> None:
        """
        Draws heliostat vector field.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = "Heliostat Vector Field"
        caption = "Rendering of the normal vector at the heliostat origin, for each heliostat in a field of tracking heliostats."
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_vector_field(color="b")

        # Comment
        comments.append("Each heliostat's surface normal, which can be viewed as a vector field.")

        # # Draw and produce output for 3d
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(), number_in_name=False,
        #                                          input_prefix=self.figure_prefix(13),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        # aimpoint_xyz.draw_point(fig_record.view, style=rcps.marker(color='tab:orange'), labels='aimpoint_xyz')
        # solar_field.draw(fig_record.view, solar_field_style)
        # self.show_save_and_check_figure(fig_record)

        # # Draw and produce output for xy
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_xy(), number_in_name=False,
        #                                          input_prefix=self.figure_prefix(14),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        # aimpoint_xyz.draw_point(fig_record.view, style=rcps.marker(color='tab:orange'), labels='aimpoint_xyz')
        # solar_field.draw(fig_record.view, solar_field_style)
        # self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                15
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        aimpoint_xyz.draw_points(fig_record.view, style=rcps.marker(color="tab:orange"), labels="aimpoint_xyz")
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # # Draw and produce output for yz
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_yz(), number_in_name=False,
        #                                          input_prefix=self.figure_prefix(16),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        # aimpoint_xyz.draw_point(fig_record.view, style=rcps.marker(color='tab:orange'), labels='aimpoint_xyz')
        # solar_field.draw(fig_record.view, solar_field_style)
        # self.show_save_and_check_figure(fig_record)

    def test_dense_vector_field(self) -> None:
        """
        Draws dense vector field.
        """
        # Initialize test.
        self.start_test()

        # View setup (xy only)
        title = "Dense Tracking Vector Field"
        caption = (
            "Grey shows heliostat outlines and surface normals for heliostats tracking to a given aim point.  "
            "Blue shows a dense field of hypothetical heliostat origins, with the surface normal that would "
            "result from tracking to the same aim point.  Both the actual heliostat surface normals and the "
            "hypothetical heliostat surface normals may be thought of as concrete instances within a continuous "
            "vector field defined by the tracking aim point."
        )
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_vector_field_outlines(color="grey")

        # Comment
        comments.append("Dense vector field of tracking surface normals.")

        # Draw solar field and aim point.
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                17
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        aimpoint_xyz.draw_points(fig_record.view, style=rcps.marker(color="tab:orange"), labels="aimpoint_xyz")

        # Draw dense vector field.
        grid_xy = solar_field.heliostat_field_regular_grid_xy(40, 20)
        # grid_xydxy = [[p, sunt.tracking_surface_normal_xy(p+[0], aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)] for p in grid_xy]
        grid_xydxy = [
            [
                p,
                sun_track.tracking_surface_normal_xy(
                    Pxyz(p + [0]), aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz
                ),
            ]
            for p in grid_xy
        ]
        fig_record.view.draw_pqdpq_list(grid_xydxy, style=rcps.vector_field(color="b", vector_scale=5.0))

        # Output.
        self.show_save_and_check_figure(fig_record)


# MAIN EXECUTION
if __name__ == "__main__":
    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False  # False
    # Setup.
    test_object = TestSolarFieldOutput()
    test_object.setUpClass(interactive=interactive, verify=verify)
    test_object.setUp()
    # Tests.
    lt.info("Beginning tests...")
    # old_object.test_single_heliostat()
    # old_object.test_annotated_heliostat()
    test_object.test_multi_heliostat()
    # old_object.test_solar_field_h_names()
    # old_object.test_solar_field_h_centroids()
    test_object.test_solar_field_h_centroids_names()
    # old_object.test_solar_field_h_outlines()
    # old_object.test_annotated_solar_field()
    test_object.test_solar_field_subset()
    test_object.test_heliostat_vector_field()
    test_object.test_dense_vector_field()
    lt.info("All tests complete.")
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
