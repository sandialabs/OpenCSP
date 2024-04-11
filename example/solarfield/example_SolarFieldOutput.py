"""
Demonstrate Solar Field Plotting Routines



"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from opencsp.common.lib.csp.ufacet.Heliostat import Heliostat
import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.csp.SolarField as sf
from opencsp.common.lib.csp.SolarField import SolarField
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class ExampleSolarFieldOutput(to.TestOutput):
    @classmethod
    def setup_class(
        self,
        source_file_body: str = 'ExampleSolarFieldOutput',  # Set these here, because pytest calls
        figure_prefix_root: str = 'tsfo',  # setup_class() with no arguments.
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

        super(ExampleSolarFieldOutput, self).setup_class(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
            output_path='solarfield',
        )

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

        # Load solar field data.
        self.solar_field: SolarField = sf.sf_from_csv_files(
            name='Sandia NSTTF',
            short_name='NSTTF',
            origin_lon_lat=lln.NSTTF_ORIGIN,
            heliostat_file=dpft.sandia_nsttf_test_heliostats_origin_file(),
            facet_centroids_file=dpft.sandia_nsttf_test_facet_centroidsfile(),
        )

    def single_heliostat(self) -> None:
        """
        Draws one heliostat.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_name = '5E10'

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Face West'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + '.'
        comments = []

        # Configuration setup
        heliostat: Heliostat = self.solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_configuration(hc.face_west())
        heliostat_style = rch.normal_facet_outlines(color='g')

        # Setup render control.
        # Style setup
        heliostat_control = rce.RenderControlEnsemble(rch.mirror_surfaces(color='b'))
        heliostat_styles = rce.RenderControlEnsemble(heliostat_style)

        # comments\
        comments.append("Demonstration of heliostat drawing.")
        comments.append("Facet outlines shown, with facet names and overall heliostat surface normal.")
        comments.append('Render mirror surfaces only.')
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
        heliostat.draw(fig_record.view, heliostat_styles)
        heliostat.draw(fig_record.view, heliostat_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def annotated_heliostat(self) -> None:
        """
        Draws annotated heliostat.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_name = '5E10'

        # View setup
        title = 'Heliostat ' + heliostat_name + ', with Highlighting'
        caption = 'A single Sandia NSTTF heliostat with rendering options ' + heliostat_name + '.'
        comments = []

        # Tracking setup.
        aimpoint_xyz = [60.0, 8.8, 28.9]
        # [year, month, day, hour, minute, second, zone]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        heliostat = self.solar_field.lookup_heliostat(heliostat_name)
        heliostat.set_tracking(aimpoint_xyz, self.solar_field.origin_lon_lat, when_ymdhmsz)

        # Style setup
        default_heliostat_style = rch.normal_facet_outlines()
        default_heliostat_style.facet_styles.add_special_name(16, rcf.corner_normals_outline_name(color='c'))
        default_heliostat_style.facet_styles.add_special_names([1, 4, 7, 24, 25], rcf.normal_outline(color='r'))
        heliostat_styles = rce.RenderControlEnsemble(default_heliostat_style)

        # Comment
        comments.append("Demonstration of example heliostat annotations.")
        comments.append("Black:   Facet outlines.")
        comments.append("Black:   Overall heliostat surface normal.")
        comments.append("Red:     Highlighted facets and their surface normals.")
        comments.append("Cyan:    Highlighted facet with facet name and facet surface normal drawn at corners.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        heliostat.draw(fig_record.view, heliostat_styles)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def multi_heliostat(self) -> None:
        """
        Draws multiple heliostats.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_spec_list = [
            ['11W1', hc.face_up(), rch.name()],
            ['11E1', hc.face_up(), rch.centroid(color='r')],
            ['11E2', hc.face_up(), rch.centroid_name(color='g')],
            ['12W1', hc.face_north(), rch.facet_outlines(color='b')],
            ['12E1', hc.face_south(), rch.normal_outline(color='c')],
            ['12E2', hc.face_east(), rch.corner_normals_outline(color='m')],
            ['13W1', hc.face_west(), rch.normal_facet_outlines(color='g')],
            ['13E1', hc.face_up(), rch.facet_outlines_normals(color='c')],
            ['13E2', hc.NSTTF_stow(), rch.facet_outlines_corner_normals()],
        ]

        # View setup
        title = 'Example Poses and Styles'
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
            heliostat.set_configuration(heliostat_config)
            heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
            heliostat.draw(fig_record.view, heliostat_styles)

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

    def solar_field_h_names(self) -> None:
        """
        Draws solar field heliostats, with names.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Names'
        caption = title
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_names(color='m')

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

    def solar_field_h_centroids(self) -> None:
        """
        Draws solar field heliostats, with centroids.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Centroids'
        caption = 'Position of heliostat centroid, in heliostat face-up configuration.'
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_centroids(color='b')

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

    def solar_field_h_centroids_names(self) -> None:
        """
        Draws solar field heliostats, with centroids and names.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Labelled Centroids'
        caption = 'Heliostat centroids with adjacent heliostat name labels.'
        comments = []

        # Tracking setup
        # Not required since we're not drawing heliostat shape.
        # Style setup
        solar_field = self.solar_field
        solar_field_style = rcsf.heliostat_centroids_names(color='g')

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

    def solar_field_h_outlines(self) -> None:
        """
        Draws solar field heliostat outlines.  (Plus aim point and legend.)
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Outlines'
        caption = 'Overall outlines of each heliostat mirror array.'
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color='b')

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
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')

        # Output.
        self.show_save_and_check_figure(fig_record)

    def annotated_solar_field(self) -> None:
        """
        Draws annotated solar field.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        up_heliostats = ['6E3', '8E3']
        stowed_heliostats = ['9W1', '12E14']
        synched_heliostats = [
            '5E1',
            '5E2',
            '5E3',
            '5E4',
            '5E5',
            '5E6',
            '5E7',
            '6E1',
            '6E2',
            '6E4',
            '6E5',
            '6E6',
            '6E7',
            '7E1',
            '7E2',
            '7E3',
            '7E4',
            '7E5',
            '7E6',
            '7E7',
        ]

        # View setup
        title = 'Solar Field Situation'
        caption = (
            'Rendering of a complex solar field situation, with some heliostats tracking, some face up, and some in stow.  '
            + 'Demonstrates rendering options applied to each.'
        )
        comments = []

        # Define tracking time
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        # Define fixed heliostat orientation.
        synch_az = np.deg2rad(205)
        synch_el = np.deg2rad(30)
        # Define upward-facing heliostat orientation.
        up_az = np.deg2rad(180)
        up_el = np.deg2rad(90)

        # Configuration setup
        solar_field = self.solar_field
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
        synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        solar_field.set_heliostats_configuration(up_heliostats, up_configuration)

        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color='b')
        solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.normal_outline(color='c'))
        solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color='r'))
        solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color='g'))

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

    def solar_field_subset(self) -> None:
        """
        Draws solar field subset.
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        mirrored_heliostats = ['7E4']
        up_heliostats = ['6E3', '8E3']
        stowed_heliostats = ['6E2', '8E5']
        synched_heliostats = [
            '5E1',
            '5E2',
            '5E3',
            '5E4',
            '5E5',
            '5E6',
            '5E7',
            '6E1',
            '6E4',
            '6E5',
            '6E6',
            '6E7',
            '7E1',
            '7E2',
            '7E3',
            '7E5',
            '7E6',
            '7E7',
        ]
        tracking_heliostats = ['8E1', '8E2', '8E4', '8E6', '8E7', '9E1', '9E2', '9E3', '9E4', '9E5', '9E6', '9E7']

        # View setup
        title = 'Selected Heliostats'
        caption = 'View of a selected range of heliostats, demonstrating different rendering for tracking, face up, and stowed heliostats.'
        comments = []

        # Define tracking time
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        # Define fixed heliostat orientation.
        synch_az = np.deg2rad(205)
        synch_el = np.deg2rad(30)
        # Define upward-facing heliostat orientation.
        up_az = np.deg2rad(180)
        up_el = np.deg2rad(90)

        # Configuration setup
        solar_field = self.solar_field
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
        synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
        solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
        solar_field.set_heliostats_configuration(mirrored_heliostats, synch_configuration)
        up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
        solar_field.set_heliostats_configuration(up_heliostats, up_configuration)

        # Style setup
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.heliostat_styles.add_special_names(mirrored_heliostats, rch.mirror_surfaces())
        solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.facet_outlines(color='c'))
        solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color='r'))
        solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color='g'))
        solar_field_style.heliostat_styles.add_special_names(tracking_heliostats, rch.facet_outlines(color='b'))

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

    def heliostat_vector_field(self) -> None:
        """
        Draws heliostat vector field.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Vector Field'
        caption = 'Rendering of the normal vector at the heliostat origin, for each heliostat in a field of tracking heliostats.'
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_vector_field(color='b')

        # Comment
        comments.append("Each heliostat's surface normal, which can be viewed as a vector field.")

        # Draw and produce output for 3d
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                13
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                14
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

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
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                16
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)

    def dense_vector_field(self) -> None:
        """
        Draws dense vector field.
        """
        # Initialize test.
        self.start_test()

        # View setup (xy only)
        title = 'Dense Tracking Vector Field'
        caption = (
            'Grey shows heliostat outlines and surface normals for heliostats tracking to a given aim point.  '
            'Blue shows a dense field of hypothetical heliostat origins, with the surface normal that would '
            'result from tracking to the same aim point.  Both the actual heliostat surface normals and the '
            'hypothetical heliostat surface normals may be thought of as concrete instances within a continuous '
            'vector field defined by the tracking aim point.'
        )
        comments = []

        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_vector_field_outlines(color='grey')

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
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')

        # Draw dense vector field.
        grid_xy = solar_field.heliostat_field_regular_grid_xy(40, 20)
        # grid_xydxy = [[p, sunt.tracking_surface_normal_xy(p+[0], aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)] for p in grid_xy]
        grid_xydxy = [
            [p, sun_track.tracking_surface_normal_xy(p + [0], aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)]
            for p in grid_xy
        ]
        fig_record.view.draw_pqdpq_list(grid_xydxy, style=rcps.vector_field(color='b', vector_scale=5.0))

        # Output.
        self.show_save_and_check_figure(fig_record)


# if __name__ == "__main__":
#     plt.close('all')

#     fm.reset_figure_management()

#     figure_control = rcfg.RenderControlFigure(tile_array=(3, 2), tile_square=False)

#     axis_control_m = rca.meters()
#     # Control flags
#     # (Also see draw_demonstration_figures() above.)
#     save_figures = False
#     show_figures = True

#     # Update fm.show_figures
#     fm.do_show_figures(show_figures)

#     # Load solar field data.
#     solar_field: SolarField = sf.sf_from_csv_files(name='Sandia NSTTF',
#                                                    short_name='NSTTF',
#                                                    origin_lon_lat=lln.NSTTF_ORIGIN,
#                                                    heliostat_file=dpft.sandia_nsttf_test_heliostats_origin_file(),
#                                                    facet_centroids_file=dpft.sandia_nsttf_test_facet_centroidsfile())

#     # Define tracking time.
#     aimpoint_xyz = [60.0, 8.8, 28.9]
#     #               year, month, day, hour, minute, second, zone]
#     when_ymdhmsz = [2021,   5,   13,   13,    2,       0,    -6]  # NSTTF solar noon

#     # Define fixed heliostat orientation.
#     synch_az = np.deg2rad(205)
#     synch_el = np.deg2rad(30)

#     # Define upward-facing heliostat orientation.
#     up_az = np.deg2rad(180)
#     up_el = np.deg2rad(90)

#     # Draw figure suite.
#     draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_3d(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)
#     draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_xy(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)
#     draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_xz(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)
#     draw_demonstration_figures(figure_control, axis_control_m, vs.view_spec_yz(), solar_field, aimpoint_xyz, when_ymdhmsz, synch_az=synch_az, synch_el=synch_el, up_az=up_az, up_el=up_el)

#     # Summarize.
#     print('\n\nFigure Summary:')
#     fm.print_figure_summary()

#     # Save figures.
#     if save_figures:
#         print('\n\nSaving figures...')
#         # Output directory.
#         output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
#         if not(os.path.exists(output_path)):
#             os.makedirs(output_path)
#         fm.save_all_figures(output_path)


#     if show_figures:
#         input("Press 'Enter' to close the figures...")
#         plt.close('all')


def example_driver():
    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = True  # False
    # Setup.
    example_object = ExampleSolarFieldOutput()
    example_object.setup_class(interactive=interactive, verify=verify)
    # Tests.
    lt.info('Beginning tests...')
    example_object.single_heliostat()
    example_object.annotated_heliostat()
    example_object.multi_heliostat()
    example_object.solar_field_h_names()
    # TODO, fix image diff: example_object.solar_field_h_centroids()
    # TODO: fix image diff: example_object.solar_field_h_centroids_names()
    example_object.solar_field_h_outlines()
    example_object.annotated_solar_field()
    example_object.solar_field_subset()
    example_object.heliostat_vector_field()
    example_object.dense_vector_field()
    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    example_object.teardown_method()


# MAIN EXECUTION
if __name__ == "__main__":
    example_driver()
