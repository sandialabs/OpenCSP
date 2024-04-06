"""
Demonstrate Solar Field Plotting Routines



"""

from datetime import datetime
import numpy as np
import os

from opencsp.common.lib.csp.ufacet.Heliostat import Heliostat
import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.csp.SolarField as sf
from opencsp.common.lib.csp.SolarField import SolarField
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
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


class TestSolarFieldOutput(to.TestOutput):
    @classmethod
    def setUpClass(
        cls,
        source_file_body: str = 'TestSolarFieldOutput',  # Set these here, because pytest calls
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

        super(TestSolarFieldOutput, cls).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
        )

    def setUp(self):
        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

        self.solar_field: sf.SolarField = None

    def get_entire_solar_field(self) -> sf.SolarField:
        if self.solar_field == None:
            import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
            import opencsp.common.lib.geo.lon_lat_nsttf as lln

            self.solar_field = sf.sf_from_csv_files(
                name='Sandia NSTTF Partition',
                short_name='NSTTF',
                origin_lon_lat=lln.NSTTF_ORIGIN,
                heliostat_file=dpft.sandia_nsttf_test_heliostats_origin_file(),
                facet_centroids_file=dpft.sandia_nsttf_test_facet_centroidsfile(),
            )
        return self.solar_field

    def test_multi_heliostat(self) -> None:
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
        solar_field_csv = os.path.join(self.actual_output_dir, 'test_multi_heliostat.csv')
        solar_field = stest.load_solar_field_partition([name[0] for name in heliostat_spec_list], solar_field_csv)

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
            heliostat = solar_field.lookup_heliostat(heliostat_name)
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
        self.show_save_and_check_figure(fig_record, dpi=150)

    def test_solar_field_h_centroids_names(self) -> None:
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
        solar_field = self.get_entire_solar_field()
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
        self.show_save_and_check_figure(fig_record, dpi=150)

    def test_solar_field_subset(self) -> None:
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

        solar_field_csv = os.path.join(self.actual_output_dir, 'test_solar_field_subset.csv')
        solar_field = stest.load_solar_field_partition(
            mirrored_heliostats + up_heliostats + stowed_heliostats + synched_heliostats + tracking_heliostats,
            solar_field_csv,
        )

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
        self.show_save_and_check_figure(fig_record, dpi=150)

    def test_heliostat_vector_field(self) -> None:
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
        solar_field = self.get_entire_solar_field()
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021, 5, 13, 13, 2, 0, -6]  # NSTTF solar noon
        # [year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Style setup
        solar_field_style = rcsf.heliostat_vector_field(color='b')

        # Comment
        comments.append("Each heliostat's surface normal, which can be viewed as a vector field.")

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
        self.show_save_and_check_figure(fig_record, dpi=150)

    def test_dense_vector_field(self) -> None:
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
        solar_field = self.get_entire_solar_field()
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
        self.show_save_and_check_figure(fig_record, dpi=150)


# MAIN EXECUTION
if __name__ == "__main__":
    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = True  # False
    # Setup.
    test_object = TestSolarFieldOutput()
    test_object.setUpClass(interactive=interactive, verify=verify)
    test_object.setUp()
    # Tests.
    lt.info('Beginning tests...')
    test_object.test_single_heliostat()
    test_object.test_annotated_heliostat()
    test_object.test_multi_heliostat()
    test_object.test_solar_field_h_names()
    test_object.test_solar_field_h_centroids()
    test_object.test_solar_field_h_centroids_names()
    test_object.test_solar_field_h_outlines()
    test_object.test_annotated_solar_field()
    test_object.test_solar_field_subset()
    test_object.test_heliostat_vector_field()
    test_object.test_dense_vector_field()
    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
