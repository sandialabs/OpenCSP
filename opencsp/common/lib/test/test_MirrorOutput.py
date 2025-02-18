"""
Test generation and rendering of mirror surfaces, their surface normals, etc.
"""

import copy
import datetime
from typing import Callable

import numpy as np
import pytz
from scipy.spatial.transform import Rotation
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Pxy import Pxy

from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.render_control.RenderControlLightPath as rclp
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.SolarField import SolarField
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.geo.lon_lat_nsttf as lln


PI = np.pi
DEG2RAD = PI / 180
UP = Vxyz([0, 0, 1])


class TestMirrorOutput(to.TestOutput):

    @classmethod
    def setUpClass(
        cls,
        source_file_body: str = "TestMirrorOutput",  # Set these here, because pytest calls
        figure_prefix_root: str = "tmo",  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):
        # Generic setup.
        super(TestMirrorOutput, cls).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
        )

    def setUp(self):
        # create a scnene for placing optics
        self.scene = Scene()

        # Mirror, based on a parameteric model.
        self.m1_focal_length = 2.0  # meters
        self.m1_fxn = self.lambda_symmetric_paraboloid(
            self.m1_focal_length
        )  # Include self as a parameter, because this setup_class() function is a @classmethod.
        self.m1_len_x = 2.0  # m
        self.m1_len_y = 3.0  # m
        self.m1_rectangle_xy = (self.m1_len_x, self.m1_len_y)
        self.m1 = MirrorParametricRectangular(self.m1_fxn, self.m1_rectangle_xy)
        self.m1_shape_description = "rectangle " + str(self.m1_len_x) + "m x " + str(self.m1_len_y) + "m"
        self.m1_title = "Mirror (" + self.m1_shape_description + ", f=" + str(self.m1_focal_length) + "m), Face Up"
        self.m1_caption = (
            "A single mirror of shape ("
            + self.m1_shape_description
            + "), analytically defined with focal length f="
            + str(self.m1_focal_length)
            + "m."
        )
        self.m1_comments = []

        # Pentagonal mirror, based on a parametric model.
        pentagon_vertices = Pxy(
            [
                [0.0, -0.95105652, -0.58778525, 0.58778525, 0.95105652],
                [1.0, 0.30901699, -0.80901699, -0.80901699, 0.30901699],
            ]
        )
        pentagon_region = RegionXY.from_vertices(pentagon_vertices)
        self.m_pentagon = MirrorParametric(self.m1_fxn, pentagon_region)

        # Facet, based on a parameteric mirror.
        self.f1 = Facet(self.m1)
        self.f1_title = "Facet, from " + self.m1_title
        self.f1_caption = (
            "A facet defined from a parameteric mirror of shape ("
            + self.m1_shape_description
            + "), with focal length f="
            + str(self.m1_focal_length)
            + "m."
        )
        self.f1_comments = []

        # Simple 2x2 heliostat, with parameteric facets.
        self.h2x2_f1 = Facet(copy.deepcopy(self.m1))
        self.h2x2_f2 = Facet(copy.deepcopy(self.m1))
        self.h2x2_f3 = Facet(copy.deepcopy(self.m1))
        self.h2x2_f4 = Facet(copy.deepcopy(self.m1))
        fe2x2 = FacetEnsemble([self.h2x2_f1, self.h2x2_f2, self.h2x2_f3, self.h2x2_f4])
        facet_positions = Pxyz([[-1.1, 1.1, -1.1, 1.1], [1.6, 1.6, -1.6, -1.6], [0, 0, 0, 0]])
        fe2x2.set_facet_positions(facet_positions)  # fe2x2 := facet emsenble, two by two

        # Set canting angles.
        cos5 = np.cos(np.deg2rad(8))
        sin5 = np.sin(np.deg2rad(8))
        tilt_up = Rotation.from_matrix(np.asarray([[1, 0, 0], [0, cos5, -sin5], [0, sin5, cos5]]))
        tilt_down = Rotation.from_matrix(np.asarray([[1, 0, 0], [0, cos5, sin5], [0, -sin5, cos5]]))
        tilt_left = Rotation.from_matrix(np.asarray([[cos5, 0, sin5], [0, 1, 0], [-sin5, 0, cos5]]))
        tilt_right = Rotation.from_matrix(np.asarray([[cos5, 0, -sin5], [0, 1, 0], [sin5, 0, cos5]]))
        fe_2x2_canting_rotations = [
            tilt_left * tilt_up,
            tilt_right * tilt_up,
            tilt_left * tilt_down,
            tilt_right * tilt_down,
        ]
        fe2x2.set_facet_cantings(fe_2x2_canting_rotations)

        self.h2x2 = HeliostatAzEl(fe2x2, name="Simple 2x2 Heliostat")
        self.h2x2_title = "Heliostat with Parametrically Defined Facets"
        self.h2x2_caption = (
            "Heliostat with four facets ("
            + self.m1_shape_description
            + "), with focal length f="
            + str(self.m1_focal_length)
            + "m."
        )
        self.h2x2_comments = []

        # Simple solar field, with two simple heliostats.
        self.sf2x2_h1 = HeliostatAzEl(copy.deepcopy(fe2x2), "Heliostat 1")
        self.sf2x2_h2 = HeliostatAzEl(copy.deepcopy(fe2x2), "Heliostat 2")
        self.sf2x2_heliostats = [self.sf2x2_h1, self.sf2x2_h2]
        self.sf2x2 = SolarField(self.sf2x2_heliostats, [-106.509606, 34.962276], "Test Field", "test")
        heliostat_positions = Pxyz([[0, 0], [0, 10], [0, 0]])
        self.sf2x2.set_heliostat_positions(heliostat_positions)
        self.sf2x2_title = "Two Heliostats"
        self.sf2x2_caption = "Two 4-facet heliostats, tracking."
        self.sf2x2_comments = []

    def lambda_symmetric_paraboloid(self, focal_length: float) -> Callable[[float, float], float]:
        """Returns a callable for a symmetric paraboloid surface

        Parameters
        ----------
        focal_length : float
            Focal length

        Returns
        -------
        Callable[[float, float], float]
            Ouptuts z coordinate when input x, y coordinate
        """
        a = 1.0 / (4 * focal_length)
        return lambda x, y: a * (x**2 + y**2)

    def test_mirror_halfpi_rotation(self) -> None:
        """
        Draws a pentagonal mirror that is rotated 90 deg in space. Should look like a mirror with its normal parallel in the xy plane.
        """
        # Initialize test.
        self.start_test()
        local_comments = self.m1_comments.copy()

        # Position/Rotation in space.
        tran = Vxyz([0, 0, 0])
        rot = Rotation.from_euler("x", 45, True)
        transform = TransformXYZ.from_R_V(rot, tran)

        local_comments.append("Oriented face 45 deg up from level.")
        self.scene.add_object(self.m_pentagon)
        self.scene.set_position_in_space(self.m_pentagon, transform)

        # Setup render control.
        mirror_control = rcm.RenderControlMirror()
        local_comments.append("Render surface only.")

        # Draw.
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(2),
            title=self.m1_title + ", Face Horizon",
            caption=st.add_to_last_sentence(self.m1_caption, ", facing the horizon"),
            comments=local_comments,
            code_tag=self.code_tag,
        )
        self.m_pentagon.draw(fig_record.view, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_facet(self) -> None:
        """
        Draws a facet that contains a mirror. Includes the surface normals for multiple points on the mirror.
        """
        # Initialize test.
        self.start_test()
        local_comments = self.f1_comments.copy()

        # Position in space.
        tran = Vxyz([0, 0, 0])
        rot_mat = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rot = Rotation.from_matrix(rot_mat)
        transform = TransformXYZ.from_R_V(rot, tran)

        self.scene.add_object(self.f1)
        self.scene.set_position_in_space(self.f1, transform)
        local_comments.append("Oriented face horizon.")

        # Setup render control.
        mirror_control = rcm.RenderControlMirror(surface_normals=True, norm_len=1)
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control,
            draw_outline=True,
            draw_surface_normal=False,
            # draw_surface_normal_at_corners=True,
        )
        local_comments.append("Render mirror surface with normals, facet outline with corner normals.")

        # Draw.
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(4),
            title=self.f1_title,
            caption=self.f1_caption,
            comments=local_comments,
            code_tag=self.code_tag,
        )
        self.f1.draw(fig_record.view, facet_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_solar_field(self) -> None:
        """
        Shows a solar field with two heliostats, while tracking.
        """
        # Initialize test.
        self.start_test()
        local_comments = self.h2x2_comments.copy()

        # Set configurations.
        # self.sf2x2_h1.set_orientation_from_az_el(180 * DEG2RAD, 0)
        # local_comments.append('Heliostat 1 oriented initially face west.')  # Overriden by tracking below.
        # self.sf2x2_h2.set_orientation_from_az_el(90 * DEG2RAD, 0)
        # local_comments.append('Heliostat 2 oriented initially face south.')  # Overriden by tracking below.

        # Define tracking time.
        aimpoint = Pxyz([60.0, 8.8, 28.9])
        #               year, month, day, hour, minute, second, zone]
        when_ymdhmsz = (2021, 5, 13, 13, 0, 0, -6)
        self.sf2x2.set_full_field_tracking(aimpoint, when_ymdhmsz)
        local_comments.append("Heliostats set to track to " + str(aimpoint) + " at ymdhmsz =" + str(when_ymdhmsz))

        # Setup render control.
        mirror_control = rcm.RenderControlMirror(surface_normals=False)
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control,
            draw_outline=False,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control, draw_centroid=True, draw_outline=True, draw_normal_vector=True
        )
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=True, facet_ensemble_style=facet_ensemble_control, draw_facet_ensemble=True
        )
        solar_field_control = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control)
        local_comments.append("Render mirror surfaces, facet centroids, and heliostat outline and surface normal.")
        light_path_control = rclp.RenderControlLightPath(current_length=3)

        # Draw.
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(7),
            title=self.sf2x2_title,
            caption=self.sf2x2_caption,
            comments=local_comments,
            code_tag=self.code_tag,
        )
        self.sf2x2.draw(fig_record.view, solar_field_control)

        scene = Scene()
        tz = pytz.timezone("US/Mountain")
        time = datetime.datetime(2021, 5, 13, 13, 0, 0, 0, tz)
        sun = LightSourceSun.from_location_time([34.962276, -106.509606], time, 2)
        scene.add_light_source(sun)

        # debug
        # print("The direction of the center sunray is: ", sun.get_incident_rays(Pxyz([0, 0, 0])))
        # print("sop from he:", self.sf2x2.heliostats[1].facet_ensemble.survey_of_points(Resolution.pixelX(3)))
        # print("sop from heliostat:", self.sf2x2.heliostats[1].survey_of_points(Resolution.pixelX(3)))
        # print("sop from sf:", self.sf2x2.survey_of_points(Resolution.pixelX(3)))

        # scene.add_object(self.sf2x2)
        # trace = rt.trace_scene(scene, Resolution.center())

        # trace.draw(fig_record.view, rcrt.RenderControlRayTrace(light_path_control))
        # aimpoint.draw_point(fig_record.view)

        # Output.
        self.show_save_and_check_figure(fig_record)

    def test_heliostat_05W01_and_14W01(self) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        # Define scenario.
        nsttf_pivot_height = 4.02  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_pivot_offset = 0.1778  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        # 5W01.
        x_5W01 = -4.84  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        y_5W01 = 57.93  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        z_5W01 = 3.89  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length_5W01 = 55  # meters
        name_5W01 = "5W01"
        title_5W01 = "NSTTF Heliostat " + name_5W01
        caption_5W01 = "5W01 modeled as a symmetric paraboloid with focal length f=" + str(focal_length_5W01) + "m."
        # 14W01.
        x_14W01 = -4.88  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        y_14W01 = 194.71  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        z_14W01 = 4.54  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length_14W01 = 186.8  # meters
        name_14W01 = "14W01"
        title_14W01 = "NSTTF Heliostat " + name_14W01
        caption_14W01 = "14W01 modeled as a symmetric paraboloid with focal length f=" + str(focal_length_14W01) + "m."
        # Solar field.
        short_name_sf = "Mini NSTTF"
        name_sf = "Mini NSTTF with " + name_5W01 + " and " + name_14W01
        title_sf = "Two NSTTF Heliostats: " + name_5W01 + " and " + name_14W01
        caption_sf = (
            "Two NSTTF heliostats."
            + "  "
            + caption_5W01
            + "  "
            + caption_14W01
            + "  "
            + "Facet surfaces and canting have the same focal length."
        )
        comments = []

        # Construct heliostat objects and solar field object.
        def fn_5W01(x, y):
            return (x**2) / (4 * focal_length_5W01) + (y**2) / (4 * focal_length_5W01)

        # helio.h_from_facet_centroids(name_5W01, np.asarray([x_5W01, y_5W01, z_5W01]), 25, 5, 5,
        #                                       dpft.sandia_nsttf_test_facet_centroidsfile(),
        #                                       pivot_height=nsttf_pivot_height,
        #                                       pivot_offset=nsttf_pivot_offset,
        #                                       facet_width=nsttf_facet_width,
        #                                       facet_height=nsttf_facet_height,
        #                                       default_mirror_shape=fn_5W01)

        h_5W01_mirror = MirrorParametricRectangular(fn_5W01, (nsttf_facet_width, nsttf_facet_height))
        h_5W01, location_5W01 = HeliostatAzEl.from_csv_files(
            "5W1",
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h_5W01_mirror,
        )

        h_5W01.set_canting_from_equation(fn_5W01)

        def fn_14W01(x, y):
            return (x**2) / (4 * focal_length_14W01) + (y**2) / (4 * focal_length_14W01)

        # helio.h_from_facet_centroids("NSTTF Heliostat 14W01", np.asarray([x_14W01, y_14W01, z_14W01]), 25, 5, 5,
        #                                        dpft.sandia_nsttf_test_facet_centroidsfile(),
        #                                        pivot_height=nsttf_pivot_height,
        #                                        pivot_offset=nsttf_pivot_offset,
        #                                        facet_width=nsttf_facet_width,
        #                                        facet_height=nsttf_facet_height,
        #                                        default_mirror_shape=fn_14W01)
        h_14W01_mirror = MirrorParametricRectangular(fn_14W01, (nsttf_facet_width, nsttf_facet_height))
        h_14W01, location_14W01 = HeliostatAzEl.from_csv_files(
            "14W1",
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h_14W01_mirror,
        )

        h_14W01.set_canting_from_equation(fn_14W01)

        heliostats = [h_5W01, h_14W01]
        heliostat_locations = [location_5W01, location_14W01]

        UP = Pxyz([0, 0, 1])
        h_5W01.set_orientation_from_az_el(0, np.pi / 2)
        h_14W01.set_orientation_from_az_el(0, np.pi / 2)

        sf = SolarField(heliostats, lln.NSTTF_ORIGIN, name_sf)
        sf.set_heliostat_positions(heliostat_locations)

        comments_long = comments.copy()  # We'll add a different comment for the plots with long normals.
        comments_very_long = comments.copy()  # We'll add a different comment for the plots with very long normals.
        comments_exaggerated_z = (
            comments.copy()
        )  # We'll add a different comment for the plots with an exaggerated z axis.
        comments.append("Render mirror surfaces and normals, facet outlines, and heliostat centroid.")

        # Setup render control (long normals).
        mirror_control_long = rcm.RenderControlMirror(surface_normals=True, norm_len=12, norm_res=3)
        facet_control_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_long = rcfe.RenderControlFacetEnsemble(facet_control_long)
        heliostat_control_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_long, draw_facet_ensemble=True
        )
        # # Setup render control.
        # mirror_control = rcm.RenderControlMirror(surface_normals=False)
        # facet_control = rcf.RenderControlFacet(draw_mirror_curvature=True,
        #                                        mirror_styles=mirror_control,
        #                                        draw_outline=False,
        #                                        draw_surface_normal=False,
        #                                        draw_surface_normal_at_corners=False,
        #                                        draw_name=False)
        # facet_ensemble_control = rcfe.RenderControlFacetEnsemble(default_style=facet_control,
        #                                                          draw_centroid=True)
        # heliostat_control = rch.RenderControlHeliostat(draw_centroid=True,
        #                                                draw_outline=True,
        #                                                draw_surface_normal=False,
        #                                                draw_surface_normal_at_corners=False,
        #                                                facet_ensemble_style=facet_ensemble_control,
        #                                                draw_facets=True)
        # solar_field_control = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control)

        comments_long.append("Render mirror surfaces and long normals, facet outlines, and heliostat centroid.")

        # Draw and output 5W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(17),
            title=title_5W01 + " (long normals)",
            caption=caption_5W01,
            comments=comments,
            code_tag=self.code_tag,
        )
        h_5W01.draw(fig_record.view, heliostat_control_long)
        self.show_save_and_check_figure(fig_record)

        # Draw and output 14W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(18),
            title=title_14W01 + " (long normals)",
            caption=caption_14W01,
            comments=comments,
            code_tag=self.code_tag,
        )
        h_14W01.draw(fig_record.view, heliostat_control_long)
        self.show_save_and_check_figure(fig_record)

        # Setup render control (very long normals).
        mirror_control_very_long = rcm.RenderControlMirror(
            surface_normals=True,
            norm_len=(2 * focal_length_14W01),  # Twice the focal length is the center of curvature.
            norm_res=2,
        )
        facet_control_very_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_very_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_very_long = rcfe.RenderControlFacetEnsemble(facet_control_very_long)
        heliostat_control_very_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_very_long, draw_facet_ensemble=True
        )
        solar_field_control_very_long = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control_very_long)
        comments_very_long.append(
            "Render mirror surfaces and very long normals, facet outlines, and heliostat centroid."
        )

        # Draw and output solar_field figure (very long normals, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(19),
            title=title_sf + " (very long normals)",
            caption=caption_sf,
            comments=comments,
            code_tag=self.code_tag,
        )
        sf.draw(fig_record.view, solar_field_control_very_long)
        self.show_save_and_check_figure(fig_record)

        # Setup render control (exaggerated z).
        z_exaggerated_margin = 0.35  # meters, plus or minus reference height.
        decimal_factor = 100.0
        # Different z limits for each heliostat, because they are at different elevations on the sloped field.
        z_min_5W01 = np.floor(decimal_factor * ((z_5W01 + nsttf_pivot_offset) - z_exaggerated_margin)) / decimal_factor
        z_max_5W01 = np.ceil(decimal_factor * ((z_5W01 + nsttf_pivot_offset) + z_exaggerated_margin)) / decimal_factor
        exaggerated_z_limits_5W01 = [z_min_5W01, z_max_5W01]
        z_min_14W01 = (
            np.floor(decimal_factor * ((z_14W01 + nsttf_pivot_offset) - z_exaggerated_margin)) / decimal_factor
        )
        z_max_14W01 = np.ceil(decimal_factor * ((z_14W01 + nsttf_pivot_offset) + z_exaggerated_margin)) / decimal_factor
        exaggerated_z_limits_14W01 = [z_min_14W01, z_max_14W01]

        mirror_control_exaggerated_z = rcm.RenderControlMirror(surface_normals=False)
        facet_control_exaggerated_z = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_exaggerated_z,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control_exaggerated_z = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control_exaggerated_z
        )
        heliostat_control_exaggerated_z = rch.RenderControlHeliostat(
            draw_centroid=False,
            facet_ensemble_style=facet_ensemble_control_exaggerated_z,
            draw_facet_ensemble=True,
            post=False,
        )
        comments_exaggerated_z.append("Render heliostat with exaggerated z axis.")

        # # Draw and output 5W01 figure (exaggerated z).
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          number_in_name=False, input_prefix=self.figure_prefix(20),
        #                                          title=title_5W01 + ' (exaggerated z)', caption=caption_5W01, comments=comments, code_tag=self.code_tag)
        # fig_record.z_limits = exaggerated_z_limits_5W01
        # h_5W01.draw(fig_record.view, heliostat_control_exaggerated_z)
        # self.show_save_and_check_figure(fig_record)

        # # Draw and output 5W01 figure (exaggerated z, yz view).
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_yz(),
        #                                          # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          number_in_name=False, input_prefix=self.figure_prefix(21),
        #                                          title=title_5W01 + ' (exaggerated z)', caption=caption_5W01, comments=comments, code_tag=self.code_tag)
        # fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        # fig_record.z_limits = exaggerated_z_limits_5W01  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        # h_5W01.draw(fig_record.view, heliostat_control_exaggerated_z)
        # self.show_save_and_check_figure(fig_record)

        # Draw and output 14W01 figure (exaggerated z).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(22),
            title=title_14W01 + " (exaggerated z)",
            caption=caption_14W01,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.z_limits = exaggerated_z_limits_14W01
        h_14W01.draw(fig_record.view, heliostat_control_exaggerated_z)
        self.show_save_and_check_figure(fig_record)

        # Draw and output 14W01 figure (exaggerated z, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(23),
            title=title_14W01 + " (exaggerated z)",
            caption=caption_14W01,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        fig_record.z_limits = (
            exaggerated_z_limits_14W01  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        )
        h_14W01.draw(fig_record.view, heliostat_control_exaggerated_z)
        self.show_save_and_check_figure(fig_record)

    def test_heliostat_stages(self) -> None:
        """
        Shows construction stages from a flat heliostat to a canted and tracking heliostat.
        """
        # Initialize test.
        self.start_test()

        # Define scenario.
        nsttf_pivot_height = 4.02  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_pivot_offset = 0.1778  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length = 5  # meters.  A hyopthetical short-focal legnth heliostat, so curvature is more apparent.
        name = "Short Focal Length"
        title = "NSTTF Heliostat " + name
        caption = (
            "Hypothetical short-focal length NSTTF heliostat modeled as a symmetric paraboloid with focal length f="
            + str(focal_length)
            + "m."
        )
        # Solar field.
        short_name_sf = "Mini NSTTF"
        name_sf = "Mini NSTTF with " + name
        comments = []

        # Construct heliostat objects and solar field object.
        def fn(x: float, y: float):
            return (x**2) / (4 * focal_length) + (y**2) / (4 * focal_length)

        m1 = MirrorParametricRectangular(fn, (nsttf_facet_width, nsttf_facet_height))
        h1, _ = HeliostatAzEl.from_csv_files(
            "5W1", dpft.sandia_nsttf_test_heliostats_origin_file(), dpft.sandia_nsttf_test_facet_centroidsfile(), m1
        )
        # h1.set_orientation_from_pointing_vector(UP)
        h1.set_orientation_from_az_el(0, np.pi / 2)

        heliostats = [h1]

        # sf = SolarField(name_sf, short_name_sf, [-106.509606, 34.962276], heliostats)
        sf = SolarField(heliostats, [-106.509606, 34.962276], name=name_sf)

        # Define tracking time.
        aimpoint_xyz = Pxyz([60.0, 8.8, 28.9])
        #               year, month, day, hour, minute, second, zone]
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)

        # Setup render control.
        mirror_control = rcm.RenderControlMirror(surface_normals=True, norm_len=4, norm_res=2)
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control = rcfe.RenderControlFacetEnsemble(default_style=facet_control)
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=True, facet_ensemble_style=facet_ensemble_control, draw_facet_ensemble=True
        )

        # # Face up heliostat, no canting.
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          number_in_name=False, input_prefix=self.figure_prefix(24),
        #                                          title=title + ', Without Canting', caption=st.add_to_last_sentence(caption, ' without canting'), comments=comments, code_tag=self.code_tag)
        # h1.draw(fig_record.view, heliostat_control)
        # self.show_save_and_check_figure(fig_record)

        # Draw and output face up heliostat, with canting and lifted.
        h2 = copy.deepcopy(h1)
        comments_lifted = comments.copy()

        new_positions = []
        for p in h2.facet_ensemble.facet_positions:
            x, y = p.x[0], p.y[0]
            z = fn(x, y)
            new_positions.append(Pxyz([x, y, z]))

        h2.facet_ensemble.set_facet_positions(new_positions)

        h2.set_canting_from_equation(fn)
        comments_lifted.append("Set canting angle and lifted.")
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(25),
            title=title + ", Canted and Lifted",
            caption=st.add_to_last_sentence(caption, " with canting angle and lifted"),
            comments=comments,
            code_tag=self.code_tag,
        )
        h2.draw(fig_record.view, heliostat_control)
        self.show_save_and_check_figure(fig_record)

        # # Draw and output face up heliostat, with canting.
        # # h1.flatten()
        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          number_in_name=False, input_prefix=self.figure_prefix(26),
        #                                          title=title + ', With Canting', caption=st.add_to_last_sentence(caption, ' with canting'), comments=comments, code_tag=self.code_tag)
        # h1.draw(fig_record.view, heliostat_control)
        # self.show_save_and_check_figure(fig_record)

        # Tracking heliostat.

        h1.set_canting_from_equation(fn)
        sf.set_full_field_tracking(aimpoint_xyz, when_ymdhmsz)
        comments.append("Heliostats set to track to " + str(aimpoint_xyz) + " at ymdhmsz =" + str(when_ymdhmsz))
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(27),
            title=title + ", With Canting and Tracking",
            caption=st.add_to_last_sentence(caption, " with canting and tracking"),
            comments=comments,
            code_tag=self.code_tag,
        )
        h1.draw(fig_record.view, heliostat_control)
        self.show_save_and_check_figure(fig_record)


# MAIN EXECUTION
if __name__ == "__main__":

    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False
    # Setup.
    test_object = TestMirrorOutput()
    test_object.setUpClass(interactive=interactive, verify=verify)
    test_object.setUp()  # Tests.

    lt.info("Beginning tests...")
    test_object.test_mirror_halfpi_rotation()
    test_object.test_facet()
    test_object.test_solar_field()
    test_object.test_heliostat_05W01_and_14W01()
    test_object.test_heliostat_stages()
    lt.info("All tests complete.")
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
