"""
Test generation and rendering of ray trace outputs on opencsp mirrors.
"""

import copy
import logging
import math
import os
import time
from cmath import sin
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation

import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as st
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
from opencsp.common.lib.render_control.RenderControlEnsemble import RenderControlEnsemble
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface


class TestRayTraceOutput(to.TestOutput):

    @classmethod
    def setUpClass(
        cls,
        source_file_body: str = "TestRayTraceOutput",  # Set these here, because pytest calls
        figure_prefix_root: str = "trto",  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):
        # Generic setup.
        super(TestRayTraceOutput, cls).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
        )

    def setUp(self):
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

        # Facet, based on a parameteric mirror.
        self.f1 = Facet(self.m1, "1")
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
        self.h2x2_f1 = Facet(copy.deepcopy(self.m1), "1")
        self.h2x2_f2 = Facet(copy.deepcopy(self.m1), "2")
        self.h2x2_f3 = Facet(copy.deepcopy(self.m1), "3")
        self.h2x2_f4 = Facet(copy.deepcopy(self.m1), "4")

        # Set canting angles.
        cos5 = np.cos(np.deg2rad(8))
        sin5 = np.sin(np.deg2rad(8))
        tilt_up = Rotation.from_matrix(np.asarray([[1, 0, 0], [0, cos5, -sin5], [0, sin5, cos5]]))
        tilt_down = Rotation.from_matrix(np.asarray([[1, 0, 0], [0, cos5, sin5], [0, -sin5, cos5]]))
        tilt_left = Rotation.from_matrix(np.asarray([[cos5, 0, sin5], [0, 1, 0], [-sin5, 0, cos5]]))
        tilt_right = Rotation.from_matrix(np.asarray([[cos5, 0, -sin5], [0, 1, 0], [sin5, 0, cos5]]))
        self.h2x2_canting = [tilt_left * tilt_up, tilt_right * tilt_up, tilt_left * tilt_down, tilt_right * tilt_down]
        # self.h2x2_f2.canting = tilt_right * tilt_up
        # self.h2x2_f3.canting = tilt_left * tilt_down
        # self.h2x2_f4.canting = tilt_right * tilt_down
        self.h2x2_facets = [self.h2x2_f1, self.h2x2_f2, self.h2x2_f3, self.h2x2_f4]
        self.fe2x2 = FacetEnsemble(self.h2x2_facets)
        fe2x2_positions = Pxyz([[-1.1, 1.1, -1.1, 1.1], [1.6, 1.6, -1.6, -1.6], [0, 0, 0, 0]])
        self.fe2x2.set_facet_positions(fe2x2_positions)
        self.fe2x2.set_facet_cantings(self.h2x2_canting)
        self.h2x2 = HeliostatAzEl(self.fe2x2, "Simple 2x2 Heliostat")
        self.h2x2.pivot = 0
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
        # self.sf2x2_h1 = HeliostatAzEl('Heliostat 1', [0, 0, 0], 4, 2, 2, copy.deepcopy(self.h2x2_facets), 4.02, 0.1778)
        # self.sf2x2_h2 = HeliostatAzEl('Heliostat 2', [0, 10, 0], 4, 2, 2, copy.deepcopy(self.h2x2_facets), 4.02, 0.1778)
        self.sf2x2_h1 = HeliostatAzEl(copy.deepcopy(self.fe2x2), "Heliostat 1")
        self.sf2x2_h2 = HeliostatAzEl(copy.deepcopy(self.fe2x2), "Heliostat 2")
        self.sf2x2_heliostats = [self.sf2x2_h1, self.sf2x2_h2]
        self.sf2x2 = SolarField(self.sf2x2_heliostats, [-106.509606, 34.962276], "Test Field", "test")
        h1_pos = Pxyz([0, 0, 0])
        h2_pos = Pxyz([0, 10, 0])
        self.sf2x2.set_heliostat_positions(Pxyz.merge([h1_pos, h2_pos]))
        self.sf2x2_title = "Two Heliostats"
        self.sf2x2_caption = "Two 4-facet heliostats, tracking."
        self.sf2x2_comments = []

    def lambda_symmetric_paraboloid(self, focal_length: float) -> Callable[[float, float], float]:
        """
        Helper function that makes lambdas of paraboloids of a given focal length.
        """
        a = 1.0 / (4 * focal_length)
        return lambda x, y: a * (x**2 + y**2)

    # draw simple rays
    def test_draw_simple_ray(self) -> None:

        self.start_test()

        local_comments = self.m1_comments
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(1),
            title="draw_simple_light_path",
            caption=self.f1_caption,
            comments=local_comments,
            code_tag=self.code_tag,
        )

        view = fig_record.view

        points = Pxyz(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 1, 0, 3]]
        )  # [np.array([0,0,0]),np.array([0,0,1]),np.array([0,2,0]),np.array([0,0,3])]
        normal_vector = Uxyz([0, 0, 1])
        incoming_vector = Vxyz([0, 1, -1])
        ref_vec = rt.calc_reflected_ray(normal_vector, incoming_vector)
        ray = LightPath(points, incoming_vector, ref_vec)
        light_path_control = RenderControlLightPath(line_render_control=rcps.RenderControlPointSeq(color="y"))
        ray.draw(view, light_path_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

    # demonstration of ray tracing a single mirror
    def test_mirror_trace(self) -> None:

        self.start_test()

        local_comments = []
        fn = self.lambda_symmetric_paraboloid(2)
        m1 = MirrorParametricRectangular(fn, (2, 2))

        rot_id = Rotation.identity()
        rot_45_deg = Rotation.from_euler("x", 45, degrees=True)

        light_path_control = RenderControlLightPath(current_length=4)
        mirror_control = rcm.RenderControlMirror()

        ls = LightSourceSun()

        # Face Up, Parallel Beams 3d

        ls.incident_rays = LightPath.many_rays_from_many_vectors(None, Vxyz([0, 0, -1]))  # straight down

        scene1 = Scene()
        scene1.add_light_source(ls)
        scene1.add_object(m1)

        trans1 = TransformXYZ.from_R(rot_id)
        scene1.set_position_in_space(m1, trans1)

        trace1 = rt.trace_scene(scene1, Resolution.pixelX(8))

        # Face Up, Parallel Beams yz

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(3),
            title="mirror_facing_up_side_view",
            caption="mirror_facing_up_side_view",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view1_yz = fig_record.view

        trace1.draw(view1_yz, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view1_yz, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # Many rays at a 45 degree angle

        test_vecs = Uxyz([[0, 0, 0, 0.1, -0.1], [0, 0.1, -0.1, 0, 0], [-1, -1, -1, -1, -1]])

        ls.incident_rays = LightPath.many_rays_from_many_vectors(None, test_vecs.rotate(rot_45_deg))

        trans4 = TransformXYZ.from_R(rot_45_deg)
        scene1.set_position_in_space(m1, trans4)

        trace4 = rt.trace_scene(scene1, Resolution.pixelX(8))

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(8),
            title="mirror_tilted_many_rays",
            caption="mirror_tilted_many_rays",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view4 = fig_record.view

        trace4.draw(view4, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view4, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        return

    # demonstration of ray tracing a single heliostat
    def test_heliostat_trace(self) -> None:

        self.start_test()

        def curved_func(x, y):
            return x**2 / 40 + y**2 / 40

        m_curved = MirrorParametricRectangular(curved_func, (1.2192, 1.2192))

        loc = (-106.509606, 34.962276)

        local_comments = []

        h_curved, h_curved_loc = HeliostatAzEl.from_csv_files(
            "5W1",
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            m_curved,
        )
        h_canted = copy.deepcopy(h_curved)
        sf_curved = sf.SolarField([h_curved], loc, "mini Nsttf with 5W1 and 14W1", "mini Field")
        sf_curved.set_heliostat_positions([h_curved_loc])

        h_canted.set_canting_from_equation(curved_func)
        sf_canted = sf.SolarField([h_canted], loc, "mini Nsttf with 5W1 and 14W1", "mini Field")
        sf_canted.set_heliostat_positions([h_curved_loc])

        mirror_control = rcm.RenderControlMirror(
            surface_normals=False, norm_len=8, norm_res=2, resolution=Resolution.pixelX(3)
        )
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control,
            draw_outline=True,
            draw_name=False,
            draw_centroid=False,
            draw_surface_normal=False,
        )
        fe_control = rcfe.RenderControlFacetEnsemble(default_style=facet_control)
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=True, draw_facet_ensemble=True, facet_ensemble_style=fe_control
        )
        solar_field_style = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control)

        h_curved.set_orientation_from_az_el(0, np.pi / 2)
        h_canted.set_orientation_from_az_el(0, np.pi / 2)

        # RAY TRACING

        # set of inc vectors to test
        test_vecs = Uxyz([[0, 0, 0, 0.1, -0.1], [0, 0.1, -0.1, 0, 0], [-1, -1, -1, -1, -1]])

        sun = LightSourceSun()
        # sun.set_incident_rays(loc, when_ymdhmsz, 3)
        sun.incident_rays = LightPath.many_rays_from_many_vectors(None, test_vecs)

        path_control = RenderControlLightPath(current_length=50, init_length=4)
        trace_control = RenderControlRayTrace(light_path_control=path_control)

        # CURVED

        scene1 = Scene()
        scene1.add_object(sf_curved)
        scene1.add_light_source(sun)
        trace1 = rt.trace_scene(scene1, obj_resolution=Resolution.pixelX(10))

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(13),
            title="curved_heliostat_trace",
            caption="curved_heliostat_trace",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view1_yz = fig_record.view
        sf_curved.draw(view1_yz, solar_field_style)
        trace1.draw(view1_yz, trace_control)
        self.show_save_and_check_figure(fig_record)

        # CANTED

        scene2 = Scene()
        scene2.add_object(sf_canted)
        scene2.add_light_source(sun)
        trace2 = rt.trace_scene(scene2, obj_resolution=Resolution.pixelX(10))

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(15),
            title="canted_heliostat_trace",
            caption="canted_heliostat_trace",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view2_yz = fig_record.view
        sf_canted.draw(view2_yz, solar_field_style)
        trace2.draw(view2_yz, trace_control)
        self.show_save_and_check_figure(fig_record)

        return

    def test_changing_time_of_day(self) -> None:
        # create a figure that shows 5w1 reflecting the sun towards an aimpoint -- TODO TJL:sun rays coming from wrong direction
        def _heliostat_at_moment(name: str, aimpoint_xyz: Pxyz, when_ymdhmsz: tuple, i: int) -> None:

            self.start_test()

            local_comments = []

            def fn_5w1(x, y):
                return x**2 / (4 * 65) + y**2 / (4 * 65)

            m_5w1 = MirrorParametricRectangular(fn_5w1, (1.2192, 1.2192))

            h_05w01, location_05w01 = HeliostatAzEl.from_csv_files(
                "5W1",
                dpft.sandia_nsttf_test_heliostats_origin_file(),
                dpft.sandia_nsttf_test_facet_centroidsfile(),
                m_5w1,
            )

            h_05w01.set_canting_from_equation(fn_5w1)

            heliostats = [h_05w01]

            sf1 = sf.SolarField(heliostats, lln.NSTTF_ORIGIN, "mini Nsttf with 5W1", "mini Field")
            sf1.set_heliostat_positions(location_05w01)

            heliostat_control = rch.facet_outlines()
            solar_field_style = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control)

            sf1.set_full_field_tracking(aimpoint_xyz, when_ymdhmsz)

            # light source
            sun = LightSourceSun()
            sun.set_incident_rays(lln.NSTTF_ORIGIN, when_ymdhmsz, 3)

            # make scene
            scene = Scene()
            scene.add_object(sf1)
            scene.add_light_source(sun)

            # drawing
            path_control = RenderControlLightPath(current_length=100, init_length=20)
            trace_control = RenderControlRayTrace(light_path_control=path_control)
            trace = rt.trace_scene(scene, obj_resolution=Resolution.center())

            def _draw_helper(view: View3d) -> None:
                sf1.draw(view, solar_field_style)
                trace.draw(view, trace_control)
                aimpoint_xyz.draw_points(view, style=rcps.marker(color="tab:orange"))

                # debug
                heliostat_origin = sf1.heliostats[0].self_to_global_tranformation.apply(Pxyz.origin())
                pointing_vector = st.tracking_surface_normal_xyz(
                    heliostat_origin, aimpoint_xyz, lln.NSTTF_ORIGIN, when_ymdhmsz
                )
                Vxyz.merge([heliostat_origin, heliostat_origin + pointing_vector * 10]).draw_line(view)
                # debug

                self.show_save_and_check_figure(fig_record)

            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_3d(),
                # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                number_in_name=False,
                input_prefix=self.figure_prefix(i),
                title=name,
                caption=f"3d view of the ratrace of {name}",
                comments=local_comments,
                code_tag=self.code_tag,
            )
            view_3d = fig_record.view
            _draw_helper(view_3d)

            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_xz(),
                # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                number_in_name=False,
                input_prefix=self.figure_prefix(i + 1),
                title=name,
                caption=f"xz view of the ratrace of {name}",
                comments=local_comments,
                code_tag=self.code_tag,
            )
            view_xz = fig_record.view
            _draw_helper(view_xz)

        # _heliostat_at_moment("5w1 at 11:02", Pxyz([60.0, 8.8, 28.9]), (2021, 5, 13, 11, 2, 0, -6), 16)
        # _heliostat_at_moment("5w1 at 12:02", [60.0,8.8,28.9], (2021,5,13,12,2,0,-6), 18)
        # _heliostat_at_moment("5w1 at 13:02", [60.0,8.8,28.9], (2021,5,13,13,2,0,-6), 20)
        # _heliostat_at_moment("5w1 at 14:02", [60.0,8.8,28.9], (2021,5,13,14,2,0,-6), 22)
        _heliostat_at_moment("5w1 at 15:02", Pxyz([60.0, 8.8, 28.9]), (2021, 5, 13, 15, 2, 0, -6), 24)
        # test__changing-time_of_day_helper("5w1 at solar noon, pointing at origin",[0,0,28.9], (2021,5,13,13,2,0,-6), figure_control, axis_control_m)
        return

    # partial field test
    def test_partial_field_trace(self) -> None:

        self.start_test()

        local_comments = []
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(26),
            title="partial_field_trace",
            caption="A partial trace of the NSTTF field.",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view = fig_record.view

        heliostat_list = [
            "5E10",
            "5E9",
            "5E8",
            "5E7",
            "5E6",
            "5E5",
            "5E4",
            "5E3",
            "5E2",
            "5E1",
            "5W1",
            "5W2",
            "5W3",
            "5W4",
            "5W5",
        ]

        # SOLAR FIELD SETUP
        aimpoint_xyz = Pxyz([0, 0, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # solar noon

        solar_field: sf.SolarField = sf.SolarField.from_csv_files(
            lln.NSTTF_ORIGIN,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
        )

        # Tracking setup
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)

        # Comment
        fig_record.comments.append("Partial Solar Field Trace.")
        fig_record.comments.append("Using 1 ray per surface normal and one surface normal per mirror.")
        fig_record.comments.append(
            "Mirror curvature and canting is defined per heliostat. They are both parabolic and have focal lengths based on the distance of the heliostat to the tower."
        )
        fig_record.comments.append("Traces one in every 13 heliostats in the NSTTF field.")

        # Draw
        fe_control = rcfe.facet_ensemble_outline()
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=fe_control, draw_facet_ensemble=True
        )

        sun = LightSourceSun()
        sun.set_incident_rays(lln.NSTTF_ORIGIN, when_ymdhmsz, 1)

        scene = Scene()
        scene.add_light_source(sun)

        for h_name in heliostat_list:
            h = solar_field.lookup_heliostat(h_name).no_parent_copy()
            h_loc = solar_field.lookup_heliostat(h_name)._self_to_parent_transform
            scene.add_object(h)
            scene.set_position_in_space(h, h_loc)
            h.draw(view, heliostat_control)

        trace = rt.trace_scene(scene, Resolution.center(), verbose=False)
        trace.draw(view, RenderControlRayTrace(RenderControlLightPath(15, 200)))

        aimpoint_xyz.draw_points(view, rcps.RenderControlPointSeq(color="orange", marker="."))

        self.show_save_and_check_figure(fig_record)


# MAIN EXECUTION

if __name__ == "__main__":
    # Control flags.
    interactive = True
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False
    # Setup.
    test_object = TestRayTraceOutput()
    test_object.setUpClass(interactive=interactive, verify=verify)
    test_object.setUp()
    # Tests.
    lt.info("Beginning tests...")
    test_object.test_draw_simple_ray()
    test_object.test_mirror_trace()
    test_object.test_heliostat_trace()
    test_object.test_changing_time_of_day()
    test_object.test_partial_field_trace()

    lt.info("All tests complete.")
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
