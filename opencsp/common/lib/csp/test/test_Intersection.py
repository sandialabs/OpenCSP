"""Unit test to test the MirrorPoint class"""

import numpy as np
import unittest


from opencsp.common.lib.csp.LightPathEnsemble import LightPathEnsemble
from opencsp.common.lib.csp.LightSourcePoint import LightSourcePoint
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.RayTrace import RayTrace
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Intersection import Intersection
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.csp.RayTrace as rt


class TestIntersection(unittest.TestCase):
    """Test class for testing Intersection class. MirrorParametric and RayTrac must also be working."""

    def test_rays_intersect_properly(self):
        """two different intersection planes should
        should prduce the same intersection point for a single ray
        that passes through an intersection point on the plane."""

        UP = Vxyz([0, 0, 1])
        WEST = Vxyz([-1, 0, 0])
        plane1 = (Pxyz.origin(), UP)
        plane2 = (Pxyz.origin(), WEST)

        light_ray = LightPathEnsemble.from_parts(
            init_directions=Vxyz([1, 0, 1]).normalize(),
            points=[Pxyz([-1, 0, -1])],
            curr_directions=Vxyz([1, 0, 1]).normalize(),
        )

        raytrace = RayTrace()
        raytrace.light_paths_ensemble = light_ray

        intersection1 = Intersection.plane_intersect_from_ray_trace(raytrace, plane1)
        intersection2 = Intersection.plane_intersect_from_ray_trace(raytrace, plane2)

        # Test
        np.testing.assert_array_almost_equal(
            intersection1.intersection_points.data, intersection2.intersection_points.data
        )

    def test_parallel_reflection(self):
        """Test to see if the intersection properly finds rays that
        should be parallel."""

        # Define mirror
        mirror = MirrorParametricRectangular.from_focal_length(2, (1, 1))

        # Define plane
        plane = (Pxyz([0, 0, 2]), Vxyz([0, 0, -1]))

        # Make scene and light
        light = LightSourcePoint(Pxyz([0, 0, 2]))

        scene = Scene()
        scene.add_object(mirror)
        scene.add_light_source(light)

        # Trace
        resolution = Resolution.separation(0.5)

        trace = rt.trace_scene(scene, obj_resolution=resolution)
        intersection = Intersection.plane_intersect_from_ray_trace(trace, plane)

        # Define Expectation
        # we expect all the rays intersections to be the same x and y as the mirror points
        expected_points, expected_directions = mirror.survey_of_points(resolution)

        # Test
        np.testing.assert_array_almost_equal(intersection.intersection_points.x, expected_points.x)
        np.testing.assert_array_almost_equal(intersection.intersection_points.y, expected_points.y)

    def test_converging_reflection(self):
        """Test to see if the intersection properly finds rays that
        should be converge to a point."""

        # Define mirror
        mirror = MirrorParametricRectangular.from_focal_length(2, (1, 1))

        # Define plane
        plane = (Pxyz([0, 0, 2]), Vxyz([0, 0, -1]))

        # Make scene and light
        light = LightSourceSun.from_given_sun_position(Vxyz([0, 0, -1]), 1)

        scene = Scene()
        scene.add_object(mirror)
        scene.add_light_source(light)

        # Trace
        resolution = Resolution.separation(0.5)

        trace = rt.trace_scene(scene, obj_resolution=resolution)
        intersection = Intersection.plane_intersect_from_ray_trace(trace, plane)

        # Define Expectation
        # we expect all points to converge at (0,0,2)
        expected_points = Pxyz([[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2]])

        # Test
        np.testing.assert_array_almost_equal(intersection.intersection_points._data, expected_points.data)

    def test_bad_params_raises_error(self):
        """Make sure that passing in bad parameters causes an exception"""
        # too many values for plane
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 0])
        plane_point = Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))
        plane_normal = Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

        # unmatched dimensions for plane
        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection((lines_points, lines_vecs), (Pxyz([0, 0, 0]), plane_normal))
        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, Pxyz([0, 0, 0])))

        # unmatched dimensions for lines
        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection(
                (Pxyz(np.array([[0, 0], [0, 0], [0, 0]])), lines_vecs), (plane_point, plane_normal)
            )
        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection(
                (lines_points, Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))), (plane_point, plane_normal)
            )

    def test_parallel_line_plane(self):
        """Raise an exception when the line and the plane are parallel"""
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([1, 1, 1])
        plane_normal = Uxyz([0, 1, 0])

        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

    def test_parallel_coincident_line_plane(self):
        """Raise an exception when the line and the plane are parallel, including the coincident edge case."""
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([0, 0, 0])
        plane_normal = Pxyz([0, 1, 0])

        with self.assertRaises(ValueError):
            Intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

    def test_x_plane(self):
        """Verify that we can find intersections for a plane with a normal == x-axis"""
        lines_points = Pxyz([1, 0, 1])
        lines_vecs = Vxyz([1, 0, 0])
        plane_point = Pxyz([0, 0, 0])
        plane_normal = Pxyz([1, 0, 0])

        intersection_points = Intersection.plane_lines_intersection(
            (lines_points, lines_vecs), (plane_point, plane_normal)
        )
        self.assertEqual(1.0, intersection_points.z)

    def test_y_plane(self):
        """Verify that we can find intersections for a plane with a normal == y-axis"""
        lines_points = Pxyz([1, 1, 0])
        lines_vecs = Vxyz([0, 1, 0])
        plane_point = Pxyz([0, 0, 0])
        plane_normal = Pxyz([0, 1, 0])

        intersection_points = Intersection.plane_lines_intersection(
            (lines_points, lines_vecs), (plane_point, plane_normal)
        )
        self.assertEqual(1.0, intersection_points.x)

    def test_z_plane(self):
        """Verify that we can find intersections for a plane with a normal == z-axis"""
        lines_points = Pxyz([0, 1, 1])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([0, 0, 0])
        plane_normal = Pxyz([0, 0, 1])

        intersection_points = Intersection.plane_lines_intersection(
            (lines_points, lines_vecs), (plane_point, plane_normal)
        )
        self.assertEqual(1.0, intersection_points.y)

    def test_multiple_lines(self):
        """Test with two lines and with three lines. We should get the same number of intersection points as the number of lines."""
        lines_points = Pxyz([[-1, 0, 1], [0, 0, 0], [1, 1, 1]])
        lines_vecs = Vxyz([[1, 0, 0], [0, 1, 0], [-1, -1, -1]])
        plane_point = Pxyz([0, 0, 0])
        plane_normal = Pxyz([0, 0, 1])

        intersection_points = Intersection.plane_lines_intersection(
            (lines_points, lines_vecs), (plane_point, plane_normal)
        )
        self.assertEqual(3, intersection_points.data.shape[1])


if __name__ == '__main__':
    Test = TestIntersection()
    Test.test_rays_intersect_properly()
    Test.test_parallel_reflection()
    Test.test_converging_reflection()
    unittest.main()
