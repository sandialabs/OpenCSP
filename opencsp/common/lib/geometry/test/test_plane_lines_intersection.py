import numpy as np
import unittest

from opencsp.common.lib.geometry import plane_lines_intersection as intersection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz



class test_planes_line_intersection(unittest.TestCase):
    def test_bad_params_raises_error(self):
        """Make sure that passing in bad parameters causes an exception"""
        # too many values for plane
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 0])
        plane_point = Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))
        plane_normal = Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

        # unmatched dimensions for plane
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((lines_points, lines_vecs), (Pxyz([0, 0, 0]), plane_normal))
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, Pxyz([0, 0, 0])))

        # unmatched dimensions for lines
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((Pxyz(np.array([[0, 0], [0, 0], [0, 0]])),
                                     lines_vecs), (plane_point, plane_normal))
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection(
                (lines_points, Pxyz(np.array([[0, 0], [0, 0], [0, 0]]))), (plane_point, plane_normal))


    def test_parallel_line_plane(self):
        """Raise an exception when the line and the plane are parallel"""
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([1,1,1])
        plane_normal = Uxyz([0,1,0])
        
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

    def test_parallel_coincident_line_plane(self):
        """Raise an exception when the line and the plane are parallel, including the coincident edge case."""
        lines_points = Pxyz([0, 0, 0])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([0,0,0])
        plane_normal = Pxyz([0,1,0])
        
        with self.assertRaises(ValueError):
            intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))

    def test_x_plane(self):
        """Verify that we can find intersections for a plane with a normal == x-axis"""
        lines_points = Pxyz([1,0,1])
        lines_vecs = Vxyz([1, 0, 0])
        plane_point = Pxyz([0,0,0])
        plane_normal = Pxyz([1,0,0])
        
        intersection_points = intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))
        self.assertEqual(1., intersection_points.z)

    def test_y_plane(self):
        """Verify that we can find intersections for a plane with a normal == y-axis"""
        lines_points = Pxyz([1,1,0])
        lines_vecs = Vxyz([0, 1, 0])
        plane_point = Pxyz([0,0,0])
        plane_normal = Pxyz([0,1,0])
        
        intersection_points = intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))
        self.assertEqual(1., intersection_points.x)

    def test_z_plane(self):
        """Verify that we can find intersections for a plane with a normal == z-axis"""
        lines_points = Pxyz([0,1,1])
        lines_vecs = Vxyz([0, 0, 1])
        plane_point = Pxyz([0,0,0])
        plane_normal = Pxyz([0,0,1])
        
        intersection_points = intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))
        self.assertEqual(1., intersection_points.y)

    def test_multiple_lines(self):
        """Test with two lines and with three lines. We should get the same number of intersection points as the number of lines."""
        lines_points = Pxyz([[-1,0,1],[0, 0, 0], [1,1,1]])
        lines_vecs = Vxyz([[1,0,0],[0,1,0],[-1,-1,-1]])
        plane_point = Pxyz([0,0,0])
        plane_normal = Pxyz([0,0,1])

        intersection_points = intersection.plane_lines_intersection((lines_points, lines_vecs), (plane_point, plane_normal))
        self.assertEqual(3, intersection_points.data.shape[1])


if __name__ == '__main__':
    unittest.main()
