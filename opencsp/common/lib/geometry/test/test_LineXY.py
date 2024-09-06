import numpy as np
import unittest

from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.Vxy import Vxy


class TestLineXY(unittest.TestCase):
    def test_ABC(self):
        ABC = np.array([1, 2, 3])
        line = LineXY(*ABC)
        ABC_exp = ABC / np.sqrt(np.sum(ABC[:2] ** 2))
        np.testing.assert_equal(line.ABC, ABC_exp)

    def test_n_vec(self):
        line = LineXY(1, 2, 3)
        n_vec = Vxy((1, 2)).normalize()
        np.testing.assert_almost_equal(line.n_vec.data, n_vec.data)

    def test_from_points(self):
        pts = Vxy(
            [np.arange(16), [1.1, 2.4, 3.1, 4.6, 5.2, 6.8, 7.3, 8.7, 9.9, 10.3, 11.5, 12.1, 13.8, 14.4, 15.0, 16.5]]
        )

        with np.testing.assert_raises(ValueError):
            LineXY.fit_from_points(pts[:5])

        # Test fitting a line
        line = LineXY.fit_from_points(pts)
        ABC_exp = np.array([0.7097406177, -0.7044630974, 0.9598756044])
        np.testing.assert_almost_equal(line.ABC, ABC_exp)

    def test_from_two_points(self):
        pt1 = Vxy((1, 0))
        pt2 = Vxy((0, 1))
        line = LineXY.from_two_points(pt1, pt2)
        ABC_exp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)])
        np.testing.assert_almost_equal(line.ABC, ABC_exp)

    def test_from_two_points_edge_case(self):
        pth1 = Vxy((0, 1))
        pth2 = Vxy((2, 1))
        ptv1 = Vxy((1, 0))
        ptv2 = Vxy((1, 2))

        # horizontal line
        # y = (-Ax - C) / B
        lineh = LineXY.from_two_points(pth1, pth2)
        self.assertAlmostEqual(-lineh.C / lineh.B, 1)
        self.assertAlmostEqual(lineh.A, 0)

        # vertical line
        # x = (-By - C) / A
        linev = LineXY.from_two_points(ptv1, ptv2)
        self.assertAlmostEqual(-linev.C / linev.A, 1)
        self.assertAlmostEqual(linev.B, 0)

    def test_from_rho_theta(self):
        # vertical line
        l1 = LineXY.from_rho_theta(1, 0)
        self.assertAlmostEqual(l1.x_from_y(-1), 1)
        self.assertAlmostEqual(l1.x_from_y(1), 1)

        # 45-degree downward slope
        l2 = LineXY.from_rho_theta(np.sqrt(2) / 2, np.pi / 4)
        self.assertAlmostEqual(l2.y_from_x(0), 1)
        self.assertAlmostEqual(l2.y_from_x(1), 0)

        # horizontal line
        l3 = LineXY.from_rho_theta(1, np.pi / 2)
        self.assertAlmostEqual(l3.y_from_x(-1), 1)
        self.assertAlmostEqual(l3.y_from_x(1), 1)

    def test_from_location_angle(self):
        # horizontal line
        l1 = LineXY.from_location_angle(Vxy([1, 1]), 0)
        self.assertAlmostEqual(l1.y_from_x(-1), 1)
        self.assertAlmostEqual(l1.y_from_x(2), 1)

        # 45-degree upward slope
        l2 = LineXY.from_location_angle(Vxy([1, 1]), np.pi / 4)
        self.assertAlmostEqual(l2.y_from_x(0), 0)
        self.assertAlmostEqual(l2.y_from_x(2), 2)

        # vertical line
        l3 = LineXY.from_location_angle(Vxy([1, 1]), np.pi / 2)
        self.assertAlmostEqual(l3.x_from_y(-1), 1)
        self.assertAlmostEqual(l3.x_from_y(2), 1)

    def test_y_from_x(self):
        # Line y = -x
        line = LineXY(1, 1, 0)
        xs = np.arange(-10, 10)
        ys = line.y_from_x(xs)
        ys_exp = -xs
        np.testing.assert_almost_equal(ys, ys_exp)

        # Horizontal line
        line = LineXY(0, 1, -5)
        xs = np.arange(-10, 10)
        ys = line.y_from_x(xs)
        ys_exp = np.ones(ys.size) * 5
        np.testing.assert_almost_equal(ys, ys_exp)

    def test_y_from_x_return_type(self):
        # Line y = -x
        line = LineXY(1, 1, 0)

        # test that providing a numpy array also returns a numpy array
        xs = np.arange(-10, 10)
        ys = line.y_from_x(xs)
        ys_exp = -xs
        self.assertTrue(isinstance(ys, np.ndarray))
        np.testing.assert_almost_equal(ys, ys_exp)

        # test that providing a float also returns a float
        x = -10
        y = line.y_from_x(x)
        y_exp = -x
        self.assertTrue(isinstance(y, float))
        np.testing.assert_almost_equal(y, y_exp)

    def test_x_from_y(self):
        # Line y = -x
        line = LineXY(1, 1, 0)
        ys = np.arange(-10, 10)
        xs = line.x_from_y(ys)
        xs_exp = -ys
        np.testing.assert_almost_equal(xs, xs_exp)

        # Vertical line
        line = LineXY(1, 0, -5)
        ys = np.arange(-10, 10)
        xs = line.x_from_y(ys)
        xs_exp = np.ones(xs.size) * 5
        np.testing.assert_almost_equal(xs, xs_exp)

    def test_dist_from(self):
        # Line y = -x
        line = LineXY(1, 1, 0)
        pts = Vxy(([-1, 0, 1], [-1, 0, 1]))
        dists = line.dist_from(pts)
        dists_exp = np.array([np.sqrt(2), 0, np.sqrt(2)])
        np.testing.assert_almost_equal(dists, dists_exp)

        # Horizontal line
        line = LineXY(0, 1, -5)
        pts = Vxy(([0, 0, 0, 0, 0], [7, 6, 5, 4, 3]))
        dists = line.dist_from(pts)
        dists_exp = np.array([2, 1, 0, 1, 2])
        np.testing.assert_almost_equal(dists, dists_exp)

    def test_dist_from_signed(self):
        # Line y = -x
        line = LineXY(1, 1, 0)
        pts = Vxy(([-1, 0, 1], [-1, 0, 1]))
        dists = line.dist_from_signed(pts)
        dists_exp = np.array([-np.sqrt(2), 0, np.sqrt(2)])
        np.testing.assert_almost_equal(dists, dists_exp)

        # Horizontal line
        line = LineXY(0, 1, -5)
        pts = Vxy(([0, 0, 0, 0, 0], [7, 6, 5, 4, 3]))
        dists = line.dist_from_signed(pts)
        dists_exp = np.array([2, 1, 0, -1, -2])
        np.testing.assert_almost_equal(dists, dists_exp)

    def test_intersect(self):
        # x = (-By - C) / A
        # y = (-Ax - C) / B
        line_h = LineXY(0, 1, -1)  # horizontal line through y=1
        line_v = LineXY(1, 0, -1)  # vertical line through x=1

        p1 = line_h.intersect_with(line_v)
        p2 = line_v.intersect_with(line_h)

        self.assertAlmostEqual(p1.x, 1.0)
        self.assertAlmostEqual(p1.y, 1.0)
        self.assertAlmostEqual(p2.x, 1.0)
        self.assertAlmostEqual(p2.y, 1.0)

    def intersect_with(self):
        line1 = LineXY(0, 1, -5)
        line2 = LineXY(1, 0, -5)

        # Line 1 with line 2
        pt = line1.intersect_with(line2)
        pt_exp = np.array([[5], [5]])
        np.testing.assert_almost_equal(pt.data, pt_exp)

        # Line 2 with line 1
        pt = line2.intersect_with(line1)
        pt_exp = np.array([[5], [5]])
        np.testing.assert_almost_equal(pt.data, pt_exp)

    def flip(self):
        ABC = np.array([1, 2, 3])
        line = LineXY(*ABC.copy())
        line_flip = line.flip()
        np.testing.assert_almost_equal(line_flip.ABC, -ABC)

    def flip_in_place(self):
        ABC = np.array([1, 2, 3])
        line = LineXY(*ABC.copy())
        line.flip()
        np.testing.assert_almost_equal(line.ABC, -ABC)

    def test_slope(self):
        # flat line to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, 0)))
        self.assertAlmostEqual(line.slope, 0)

        # flat line to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, 0)))
        self.assertAlmostEqual(line.slope, 0)

        # vertical line up
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((0, 1)))
        self.assertTrue(np.isinf(line.slope) and not np.isneginf(line.slope))

        # vertical line down
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((0, -1)))
        self.assertTrue(np.isneginf(line.slope))

        # 45-degree, up and to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, 1)))
        self.assertAlmostEqual(line.slope, 1)

        # 135-degree, up and to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, 1)))
        self.assertAlmostEqual(line.slope, -1)

        # 225-degree, down and to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, -1)))
        self.assertAlmostEqual(line.slope, 1)

        # 315-degree, down and to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, -1)))
        self.assertAlmostEqual(line.slope, -1)

    def test_angle(self):
        # flat line to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, 0)))
        self.assertAlmostEqual(line.angle, 0)

        # flat line to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, 0)))
        self.assertAlmostEqual(line.angle, np.pi)

        # vertical line up
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((0, 1)))
        self.assertAlmostEqual(line.angle, np.pi / 2)

        # vertical line down
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((0, -1)))
        self.assertAlmostEqual(line.angle, np.pi * 3 / 2)

        # 45-degree, up and to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, 1)))
        self.assertAlmostEqual(line.angle, np.pi / 4)

        # 135-degree, up and to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, 1)))
        self.assertAlmostEqual(line.angle, np.pi * 3 / 4)

        # 225-degree, down and to the left
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((-1, -1)))
        self.assertAlmostEqual(line.angle, np.pi * 5 / 4)

        # 315-degree, down and to the right
        line = LineXY.from_two_points(Vxy((0, 0)), Vxy((1, -1)))
        self.assertAlmostEqual(line.angle, np.pi * 7 / 4)


if __name__ == '__main__':
    unittest.main()
