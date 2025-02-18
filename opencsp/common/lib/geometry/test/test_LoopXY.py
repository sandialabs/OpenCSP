import matplotlib.pyplot as plt
import numpy as np
import unittest

from opencsp.common.lib.geometry.EdgeXY import EdgeXY
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.render.figure_management as fm


class TestLoopXY(unittest.TestCase):
    def test_is_closed(self):
        # Closed
        e1 = EdgeXY(Vxy(([0, 1], [0, 0])))
        e2 = EdgeXY(Vxy(([1, 1], [0, 1])))
        e3 = EdgeXY(Vxy(([1, 0], [1, 0])))
        LoopXY([e1, e2, e3])

        # Open
        e1 = EdgeXY(Vxy(([0, 1], [0, 0])))
        e2 = EdgeXY(Vxy(([1, 1], [0, 1])))
        e3 = EdgeXY(Vxy(([1, 0], [1, 1])))
        with np.testing.assert_raises(ValueError):
            LoopXY([e1, e2, e3])

    def test_is_convex(self):
        # Non-convex
        e1 = EdgeXY(Vxy(([0, 2], [0, 0])))
        e2 = EdgeXY(Vxy(([2, 0], [0, 2])))
        e3 = EdgeXY(Vxy(([0, 1], [2, 1])))
        e4 = EdgeXY(Vxy(([1, 0], [1, 0])))
        with np.testing.assert_raises(ValueError):
            LoopXY([e1, e2, e3, e4])

    def test_from_lines(self):
        v1 = Vxy((0, 0))
        v2 = Vxy((1, 0))
        v3 = Vxy((1, 1))
        v4 = Vxy((0, 1))

        l1 = LineXY.from_two_points(v1, v2)
        l2 = LineXY.from_two_points(v2, v3)
        l3 = LineXY.from_two_points(v3, v4)
        l4 = LineXY.from_two_points(v4, v1)

        loop = LoopXY.from_lines([l1, l2, l3, l4])
        vert_exp = np.array(([1, 0, 0, 1], [1, 1, 0, 0]), dtype=float)

        np.testing.assert_allclose(loop.vertices.data, vert_exp)

    def test_from_vertices(self):
        verts = Vxy(([1, 0, 0, 1], [1, 0, 1, 0]))

        loop = LoopXY.from_vertices(verts)
        vert_exp = np.array(([1, 0, 0, 1], [1, 1, 0, 0]), dtype=float)

        np.testing.assert_allclose(loop.vertices.data, vert_exp)

    def test_from_rectangle(self):
        loop = LoopXY.from_rectangle(2, 3, 4, 5)
        vert_exp = np.array(([6, 2, 2, 6], [8, 8, 3, 3]), dtype=float)

        np.testing.assert_allclose(loop.vertices.data, vert_exp)

    def test_positive_orientation(self):
        # Positive orientation
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        ori = loop.is_positive_orientation
        assert ori is True

    def test_flip_orientation(self):
        # Positive orientation
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        ori_pos_1 = loop.is_positive_orientation

        # Flip
        loop_neg = loop.flip()
        ori_neg_1 = loop_neg.is_positive_orientation

        # Flip in place
        loop_neg.flip_in_place()
        ori_pos_2 = loop_neg.is_positive_orientation

        assert ori_pos_1 is True
        assert ori_pos_2 is True
        assert ori_neg_1 is False

    def test_as_mask(self):
        # Square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        vx = vy = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5])

        mask_exp = np.array(
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
            ]
        )
        mask = loop.as_mask(vx, vy)

        np.testing.assert_array_equal(mask, mask_exp)

    def test_is_inside(self):
        # Square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)

        pts = Vxy(([0, 0.5, 1, 2], [0.5, 0.5, 0.5, 0.5]))

        mask = loop.is_inside(pts)
        mask_exp = np.array([False, True, False, False])

        np.testing.assert_array_equal(mask, mask_exp)

    def test_draw(self):
        # Square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)

        fig = fm.mpl_pyplot_figure()
        loop.draw(fig.gca())
        plt.close(fig)

    def test_intersect_line(self):
        """For the lines:
            - horizontal y=1,
            - diagonal x=y
        Get their intersection points through a 2x2 square loop."""
        loop = LoopXY.from_rectangle(0, 0, 2, 2)
        line_h = LineXY(0, 1, -1)
        line_xy = LineXY(-1, 1, 0)

        verts_h_act = loop.intersect_line(line_h).data
        verts_xy_act = loop.intersect_line(line_xy).data
        verts_h_act = np.sort(verts_h_act, axis=1)  # sort by x value
        verts_xy_act = np.sort(verts_xy_act, axis=1)

        self.assertEqual(verts_h_act.shape[0], 2)
        self.assertEqual(verts_xy_act.shape[0], 2)
        np.testing.assert_almost_equal(verts_h_act, np.array([[0, 2], [1, 1]]))
        np.testing.assert_almost_equal(verts_xy_act, np.array([[0, 2], [0, 2]]))


if __name__ == "__main__":
    unittest.main()
