import numpy as np

from opencsp.common.lib.geometry.EdgeXY import EdgeXY
from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.Vxy import Vxy


class TestEdgeXY:
    def test_EdgeXY_instantiation(self):
        # Proper edge
        pts = Vxy(([0, 1], [0, 0]))
        EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        # Not length 2 segment
        pts = Vxy(([0, 1, 2], [0, 0, 0]))
        with np.testing.assert_raises(ValueError):
            EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        # Not closed loop
        pts = Vxy(([0, 1], [0, 0]))
        with np.testing.assert_raises(NotImplementedError):
            EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=True)

        # Not type line edge
        pts = Vxy(([0, 1], [0, 0]))
        with np.testing.assert_raises(ValueError):
            EdgeXY(vertices=pts, curve_data={"type": "other"}, closed=False)

    def test_vertices(self):
        # Sample edge
        pts = Vxy(([0, 1], [0, 0]))
        edge = EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        np.testing.assert_equal(edge.vertices.data, pts.data)

    def test_curve(self):
        # Sample edge
        pts = Vxy(([0, 1], [0, 0]))
        edge = EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        assert type(edge.curve) is LineXY

    def test_closed(self):
        # Sample edge
        pts = Vxy(([0, 1], [0, 0]))
        closed = False
        edge = EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=closed)

        assert edge.is_closed is closed

    def test_flip(self):
        # Sample edge
        pts = Vxy(([0, 1], [0, 0]))
        pts_flip_exp = Vxy(([1, 0], [0, 0]))
        edge = EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        # Flip edge
        edge_flip = edge.flip()
        np.testing.assert_array_equal(pts_flip_exp.data, edge_flip.vertices.data)

    def test_flip_in_place(self):
        # Sample edge
        pts = Vxy(([0, 1], [0, 0]))
        pts_flip_exp = Vxy(([1, 0], [0, 0]))
        edge = EdgeXY(vertices=pts, curve_data={"type": "line"}, closed=False)

        # Flip edge
        edge.flip_in_place()
        np.testing.assert_array_equal(pts_flip_exp.data, edge.vertices.data)
