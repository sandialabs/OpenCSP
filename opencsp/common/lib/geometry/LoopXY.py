import copy
import numbers

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.geometry.EdgeXY import EdgeXY
from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class LoopXY:
    """Representation of 2D loop. The loop created by the given edges must
    satisfy the following:

     * Closed geometry (currently checked)
     * Must be convex (currently not checked)
     * Linear boundary types (only linear is supported)
     * The orientation of the edges must be consistent (all CCW/CW)
    """

    def __init__(self, edges: list[EdgeXY]):
        """Instantiate a loop

        Parameters
        ----------

        edges : list[EdgesXY, ...]
            Oriented edges of loop.

        """
        # Save number of edges
        self.num_edges = len(edges)

        # Save edges
        self._edges = edges

        # Check 1) Check loop is closed
        self._check_closed_loop()

        # Check 2) Check that the edges do not cross each other except at vertices
        self._check_convex()

        # Check 3) Check there are no measure zero regions of the loop
        # TODO

    def __len__(self):
        return len(self._edges)

    def _check_closed_loop(self) -> None:
        """
        Checks that the loop is closed. Check the second vertex of the first
        edge matches the first vertex of the second edge and so on.

        """
        for idx2 in range(self.num_edges):
            idx1 = np.mod(idx2 - 1, self.num_edges)
            P_1 = self._edges[idx1].vertices[1]
            P_2 = self._edges[idx2].vertices[0]
            if P_1.x != P_2.x or P_1.y != P_2.y:
                raise ValueError(
                    "The second vertex of line index {:d} does not match the first vertex of line index {:d}.".format(
                        idx1, idx2
                    )
                )

    def _check_convex(self) -> None:
        """
        Checks that straight lines connecting each vertex makes convex loop.

        """
        vertex_angles = self._vertex_to_vertex_angles()
        vertex_positive = np.unique(vertex_angles > 0)
        if vertex_positive.size > 1:
            raise ValueError("Loop may not be convex or edges cross within loop.")

    def _vertex_to_vertex_angles(self):
        """
        Calculates the sin of the angle between the straight line drawn from
        vertex to vertex.

        """
        cross_prod_data = np.zeros(self.num_edges)
        for idx1 in range(self.num_edges):
            idx2 = np.mod(idx1 + 1, self.num_edges)
            # Calcualte edge vectors
            V_1 = (self._edges[idx1]._vertices[1] - self._edges[idx1]._vertices[0]).normalize()
            V_2 = (self._edges[idx2]._vertices[1] - self._edges[idx2]._vertices[0]).normalize()
            # Calculate cross product
            cross_prod_data[idx2] = V_1.cross(V_2)[0]

        return cross_prod_data

    @classmethod
    def from_lines(cls, lines: list[LineXY]):
        """
        Returns LoopXY from list of 2d lines. Lines are intersected with each
        other in the order given to calculate the loop vertices. The vertices
        are sorted to be CCW starting from the +x axis.

        Parameters
        ----------
        lines : list[LineXY, ...]
            2d lines defining boundary of loop.

        Returns
        -------
        LoopXY

        """
        # Create vertices from lines
        vertices = []
        for idx in range(len(lines)):
            # Get two lines
            l_1: LineXY = lines[np.mod(idx - 1, len(lines))]
            l_2: LineXY = lines[idx]

            # Intersect line
            vertices.append(l_1.intersect_with(l_2).data)

        # Concatenate vertices
        vertices = Vxy(np.concatenate(vertices, axis=1))

        # Sort vertices
        vertices = cls._process_vertices(vertices)

        # Create list of edges from vertices
        edges = []
        for idx1 in range(len(vertices)):
            idx2 = np.mod(idx1 + 1, len(vertices))
            edges.append(EdgeXY(vertices=vertices[[idx1, idx2]], curve_data={"type": "line"}, closed=False))

        return cls(edges=edges)

    @classmethod
    def from_vertices(cls, vertices: Vxy):
        """
        Returns LoopXY defined from 2d points. Points are sorted to be CCW
        starting from the +x axis. Assumes 2d lines are the curves connecting
        each vertex.

        Parameters
        ----------
        vertices : Vxy
            2d points defining boundary of loop.

        Returns
        -------
        LoopXY

        """
        # Sort vertices
        vertices = cls._process_vertices(vertices)

        # Create list of edges from vertices
        edges = []
        for idx1 in range(len(vertices)):
            idx2 = np.mod(idx1 + 1, len(vertices))
            edges.append(EdgeXY(vertices=vertices[[idx1, idx2]], curve_data={"type": "line"}, closed=False))

        return cls(edges=edges)

    @classmethod
    def from_rectangle(cls, x: float, y: float, width: float, height: float) -> "LoopXY":
        """Returns rectangular loop

        Parameters
        ----------
        x/y : float
            Coordinates of lower left (minimum xy values) corner of rectangle
        width : float
            Width of rectangle (x direction)
        height : float
            Height of rectangle (y direction)

        Returns
        -------
        LoopXY
        """
        vertices = Vxy(([x, x + width, x + width, x], [y, y, y + height, y + height]))
        return cls.from_vertices(vertices)

    @property
    def vertices(self) -> Vxy:
        """
        Returns the vertices of the loop.

        """
        vertex_xy_data = np.zeros((2, self.num_edges))
        for idx, edge in enumerate(self._edges):
            vertex_xy_data[:, idx : idx + 1] = edge.vertices[0].data
        return Vxy(vertex_xy_data)

    @property
    def is_positive_orientation(self) -> bool:
        """
        Checks the orientation of the loop.

        """
        # Calculate first vertex-to-vertex angle (should all have same sign)
        vertex_angle = self._vertex_to_vertex_angles()[0]
        return bool(vertex_angle > 0)

    def flip(self):
        """
        Flips the orientation of the loop. Returns a copy of the flipped loop.

        """
        loop = LoopXY(self._edges)
        loop.flip_in_place()
        return loop

    def flip_in_place(self):
        """
        Flips the orientation of the loop. Flips the internal data of the loop.

        """
        # Flip edges
        for edge in self._edges:
            edge.flip_in_place()
        # Flip order of edges
        self._edges = self._edges[::-1]

    def as_mask(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """
        Returns 2d mask given sample points on x and y axis.

        Parameters
        ----------
        xv : np.ndarray
            1d array, x sample points.
        yv : np.ndarray
            1d array, y sample points.

        Returns
        -------
        np.ndarray
            Boolean 2d mask with shape (yv.size, xv.size). True for inside this
            loop, False for outside this loop.

        """
        # Create XY coordinates
        x, y = np.meshgrid(vx, vy)
        pts = Vxy((x.flatten(), y.flatten()))

        # Calculate mask
        mask = self.is_inside(pts)

        return mask.reshape((vy.size, vx.size))

    def is_inside(self, points: Vxy) -> np.ndarray:
        """
        Calculates which of the given points are within the loop. Points on the
        edge of the loop are not considered inside.

        Parameters
        ----------
        points : Vxy
            Given points to compare to loop.

        Returns
        -------
        mask : np.ndarray
            1D boolean array of same length as input points. True where the
            points are inside, False where the points are outside.

        """
        mask = np.ones(len(points), dtype=bool)
        positive_orientation = self.is_positive_orientation

        # Find distances from curves
        for edge in self._edges:
            ds = edge.curve.dist_from_signed(points)
            if positive_orientation:
                mask = np.logical_and(mask, ds < 0)
            else:
                mask = np.logical_and(mask, ds > 0)

        return mask

    def is_inside_or_on_border(self, points: Vxy, thresh: float = 1e-6) -> np.ndarray:
        """
        Calculates which of the given points are within the loop. Points on the
        edge of the loop are not considered inside.

        Parameters
        ----------
        points : Vxy
            Given points to compare to loop.

        Returns
        -------
        mask : np.ndarray
            1D boolean array of same length as input points.

        """
        mask = np.ones(len(points), dtype=bool)
        positive_orientation = self.is_positive_orientation

        # Find distances from curves
        for edge in self._edges:
            ds = edge.curve.dist_from_signed(points)
            if positive_orientation:
                mask = np.logical_and(mask, ds <= thresh)
            else:
                mask = np.logical_and(mask, ds >= -thresh)

        return mask

    def draw(self, ax: plt.Axes = None, style: rcps.RenderControlPointSeq = None) -> None:
        """
        Draws lines as arrows and marks starting point.

        Parameters
        ----------
        ax : Axes, optional
            The axes to draw on. If not given, uses current axes.
        style : str, optional
            The style used to draw this region. Default rcps.default().

        """
        if ax is None:
            ax = plt.gca()
        if style is None:
            style = rcps.default(marker='arrow')

        # Draw arrows
        first_vert_np = np.array([self.vertices.x[:1], self.vertices.y[:1]])
        closed_loop_verts = Vxy(np.concatenate((self.vertices.data, first_vert_np), axis=1))
        view = v3d.View3d(ax.figure, ax, vs.view_spec_xy())
        view.draw_pq((closed_loop_verts.x, closed_loop_verts.y), style)

        # Plot starting point as green dot
        ax.scatter(*self.vertices.data[:, 0:1], color="green")

    def edge_sample(self, count: int) -> Vxy:
        """Returns a Vxy of count points per edge= defining the loop"""
        return Vxy.merge([edge.sample(count) for edge in self._edges])

    @staticmethod
    def _process_vertices(vertices: Vxy):
        """
        Performs the following on the given vertices
            - Orders points CCW
            - Sorts the points CCW from the +x axis
            - Checks for convexity (unimplimented)

        Parameters
        ----------
        vertices: Vxy
            Input vertices to define 2d loop.

        """
        # Calculate centroid
        cent = Vxy(vertices.data.mean(1))

        thetas = np.zeros(len(vertices))
        for idx in range(len(vertices)):
            # Calculate CCW angle from +X axis
            pt = vertices[idx] - cent
            theta = np.arctan2(pt.y, pt.x)[0]  # radians
            theta = np.mod(theta, 2 * np.pi)  # radians
            thetas[idx] = theta

        # Order
        order = np.argsort(thetas)
        return vertices[order]

    def aabbox(self, *args, **kwargs) -> tuple[float, float, float, float]:
        """
        Alias for LoopXY.minnimum_bounding_envelope.
        """
        return self.axis_aligned_bounding_box(*args, **kwargs)

    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Gives the minnimum bounding envelope for the region. The minnimum
        bounding envelope is the smallest rectangle that can fit the LoopXY
        in question where all sides of the rectangle are parrallel to either the X or Y axes.

        Returns
        ---------
        (left, right, bottom, top): tuple[float, float, float, float]
            the x and y values for the bounds
        """
        vertices = self.vertices
        xs = vertices.x
        ys = vertices.y
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)
        return (left, right, bottom, top)

    def intersect_line(self, line: LineXY):
        """Returns intersection points with line

        Parameters
        ----------
        line : LineXY
            Line to intersect with loop

        Returns
        -------
        Vxy
            Intersection points

        Raises
        ------
        NotImplementedError
            If loop enge is not LineXY
        """
        intersect_xs: list[float] = []
        intersect_ys: list[float] = []

        # Get all edge intersections
        for edge in self._edges:
            if isinstance(edge._curve, LineXY):
                other: LineXY = edge._curve
                intersect_point = other.intersect_with(line)
                if intersect_point is not None:
                    intersect_xs.append(intersect_point.x[0])
                    intersect_ys.append(intersect_point.y[0])
            else:
                raise NotImplementedError("Intersections of non-line edges not yet supported in this method")
        intersect_points = Vxy((intersect_xs, intersect_ys))

        # Limit to internal (or border) intersections
        intersect_points = intersect_points[self.is_inside_or_on_border(intersect_points)]

        # De-duplicate intersections
        keep_xs: list[float] = []
        keep_ys: list[float] = []
        for intersect_point in intersect_points:
            close_xs = np.isclose([intersect_point.x[0]] * len(keep_xs), keep_xs)
            close_ys = np.isclose([intersect_point.y[0]] * len(keep_ys), keep_ys)
            if not any(close_xs & close_ys):
                keep_xs.append(intersect_point.x[0])
                keep_ys.append(intersect_point.y[0])
        intersect_points = Vxy((keep_xs, keep_ys))

        return intersect_points

    def intersect_edge(self, edge: EdgeXY, thresh=1e-6) -> Vxy:
        """Returns intersection points with edge.

        Parameters
        ----------
        edge : EdgeXY
            Edge to intersect with loop.
        thresh : float, optional
            Boundary around the extents of the edge to be considered as part of
            the edge when testing points against the edge limits. Default is 1e-6.

        Returns
        -------
        Vxy
            Intersection points with the edge. May be empty.

        Raises
        ------
        ValueError
            If the type of edge is not compatible with the current version of
            the algorithm.
        """
        # check for algorithm compatibility
        if edge._curve_data['type'] != 'line':
            lt.error_and_raise(
                ValueError,
                "Error in LoopXY.intersect_edge(): "
                + f"The current version of this algorithm only supports edges with type 'line', but {edge['type']=}.",
            )

        # get the line intersections
        edge_as_line = LineXY.from_two_points(edge.vertices[0], edge.vertices[1])
        line_intersections = self.intersect_line(edge_as_line)

        # limit to the region containing the edge extents
        bottom = np.min(edge.vertices.y) - thresh
        top = np.max(edge.vertices.y) + thresh
        left = np.min(edge.vertices.x) - thresh
        right = np.max(edge.vertices.x) + thresh
        keep_x = np.logical_and(line_intersections.x >= left, line_intersections.x <= right)
        keep_y = np.logical_and(line_intersections.y >= bottom, line_intersections.y <= top)
        keep = np.logical_and(keep_x, keep_y)

        return Vxy([line_intersections.x[keep], line_intersections.y[keep]])

    def intersect_loop(self, other: 'LoopXY', thresh=1e-6) -> 'LoopXY':
        """Returns the overlapping area between this loop and the other loop.

        Parameters
        ----------
        other: LoopXY
            Other loop to find the overlapping area with.

        Returns
        -------
        intersection: LoopXY
            A new loop composed of the overlapping area.
        thresh: float, optional
            The acceptable tolerance for overlapping regions.

        Raises
        ------
        ValueError
            If the edges for the other loop aren't supported by the code as
            currently implemented.
        """

        def _remove_duplicates_from_sorted_list(points: list[Vxy], thresh=1e-6) -> list[Vxy]:
            points = copy.copy(points)

            to_remove: list[Vxy] = []
            for n in range(len(points)):
                m = (n + 1) % len(points)
                pt_n, pt_m = points[n], points[m]
                if (np.abs(pt_n.x[0] - pt_m.x[0]) < thresh) and (np.abs(pt_n.y[0] - pt_m.y[0]) < thresh):
                    to_remove.append(pt_m)

            for pt in to_remove:
                points.remove(pt)

            return points

        # get the vertices that overlap in both loops
        a, b = self, other
        a_contains_b = self.is_inside_or_on_border(b.vertices, thresh)
        b_contains_a = b.is_inside_or_on_border(a.vertices, thresh)
        a_vertices = Vxy([a.vertices.x[b_contains_a], a.vertices.y[b_contains_a]])
        b_vertices = Vxy([b.vertices.x[a_contains_b], b.vertices.y[a_contains_b]])
        ab_vertices = Vxy(np.concatenate((a_vertices.data, b_vertices.data), axis=1))

        # include the intersection vertices for the loop edges
        for edge in other._edges:
            edge_intersections = self.intersect_edge(edge)
            if len(edge_intersections) > 0:
                ab_vertices = Vxy(np.concatenate((ab_vertices.data, edge_intersections.data), axis=1))

        # sort by x/y and remove duplicate vertices
        ab_verts_list = [Vxy([ab_vertices[i].x, ab_vertices[i].y]) for i in range(len(ab_vertices))]
        ab_verts_list = sorted(ab_verts_list, key=lambda pt: pt.y[0])
        ab_verts_list = sorted(ab_verts_list, key=lambda pt: pt.x[0])
        ab_verts_list = _remove_duplicates_from_sorted_list(ab_verts_list, thresh)

        # sort by angle
        origin = ab_verts_list.pop(0)
        ab_verts_list = [origin] + sorted(ab_verts_list, key=lambda pt: (pt - origin).angle()[0])

        # return a new loop
        ab_verts = Vxy.from_list(ab_verts_list)
        return LoopXY.from_vertices(ab_verts)

    def __add__(self, other: Vxy | numbers.Number) -> "LoopXY":
        if isinstance(other, Vxy) or isinstance(other, numbers.Number):
            pass
        else:
            lt.error_and_raise(
                TypeError,
                "Error in LoopXY.__add__(): "
                + f"secondary value in addition must be of type Vxy or Number, "
                + f"but is {type(other)}",
            )
        if isinstance(other, Vxy) and len(other) != 1:
            lt.error_and_raise(
                ValueError,
                "Error in LoopXY.__add__(): " + f"other value Vxy must have length 1, " + f"but {len(other)=}",
            )

        if isinstance(other, Vxy):
            verticies = self.vertices + other
        else:
            verticies = self.vertices + Vxy([other, other])

        return LoopXY.from_vertices(verticies)

    def __sub__(self, other: Vxy) -> "LoopXY":
        return self + (-other)
