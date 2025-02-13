import numpy as np
from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.Pxy import Vxy


class EdgeXY:
    """
    Representation of a 2D edge.
    """

    def __init__(self, vertices: Vxy, curve_data: dict = {"type": "line"}, closed: bool = False):
        """
        Representation of a 2D edge.

        Parameters
        ----------
        vertices : Vxy
            A length 2 vector of the beginning and end vertex of the edge.
        curve_data : dict, optional
            Additional curve data. The default is {'type': 'line'}.
            This is reserved for further future implementations of non-linear
            edge types.
        closed : bool, optional
            Flag for weather or not the edge is closed or not. The default is
            False. This is reserved for further future implimentations of non-
            linear edge types.

        """
        # Check inputs
        if len(vertices) != 2:
            raise ValueError("Input vertices must have length 2, not {:d}.".format(len(vertices)))
        if closed is True:
            raise NotImplementedError("Curves that are not closed are not currently supported.")

        # Save properties
        self._vertices = vertices
        self._curve_data = curve_data
        self._closed = closed

        # Create curve objects
        if curve_data["type"] == "line":
            self._curve = LineXY.from_two_points(vertices[0], vertices[1])
        else:
            raise ValueError("Curve type {:s} not currently supported.".format(curve_data["type"]))

    @property
    def vertices(self) -> Vxy:
        """
        Returns vertices of edge as length 2 Vxy.

        """
        return self._vertices

    @property
    def curve(self):
        """
        Returns the curve object used to define the edge.

        """
        return self._curve

    @property
    def is_closed(self) -> bool:
        """
        Returns if the edge is closed or not.

        """
        return self._closed

    def sample(self, count: int) -> Vxy:
        """
        Returns a sample of 'count' points evenly spaced on the edge

        count must be greater than or equal to 2. egde.sample(2) should be the same as edge.vertices
        """
        if self._curve_data["type"] == "line":
            if count <= 2:
                raise ValueError("count must an integer greater than or equal to 2.")
            v = self._vertices
            xs, ys = v.x, v.y
            xs = np.linspace(xs[0], xs[1], count)
            ys = np.linspace(ys[0], ys[1], count)
            return Vxy(np.array([xs, ys]))
        else:
            raise NotImplementedError("EdgeXY.sample only supports line edges.")

    def flip(self):
        """
        Returns a copy of the edge with its orientation flipped.

        """
        edge = EdgeXY(self._vertices, self._curve_data, self._closed)
        edge.flip_in_place()
        return edge

    def flip_in_place(self):
        """
        Flips the edge. The data is updated within the object.

        """
        # Flip order of vertices
        self._vertices = Vxy(self._vertices.data[:, ::-1])
        # Flip orientation of line
        self._curve.flip_in_place()
