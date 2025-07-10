from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class Pxyz(Vxyz):
    """
    A class representing a 3D point in space, inheriting from the Vxyz class.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, data, dtype=float):
        """
        Initializes a Pxyz object with the given data and data type.

        Parameters
        ----------
        data : array-like
            The coordinates of the point in 3D space.
        dtype : type, optional
            The data type of the point coordinates (default is float).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return "3D Point:\n" + self._data.__repr__()

    def distance(self, data_in: "Pxyz") -> float:
        """
        Calculates the Euclidean distance between this point and another Pxyz point.

        Parameters
        ----------
        data_in : Pxyz
            The point to which the distance is to be calculated.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self._check_is_Vxyz(data_in)
        return (self - data_in).magnitude()[0]

    def as_Vxyz(self):
        """
        Converts this Pxyz point to a Vxyz object.

        Returns
        -------
        Vxyz
            A Vxyz object representing the same point in 3D space.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return Vxyz(self._data, self.dtype)

    @classmethod
    def empty(cls):
        """
        Creates and returns an empty Pxyz object.

        Returns
        -------
        Pxyz
            An empty Pxyz object with no coordinates.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return Pxyz([[], [], []])

    def draw_point(
        self,
        figure: rcfr.RenderControlFigureRecord | v3d.View3d,
        style: rcps.RenderControlPointSeq = None,
        labels: list[str] = None,
    ):
        """Calls figure.draw_xyz(p) for all points in this instance, and with
        the default arguments in place for any None's."""
        if style is None:
            style = rcps.default(markersize=2)
        if labels is None:
            labels = [None] * len(self)
        view = figure if isinstance(figure, v3d.View3d) else figure.view
        for x, y, z, label in zip(self.x, self.y, self.z, labels):
            view.draw_xyz((x, y, z), style, label)
