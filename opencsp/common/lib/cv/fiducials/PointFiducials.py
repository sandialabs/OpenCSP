import scipy.spatial

from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class PointFiducials(AbstractFiducials):
    """
    A collection of pixel locations where points of interest are located in an image.
    """

    def __init__(self, style: rcps.RenderControlPointSeq = None, points: p2.Pxy = None):
        """
        Initializes the PointFiducials with a specified style and points.

        Parameters
        ----------
        style : rcps.RenderControlPointSeq, optional
            The rendering style for the control points. Defaults to None.
        points : p2.Pxy, optional
            The pixel locations of the points of interest. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super().__init__(style)
        self.points = points

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """
        Get the bounding box for a specific point by index.

        Parameters
        ----------
        index : int, optional
            The index of the point for which to retrieve the bounding box. Defaults to 0.

        Returns
        -------
        reg.RegionXY
            The bounding box as a RegionXY object.

        Notes
        -----
        This method is untested and may require validation.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # TODO untested
        return reg.RegionXY.from_vertices(p2.Pxy((self.points.x[index], self.points.y[index])))

    @property
    def origin(self) -> p2.Pxy:
        """
        Get the origin of the fiducials.

        Returns
        -------
        p2.Pxy
            The pixel locations of the points of interest.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return self.points

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Get the rotation of the fiducials.

        Raises
        ------
        NotImplementedError
            If the orientation is not yet implemented for this class.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        raise NotImplementedError("Orientation is not yet implemented for PointFiducials")

    @property
    def size(self) -> list[float]:
        """
        Get the size of the fiducials.

        Returns
        -------
        list[float]
            A list of sizes for each point. The default implementation for PointFiducials returns a list of zeros.

        Notes
        -----
        This property is untested and may require validation.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # TODO untested
        return [0] * len(self.points)

    @property
    def scale(self) -> list[float]:
        """
        Get the scale of the fiducials.

        Returns
        -------
        list[float]
            A list of scales for each point. The default implementation for PointFiducials returns a list of zeros.

        Notes
        -----
        This property is untested and may require validation.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # TODO untested
        return [0] * len(self.points)
