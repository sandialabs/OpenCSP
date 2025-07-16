import matplotlib.axes
import matplotlib.patches
import numpy as np
import scipy.spatial

from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
import opencsp.common.lib.geometry.LoopXY as loop
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render_control.RenderControlBcs as rcb
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr


class BcsFiducial(AbstractFiducials):
    """
    Fiducial for indicating where the BCS target is in an image.
    """

    def __init__(
        self, origin_px: p2.Pxy, radius_px: float, style: rcb.RenderControlBcs = None, pixels_to_meters: float = 0.1
    ):
        """
        Initializes the BcsFiducial with the specified origin, radius, style, and pixel-to-meter conversion.

        Parameters
        ----------
        origin_px : p2.Pxy
            The center point of the BCS target, in pixels.
        radius_px : float
            The radius of the BCS target, in pixels.
        style : rcb.RenderControlBcs, optional
            The rendering style for the fiducial. Defaults to None.
        pixels_to_meters : float, optional
            A conversion factor for how many meters a pixel represents, for use in scale(). Defaults to 0.1.
        """
        # "ChatGPT 4o" assisted with generating this docstring.

        super().__init__(style=style)
        self.origin_px = origin_px
        self.radius_px = radius_px
        self.pixels_to_meters = pixels_to_meters

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """
        Get the bounding box of the BCS target.

        Parameters
        ----------
        index : int, optional
            Ignored for BcsFiducials.

        Returns
        -------
        reg.RegionXY
            The bounding box as a RegionXY object.

        Notes
        -----
        The bounding box is calculated based on the origin and radius of the BCS target.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        x1, x2 = self.origin.x[0] - self.radius_px, self.origin.x[0] + self.radius_px
        y1, y2 = self.origin.y[0] - self.radius_px, self.origin.y[0] + self.radius_px
        return reg.RegionXY(loop.LoopXY.from_rectangle(x1, y1, x2 - x1, y2 - y1))

    @property
    def origin(self) -> p2.Pxy:
        """
        Get the origin of the BCS fiducial.

        Returns
        -------
        p2.Pxy
            The center point of the BCS target in pixels.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return self.origin_px

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Get the rotation of the BCS fiducial.

        Raises
        ------
        NotImplementedError
            Rotation is not yet implemented for BcsFiducial.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        raise NotImplementedError("rotation is not yet implemented for PointFiducials")

    @property
    def size(self) -> list[float]:
        """
        Get the size of the BCS fiducial.

        Returns
        -------
        list[float]
            A list containing a single value: the diameter of the BCS target.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return [self.radius_px * 2]

    @property
    def scale(self) -> list[float]:
        """
        Get the scale of the BCS fiducial.

        Returns
        -------
        list[float]
            A list containing a single value: the size of the BCS target, in meters.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return [self.size * self.pixels_to_meters]

    def render_to_figure(self, fig: rcfr.RenderControlFigureRecord, image: np.ndarray, include_label=False):
        # This method adds a circle and a marker to the axes based on the style defined for the fiducial.
        label = self.get_label(include_label)

        if self.style.linestyle is not None:
            circ = matplotlib.patches.Circle(
                self.origin.data.tolist(),
                self.radius_px,
                color=self.style.color,
                linestyle=self.style.linestyle,
                linewidth=self.style.linewidth,
                fill=False,
                label=label,
            )
            fig.view.axis.add_patch(circ)
            label = None

        if self.style.marker is not None:
            fig.view.draw_pq(([self.origin.x], [self.origin.y]), style=self.style, label=label)
            label = None
