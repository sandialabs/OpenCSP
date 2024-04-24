from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class AbstractFiducials(ABC):
    def __init__(self, style=None, pixels_to_meters: Callable[[p2.Pxy], v3.Vxyz] = None):
        """
        A collection of markers (such as an ArUco board) that is used to orient the camera relative to observed objects
        in the scene. It is suggested that each implementing class be paired with a complementary FiducialLocator or
        PredictingFiducialLocator class.

        Parameters
        ----------
        style : RenderControlPointSeq, optional
            How to render this fiducial when using the defaul render_to_plot() method. By default rcps.default().
        pixels_to_meters : Callable[[p2.Pxy], v3.Vxyz], optional
            Conversion function to get the physical point in space for the given x/y position information. Used in the
            default self.scale implementation. Defaults to 1 meter per pixel.
        """
        self.style = style if style is not None else rcps.default()
        self.pixels_to_meters = pixels_to_meters

    @abstractmethod
    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """The X/Y bounding box(es) of this instance, in pixels."""

    @property
    @abstractmethod
    def origin(self) -> p2.Pxy:
        """The origin point(s) of this instance, in pixels."""

    @property
    @abstractmethod
    def orientation(self) -> v3.Vxyz:
        """The orientation(s) of this instance, in radians. This is relative to
        the source image, where x is positive to the right, y is positive down,
        and z is positive in (away from the camera)."""

    @property
    @abstractmethod
    def size(self) -> list[float]:
        """The scale(s) of this fiducial, in pixels, relative to its longest axis.
        For example, if the fiducial is a square QR-code and is oriented tangent
        to the camera, then the scale will be the number of pixels from one
        corner to the other."""  # TODO is this a good definition?

    @property
    def scale(self) -> list[float]:
        """
        The scale(s) of this fiducial, in meters, relative to its longest axis.
        This can be used to determine the distance and orientation of the
        fiducial relative to the camera.
        """
        ret = []

        for i in range(len(self.origin)):
            bb = self.get_bounding_box(i)
            left_px, right_px, bottom_px, top_px = bb.loops[0].axis_aligned_bounding_box()
            top_left_m = self.pixels_to_meters(p2.Pxy([left_px, top_px]))
            bottom_right_m = self.pixels_to_meters(p2.Pxy([right_px, bottom_px]))
            scale = (bottom_right_m - top_left_m).magnitude()[0]
            ret.append(scale)

        return ret

    def _render(self, axes: matplotlib.axes.Axes):
        """
        Called from render(). The parameters are always guaranteed to be set.
        """
        axes.scatter(
            self.origin.x,
            self.origin.y,
            linewidth=self.style.linewidth,
            marker=self.style.marker,
            s=self.style.markersize,
            c=self.style.markerfacecolor,
            edgecolor=self.style.markeredgecolor,
        )

    def render(self, axes: matplotlib.axes.Axes = None):
        """
        Renders this fiducial to the active matplotlib.pyplot plot.

        The default implementation uses plt.scatter().

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            The plot to render to. Uses the active plot if None. Default is None.
        """
        if axes is None:
            axes = plt.gca()
        self._render(axes)

    def render_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Renders this fiducial to the a new image on top of the given image.

        The default implementation creates a new matplotlib plot, and then renders to it with self.render_to_plot().
        """
        # Create the figure to plot to
        dpi = 300
        width = image.shape[1]
        height = image.shape[0]
        fig = fm.mpl_pyplot_figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        try:
            # A portion of this code is from:
            # https://stackoverflow.com/questions/35355930/figure-to-image-as-a-numpy-array

            # Get the axis and canvas
            axes = fig.gca()
            canvas = fig.canvas

            # Image from plot
            axes.axis('off')
            fig.tight_layout(pad=0)

            # To remove the huge white borders
            axes.margins(0)

            # Prepare the image and the feature points
            axes.imshow(image)
            self.render(axes)

            # Render
            canvas.draw()

            # Convert back to a numpy array
            new_image = np.asarray(canvas.buffer_rgba())
            new_image = new_image.astype(image.dtype)

            # Return the updated image
            return new_image

        except Exception as ex:
            lt.error("Error in AnnotationImageProcessor.render_points(): " + repr(ex))
            raise

        finally:
            plt.close(fig)
