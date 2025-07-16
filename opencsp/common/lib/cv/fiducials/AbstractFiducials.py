from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class AbstractFiducials(ABC):
    """
    A collection of markers (such as an ArUco board) that is used to orient the camera relative to observed objects
    in the scene. It is suggested that each implementing class be paired with a complementary locator method or
    :py:class:`opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor`.
    """

    def __init__(self, style: rcps.RenderControlPointSeq = None, pixels_to_meters: Callable[[p2.Pxy], v3.Vxyz] = None):
        """
        Initializes the AbstractFiducials with a specified rendering style and pixel-to-meter conversion function.

        Parameters
        ----------
        style : rcps.RenderControlPointSeq, optional
            How to render this fiducial when using the default render_to_plot() method. Defaults to rcps.default().
        pixels_to_meters : Callable[[p2.Pxy], v3.Vxyz], optional
            Conversion function to get the physical point in space for the given x/y position information. Used in the
            default self.scale implementation. A good implementation of this function will correct for many factors such
            as relative camera position and camera distortion. For extreme accuracy, this will also account for
            non-uniformity in the target surface. Defaults to a simple 1 meter per pixel model.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.style = style if style is not None else rcps.default()
        self.pixels_to_meters = pixels_to_meters

    @abstractmethod
    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """
        Get the X/Y bounding box of this instance, in pixels.

        Parameters
        ----------
        index : int, optional
            The index of the fiducial for which to retrieve the bounding box, for fiducials that have more than one bounding box. Defaults to 0.

        Returns
        -------
        reg.RegionXY
            The bounding box of the fiducial.
        """

        # "ChatGPT 4o" assisted with generating this docstring.

    @property
    @abstractmethod
    def origin(self) -> p2.Pxy:
        """
        Get the origin point(s) of this instance, in pixels.

        Returns
        -------
        p2.Pxy
            The origin point(s) of the fiducial.
        """

        # "ChatGPT 4o" assisted with generating this docstring.

    @property
    @abstractmethod
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Get the pointing of the normal vector(s) of this instance.

        This is relative to the camera's reference frame, where x is positive
        to the right, y is positive down, and z is positive in (away from the
        camera)

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation of the fiducial relative to the camera's reference frame.

        Notes
        -----
        This can be used to describe the forward transformation from the camera's perspective. For example, an ArUco
        marker whose origin is in the center of the image and is facing towards the camera could have the rotation
        defined as:

        .. code-block:: python

            Rotation.from_euler('y', np.pi)

        If that same ArUco marker was also placed upside down, then its rotation could be defined as:

        .. code-block:: python

            Rotation.from_euler(
                'yz',
                [[np.pi, 0],
                [0,     np.pi]]
            )

        Note that this just describes rotation, and not the translation. We call the rotation and translation together
        the orientation.
        """

    # "ChatGPT 4o" assisted with generating this docstring.

    @property
    @abstractmethod
    def size(self) -> list[float]:
        """
        Get the scale(s) of this fiducial, in pixels, relative to its longest axis.

        As an example, if the fiducial is a square QR-code and is oriented tangent
        to the camera, then the scale will be the number of pixels from one
        corner to the other.

        Returns
        -------
        list[float]
            The sizes of the fiducial in pixels.
        """

        # "ChatGPT 4o" assisted with generating this docstring.

    @property
    def scale(self) -> list[float]:
        """
        Get the scale(s) of this fiducial, in meters, relative to its longest axis.

        This value, together with the size, can potentially be used to determine the
        distance and rotation of the fiducial relative to the camera.

        Returns
        -------
        list[float]
            The scales of the fiducial in meters.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
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
        # Render the fiducial on the given axes.
        #
        # Parameters
        # ----------
        # axes : matplotlib.axes.Axes
        #    The axes on which to render the fiducial.

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
        axes : matplotlib.axes.Axes, optional
            The plot to render to. Uses the active plot if None. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if axes is None:
            axes = plt.gca()
        self._render(axes)

    def render_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Renders this fiducial to a new image on top of the given image.

        The default implementation creates a new matplotlib plot, and then renders to it with self.render().

        Parameters
        ----------
        image : np.ndarray
            The original image to which the fiducial will be rendered.

        Returns
        -------
        np.ndarray
            The updated image with the fiducial rendered on top.

        Raises
        ------
        Exception
            If an error occurs during the rendering process.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
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
            axes.axis("off")
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
