from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st


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
        style : RenderControlPointSeq, optional
            How to render this fiducial when using the default
            :py:meth:`render_to_plot` method. By default rcps.default().
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

    def get_label(self, include_label=False) -> str | None:
        """
        Get the label either from self.label (if set) or from the class name.
        Returns None if include_label is False.
        """
        if not include_label:
            label = None
        if hasattr(self, "label"):
            label = self.label
        else:
            class_name = self.__class__.__name__

            class_name_endings = ["Fiducial", "Fiducials", "Annotation", "Annotations"]
            for class_name_ending in class_name_endings:
                if class_name.endswith(class_name_ending):
                    class_name = class_name[: -len(class_name_ending)]

            label = " ".join(st.camel_case_split(class_name))

        return label

    def render_to_figure(
        self, fig_record: rcfr.RenderControlFigureRecord, image: np.ndarray = None, include_label=False
    ):
        """
        Renders a visual representation of this fiducial to the given fig_record.

        The given image should have already been rendered to the figure record
        if it is set. If this has been called from :py:meth:`render_to_image`
        then image is guaranteed to be set.

        The default version of this method renders the origin point. Overwrite
        this method for a custom implementation.

        Parameters
        ----------
        fig_record : rcfr.RenderControlFigureRecord
            The record to render with. Most render methods should be available
            via fig_record.view.draw_*().
        image : np.ndarray, optional
            The image that was already rendered to the figure record, or None if
            there hasn't been an image rendered or that data just isn't
            available. By default None, or the image passed in to
            :py:meth:`render_to_image` if being called from that method.
        include_label: bool, optional
            True if this fiducial should add a label during it's plot method. By
            default False.
        """
        label = self.get_label(include_label)
        fig_record.view.draw_pq(([self.origin.x[0]], [self.origin.y[0]]), style=self.style, label=label)

    def render_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Renders this fiducial to a new image on top of the given image.

        The default implementation creates a new matplotlib plot, and then
        renders to it with either :py:meth:`render_to_figure` or
        :py:meth:`render_to_plot`, depending on which has been implemented.

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
        (height_px, width_px), nchannel = it.dims_and_nchannels(image)
        figsize = rcfg.RenderControlFigure.pixel_resolution_inches(width_px, height_px)
        figure_control = rcfg.RenderControlFigure(
            tile=False, figsize=figsize, grid=False, draw_whitespace_padding=False
        )
        view_spec_2d = vs.view_spec_im()

        fig_record = fm.setup_figure_for_3d_data(
            figure_control,
            rca.image(draw_axes=False, grid=False),
            view_spec_2d,
            equal=False,
            name="Fiducials and Annotations",
            code_tag=f"{__file__}.render_fiducials_to_image()",
        )

        try:
            # A portion of this code is from:
            # https://stackoverflow.com/questions/35355930/figure-to-image-as-a-numpy-array

            # Prepare the image
            fig_record.view.imshow(image)

            # render
            self.render_to_figure(fig_record, image)

            # Convert back to a numpy array
            new_image = fig_record.to_array()
            new_image = new_image.astype(image.dtype)

            # Return the updated image
            return new_image

        except Exception as ex:
            lt.error("Error in AbstractFiducials.render_to_image(): " + repr(ex))
            raise

        finally:
            plt.close(fig_record.figure)
