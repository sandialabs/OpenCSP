from typing import TYPE_CHECKING
import numpy as np
import scipy.spatial.transform

from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
from contrib.common.lib.cv.annotations.CircularAnnotations import CircularAnnotations
from contrib.common.lib.cv.annotations.RectangleAnnotations import RectangleAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps

if TYPE_CHECKING:
    # don't import at runtime in order to avoid cyclic dependencies
    from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class EnclosedEnergyAnnotations(AbstractAnnotations):
    """
    A collection of pixel circles and/or rectangles where points of interest are located in an image.
    """

    def __init__(
        self,
        style: rcps.RenderControlPointSeq = None,
        centers_radiuses: tuple[p2.Pxy, list[int]] = None,
        enclosed_shape: str = "circle",
        meters_per_pixel: float = None,
    ):
        """
        Parameters
        ----------
        style : RenderControlBcs, optional
            The rendering style, by default {magenta, no corner markers}
        centers_radiuses : tuple[Pxy, list[int]]
            The center(s) and radius(es) for this annotation, in pixels
        enclosed_shape : str, optional
            The shape used to determine the enclosed energy. Supports "circle" and "square". Default is "circle".
        meters_per_pixel : float, optional
            A simple conversion method for how many meters a pixel represents,
            for use in scale(). By default None.
        """
        if style is None:
            style = rcps.default(marker=None, color=color.magenta())
        super().__init__(style)

        # validate the input
        if enclosed_shape not in ["circle", "square"]:
            raise RuntimeError(
                "Error in EnclosedEnergyAnnotations.render_to_figure() "
                + "expected enclosed shape to be one of 'circle' or 'square', "
                + f"but is instead '{self.enclosed_shape}'"
            )

        self._p2r = centers_radiuses
        self.enclosed_shape = enclosed_shape
        self.meters_per_pixel = meters_per_pixel

        self.label = f"En{enclosed_shape}d Energy"  # Encircled, Ensquared

        r = p2.Pxy((self._p2r[1], self._p2r[1]))
        upper_left = self._p2r[0] - r
        lower_right = self._p2r[0] + r
        self._representative_circle = CircularAnnotations(style, centers_radiuses, meters_per_pixel)
        self._representative_square = RectangleAnnotations(style, (upper_left, lower_right), meters_per_pixel)

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        return self._representative_circle.get_bounding_box(index)

    @property
    def origin(self) -> p2.Pxy:
        return self._representative_circle.origin

    def translate(self, translation: p2.Pxy):
        centers, radiuses = self._p2r
        centers += translation
        return self.__class__(self.style, (centers, radiuses), self.enclosed_shape, self.meters_per_pixel)

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        return self._representative_circle.rotation

    @property
    def size(self) -> list[float]:
        return self._representative_circle.size

    @property
    def scale(self) -> list[float]:
        return self._representative_circle.scale

    def render_to_figure(
        self,
        fig: rcfr.RenderControlFigureRecord,
        image: np.ndarray = None,
        include_label=False,
        operable: "SpotAnalysisOperable" = None,
    ):
        if self.enclosed_shape == "circle":
            return self._representative_circle.render_to_figure(fig, image, include_label, operable)
        elif self.enclosed_shape == "square":
            return self._representative_square.render_to_figure(fig, image, include_label, operable)
        else:
            raise RuntimeError(
                "Error in EnclosedEnergyAnnotations.render_to_figure() "
                + "expected enclosed shape to be one of 'circle' or 'square', "
                + f"but is instead '{self.enclosed_shape}'"
            )
