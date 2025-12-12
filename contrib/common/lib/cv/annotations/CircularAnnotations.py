from typing import TYPE_CHECKING
import numpy as np
import scipy.spatial.transform

from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
import opencsp.common.lib.geometry.LoopXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt

if TYPE_CHECKING:
    # don't import at runtime in order to avoid cyclic dependencies
    from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class CircularAnnotations(AbstractAnnotations):
    """
    A collection of pixel circles where points of interest are located in an image.
    """

    def __init__(
        self,
        style: rcps.RenderControlPointSeq = None,
        centers_radiuses: tuple[p2.Pxy, list[int]] = None,
        meters_per_pixel: float = None,
    ):
        """
        Parameters
        ----------
        style : RenderControlBcs, optional
            The rendering style, by default {magenta, no corner markers}
        centers_radiuses : tuple[Pxy, list[int]]
            The center(s) and radius(es) for this annotation, in pixels
        meters_per_pixel : float, optional
            A simple conversion method for how many meters a pixel represents,
            for use in scale(). By default None.
        """
        if style is None:
            style = rcps.default(marker=None, color=color.magenta())
        super().__init__(style)

        self.p2r = centers_radiuses
        self.meters_per_pixel = meters_per_pixel

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        x = self.p2r[0].x[index]
        y = self.p2r[0].y[index]
        r = self.p2r[1][index]

        return reg.RegionXY(l2.LoopXY.from_rectangle(x - r, y - r, r * 2, r * 2))

    @property
    def origin(self) -> p2.Pxy:
        return self.p2r[0]

    def translate(self, translation: p2.Pxy):
        centers, radiuses = self.p2r
        p2r = (centers + translation, radiuses)
        return self.__class__(self.style, p2r, self.meters_per_pixel)

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError("Orientation is not yet implemented for CircularAnnotations")

    @property
    def size(self) -> list[float]:
        return [r * 2 for r in self.p2r[1]]

    @property
    def scale(self) -> list[float]:
        if self.meters_per_pixel is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in CircularAnnotations.scale(): "
                + "no meters_per_pixel conversion ratio is set, so scale can't be estimated",
            )
        return [d * self.meters_per_pixel for d in self.size]

    def render_to_figure(
        self,
        fig: rcfr.RenderControlFigureRecord,
        image: np.ndarray = None,
        include_label=False,
        operable: "SpotAnalysisOperable" = None,
    ):
        label = self.get_label(include_label)

        # draw the circles
        print(f"{len(self.p2r[0])=}")
        print(f"{self.p2r[1]=}")
        for index in range(len(self.p2r[0])):
            x = self.p2r[0].x[index]
            y = self.p2r[0].y[index]
            r = self.p2r[1][index]

            pq_list = []
            nverticies = 100
            for seg in range(nverticies):
                p = (np.sin(seg / nverticies * np.pi * 2) * r) + x
                q = (np.cos(seg / nverticies * np.pi * 2) * r) + y
                if operable is not None:
                    p, q = operable.transform_coordinates(p2.Pxy((p, q)))[1].astuple()
                pq_list.append((p, q))

            fig.view.draw_pq_list(pq_list, close=True, style=self.style, label=label)

            # only add the label once
            label = None
