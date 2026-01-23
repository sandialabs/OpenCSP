from typing import TYPE_CHECKING
import matplotlib.patches
import numpy as np
import scipy.spatial.transform

from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Vxy as v2

import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlSpotSize as rcss

if TYPE_CHECKING:
    # don't import at runtime in order to avoid cyclic dependencies
    from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class SpotWidthAnnotation(AbstractAnnotations):
    def __init__(
        self,
        spot_width_technique: str,
        centroid_loc: p2.Pxy,
        width: float,
        long_axis_rotation: float = None,
        long_axis_center: p2.Pxy = None,
        orthogonal_axis_width: float = None,
        style: rcss.RenderControlSpotSize = None,
    ):
        # set defaults
        if style is None:
            style = rcss.default()

        self.spot_width_technique = spot_width_technique
        self.centroid_loc = centroid_loc
        self.width = width
        self.long_axis_rotation = long_axis_rotation
        self.long_axis_center = long_axis_center
        self.orthogonal_axis_width = orthogonal_axis_width
        self.style = style

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        def width_for_rotation(width: float, rotation: float) -> float:
            vec = v2.Vxy([width, 0])
            vec = vec.rotate(scipy.spatial.transform.Rotation.from_euler("z", rotation))
            return np.max(vec.x[0], vec.y[0])

        def single_width_bounding_box(center: p2.Pxy, width: float) -> reg.RegionXY:
            half_width = int(width / 2)
            area = reg.RegionXY.rectangle(size=width) + half_width
            offset = center - p2.Pxy([half_width, half_width])
            return area + offset

        if self.spot_width_technique == "fwhm":
            long_axis_width = width_for_rotation(self.width, self.long_axis_rotation)
            orthogonal_axis_width = width_for_rotation(self.width, self.long_axis_rotation + (np.pi / 2))
            long_bbox_verts = single_width_bounding_box(self.long_axis_center, long_axis_width).loops[0].vertices
            orthogonal_bbox_verts = (
                single_width_bounding_box(self.long_axis_center, orthogonal_axis_width).loops[0].vertices
            )

            l = np.min(np.min(long_bbox_verts.x), np.min(orthogonal_bbox_verts.x))
            r = np.max(np.max(long_bbox_verts.x), np.max(orthogonal_bbox_verts.x))
            t = np.min(np.min(long_bbox_verts.y), np.min(orthogonal_bbox_verts.y))
            b = np.max(np.max(long_bbox_verts.y), np.max(orthogonal_bbox_verts.y))

            area = reg.RegionXY.rectangle(size=(r - l, b - t))
            offset = p2.Pxy([l, t])
            bounding_box = area + offset
        else:
            bounding_box = single_width_bounding_box(self.centroid_loc, self.width)

        return bounding_box

    @property
    def origin(self) -> p2.Pxy:
        return self.centroid_loc

    def translate(self, translation: p2.Pxy):
        centroid_loc = self.centroid_loc + translation
        long_axis_center = self.long_axis_center + translation
        return self.__class__(
            self.spot_width_technique,
            centroid_loc,
            self.width,
            self.long_axis_rotation,
            long_axis_center,
            self.orthogonal_axis_width,
            self.style,
        )

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        if self.spot_width_technique == "fwhm":
            return scipy.spatial.transform.Rotation.from_euler("z", self.long_axis_rotation)
        return None

    @property
    def size(self) -> list[float]:
        raise NotImplementedError

    def render_to_figure(
        self,
        fig: rcfr.RenderControlFigureRecord,
        image: np.ndarray,
        include_label=False,
        operable: "SpotAnalysisOperable" = None,
    ):
        label = self.get_label(include_label)

        # draw the full-width boundary
        if self.style.full_width_style != "None":
            style_params = {
                "facecolor": "None",
                "edgecolor": self.style.full_width_style.color,
                "linestyle": self.style.full_width_style.linestyle,
                "linewidth": self.style.full_width_style.linewidth,
                "alpha": self.style.full_width_style.markeralpha,
                "label": label,
            }
            label = None
            if self.spot_width_technique == "fwhm":
                center = self.long_axis_center
                if operable is not None:
                    center = operable.transform_coordinates(center)[1]
                ellipse = matplotlib.patches.Ellipse(
                    xy=center.astuple(),
                    width=self.width,
                    height=self.orthogonal_axis_width,
                    angle=np.rad2deg(self.long_axis_rotation),
                    **style_params
                )
                fig.view.axis.add_patch(ellipse)
            else:
                center = self.centroid_loc
                if operable is not None:
                    center = operable.transform_coordinates(center)[1]
                ellipse = matplotlib.patches.Ellipse(
                    xy=center.astuple(), width=self.width, height=self.width, **style_params
                )
                fig.view.axis.add_patch(ellipse)

        # draw the bounding box
        if self.style.bounding_box_style != "None":
            bbox = self.get_bounding_box()
            for loop in bbox.loops:
                loop_verts = list(zip(loop.vertices.x, loop.vertices.y))
                loop_verts = [(int(x), int(y)) for x, y in loop_verts]
                if operable is not None:
                    loop_verts = [operable.transform_coordinates(p2.Pxy((x, y)))[1].astuple() for x, y in loop_verts]
                fig.view.draw_pq_list(loop_verts, close=True, style=self.style.bounding_box_style, label=label)
                label = None

        # draw the centroid
        if self.style.center_style != "None":
            xy = self.centroid_loc
            if operable is not None:
                xy = operable.transform_coordinates(xy)[1]
            fig.view.draw_pq(xy.data, self.style.center_style, label=label)
            label = None
