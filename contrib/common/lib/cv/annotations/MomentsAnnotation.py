import copy
from functools import cache, cached_property

import numpy as np
import scipy.spatial.transform

from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
import opencsp.common.lib.geometry.LineXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import contrib.common.lib.geometry.RectXY as r2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class MomentsAnnotation(AbstractAnnotations):
    def __init__(
        self,
        moments,
        centroid_style: rcps.RenderControlPointSeq = None,
        rotation_style: rcps.RenderControlPointSeq = None,
    ):
        """
        centroid_style: RenderControlPointSeq, optional
            Style used for render the centroid point. By default
            RenderControlPointSeq.defualt(color=magenta).
        rotation_style: RenderControlPointSeq, optional
            Style used for render the rotation line. By default centroid_style.
        """
        if centroid_style is None:
            centroid_style = rcps.default(color=color.magenta())
        if rotation_style is None:
            rotation_style = copy.deepcopy(centroid_style)
        super().__init__(centroid_style)

        self.moments = moments
        self.rotation_style = rotation_style

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """
        Returns a rectangle with 0 area and the same point for the
        upper-left/lower-right corners. Moments don't have a bounding box.
        """
        return r2.RectXY(self.centroid, self.centroid)

    @property
    def origin(self) -> p2.Pxy:
        return self.centroid

    def translate(self, translations: p2.Pxy):
        old_m00 = self.moments["m00"]
        old_m10 = self.moments["m10"]
        old_m01 = self.moments["m01"]
        old_u11 = self.central_moment(1, 1)
        old_u20 = self.central_moment(2, 0)
        old_u02 = self.central_moment(0, 2)
        old_u21 = self.central_moment(2, 1)
        old_u12 = self.central_moment(1, 2)
        old_u30 = self.central_moment(3, 0)
        old_u03 = self.central_moment(0, 3)

        Tx = translations.x[0]
        Ty = translations.y[0]

        # Example calculated values for an elipse at 135 degrees, size 100x50:
        # center at 50, 50       at 60, 70
        # m00    1031985.0       1031985.0
        # m10    51626280.0      61946130.0
        # m01    51634950.0      72274650.0
        # m11    2825640210.0    4580912310.0
        # m20    2994052920.0    4129777020.0
        # m02    2995071900.0    5473263900.0
        # m12    174088606080.0  362497365480.0
        # m21    174061884630.0  318333724230.0
        # m30    190942591890.0  297284048490.0
        # m03    191033689110.0  440955823110.0
        # u00    1031985.0       1031985.0
        # u01    0               0
        # u10    0               0
        # u11    242540274.9369  242540274.9369
        # u20    411386712.0237  411386712.0237
        # u02    411538165.0111  411538165.0111
        # u21    -11069058.1853  -11069058.1854
        # u12    -14244705.8617  -14244705.8616
        # u30    1303451.534790  1303451.534790
        # u03    -5805601.44134  -5805601.44128

        # central_moment formulas:
        # u00 = m00
        # u01 = 0
        # u10 = 0
        # u11 = m11 - m01 / m00 * m10
        # u20 = m20 - m10**2 / m00
        # u02 = m02 - m01**2 / m00
        # u21 = m21 - 2 * m10 / m00 * m11 - m01 / m00 * m20 + 2 * m10**2 / m00**2 * m01
        # u12 = m12 - 2 * m01 / m00 * m11 - m10 / m00 * m02 + 2 * m01**2 / m00**2 * m10
        # u30 = m30 - 3 * m10 / m00 * m20 + 2 * m10**3 / m00**2
        # u03 = m03 - 3 * m01 / m00 * m02 + 2 * m01**3 / m00**2

        # fmt: off
        new_m00 = old_m00
        new_m10 = old_m10 + old_m00 * Tx
        new_m01 = old_m01 + old_m00 * Ty
        new_m11 = old_u11 + new_m01 / new_m00 * new_m10
        new_m20 = old_u20 + new_m10**2 / new_m00
        new_m02 = old_u02 + new_m01**2 / new_m00
        new_m21 = old_u21 + 2 * new_m10 / new_m00 * new_m11 + new_m01 / new_m00 * new_m20 - 2 * new_m10**2 / new_m00**2 * new_m01
        new_m12 = old_u12 + 2 * new_m01 / new_m00 * new_m11 + new_m10 / new_m00 * new_m02 - 2 * new_m01**2 / new_m00**2 * new_m10
        new_m30 = old_u30 + 3 * new_m10 / new_m00 * new_m20 - 2 * new_m10**3 / new_m00**2
        new_m03 = old_u03 + 3 * new_m01 / new_m00 * new_m02 - 2 * new_m01**3 / new_m00**2
        # fmt: on

        new_moments = copy.copy(self.moments)
        new_moments["m00"] = new_m00
        new_moments["m10"] = new_m10
        new_moments["m01"] = new_m01
        new_moments["m11"] = new_m11
        new_moments["m20"] = new_m20
        new_moments["m02"] = new_m02
        new_moments["m21"] = new_m21
        new_moments["m12"] = new_m12
        new_moments["m30"] = new_m30
        new_moments["m03"] = new_m03

        return self.__class__(new_moments, self.style, self.rotation_style)

    @cached_property
    def rotation_angle_2d(self) -> float:
        # from https://en.wikipedia.org/wiki/Image_moment
        u00, u20, u02, u11 = (
            self.central_moment(0, 0),
            self.central_moment(2, 0),
            self.central_moment(0, 2),
            self.central_moment(1, 1),
        )
        up20, up02, up11 = u20 / u00, u02 / u00, u11 / u00

        # phi = 0.5 * np.arctan((2*up11) / (up20 - up02))
        phi = 0.5 * np.arctan2((2 * up11), (up20 - up02))

        # convert to our standard angle coordinates, with 0 being on the
        # positive x-axis and the angle increasing counter-clockwise
        if phi < 0 and phi >= -np.pi / 2:
            phi = -phi
        elif phi <= np.pi / 2:
            phi = (np.pi / 2 - phi) + np.pi / 2
        else:
            lt.error_and_raise(
                RuntimeError,
                "Error in MomentsAnnotation.rotation_angle_2d: "
                + f"expected moments angle from atan2'/:: to be between -π/2 and π/2, but got {phi}.",
            )

        return phi

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        The orientation of the primary axis of a spot in the 2d image, in
        radians, about the z-axis.

        This is a decent approximation of actual orientation for images with a
        single spot that is shaped like an enlongated circle.
        """
        phi = self.rotation_angle_2d
        return scipy.spatial.transform.Rotation.from_euler("z", [phi])

    @property
    def size(self) -> list[float]:
        """Returns 0, always."""
        return [0]

    @property
    def cX(self) -> float:
        """Centroid X value"""
        return self.moments["m10"] / self.moments["m00"]

    @property
    def cY(self) -> float:
        """Centroid Y value"""
        return self.moments["m01"] / self.moments["m00"]

    @cached_property
    def centroid(self) -> p2.Pxy:
        """
        The centroid of the image. The centroid is similar in concept to the
        center of mass.
        """
        return p2.Pxy([[self.cX], [self.cY]])

    @cached_property
    def eccentricity_untested(self) -> float:
        """
        How elongated the image is.

        TODO The results from this method are untested. Someone who knows what
        this value means should contribute unit tests.
        """
        # from https://en.wikipedia.org/wiki/Image_moment
        u00, u20, u02, u11 = (
            self.central_moment(0, 0),
            self.central_moment(2, 0),
            self.central_moment(0, 2),
            self.central_moment(1, 1),
        )
        up20, up02, up11 = u20 / u00, u02 / u00, u11 / u00
        eigenval_1 = ((up20 + up02) / 2) + (np.sqrt(4 * up11 * up11 + (up20 - up02) ** 2) / 2)
        eigenval_2 = ((up20 + up02) / 2) - (np.sqrt(4 * up11 * up11 + (up20 - up02) ** 2) / 2)
        eccentricity = np.sqrt(1 - (eigenval_2 / eigenval_1))
        return eccentricity

    @cache
    def central_moment(self, p: int, q: int) -> float:
        """
        Returns the central moment for the given order p and order q.
        """
        # from https://en.wikipedia.org/wiki/Image_moment
        if p == 0 and q == 0:
            return self.moments["m00"]
        elif p == 0 and q == 1:
            return 0
        elif p == 1 and q == 0:
            return 0
        elif p == 1 and q == 1:
            return self.moments["m11"] - self.cY * self.moments["m10"]
        elif p == 2 and q == 0:
            return self.moments["m20"] - self.cX * self.moments["m10"]
        elif p == 0 and q == 2:
            return self.moments["m02"] - self.cY * self.moments["m01"]
        elif p == 2 and q == 1:
            m21, m11, m20, m01 = self.moments["m21"], self.moments["m11"], self.moments["m20"], self.moments["m01"]
            return m21 - 2 * self.cX * m11 - self.cY * m20 + 2 * self.cX * self.cX * m01
        elif p == 1 and q == 2:
            m12, m11, m02, m10 = self.moments["m12"], self.moments["m11"], self.moments["m02"], self.moments["m10"]
            return m12 - 2 * self.cY * m11 - self.cX * m02 + 2 * self.cY * self.cY * m10
        elif p == 3 and q == 0:
            m30, m20, m10 = self.moments["m30"], self.moments["m20"], self.moments["m10"]
            return m30 - 3 * self.cX * m20 + 2 * self.cX * self.cX * m10
        elif p == 0 and q == 3:
            m03, m02, m01 = self.moments["m03"], self.moments["m02"], self.moments["m01"]
            return m03 - 3 * self.cY * m02 + 2 * self.cY * self.cY * m01
        else:
            lt.error_and_raise(
                ValueError,
                "Error in MomentsAnnotation.central_moment(): "
                + f"formula for central moment with order (p={p}, q={q}) has not been implemented.",
            )

    # other uses of moments...

    def render_to_figure(
        self, fig_record: rcfr.RenderControlFigureRecord, image: np.ndarray = None, include_label: bool = False
    ):
        # draw the centroid marker
        label = None if not include_label else "centroid"
        fig_record.view.draw_pq(([self.cX], [self.cY]), self.style, label=label)

        # start by assuming that the plot is >= 30 pixels
        height, width, rotation_arrow_dist = None, None, 30
        if image is not None:
            height, width = it.dims_and_nchannels(image)[0]
            rotation_arrow_dist = min(width, height) * 0.2
        rot_mat_2d = self.rotation.as_matrix()[:, :2, :2].squeeze()
        rot_mat_2d_rev = rot_mat_2d.transpose()
        rotation_endpoints_1 = p2.Pxy([rotation_arrow_dist, 0]).rotate(rot_mat_2d) + self.centroid
        rotation_endpoints_2 = p2.Pxy([rotation_arrow_dist, 0]).rotate(rot_mat_2d_rev) + self.centroid
        rotation_endpoints_list = [(pnt.x[0], pnt.y[0]) for pnt in [rotation_endpoints_1, rotation_endpoints_2]]

        # Use the bounds of the image in determining where to put the end point
        # of the rotation arrow.
        if image is not None:
            image_tl = p2.Pxy([0, 0])
            image_br = p2.Pxy([width, height])
            if width > 80 and height > 80:
                if self.cX > 20 and self.cX < width - 20 and self.cY > 20 and self.cY < height - 20:
                    # Find where the rotation line intersects close to the edge of the
                    # image. We use a small buffer from the image edge in order to
                    # prevent Matplotlib from drawing outside the bounds of the image.
                    image_tl = p2.Pxy([20, 20])
                    image_br = p2.Pxy([width - 20, height - 20])
            image_bounds = r2.RectXY(image_tl, image_br)
            angle = self.rotation_angle_2d
            rotation_line = l2.LineXY.from_location_angle(self.centroid, -angle)  # images use an inverted y-axis
            intersections = p2.Pxy(image_bounds.loops[0].intersect_line(rotation_line))

            if len(intersections) > 0:
                rotation_endpoints_list = [(intersections.x[i], intersections.y[i]) for i in range(len(intersections))]

        # draw the rotation as an arrow
        style = copy.deepcopy(self.rotation_style)
        style.marker = "arrow"
        style.linestyle = "-"
        fig_record.view.draw_pq_list(rotation_endpoints_list, style=style)

    def __str__(self):
        return f"MomentsAnnotation"
