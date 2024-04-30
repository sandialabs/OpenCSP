import scipy.spatial

from opencsp.common.lib.cv.AbstractFiducials import AbstractFiducials
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class PointFiducials(AbstractFiducials):
    def __init__(self, style: rcps.RenderControlPointSeq = None, points: p2.Pxy = None):
        """
        A collection of pixel locations where points of interest are located in an image.
        """
        super().__init__(style)
        self.points = points

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        # TODO untested
        return reg.RegionXY.from_vertices(p2.Pxy((self.points.x[index], self.points.y[index])))

    @property
    def origin(self) -> p2.Pxy:
        return self.points

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError("Orientation is not yet implemented for PointFiducials")

    @property
    def size(self) -> list[float]:
        # TODO untested
        return [0] * len(self.points)

    @property
    def scale(self) -> list[float]:
        # TODO untested
        return [0] * len(self.points)
