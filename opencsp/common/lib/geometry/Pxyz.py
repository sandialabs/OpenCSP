from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class Pxyz(Vxyz):
    def __init__(self, data, dtype=float):
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return '3D Point:\n' + self._data.__repr__()

    def distance(self, data_in: "Pxyz") -> float:
        """Calculates the euclidian distance between this point and the data_in point."""
        self._check_is_Vxyz(data_in)
        return (self - data_in).magnitude()[0]

    def as_Vxyz(self):
        return Vxyz(self._data, self.dtype)

    @classmethod
    def empty(cls):
        return Pxyz([[], [], []])

    def draw_point(
        self,
        figure: rcfr.RenderControlFigureRecord | v3d.View3d,
        style: rcps.RenderControlPointSeq = None,
        labels: list[str] = None,
    ):
        """Calls figure.draw_xyz(p) for all points in this instance, and with
        the default arguments in place for any None's."""
        if style is None:
            style = rcps.default(markersize=2)
        if labels is None:
            labels = [None] * len(self)
        view = figure if isinstance(figure, v3d.View3d) else figure.view
        for x, y, z, label in zip(self.x, self.y, self.z, labels):
            view.draw_xyz((x, y, z), style, label)
