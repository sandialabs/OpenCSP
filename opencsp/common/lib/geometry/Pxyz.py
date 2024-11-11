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
