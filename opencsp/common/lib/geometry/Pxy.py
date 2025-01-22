import numpy as np
import numpy.typing as npt

from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.tool.log_tools as lt


class Pxy(Vxy):
    def __init__(self, data, dtype=float):
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return '2D Point:\n' + self._data.__repr__()

    def distance(self, data_in: "Pxy") -> npt.NDArray[np.float_]:
        """Calculates the euclidian distance between this point and the data_in point."""
        self._check_is_Vxy(data_in)

        # broadcast input point to the same number of points as self
        if len(data_in) != len(self):
            if len(data_in) == 1:
                data_in = Pxy([[data_in.x[0]] * len(self), [data_in.y[0]] * len(self)])
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in Pxy.angle_from(): "
                    + f"'data_in' must be of length 1, or the same length as the destination instance ({len(self)=}).",
                )

        return (self - data_in).magnitude()

    def angle_from(self, origin: "Pxy") -> npt.NDArray[np.float_]:
        """
        Returns the rotation angle in which this point lies relative to the
        given origin point.
        """
        self._check_is_Vxy(origin)

        # broadcast input point to the same number of points as self
        if len(origin) != len(self):
            if len(origin) == 1:
                origin = Pxy([[origin.x[0]] * len(self), [origin.y[0]] * len(self)])
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in Pxy.angle_from(): "
                    + f"'origin' must be of length 1, or the same length as the destination instance ({len(self)=}).",
                )

        return (self - origin).angle()

    def as_Vxy(self):
        return Vxy(self._data, self.dtype)
