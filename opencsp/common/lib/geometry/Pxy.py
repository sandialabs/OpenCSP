import numpy as np
import numpy.typing as npt

from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.tool.log_tools as lt


class Pxy(Vxy):
    """
    A class representing a 2D point in space, inheriting from the Vxy class.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, data, dtype=float):
        """
        Initializes a Pxy object with the given data and data type.

        Parameters
        ----------
        data : array-like
            The coordinates of the point in 2D space.
        dtype : type, optional
            The data type of the point coordinates (default is float).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return "2D Point:\n" + self._data.__repr__()

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
        """
        Converts this Pxy point to a Vxy object.

        Returns
        -------
        Vxy
            A Vxy object representing the same point in 2D space.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return Vxy(self._data, self.dtype)
