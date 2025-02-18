from opencsp.common.lib.geometry.Vxy import Vxy


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

    def distance(self, data_in: "Pxy") -> float:
        """
        Calculates the Euclidean distance between this point and another Pxy point.

        Parameters
        ----------
        data_in : Pxy
            The point to which the distance is to be calculated.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self._check_is_Vxy(data_in)
        return (self - data_in).magnitude()[0]

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
