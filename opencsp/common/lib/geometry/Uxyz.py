from opencsp.common.lib.geometry.Vxyz import Vxyz


class Uxyz(Vxyz):
    """
    A class representing a 3D unit vector.

    The Uxyz class extends the Vxyz class to specifically represent unit vectors
    in three-dimensional space. Upon initialization, the vector is normalized to
    ensure it has a magnitude of 1.

    Attributes
    ----------
    _data : np.ndarray
        The underlying data representing the vector, normalized to unit length.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, data, dtype=float):
        """
        Instantiate class for representing a 3D unit vector.

        Parameters
        ----------
        data : array-like
            The input data for the vector, which can be a list, tuple, or
            NumPy array of length 3.
        dtype : data type, optional
            The data type of the vector elements. The default is float.

        Returns
        -------
        Uxyz
            An instance of the Uxyz class representing a unit vector.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Initialize vector
        super().__init__(data, dtype)

        # Normalize
        self.normalize_in_place()

    def __repr__(self):
        return "3D Unit Vector:\n" + self._data.__repr__()

    def cross(self, V) -> Vxyz:
        """
        Calculates cross product. Similar to Vxyz.cross(), but the output is
        not normalized. See Vxyz.cross() for more information.

        Returns
        -------
        Vxyz
            3D vector. The output is not normalized.

        """
        # Convert inputs to Vxyz
        a = self.as_Vxyz()
        if type(V) is Uxyz:
            b = V.as_Vxyz()
        else:
            b = V

        return a.cross(b)

    def as_Vxyz(self) -> Vxyz:
        """
        Converts Uxyz to Vxyz.

        Returns
        -------
        Vxyz
            Vxyz version of Uxyz object.

        """
        return Vxyz(self._data, self.dtype)
