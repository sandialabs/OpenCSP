from opencsp.common.lib.geometry.Vxy import Vxy


class Uxy(Vxy):
    """
    A class representing a 2D unit vector.

    This class extends the Vxy class to ensure that the vector is normalized upon initialization,
    representing a direction in 2D space with a magnitude of 1.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, data, dtype=float):
        """
        Initializes the Uxy instance and normalizes the vector.

        Parameters
        ----------
        data : array-like
            The initial data for the vector, which should be a 2D vector.
        dtype : type, optional
            The data type of the vector elements. Defaults to float.

        Raises
        ------
        ValueError
            If the provided data does not represent a valid 2D vector.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Initialize vector
        super().__init__(data, dtype)

        # Normalize
        self.normalize_in_place()

    def __repr__(self):
        return "2D Unit Vector:\n" + self._data.__repr__()

    def as_Vxy(self):
        """
        Converts the Uxy instance to a Vxy instance.

        Returns
        -------
        Vxy
            A Vxy instance representing the same vector data.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return Vxy(self._data, self.dtype)
