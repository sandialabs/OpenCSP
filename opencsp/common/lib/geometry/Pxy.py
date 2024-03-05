from opencsp.common.lib.geometry.Vxy import Vxy


class Pxy(Vxy):
    def __init__(self, data, dtype=float):
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return '2D Point:\n' + self._data.__repr__()

    def distance(self, data_in: "Pxy") -> float:
        """Calculates the euclidian distance between this point and the data_in point."""
        self._check_is_Vxy(data_in)
        return (self - data_in).magnitude()[0]

    def as_Vxy(self):
        return Vxy(self._data, self.dtype)
