from   opencsp.common.lib.geometry.Vxy import Vxy


class Uxy(Vxy):
    def __init__(self, data, dtype=float):
        # Initialize vector
        super().__init__(data, dtype)

        # Normalize
        self.normalize_in_place()

    def __repr__(self):
        return '3D Unit Vector:\n' + self._data.__repr__()

    def as_Vxy(self):
        return Vxy(self._data, self.dtype)
