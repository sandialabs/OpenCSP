from opencsp.common.lib.geometry.Vxyz import Vxyz


class Uxyz(Vxyz):
    def __init__(self, data, dtype=float):
        # Initialize vector
        super().__init__(data, dtype)

        # Normalize
        self.normalize_in_place()

    def __repr__(self):
        return '3D Unit Vector:\n' + self._data.__repr__()

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
