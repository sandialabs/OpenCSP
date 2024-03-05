import numpy as np
from warnings import warn

from opencsp.common.lib.geometry.Vxyz import Vxyz


class TranslationXYZ:
    def __init__(self) -> None:
        warn(
            'TranslationXYZ is deprecated. Replace with Vxyz.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.trans_mtrx = np.zeros((3, 1))

    def from_vector(v: Vxyz):
        pass
