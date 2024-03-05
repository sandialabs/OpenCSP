import numpy as np

from   opencsp.common.lib.geometry.Pxyz import Pxyz
from   opencsp.common.lib.geometry.Vxyz import Vxyz


class TestPxyz:
    def test_Pxyz(self):
        data = np.array([[1], [2], [3]])
        pt = Pxyz(data)
        np.testing.assert_almost_equal(pt.data, data)

    def test_as_Vxyz(self):
        data = np.array([[1], [2], [3]])
        pt = Pxyz(data).as_Vxyz()

        np.testing.assert_almost_equal(pt.data, data)
        assert type(pt) is Vxyz
