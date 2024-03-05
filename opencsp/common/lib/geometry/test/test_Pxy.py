import numpy as np

from   opencsp.common.lib.geometry.Pxy import Pxy
from   opencsp.common.lib.geometry.Vxy import Vxy


class TestPxy:
    def test_Pxy(self):
        data = np.array([[1], [2]])
        pt = Pxy(data)
        np.testing.assert_almost_equal(pt.data, data)

    def test_as_Vxy(self):
        data = np.array([[1], [2]])
        pt = Pxy(data).as_Vxy()

        np.testing.assert_almost_equal(pt.data, data)
        assert type(pt) is Vxy
