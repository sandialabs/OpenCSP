import numpy as np

from opencsp.common.lib.geometry.Uxy import Uxy
from opencsp.common.lib.geometry.Vxy import Vxy


class TestUxyz:
    def test_Uxy(self):
        data = np.array([[1], [2]])
        pt = Uxy(data)
        data_exp = data / np.sqrt(np.sum(data**2))
        np.testing.assert_almost_equal(pt.data, data_exp)

    def test_Uxyz_non_zero(self):
        with np.testing.assert_raises(ValueError):
            Uxy([0, 0])

    def test_as_Vxy(self):
        data = np.array([[1], [2]])
        pt = Uxy(data).as_Vxy()
        data_exp = data / np.sqrt(np.sum(data**2))

        np.testing.assert_almost_equal(pt.data, data_exp)
        assert type(pt) is Vxy
