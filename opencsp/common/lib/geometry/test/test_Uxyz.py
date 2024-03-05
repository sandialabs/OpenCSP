import numpy as np

from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestUxyz:
    def test_Uxyz(self):
        data = np.array([[1], [2], [3]])
        pt = Uxyz(data)
        data_exp = data / np.sqrt(np.sum(data**2))
        np.testing.assert_almost_equal(pt.data, data_exp)

    def test_Uxyz_non_zero(self):
        with np.testing.assert_raises(ValueError):
            Uxyz([0, 0, 0])

    def test_as_Vxyz(self):
        data = np.array([[1], [2], [3]])
        pt = Uxyz(data).as_Vxyz()
        data_exp = data / np.sqrt(np.sum(data**2))

        np.testing.assert_almost_equal(pt.data, data_exp)
        assert type(pt) is Vxyz
