import numpy as np
from   scipy.spatial.transform import Rotation

from   opencsp.common.lib.geometry.Vxyz import Vxyz
from   opencsp.common.lib.geometry.TransformXYZ import TransformXYZ


class TestTransformXYZ:
    @classmethod
    def setup_class(cls):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        V = Vxyz((4, 5, 6))
        t = TransformXYZ.from_R_V(R, V)
        cls.transform = t
        cls.matrix = t.matrix
        return cls

    def test_mul(self):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        V = Vxyz((1., 2., 3.))
        t1 = TransformXYZ.from_R_V(R, V)

        R = Rotation.from_rotvec([0.4, 0.5, 0.6])
        V = Vxyz((4., 5., 6.))
        t2 = TransformXYZ.from_R_V(R, V)

        matrix = (t1 * t2).matrix
        matrix_exp = t1.matrix @ t2.matrix

        np.testing.assert_almost_equal(matrix, matrix_exp)

    def test_zero_zero(self):
        t = TransformXYZ.from_zero_zero()
        np.testing.assert_almost_equal(t.matrix, np.eye(4))

    def test_from_R_V(self):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        V = Vxyz((4, 5, 6))
        t = TransformXYZ.from_R_V(R, V)
        matrix_exp = np.eye(4)
        matrix_exp[:3, :3] = R.as_matrix()
        matrix_exp[:3, 3:4] = V.data

        np.testing.assert_almost_equal(t.matrix, matrix_exp)

    def test_from_R(self):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        t = TransformXYZ.from_R(R)
        matrix_exp = np.eye(4)
        matrix_exp[:3, :3] = R.as_matrix()

        np.testing.assert_almost_equal(t.matrix, matrix_exp)

    def test_from_V(self):
        V = Vxyz((4, 5, 6))
        t = TransformXYZ.from_V(V)
        matrix_exp = np.eye(4)
        matrix_exp[:3, 3:4] = V.data

        np.testing.assert_almost_equal(t.matrix, matrix_exp)

    def test_R(self):
        R = self.transform.R
        assert type(R) is Rotation
        rmat_exp = self.matrix[:3, :3]

        np.testing.assert_almost_equal(self.transform.R.as_matrix(), rmat_exp)
        np.testing.assert_almost_equal(self.transform.R_matrix, rmat_exp)

    def test_V(self):
        V = self.transform.V
        assert type(V) is Vxyz
        v_exp = self.matrix[:3, 3:4]

        np.testing.assert_almost_equal(self.transform.V.data, v_exp)
        np.testing.assert_almost_equal(self.transform.V_matrix, v_exp)

    def test_apply(self):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        V = Vxyz((4, 5, 6))
        t = TransformXYZ.from_R_V(R, V)

        V_1 = Vxyz((5, 5, 5))
        V_2 = t.apply(V_1)
        V_2_exp = V_1.rotate(R) + V

        np.testing.assert_almost_equal(V_2.data, V_2_exp.data)
