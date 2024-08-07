import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestVxyz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V1 = Vxyz((2, 2, 2))
        cls.V1_array = np.array([[2], [2], [2]])

        v2_1 = np.random.rand()
        v2_2 = np.random.rand()
        v2_3 = np.random.rand()
        cls.V2 = Vxyz((v2_1, v2_2, v2_3))
        cls.V2_array = np.array([[v2_1], [v2_2], [v2_3]])

    def test_Vxyz_length_1(self):
        # From tuple
        Vxyz((1, 2, 3))
        # From list
        Vxyz([1, 2, 3])
        # From ndarray
        Vxyz(np.array([[1], [2], [3]]))
        Vxyz(np.array([1, 2, 3]))
        Vxyz(np.zeros((1, 3)))

    def test_Vxyz_length_n(self):
        # From tuple
        x = y = z = (1, 2, 3, 4)
        Vxyz((x, y, z))
        Vxyz([x, y, z])
        # From list
        x = y = z = [1, 2, 3, 4]
        Vxyz((x, y, z))
        Vxyz([x, y, z])
        # From ndarray
        x = y = z = np.array((1, 2, 3, 4))
        Vxyz((x, y, z))
        Vxyz([x, y, z])
        Vxyz(np.zeros((3, 4)))

    def test_Vxyz_array_shape(self):
        # Raise ValueError if lelngth of first dimension is not 3
        with np.testing.assert_raises(ValueError):
            Vxyz(np.zeros((4, 4)))

    def test_Vxyz_copy_constructor(self):
        original = Vxyz([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        copy = Vxyz(original)
        self.assertEqual(copy.x.tolist(), [0, 1, 2])
        self.assertEqual(copy.y.tolist(), [3, 4, 5])
        self.assertEqual(copy.z.tolist(), [6, 7, 8])

    def test_from_list(self):
        # test single-valued Vxy instances
        a1 = Vxyz([0, 1, 2])
        b1 = Vxyz([3, 4, 5])
        c2 = Vxyz.from_list([a1, b1])
        self.assertEqual(len(c2), 2)
        self.assertEqual(c2.x.tolist(), [0, 3])
        self.assertEqual(c2.y.tolist(), [1, 4])
        self.assertEqual(c2.z.tolist(), [2, 5])

        # test multi-valued Vxyz instances
        a2 = Vxyz(list(zip([0, 1, 2], [3, 4, 5])))
        b3 = Vxyz(list(zip([6, 7, 8], [9, 10, 11], [12, 13, 14])))
        c4 = Vxyz(list(zip([15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26])))
        d9 = Vxyz.from_list([a2, b3, c4])
        self.assertEqual(len(d9), 9)
        self.assertEqual(d9.x.tolist(), [0, 3, 6, 9, 12, 15, 18, 21, 24])
        self.assertEqual(d9.y.tolist(), [1, 4, 7, 10, 13, 16, 19, 22, 25])
        self.assertEqual(d9.z.tolist(), [2, 5, 8, 11, 14, 17, 20, 23, 26])

    def test_xyz(self):
        assert self.V1.x[0] == self.V1_array[0]
        assert self.V1.y[0] == self.V1_array[1]
        assert self.V1.z[0] == self.V1_array[2]

    def test_add(self):
        # Vxyz
        V = self.V1 + self.V2
        V_exp = self.V1_array + self.V2_array
        np.testing.assert_almost_equal(V.data, V_exp)

        # Float, int, string
        with np.testing.assert_raises(TypeError):
            self.V1 + int(1)
        with np.testing.assert_raises(TypeError):
            self.V1 + 1.0
        with np.testing.assert_raises(TypeError):
            self.V1 + '1'

    def test_sub(self):
        # Vxyz
        V = self.V1 - self.V2
        V_exp = self.V1_array - self.V2_array
        np.testing.assert_almost_equal(V.data, V_exp)

        # Float, int, string
        with np.testing.assert_raises(TypeError):
            self.V1 - int(1)
        with np.testing.assert_raises(TypeError):
            self.V1 - 1.0
        with np.testing.assert_raises(TypeError):
            self.V1 - '1'

    def test_mul(self):
        # Vxyz
        V = self.V1 * self.V2
        V_exp = self.V1_array * self.V2_array
        np.testing.assert_almost_equal(V.data, V_exp)

        # Integer
        V = self.V2 * int(2)
        V_exp = self.V2_array * int(2)
        np.testing.assert_almost_equal(V.data, V_exp)

        # Float
        V = self.V2 * float(2.2)
        V_exp = self.V2_array * float(2.2)
        np.testing.assert_almost_equal(V.data, V_exp)

        # String
        with np.testing.assert_raises(TypeError):
            self.V2 * '2'

    def test_len(self):
        # Length > 3
        V = Vxyz(np.ones((3, 4)))
        assert 4 == len(V)

        # Length < 3
        V = Vxyz(np.ones((3, 2)))
        assert 2 == len(V)

        # Length = 3
        V = Vxyz(np.ones((3, 3)))
        assert 3 == len(V)

        # Length = 1
        V = Vxyz(np.ones((3, 1)))
        assert 1 == len(V)

    def test_neg(self):
        np.testing.assert_almost_equal(-(self.V1).data, -self.V1_array)

    def test_getitem(self):
        # Length 1
        V = self.V2[0]
        np.testing.assert_almost_equal(V.data, self.V2.data)

        # Length N
        x = y = z = np.arange(5)
        V = Vxyz((x, y, z))[2]
        V_exp = Vxyz([2, 2, 2])
        np.testing.assert_almost_equal(V.data, V_exp.data)

    def test_magnitude(self):
        mag = self.V1.magnitude()
        mag_exp = np.array([np.sqrt(np.sum(self.V1_array**2))])
        np.testing.assert_almost_equal(mag, mag_exp)

    def test_normalize(self):
        # Zero magnitude error
        with np.testing.assert_raises(ValueError):
            Vxyz([0, 0, 0]).normalize()

        # Normalize copy
        array = np.array([4, 5, 6])
        V = Vxyz(array)

        norm = V.normalize()
        norm_exp = array / np.sqrt(np.sum(array**2))

        np.testing.assert_almost_equal(norm.data.squeeze(), norm_exp)

    def test_normalize_in_place(self):
        # Normalize in place
        array = np.array([4, 5, 6])
        V = Vxyz(array)

        V.normalize_in_place()
        norm_exp = array / np.sqrt(np.sum(array**2))

        np.testing.assert_almost_equal(V.data.squeeze(), norm_exp)

    def test_rotate(self):
        V = Vxyz([0, 0, 1])
        R = Rotation.from_rotvec([np.pi / 2, 0, 0])

        # Type error
        with np.testing.assert_raises(TypeError):
            V_rot = V.rotate(1)

        # Rotate about X
        V_rot = V.rotate(R)
        np.testing.assert_almost_equal(V_rot.data.squeeze(), np.array([0, -1, 0]))

    def test_rotate_about(self):
        V = Vxyz((0, 10, 10))
        V_pivot = Vxyz((0, 5, 10))
        R = Rotation.from_rotvec([0, 0, np.pi / 2])

        V_rot = V.rotate_about(R, V_pivot)
        V_exp = Vxyz([-5, 5, 10])

        np.testing.assert_almost_equal(V_rot.data, V_exp.data)

    def test_rotate_in_place(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.rotate(1)

        V = Vxyz([0, 0, 1])
        R = Rotation.from_rotvec([np.pi / 2, 0, 0])

        # Rotate about X
        V.rotate_in_place(R)
        np.testing.assert_almost_equal(V.data.squeeze(), np.array([0, -1, 0]))

    def test_rotate_about_in_place(self):
        V = Vxyz((0, 10, 10))
        V_pivot = Vxyz((0, 5, 10))
        R = Rotation.from_rotvec([0, 0, np.pi / 2])

        V.rotate_about_in_place(R, V_pivot)
        V_exp = Vxyz([-5, 5, 10])

        np.testing.assert_almost_equal(V.data, V_exp.data)

    def test_dot(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.dot(1)

        # Length 1
        Vy = Vxyz((0, 2, 0))
        Vz = Vxyz((0, 0, 2))

        assert Vy.dot(Vz)[0] == 0
        assert Vy.dot(Vy)[0] == 4

        # Length N dot length N
        N = 5
        y = z = np.arange(N)
        zero = np.zeros(N)
        Vy = Vxyz((zero, y, zero))
        Vz = Vxyz((zero, zero, z))

        dot = Vy.dot(Vz)
        np.testing.assert_almost_equal(dot, np.zeros(N))

        dot = Vy.dot(Vy)
        np.testing.assert_almost_equal(dot, y**2)

        # Length N dot length 1
        N = 5
        y = z = np.arange(N)
        zero = np.zeros(N)
        Vy = Vxyz((zero, y, zero))
        Vy_1 = Vxyz((0, 2, 0))
        Vz = Vxyz((0, 0, 2))

        dot = Vy.dot(Vz)
        np.testing.assert_almost_equal(dot, np.zeros(N))

        dot = Vy.dot(Vy_1)
        np.testing.assert_almost_equal(dot, y * 2)

    def test_cross(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.cross(1)

        # Value error
        with np.testing.assert_raises(ValueError):
            V1 = Vxyz(np.ones((3, 2)))
            V2 = Vxyz(np.ones((3, 4)))
            V1.cross(V2)

        # Length 1 x length 1
        Vx = Vxyz((2, 0, 0))
        Vy = Vxyz((0, 3, 0))
        V_out = Vx.cross(Vy)
        np.testing.assert_almost_equal(V_out.data, np.array([[0], [0], [6]]))

        # Length 1 x length N
        Vx = Vxyz((2, 0, 0))
        Vy = Vxyz(([0, 0], [3, 3], [0, 0]))
        V_out = Vx.cross(Vy)
        np.testing.assert_almost_equal(V_out.data, np.array(([0, 0], [0, 0], [6, 6])))

        # Length N x length 1
        Vx = Vxyz(([2, 2], [0, 0], [0, 0]))
        Vy = Vxyz((0, 3, 0))
        V_out = Vx.cross(Vy)
        np.testing.assert_almost_equal(V_out.data, np.array(([0, 0], [0, 0], [6, 6])))

    def test_align_to(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.align_to(1)

        # Value error, 1 x N
        with np.testing.assert_raises(ValueError):
            V1 = Vxyz(np.ones((3, 1)))
            V2 = Vxyz(np.ones((3, 4)))
            V1.align_to(V2)

        # Value error, N x 1
        with np.testing.assert_raises(ValueError):
            V1 = Vxyz(np.ones((3, 4)))
            V2 = Vxyz(np.ones((3, 1)))
            V1.align_to(V2)

        # Value error, N x N
        with np.testing.assert_raises(ValueError):
            V1 = Vxyz(np.ones((3, 4)))
            V2 = Vxyz(np.ones((3, 4)))
            V1.align_to(V2)

        # Align
        Vx = Vxyz((2, 0, 0))
        Vy = Vxyz((0, 3, 0))
        r_out = Vx.align_to(Vy)
        np.testing.assert_almost_equal(r_out.as_rotvec(), np.array([0, 0, np.pi / 2]))


if __name__ == "__main__":
    unittest.main()
