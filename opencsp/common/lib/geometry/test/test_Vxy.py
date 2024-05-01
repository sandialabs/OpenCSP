import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.tool.file_tools as ft


class TestVxy(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "Vxy")
        self.out_dir = os.path.join(path, "data", "output", "Vxy")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        self.V1 = Vxy((2, 2))
        self.V1_array = np.array([[2], [2]])

        self.V2 = Vxy((1, 2))
        self.V2_array = np.array([[1], [2]])

    def test_Vxy_length_1(self):
        # From tuple
        Vxy((1, 2))
        # From list
        Vxy([1, 2])
        # From ndarray
        Vxy(np.array([[1], [2]]))
        Vxy(np.array([1, 2]))
        Vxy(np.zeros((1, 2)))

    def test_Vxy_length_n(self):
        # From tuple
        x = y = (1, 2, 3, 4)
        Vxy((x, y))
        Vxy([x, y])
        # From list
        x = y = [1, 2, 3, 4]
        Vxy((x, y))
        Vxy([x, y])
        # From ndarray
        x = y = np.array((1, 2, 3, 4))
        Vxy((x, y))
        Vxy([x, y])
        Vxy(np.zeros((2, 4)))

    def test_Vxy_array_shape(self):
        # Raise ValueError if lelngth of first dimension is not 2
        with np.testing.assert_raises(ValueError):
            Vxy(np.zeros((4, 4)))

    def test_xy(self):
        assert self.V1.x[0] == self.V1_array[0]
        assert self.V1.y[0] == self.V1_array[1]

    def test_add(self):
        # Vxy
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
        # Vxy
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
        # Vxy
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
        # Length > 2
        V = Vxy(np.ones((2, 3)))
        assert 3 == len(V)

        # Length = 2
        V = Vxy(np.ones((2, 2)))
        assert 2 == len(V)

        # Length = 1
        V = Vxy(np.ones((2, 1)))
        assert 1 == len(V)

    def test_neg(self):
        np.testing.assert_almost_equal(-(self.V1).data, -self.V1_array)

    def test_getitem(self):
        # Length 1
        V = self.V2[0]
        np.testing.assert_almost_equal(V.data, self.V2.data)

        # Length N
        x = y = np.arange(5)
        V = Vxy((x, y))[2]
        V_exp = Vxy([2, 2])
        np.testing.assert_almost_equal(V.data, V_exp.data)

    def test_magnitude(self):
        mag = self.V1.magnitude()
        mag_exp = np.array([np.sqrt(np.sum(self.V1_array**2))])
        np.testing.assert_almost_equal(mag, mag_exp)

    def test_normalize(self):
        # Zero magnitude error
        with np.testing.assert_raises(ValueError):
            Vxy([0, 0]).normalize()

        # Normalize copy
        array = np.array([4, 5])
        V = Vxy(array)

        norm = V.normalize()
        norm_exp = array / np.sqrt(np.sum(array**2))

        np.testing.assert_almost_equal(norm.data.squeeze(), norm_exp)

    def test_normalize_in_place(self):
        # Normalize in place
        array = np.array([4, 5])
        V = Vxy(array)

        V.normalize_in_place()
        norm_exp = array / np.sqrt(np.sum(array**2))

        np.testing.assert_almost_equal(V.data.squeeze(), norm_exp)

    def test_rotate(self):
        V = Vxy([0, 1])
        R = np.array([[0, -1], [1, 0]])  # CCW 90deg

        # Type error
        with np.testing.assert_raises(TypeError):
            V_rot = V.rotate(1)

        # Rotate about X
        V_rot = V.rotate(R)
        np.testing.assert_almost_equal(V_rot.data.squeeze(), np.array([-1, 0]))

    def test_rotate_in_place(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.rotate_in_place(1)

        V = Vxy([0, 1])
        R = np.array([[0, -1], [1, 0]])  # CCW 90deg

        # Rotate about X
        V.rotate_in_place(R)
        np.testing.assert_almost_equal(V.data.squeeze(), np.array([-1, 0]))

    def test_rotate_about(self):
        V = Vxy([0, 10])
        V_pivot = Vxy([0, 5])
        R = np.array([[0, -1], [1, 0]])  # CCW 90deg

        # Type error
        with np.testing.assert_raises(TypeError):
            V.rotate_about(1)

        # Rotate about X
        V_out = V.rotate_about(R, V_pivot)
        np.testing.assert_almost_equal(V_out.data.squeeze(), np.array([-5, 5]))

    def test_rotate_about_in_place(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.rotate_about_in_place(1)

        V = Vxy([0, 10])
        V_pivot = Vxy([0, 5])
        R = np.array([[0, -1], [1, 0]])  # CCW 90deg

        # Rotate about X
        V.rotate_about_in_place(R, V_pivot)
        np.testing.assert_almost_equal(V.data.squeeze(), np.array([-5, 5]))

    def test_dot(self):
        # Type error
        with np.testing.assert_raises(TypeError):
            self.V1.dot(1)

        # Length 1
        Vy = Vxy((2, 0))
        Vz = Vxy((0, 2))

        assert Vy.dot(Vz)[0] == 0
        assert Vy.dot(Vy)[0] == 4

        # Length N dot length N
        N = 5
        x = y = np.arange(N)
        zero = np.zeros(N)
        Vx = Vxy((x, zero))
        Vy = Vxy((zero, y))

        dot = Vx.dot(Vy)
        np.testing.assert_almost_equal(dot, np.zeros(N))

        dot = Vy.dot(Vy)
        np.testing.assert_almost_equal(dot, y**2)

        # Length N dot length 1
        N = 5
        x = y = np.arange(N)
        zero = np.zeros(N)
        Vx = Vxy((2, 0))
        Vy = Vxy((zero, y))
        Vy_1 = Vxy((0, 2))

        dot = Vx.dot(Vy)
        np.testing.assert_almost_equal(dot, np.zeros(N))

        dot = Vy.dot(Vy_1)
        np.testing.assert_almost_equal(dot, y * 2)

    def test_cross(self):
        # Length 1 and Length N
        a = Vxy(([1], [1])).normalize()
        b = Vxy(([1, 1, 1], [0, 0, 0])).normalize()
        res = a.cross(b)
        res_exp = np.array([-1 / np.sqrt(2)] * 3)
        np.testing.assert_almost_equal(res, res_exp)

        # Length N and Length 1
        a = Vxy(([1], [1])).normalize()
        b = Vxy(([1, 1, 1], [0, 0, 0])).normalize()
        res = b.cross(a)
        res_exp = np.array([1 / np.sqrt(2)] * 3)
        np.testing.assert_almost_equal(res, res_exp)

        # Length 2 and Length N
        a = Vxy(([1, 1], [1, 1])).normalize()
        b = Vxy(([1, 1, 1], [0, 0, 0])).normalize()
        with np.testing.assert_raises(ValueError):
            res = a.cross(b)

    def test_draw(self):
        fig = fm.mpl_pyplot_figure()
        ax = plt.gca()
        self.V1.draw(ax)
        plt.close(fig)

    def test_astuple(self):
        # Length 1
        a = Vxy((2, 3))
        ax, ay = a.astuple()
        self.assertEqual(ax, 2)
        self.assertEqual(ay, 3)

        # Length 2
        b = Vxy(np.array([[2, 4], [3, 5]]))
        with self.assertRaises(Exception):
            b.astuple()


if __name__ == '__main__':
    unittest.main()
