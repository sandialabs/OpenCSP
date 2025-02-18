import numpy as np
import unittest

import opencsp.common.lib.tool.math_tools as mt


class TestMathTools(unittest.TestCase):
    def test_overlapping_range(self):
        overlap = mt.overlapping_range([1, 5], [3, 8])
        self.assertEqual(overlap, [3, 5])
        overlap = mt.overlapping_range([1, 5], [3, 8], [0, 4])
        self.assertEqual(overlap, [3, 4])
        overlap = mt.overlapping_range([6, 7], [7, 8])
        self.assertEqual(overlap, [7, 7])
        overlap = mt.overlapping_range([6, 7], [8, 9])
        self.assertEqual(overlap, [])
        overlap = mt.overlapping_range([], [])
        self.assertEqual(overlap, [])

    def test_overlapping_range_default(self):
        overlap = mt.overlapping_range([6, 7], [8, 9], default="hello")
        self.assertEqual(overlap, "hello")
        overlap = mt.overlapping_range([], [], default="world")
        self.assertEqual(overlap, "world")

    def test_rolling_average_invalid(self):
        with self.assertRaises(ValueError):
            mt.rolling_average([1, 2, 3, 4], 0)

    def test_rolling_average_datatype(self):
        in_np = np.ones((10))
        in_list = [1] * 10
        out_np = mt.rolling_average(in_np, 5)
        out_list = mt.rolling_average(in_list, 5)
        self.assertTrue(isinstance(out_np, np.ndarray))
        self.assertTrue(isinstance(out_list, list))

    def test_rolling_average_simple(self):
        # average of all ones should be all ones
        in10 = np.ones((10))
        out10 = mt.rolling_average(in10, 5)
        self.assertAlmostEqual(np.sum(in10 - out10), 0, msg=f"A: out10 is not all 1's: {out10}")
        out10 = mt.rolling_average(in10, 6)
        self.assertAlmostEqual(np.sum(in10 - out10), 0, msg=f"B: out10 is not all 1's: {out10}")

    def test_rolling_average_full(self):
        # average of all ones should be all ones
        in10 = np.ones((10))
        out10 = mt.rolling_average(in10, 10)
        self.assertAlmostEqual(np.sum(in10 - out10), 0)

        in11 = np.ones((11))
        out11 = mt.rolling_average(in11, 11)
        self.assertAlmostEqual(np.sum(in11 - out11), 0)

    def test_rolling_average_complex(self):
        in7 = [1, 2, 3, 4, 5, 6, 7]

        out2 = mt.rolling_average(in7, 2)
        self.assertAlmostEqual(out2[0], (1) / 1)
        self.assertAlmostEqual(out2[6], (6 + 7) / 2)

        out7 = mt.rolling_average(in7, 7)
        self.assertAlmostEqual(out7[0], (1 + 2 + 3 + 4) / 4)
        self.assertAlmostEqual(out7[1], (1 + 2 + 3 + 4 + 5) / 5)
        self.assertAlmostEqual(out7[2], (1 + 2 + 3 + 4 + 5 + 6) / 6)
        self.assertAlmostEqual(out7[3], (1 + 2 + 3 + 4 + 5 + 6 + 7) / 7)
        self.assertAlmostEqual(out7[4], (2 + 3 + 4 + 5 + 6 + 7) / 6)
        self.assertAlmostEqual(out7[5], (3 + 4 + 5 + 6 + 7) / 5)
        self.assertAlmostEqual(out7[6], (4 + 5 + 6 + 7) / 4)


if __name__ == "__main__":
    unittest.main()
