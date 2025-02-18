import numpy as np
import os
import subprocess
import sys
import time
import unittest

import opencsp.common.lib.geometry.angle as angle
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestFileTools(unittest.TestCase):
    def test_normalize_return_same_type(self):
        self.assertIsInstance(angle.normalize(0.0), float)
        self.assertIsInstance(angle.normalize([0.0]), np.ndarray)
        self.assertIsInstance(angle.normalize(tuple([0.0])), np.ndarray)
        self.assertIsInstance(angle.normalize(np.array([0.0])), np.ndarray)

    def test_normalize_single_angle(self):
        twopi = np.pi * 2

        self.assertAlmostEqual(angle.normalize(0.1), 0.1, 7)
        self.assertAlmostEqual(angle.normalize(np.pi), np.pi, 7)
        self.assertAlmostEqual(angle.normalize(twopi - 0.1), twopi - 0.1, 7)

        self.assertAlmostEqual(angle.normalize(-0.1), twopi - 0.1, 7)
        self.assertAlmostEqual(angle.normalize(-np.pi), np.pi, 7)
        self.assertAlmostEqual(angle.normalize(-twopi + 0.1), 0.1, 7)

        self.assertAlmostEqual(angle.normalize(twopi + 0.1), 0.1, 7)
        self.assertAlmostEqual(angle.normalize(-np.pi), np.pi, 7)
        self.assertAlmostEqual(angle.normalize(-twopi + 0.1), 0.1, 7)

    def test_normalize_single_angle_array(self):
        twopi = np.pi * 2

        np.testing.assert_almost_equal(angle.normalize([0.1]), [0.1], 7)
        np.testing.assert_almost_equal(angle.normalize([np.pi]), [np.pi], 7)
        np.testing.assert_almost_equal(angle.normalize([twopi - 0.1]), [twopi - 0.1], 7)

        np.testing.assert_almost_equal(angle.normalize([-0.1]), [twopi - 0.1], 7)
        np.testing.assert_almost_equal(angle.normalize([-np.pi]), [np.pi], 7)
        np.testing.assert_almost_equal(angle.normalize([-twopi + 0.1]), [0.1], 7)

        np.testing.assert_almost_equal(angle.normalize([twopi + 0.1]), [0.1], 7)
        np.testing.assert_almost_equal(angle.normalize([-np.pi]), [np.pi], 7)
        np.testing.assert_almost_equal(angle.normalize([-twopi + 0.1]), [0.1], 7)

    def test_normalize_many_angles(self):
        twopi = np.pi * 2

        input_expected_vals = [
            [0.1, 0.1],
            [np.pi, np.pi],
            [twopi - 0.1, twopi - 0.1],
            [-0.1, twopi - 0.1],
            [-np.pi, np.pi],
            [-twopi + 0.1, 0.1],
            [twopi + 0.1, 0.1],
            [-np.pi, np.pi],
            [-twopi + 0.1, 0.1],
        ]
        input_vals = [v[0] for v in input_expected_vals]
        expected_vals = [v[1] for v in input_expected_vals]

        np.testing.assert_almost_equal(angle.normalize(input_vals), expected_vals, 7)


if __name__ == "__main__":
    unittest.main()
