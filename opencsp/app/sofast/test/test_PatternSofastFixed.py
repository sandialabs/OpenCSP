import unittest

import numpy as np

from opencsp.app.sofast.lib.PatternSofastFixed import PatternSofastFixed


class TestPatternSofastFixed(unittest.TestCase):
    def test_FixedPatternDisplay(self):
        # Instantiate
        pattern = PatternSofastFixed(100, 100, 10, 10)

        # Test screen fractions
        np.testing.assert_allclose(pattern.x_locs_frac, np.array([0.09, 0.29, 0.49, 0.69, 0.89]), rtol=0, atol=1e-6)
        np.testing.assert_allclose(pattern.y_locs_frac, np.array([0.09, 0.29, 0.49, 0.69, 0.89]), rtol=0, atol=1e-6)

        # Test indices
        np.testing.assert_equal(pattern.x_indices, np.array([-2, -1, 0, 1, 2]))
        np.testing.assert_equal(pattern.y_indices, np.array([-2, -1, 0, 1, 2]))

        # Calculate image
        im = pattern.get_image("uint8", 255)
        np.testing.assert_array_equal([100, 100, 3], im.shape)


if __name__ == "__main__":
    unittest.main()
