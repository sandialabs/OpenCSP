import unittest

import opencsp.common.lib.tool.time_date_tools as tdt


class TestTimeDateTools(unittest.TestCase):
    def test_conversion(self):
        when_ymdhmsz = [2020, 12, 3, 15, 44, 13, -6]  # Dec 3rd dataset
        when_dt = tdt.to_datetime(when_ymdhmsz)
        when_ymdhmsz2 = tdt.from_datetime(when_dt)

        self.assertEqual(when_ymdhmsz, when_ymdhmsz2)


if __name__ == "__main__":
    unittest.main()
