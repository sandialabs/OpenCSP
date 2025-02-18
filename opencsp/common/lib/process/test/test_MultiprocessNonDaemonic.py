import random
import time
import unittest

import opencsp.common.lib.process.MultiprocessNonDaemonic as mnd


class TestSubprocess(unittest.TestCase):
    @staticmethod
    def echo(sval):
        print(sval)

    @staticmethod
    def add_two(ival):
        return ival + 2

    @staticmethod
    def add_two_randsleep(ival):
        sleep_deciseconds = random.randint(0, 20) / 10
        time.sleep(sleep_deciseconds)
        # print(ival)
        return ival + 2

    def test_single_process(self):
        """This test just verifies that the call to starmap completes."""
        pool = mnd.MultiprocessNonDaemonic(1)
        pool.starmap(self.echo, [["Hello, world!"]])

    def test_many_processes(self):
        """This test just verifies that the call to starmap completes."""
        pool = mnd.MultiprocessNonDaemonic(6)
        pool.starmap(self.echo, [[f"Hello, world {i}!"] for i in range(20)])

    def test_single_process_retval(self):
        """Test that we get the expected return value."""
        pool = mnd.MultiprocessNonDaemonic(1)
        results = pool.starmap(self.add_two, [[0]])
        self.assertEqual(len(results), 1, "Should get one result back from the single process")
        self.assertEqual(results[0], 2, "Subprocess should have added 2 to 0")

    def test_many_process_retval(self):
        """Test that we get the expected return values."""
        inputs = [[i] for i in range(20)]
        outputs = [i + 2 for i in range(20)]
        pool = mnd.MultiprocessNonDaemonic(6)
        results = pool.starmap(self.add_two, inputs)
        self.assertEqual(len(results), 20, "Should get 20 results back from the 20 processes")
        self.assertEqual(results, outputs, "Subprocess should have added 2 to each value")

    def test_many_process_retval_randsleep(self):
        """Test that we get the expected return values, even when the subprocesses complete at random times."""
        inputs = [[i] for i in range(20)]
        outputs = [i + 2 for i in range(20)]
        pool = mnd.MultiprocessNonDaemonic(20)
        results = pool.starmap(self.add_two_randsleep, inputs)
        self.assertEqual(len(results), 20, "Should get 20 results back from the 20 processes")
        self.assertEqual(sorted(results), outputs, "Subprocess should have added 2 to each value")
        self.assertEqual(results, outputs, "Output order from pool.starmap() should be the same as input order")


if __name__ == "__main__":
    unittest.main()
