import os
import subprocess
import sys
import time
import unittest

import opencsp.common.lib.process.subprocess_tools as subt
import opencsp.common.lib.process.lib.ProcessOutputLine as pol
import opencsp.common.lib.process.test.lib.subprocess_test_helper as helper


class TestSubprocess(unittest.TestCase):
    def test_echo(self):
        output = subt.run(f"echo 'hello, world!'")
        self.assertEqual(len(output), 1, f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].val.strip().strip("'"), "hello, world!", f"Unexpected output:\n\t{output}")

    def test_success(self):
        subt.run(f"{sys.executable} {helper.__file__} --retcode=0")

    def test_failure(self):
        func = lambda: subt.run(f"{sys.executable} {helper.__file__} --retcode=1")
        self.assertRaises(subprocess.CalledProcessError, func)

    def test_get_stdout(self):
        output = subt.run(f"{sys.executable} {helper.__file__} --simple_stdout", stdout="collect")
        self.assertEqual(len(output), 2, f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].val, "Hello", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[1].val, "world!", f"Unexpected output:\n\t{output}")
        for i in range(len(output)):
            self.assertEqual(output[i].is_err, False)
            self.assertEqual(output[i].lineno, i)

    def test_get_stderr(self):
        output = subt.run(f"{sys.executable} {helper.__file__} --simple_stderr", stderr="collect")
        self.assertEqual(len(output), 2, f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].val, "Goodbye", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[1].val, "world!", f"Unexpected output:\n\t{output}")
        for i in range(len(output)):
            self.assertEqual(output[i].is_err, True)
            self.assertEqual(output[i].lineno, i)

    def test_get_mixed_stdout_stderr(self):
        if os.name == "nt":
            # can't get mixed output on windows
            return
        output = subt.run(
            f"{sys.executable} {helper.__file__} --mixed_stdout_stderr", stdout="collect", stderr="collect"
        )
        self.assertEqual(len(output), 3, f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].val, "foo", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[1].val, "bar", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[2].val, "baz", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].is_err, False)
        self.assertEqual(output[1].is_err, True)
        self.assertEqual(output[2].is_err, False)
        for i in range(len(output)):
            self.assertEqual(output[i].lineno, i)

    def test_timeout_completes(self):
        """Test that a simple execution is able to complete before the timeout kills it."""
        start_time = time.time()
        output = subt.run(
            f"{sys.executable} {helper.__file__} --simple_stdout --delay_before=0.5",
            stdout="collect",
            stderr="collect",
            timeout=5.0,
        )
        self.assertLess(time.time() - start_time, 4.9)
        self.assertEqual(len(output), 2, f"Unexpected output:\n\t{output}")
        self.assertEqual(output[0].val.strip().strip("'"), "Hello", f"Unexpected output:\n\t{output}")
        self.assertEqual(output[1].val.strip().strip("'"), "world!", f"Unexpected output:\n\t{output}")

    def test_timeout_timesout(self):
        """Test that a program that executes in 0.9 seconds is killed by a 0.4s timeout."""
        with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            start_time = time.time()
            output = subt.run(
                f"{sys.executable} {helper.__file__} --simple_stdout --simple_stderr --delay_before=0.5 --delay_after=0.4",
                stdout="collect",
                stderr="collect",
                timeout=0.4,
            )
            self.assertLess(time.time() - start_time, 0.9)
            self.assertEqual(len(output), 0, f"Unexpected output:\n\t{output}")


if __name__ == "__main__":
    unittest.main()
