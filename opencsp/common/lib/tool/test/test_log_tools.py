import os
import subprocess
import sys
import time
import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestLogTools(unittest.TestCase):
    """This test class makes heavy use of subprocesses in order to allow the
    log files to be deleted after they have been used for each test.

    The basic flow for each test is:
        0. (in setUp) Prepare a file name to log.
        1. Start _log_[test_name](log_name) as a separate process. We do this
           in a separate process so that the file can be deleted in the main
           process.
        2. Check the log contents for the expected results.
        3. (in tearDown) Delete the temporary log file.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.out_dir = os.path.join('common', 'lib', 'test', 'data', 'output', 'tool', 'log_tools')
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        super().setUpClass()

    def setUp(self) -> None:
        test_method = self.id().split(".")[-1]
        self.log_dir_body_ext = os.path.join(self.out_dir, test_method + ".txt")

    def proc_exec(self, func_name1: str, func_name2: str = None):
        stdout1, stdout2 = "", ""

        with subprocess.Popen(
            [sys.executable, __file__, "--funcname", func_name1, "--logname", self.log_dir_body_ext],
            stdout=subprocess.PIPE,
        ) as proc1:
            if func_name2 != None:
                with subprocess.Popen(
                    [sys.executable, __file__, "--funcname", func_name2, "--logname", self.log_dir_body_ext],
                    stdout=subprocess.PIPE,
                ) as proc2:
                    pass

                    if proc2 is not None and proc2.stdout is not None:
                        stdout2 = proc2.stdout.read().decode('utf-8')

            if proc1 is not None and proc1.stdout is not None:
                stdout1 = proc1.stdout.read().decode('utf-8')

        return stdout1 + stdout2

    def get_log_contents(self, preserve_lines=False):
        lines = ft.read_text_file(self.log_dir_body_ext)
        if preserve_lines:
            return lines
        return "\n".join(lines)

    def _log_single_process_logger(self, logname):
        lt.logger(logname)
        lt.info("Hello, world!")

    def test_single_process_logger(self):
        self.proc_exec("_log_single_process_logger")
        log_contents = self.get_log_contents()
        self.assertTrue("Hello, world!" in log_contents, f"Can't find hello log in log contents:\n\t\"{log_contents}\"")

    def _log_single_process_dont_delete(self, logname):
        lt.logger(logname, delete_existing_log=False)
        lt.info("Goodbye, world!")

    def test_single_process_dont_delete(self):
        with open(self.log_dir_body_ext, "w") as fout:
            fout.write("Hello, world!")
        self.proc_exec("_log_single_process_dont_delete")

        log_contents = self.get_log_contents()
        self.assertTrue("Hello, world!" in log_contents, f"Can't find hello log in log contents:\n\t\"{log_contents}\"")
        self.assertTrue(
            "Goodbye, world!" in log_contents, f"Can't find goodbye log in log contents:\n\t\"{log_contents}\""
        )

    def _log_multiprocess_logger1(self, logname):
        lt.multiprocessing_logger(logname)
        lt.info("Hello, world!")
        time.sleep(0.2)
        lt.info("Goodbye, world!")

    def _log_multiprocess_logger2(self, logname):
        time.sleep(0.1)
        lt.multiprocessing_logger(logname)
        lt.info("other process")

    def test_multiprocess_logger(self):
        self.proc_exec("_log_multiprocess_logger1", "_log_multiprocess_logger2")

        log_contents = self.get_log_contents()
        self.assertTrue("Hello, world!" in log_contents, f"Can't find hello log in log contents:\n\t\"{log_contents}\"")
        self.assertTrue("other process" in log_contents, f"Can't find other log in log contents:\n\t\"{log_contents}\"")
        self.assertTrue(
            "Goodbye, world!" in log_contents, f"Can't find goodbye log in log contents:\n\t\"{log_contents}\""
        )

    def _log_log_level_screening(self, logname):
        lt.logger(logname)
        lt.info("Hello, world!")
        lt.debug("Goodbye, world!")

    def test_log_level_screening(self):
        self.proc_exec("_log_log_level_screening")

        log_contents = self.get_log_contents()
        self.assertTrue("Hello, world!" in log_contents, f"Can't find hello log in log contents:\n\t\"{log_contents}\"")
        self.assertFalse(
            "Goodbye, world!" in log_contents, "Found goodbye log in log contents when it shouldn't be there"
        )

    def _log_set_log_level(self, logname):
        lt.logger(logname, level=lt.log.DEBUG)
        lt.info("Hello, world!")
        lt.debug("Goodbye, world!")

    def test_set_log_level(self):
        self.proc_exec("_log_set_log_level")

        log_contents = self.get_log_contents()
        self.assertTrue("Hello, world!" in log_contents, f"Can't find hello log in log contents:\n\t\"{log_contents}\"")
        self.assertTrue(
            "Goodbye, world!" in log_contents, f"Can't find goodbye log in log contents:\n\t\"{log_contents}\""
        )

    def _log_error_and_raise(self, logname):
        lt.logger(logname)
        try:
            lt.error_and_raise(RuntimeError, "Error, world!")
        except RuntimeError:
            lt.info("RuntimeError encountered")

    def test_error_and_raise(self):
        self.proc_exec("_log_error_and_raise")

        log_contents = self.get_log_contents()
        self.assertTrue("Error, world!" in log_contents, f"Can't find error log in log contents:\n\t\"{log_contents}\"")
        self.assertTrue(
            "RuntimeError encountered" in log_contents,
            f"Can't find evidence of RuntimeError in log contents:\n\t\"{log_contents}\"",
        )

    def _log_end_str(self, _):
        lt.info("Hello", end=",")
        lt.info(" world!")
        lt.info("Goodbye", end="")
        lt.info(" world!")

    def test_end_str(self):
        stdout = self.proc_exec("_log_end_str")
        self.assertTrue("Hello, world!" in stdout, f"Can't find hello log in log contents:\n\t\"{stdout}\"")
        self.assertTrue("Goodbye world!" in stdout, f"Can't find goodbye log in log contents:\n\t\"{stdout}\"")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog=__file__.rstrip(".py"), description='Testing log tools')
    parser.add_argument('--logname', help="The name of the log file to log to.")
    parser.add_argument('--funcname', help="Calls the given function")
    args = parser.parse_args()
    log_name = args.logname
    func_name = args.funcname

    if func_name != None:
        tlt = TestLogTools()
        tlt.__getattribute__(func_name)(log_name)
    else:
        unittest.main()
