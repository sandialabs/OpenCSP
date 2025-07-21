import shutil
import subprocess
import sys
import unittest

import opencsp.common.lib.tool.file_tools as ft

import unittest
import subprocess
import os


class test_cross_section(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, "data/input", name.split("test_")[-1])
        cls.out_dir = ft.join(path, "data/output", name.split("test_")[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split(".")[-1]
        self.tmp_dir = ft.join(self.out_dir, self.test_name)

        # remove the results from the previous execution
        shutil.rmtree(self.tmp_dir)
        ft.create_directories_if_necessary(self.tmp_dir)

    def _notebook_runs_without_errors(self, notebook_path):
        # Generated with Google search AI
        #
        # We test the notebook this way so that the tests for notebooks conform
        # to the standard test/test_* directory and file structure for OpenCSP.
        # TODO generalize this into an abstract testing class

        # Ensure the notebook exists
        self.assertTrue(os.path.exists(notebook_path), f"Notebook not found at: {notebook_path}")

        # Use nbconvert to execute the notebook
        python = sys.executable
        command = [
            python,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            notebook_path,
            "--output-dir",
            self.tmp_dir,
        ]

        try:
            # Run the command, capturing stdout and stderr
            process = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=600
            )  # Add timeout for safety

            # If nbconvert succeeds (exit code 0), the test passes
            self.assertEqual(process.returncode, 0, f"Notebook execution failed with exit code {process.returncode}")
            print(f"Notebook executed successfully: {notebook_path}")

        except subprocess.CalledProcessError as e:
            # If nbconvert fails (non-zero exit code), it will raise CalledProcessError
            self.fail(
                f"Notebook execution failed with errors. \n"
                f"Stderr: {e.stderr} \n"
                f"Stdout: {e.stdout} \n"
                f"Return Code: {e.returncode}"
            )
        except subprocess.TimeoutExpired as e:
            self.fail(
                f"Notebook execution timed out after {e.timeout} seconds.\n"
                f"Stderr: {e.stderr} \n"
                f"Stdout: {e.stdout}"
            )

    def test_cross_section(self):
        path, _, _ = ft.path_components(__file__)
        path = ft.join(path, "..", "cross_section.ipynb")
        self._notebook_runs_without_errors(path)


if __name__ == "__main__":
    unittest.main()
