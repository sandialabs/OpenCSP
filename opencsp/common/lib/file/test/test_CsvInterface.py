import os
import unittest

import opencsp.common.lib.file.CsvInterface as csvi
import opencsp.common.lib.tool.file_tools as ft


class TstCsvInterface(csvi.CsvInterface):
    """Simple class for testing the CsvInterface class."""

    def __init__(self, a_val: str, b_val: int, c_val: float):
        self.a_val = a_val
        self.b_val = b_val
        self.c_val = c_val

    @staticmethod
    def csv_header(delimeter=",") -> str:
        return delimeter.join(["Column A", "Column B", "Column C"])

    def to_csv_line(self, delimeter=",") -> str:
        return delimeter.join([str(val) for val in [self.a_val, self.b_val, self.c_val]])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["TstCsvInterface", list[str]]:
        ret = cls(data[0], int(data[1]), float(data[2]))
        return ret, data[3:]


class test_CsvInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.out_dir = ft.join(path, "data/output", name.split("test_")[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*")
        return super().setUpClass()

    def setUp(self) -> None:
        self.out_file = self.id().split(".")[-1]
        return super().setUp()

    def assertRowEquals(self, row: TstCsvInterface, vals: tuple[str, int, float]):
        self.assertEqual(row.a_val, vals[0])
        self.assertEqual(row.b_val, vals[1])
        self.assertEqual(row.c_val, vals[2])
        self.assertIsInstance(row.a_val, str)
        self.assertIsInstance(row.b_val, int)
        self.assertIsInstance(row.c_val, float)

    def test_read_zero_lines(self):
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "w") as fout:
            fout.write("Column A,Column B,Column C\n")

        rows = TstCsvInterface.from_csv(self.id(), self.out_dir, self.out_file + ".csv")
        self.assertEqual(len(rows), 0)

    def test_read_one_line(self):
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "w") as fout:
            fout.write("Column A,Column B,Column C\n")
            fout.write("0,0,0\n")

        rows = TstCsvInterface.from_csv(self.id(), self.out_dir, self.out_file + ".csv")
        rows: list[TstCsvInterface] = [row[0] for row in rows]
        self.assertEqual(len(rows), 1)
        self.assertRowEquals(rows[0], ("0", 0, 0.0))

    def test_read_two_lines(self):
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "w") as fout:
            fout.write("Column A,Column B,Column C\n")
            fout.write("0,1,2\n")
            fout.write("hello,4,5e10\n")

        rows = TstCsvInterface.from_csv(self.id(), self.out_dir, self.out_file + ".csv")
        rows: list[TstCsvInterface] = [row[0] for row in rows]
        self.assertEqual(len(rows), 2)
        self.assertRowEquals(rows[0], ("0", 1, 2.0))
        self.assertRowEquals(rows[1], ("hello", 4, 5e10))

    def test_write_one_line(self):
        # write the file
        TstCsvInterface("0", 0, 0.0).to_csv(self.id(), self.out_dir, self.out_file)

        # read the file
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "r") as fin:
            lines = fin.readlines()
        lines = [line.strip() for line in lines]

        # verify contents
        self.assertEqual(lines[0], "Column A,Column B,Column C")
        self.assertTrue("0,0,0" in lines[1])

    def test_write_two_lines(self):
        # write the file
        rows = [TstCsvInterface("0", 1, 2), TstCsvInterface("hello", 4, 5e10)]
        rows[0].to_csv(self.id(), self.out_dir, self.out_file, rows=rows)

        # read the file
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "r") as fin:
            lines = fin.readlines()
        lines = [line.strip() for line in lines]

        # verify contents
        self.assertEqual(lines[0], "Column A,Column B,Column C")
        self.assertTrue("0,1,2" in lines[1])
        self.assertTrue(("hello,4,50000000000" in lines[2]) or (lines[2] == "hello,4,5e10"))

    def test_overwrite(self):
        TstCsvInterface("0", 0, 0.0).to_csv(self.id(), self.out_dir, self.out_file)

        TstCsvInterface("1", 1, 1.0).to_csv(self.id(), self.out_dir, self.out_file, overwrite=True)
        with open(ft.join(self.out_dir, self.out_file + ".csv"), "r") as fin:
            lines = fin.readlines()
        self.assertTrue("1,1,1" in lines[1])

        with self.assertRaises(FileExistsError):
            TstCsvInterface("2", 2, 2.0).to_csv(self.id(), self.out_dir, self.out_file, overwrite=False)


if __name__ == '__main__':
    unittest.main()
