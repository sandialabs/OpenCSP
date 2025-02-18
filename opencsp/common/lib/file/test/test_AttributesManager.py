import json
import os
import unittest
import unittest.mock

import opencsp.common.lib.file.AttributesManager as am
import opencsp.common.lib.file.AbstractAttributeParser as aap
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.typing_tools as tt


class SimpleAttributeParser(aap.AbstractAttributeParser):
    def __init__(self):
        self.strval = "hello world"
        self.listval = ["foo", "bar", "baz"]

    def set_defaults(self, other: "SimpleAttributeParser"):
        self.strval = tt.default(self.strval, other.strval)
        self.listval = tt.default(self.listval, other.listval)

    def attributes_key(self) -> str:
        return "test_attribute_parser"

    def has_contents(self) -> bool:
        return True

    def parse_my_contents(self, file_path_name_ext: str, raw_contents: str, my_contents: any):
        self.strval = my_contents["strval"]
        self.listval = my_contents["listval"]

    def my_contents_to_json(self, file_path_name_ext: str) -> any:
        my_contents = {"strval": self.strval, "listval": self.listval}
        return my_contents


SimpleAttributeParser.RegisterClass()


class test_AttributesManager(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "AttributesManager")
        self.out_dir = os.path.join(path, "data", "output", "AttributesManager")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_save_load(self):
        file_name = os.path.join(self.out_dir, "test_save_load.txt")

        tap = SimpleAttributeParser()
        tap.save(file_name)

        tap2 = SimpleAttributeParser()
        tap2.load(file_name)

        self.assertEqual(tap.strval, tap2.strval)
        self.assertEqual(tap.listval, tap2.listval)

    def test_custom_contents(self):
        file_name = os.path.join(self.out_dir, "test_custom_contents.txt")

        tap = SimpleAttributeParser()
        tap.strval = "goodbye world"
        tap.listval = [1, 2, 3]
        tap.save(file_name)

        tap2 = SimpleAttributeParser()
        tap2.load(file_name)

        self.assertEqual(tap2.strval, "goodbye world")
        self.assertEqual(tap2.listval, [1, 2, 3])

    def test_registration(self):
        file_name = os.path.join(self.out_dir, "test_registration.txt")

        tap = SimpleAttributeParser()
        tap.strval = "test registration string"
        tap.listval = ["t", "r"]
        tap.save(file_name)

        # make sure the attributes manager was initialized with a
        # SimpleAttributeParser, and that it parsed the file correctly
        attr = am.AttributesManager()
        attr.load(file_name)
        tap2 = attr.get_parser(SimpleAttributeParser)

        self.assertEqual(tap.strval, tap2.strval)
        self.assertEqual(tap.listval, tap2.listval)

    def test_bad_file(self):
        file_name = os.path.join(self.out_dir, "bad_file.txt")
        with open(file_name, "w") as fout:
            fout.write("Not a json formatted file")

        # loading should fail because the file isn't json
        with self.assertRaises(json.decoder.JSONDecodeError):
            am.AttributesManager().load(file_name)

        # saving should fail because the file already exists
        with self.assertRaises(FileExistsError):
            am.AttributesManager().save(file_name)

        # loading a specific parser should fail because the file isn't json
        with self.assertRaises(json.decoder.JSONDecodeError):
            SimpleAttributeParser().load(file_name)

        # saving a specific parser should fail because the file isn't json
        with self.assertRaises(FileExistsError):
            SimpleAttributeParser().save(file_name)

    def test_overwrite_bad_file(self):
        file_name = os.path.join(self.out_dir, "overwrite_bad_file.txt")
        with open(file_name, "w") as fout:
            fout.write("Not a json formatted file")

        # saving a specific parser with overwrite, then loading it back, should work correctly
        out_parser = SimpleAttributeParser()
        out_parser.strval = "This should be jsonifiable"
        out_parser.save(file_name, overwrite=True)
        in_parser = SimpleAttributeParser()
        in_parser.load(file_name)
        self.assertEqual("This should be jsonifiable", in_parser.strval)


if __name__ == "__main__":
    unittest.main()
