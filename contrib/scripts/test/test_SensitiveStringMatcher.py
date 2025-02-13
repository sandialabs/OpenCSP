import os
import sys
import time
import unittest
import unittest.mock

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.time_date_tools as tdt

# setting path
sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.SensitiveStringMatcher as ssm  # nopep8


class test_SensitiveStringMatcher(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FileCache")
        self.out_dir = os.path.join(path, "data", "output", "FileCache")
        ft.create_directories_if_necessary(self.out_dir)

    def test_match(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "bar")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(2, matches[0].lineno)

    def test_partial_match(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "bar")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(1, matches[0].lineno)
        self.assertEqual(3, matches[0].colno)

        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(1, matches[0].lineno)
        self.assertEqual(0, matches[0].colno)

        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "baz")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(1, matches[0].lineno)
        self.assertEqual(6, matches[0].colno)

    def test_matches(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo", "bar", "baz")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(3, len(matches))
        self.assertEqual(1, matches[0].lineno)
        self.assertEqual(2, matches[1].lineno)
        self.assertEqual(3, matches[2].lineno)

    def test_dont_match(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo", "**dont_match", "foo")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(0, len(matches))

    def test_case_sensitive(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "**case_sensitive", "foo")
        matches = matcher.check_lines(["foO", "fOo", "fOO", "Foo", "FoO", "FOo", "FOO", "foo"])
        self.assertEqual(1, len(matches))
        self.assertEqual(8, matches[0].lineno)

    def test_single_regex(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "**next_is_regex", r"[a-z]a[a-z]")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(2, len(matches))
        self.assertEqual(2, matches[0].lineno)
        self.assertEqual("bar", matches[0].line_part)
        self.assertEqual(3, matches[1].lineno)
        self.assertEqual("baz", matches[1].line_part)

    def test_partial_single_regex(self):
        matcher = ssm.SensitiveStringMatcher("Regex Matcher", "**next_is_regex", r"[a-z]o[a-z]")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(0, matches[0].colno)
        self.assertEqual("foo", matches[0].line_part)

        matcher = ssm.SensitiveStringMatcher("Regex Matcher", "**next_is_regex", r"[a-z]{2}r")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(3, matches[0].colno)
        self.assertEqual("bar", matches[0].line_part)

        matcher = ssm.SensitiveStringMatcher("Regex Matcher", "**next_is_regex", r"[a-z]{2}z")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(6, matches[0].colno)
        self.assertEqual("baz", matches[0].line_part)

    def test_partial_multiple_regex(self):
        matcher = ssm.SensitiveStringMatcher("Regex Matcher", "**all_regex", r"[a-z]o[a-z]", r"[a-z]{2}r", r"[a-z]{2}z")
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(3, len(matches))
        self.assertEqual(0, matches[0].colno)
        self.assertEqual("foo", matches[0].line_part)
        self.assertEqual(3, matches[1].colno)
        self.assertEqual("bar", matches[1].line_part)
        self.assertEqual(6, matches[2].colno)
        self.assertEqual("baz", matches[2].line_part)

    def test_mixed_plain_regex(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo", "**next_is_regex", r"[a-z]{2}r", "baz")

        matches = matcher.check_lines(["foobarbaz"])
        self.assertLessEqual(1, len(matches))
        self.assertEqual(0, matches[0].colno)
        self.assertEqual("foo", matches[0].line_part)

        matches = matcher.check_lines(["goobarbaz"])
        self.assertLessEqual(1, len(matches))
        self.assertEqual(3, matches[0].colno)
        self.assertEqual("bar", matches[0].line_part)

        matches = matcher.check_lines(["googgrbaz"])
        self.assertLessEqual(1, len(matches))
        self.assertEqual(3, matches[0].colno)
        self.assertEqual("ggr", matches[0].line_part)

        matches = matcher.check_lines(["goobanbaz"])
        self.assertEqual(1, len(matches))
        self.assertEqual(6, matches[0].colno)
        self.assertEqual("baz", matches[0].line_part)

    def test_regex_dont_match(self):
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo", "**dont_match", "**next_is_regex", r"[a-z]o[a-z]")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(0, len(matches))

        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "**all_regex", "foo.?", "**dont_match", "**next_is_regex", r"[a-z]{4}"
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        self.assertEqual(1, len(matches))
        matches = matcher.check_lines(["foobarbaz"])
        self.assertEqual(0, len(matches))


if __name__ == "__main__":
    unittest.main()
