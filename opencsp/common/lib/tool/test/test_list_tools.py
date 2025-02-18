from collections import namedtuple
import numpy as np
import unittest

import opencsp.common.lib.tool.list_tools as listt


class test_list_tools(unittest.TestCase):
    def test_binary_search_size0(self):
        l = []
        i, v = listt.binary_search(l, "a")
        self.assertEqual(i, -1)
        self.assertEqual(v, None)

    def test_binary_search_size0_error(self):
        l = []
        with self.assertRaises(RuntimeError):
            i, v = listt.binary_search(l, "b", err_if_not_equal=True)

    def test_binary_search_size1_match(self):
        l = ["a"]
        i, v = listt.binary_search(l, "a")
        self.assertEqual(i, 0)
        self.assertEqual(v, "a")

    def test_binary_search_size1_nomatch(self):
        l = ["a"]
        i, v = listt.binary_search(l, "b")
        self.assertEqual(i, 0)
        self.assertEqual(v, "a")

    def test_binary_search_size1_nomatch_error(self):
        l = ["a"]
        with self.assertRaises(RuntimeError):
            i, v = listt.binary_search(l, "b", err_if_not_equal=True)

    def test_binary_search_size2_match(self):
        l = [10, 20]
        i, v = listt.binary_search(l, 10)
        self.assertEqual(i, 0)
        self.assertEqual(v, 10)
        i, v = listt.binary_search(l, 20)
        self.assertEqual(i, 1)
        self.assertEqual(v, 20)

    def test_binary_search_size2_nomatch(self):
        l = [10, 20]
        i, v = listt.binary_search(l, 11)
        self.assertEqual(i, 0)
        self.assertEqual(v, 10)
        i, v = listt.binary_search(l, 19)
        self.assertEqual(i, 1)
        self.assertEqual(v, 20)

    def test_binary_search_size3_nomatch(self):
        l = ["b", "d", "f"]
        i, v = listt.binary_search(l, "a")
        self.assertEqual(i, 0)
        self.assertEqual(v, "b")
        i, v = listt.binary_search(l, "b")
        self.assertEqual(i, 0)
        self.assertEqual(v, "b")
        i, v = listt.binary_search(l, "c")
        self.assertEqual(i, 0)
        self.assertEqual(v, "b")
        i, v = listt.binary_search(l, "d")
        self.assertEqual(i, 1)
        self.assertEqual(v, "d")
        i, v = listt.binary_search(l, "e")
        self.assertEqual(i, 1)
        self.assertEqual(v, "d")
        i, v = listt.binary_search(l, "f")
        self.assertEqual(i, 2)
        self.assertEqual(v, "f")
        i, v = listt.binary_search(l, "g")
        self.assertEqual(i, 2)
        self.assertEqual(v, "f")

    def test_binary_search_many(self):
        l = sorted(np.random.random(10_000))
        i, v = listt.binary_search(l, -1)
        self.assertEqual(i, 0)
        i, v = listt.binary_search(l, 2)
        self.assertEqual(i, 10_000 - 1)
        i, v = listt.binary_search(l, l[5_000])
        self.assertEqual(i, 5_000)

    def test_binary_search_named_tuple(self):
        nt = namedtuple("nt", ["f1"])
        comparator = lambda t, v: -1 if t.f1 < v else (1 if t.f1 > v else 0)
        l = [nt(1), nt(2), nt(3)]
        i, v = listt.binary_search(l, 1, comparator)
        self.assertEqual(i, 0)
        i, v = listt.binary_search(l, 2, comparator)
        self.assertEqual(i, 1)
        i, v = listt.binary_search(l, 3, comparator)
        self.assertEqual(i, 2)

    def test_binary_search_key(self):
        nt = namedtuple("nt", ["f1"])
        key = lambda v: v.f1
        l = [nt(1), nt(2), nt(3)]
        i, v = listt.binary_search(l, 1, key=key)
        self.assertEqual(i, 0)
        i, v = listt.binary_search(l, 2, key=key)
        self.assertEqual(i, 1)
        i, v = listt.binary_search(l, 3, key=key)
        self.assertEqual(i, 2)

    def test_get_range_exact(self):
        lvals = [0, 1, 2, 3, 4]
        tvals = ["a", "b", "c", "d", "e"]

        middle_vals = listt.get_range(lvals, tvals, [1, 3])
        self.assertEqual(middle_vals[0], [1, 2, 3])
        self.assertEqual(middle_vals[1], ["b", "c", "d"])

        front_vals = listt.get_range(lvals, tvals, [0, 2])
        self.assertEqual(front_vals[0], [0, 1, 2])
        self.assertEqual(front_vals[1], ["a", "b", "c"])

        back_vals = listt.get_range(lvals, tvals, [2, 4])
        self.assertEqual(back_vals[0], [2, 3, 4])
        self.assertEqual(back_vals[1], ["c", "d", "e"])

        all_vals = listt.get_range(lvals, tvals, [0, 4])
        self.assertEqual(all_vals[0], [0, 1, 2, 3, 4])
        self.assertEqual(all_vals[1], ["a", "b", "c", "d", "e"])

        overfull = listt.get_range(lvals, tvals, [-1, 5])
        self.assertEqual(overfull[0], [0, 1, 2, 3, 4])
        self.assertEqual(overfull[1], ["a", "b", "c", "d", "e"])

    def test_get_range_approximate(self):
        lvals = [0, 1, 2, 3, 4]
        tvals = ["a", "b", "c", "d", "e"]

        closest_vals_front = listt.get_range(lvals, tvals, [-100, -1])
        self.assertEqual(closest_vals_front[0], [0])
        self.assertEqual(closest_vals_front[1], ["a"])

        closest_vals_middle = listt.get_range(lvals, tvals, [1.2, 2.8])
        self.assertEqual(closest_vals_middle[0], [1, 2, 3])
        self.assertEqual(closest_vals_middle[1], ["b", "c", "d"])

        closest_vals_back = listt.get_range(lvals, tvals, [6, 100])
        self.assertEqual(closest_vals_back[0], [4])
        self.assertEqual(closest_vals_back[1], ["e"])

        middle_vals_exclusive = listt.get_range(lvals, tvals, [1.2, 2.8], exclude_outside_range=True)
        self.assertEqual(middle_vals_exclusive[0], [2])
        self.assertEqual(middle_vals_exclusive[1], ["c"])

        empty_vals_front = listt.get_range(lvals, tvals, [-100, -1], exclude_outside_range=True)
        self.assertEqual(len(empty_vals_front[0]), 0)
        self.assertEqual(len(empty_vals_front[1]), 0)

        empty_vals_middle = listt.get_range(lvals, tvals, [0.1, 0.9], exclude_outside_range=True)
        self.assertEqual(len(empty_vals_middle[0]), 0)
        self.assertEqual(len(empty_vals_middle[1]), 0)

        empty_vals_back = listt.get_range(lvals, tvals, [6, 100], exclude_outside_range=True)
        self.assertEqual(len(empty_vals_back[0]), 0)
        self.assertEqual(len(empty_vals_back[1]), 0)

    def test_natural_sort_just_strs(self):
        lvals = ["c", "b", "a"]
        expected = ["a", "b", "c"]
        actual = listt.natural_sort(lvals)
        self.assertEqual(actual, expected)

    def test_natural_sort_just_numbers(self):
        lvals = ["21", "3", "2"]
        expected = ["2", "3", "21"]
        actual = listt.natural_sort(lvals)
        self.assertEqual(actual, expected)

    def test_natural_sort_strs_nums(self):
        lvals = ["c_4_c", "c_3_c", "c_3_a", "b_05_b", "b_4_b", "b_0_c", "1_a_2", "1_a_1"]
        expected = ["1_a_1", "1_a_2", "b_0_c", "b_4_b", "b_05_b", "c_3_a", "c_3_c", "c_4_c"]
        actual = listt.natural_sort(lvals)
        self.assertEqual(actual, expected)

    def test_rindex(self):
        lvals = [1, 2, 3, 4, 5]
        self.assertEqual(listt.rindex(lvals, 6), -1)
        self.assertEqual(listt.rindex(lvals, 5), 4)
        self.assertEqual(listt.rindex(lvals, 4), 3)
        self.assertEqual(listt.rindex(lvals, 3), 2)
        self.assertEqual(listt.rindex(lvals, 2), 1)
        self.assertEqual(listt.rindex(lvals, 1), 0)
        self.assertEqual(listt.rindex(lvals, 0), -1)


if __name__ == "__main__":
    unittest.main()
