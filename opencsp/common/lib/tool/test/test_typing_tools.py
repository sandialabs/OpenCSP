import unittest

import opencsp.common.lib.tool.typing_tools as tt


@tt.strict_types
def imperative_zero_args():
    """A truly zero-arguments function. Not tied to any class and with no "self" argument."""
    pass


class UserClass1:
    pass


class UserClass1_1(UserClass1):
    pass


class UserClass2:
    pass


class TestTypingTools(unittest.TestCase):
    def raises_runtime_error(self):
        raise RuntimeError

    def raises_value_error(self):
        raise ValueError

    def test_default(self):
        self.assertEqual("a", tt.default(None, "a"))
        self.assertEqual(None, tt.default(None, None))
        self.assertEqual("a", tt.default(None, None, None, "a"))

        lval = [1, 2, 3, 4]
        dval = {1: 2, 3: None}
        self.assertEqual(4, tt.default(lambda: lval[3], "a"))
        self.assertEqual("a", tt.default(lambda: lval[4], "a"))
        self.assertEqual(2, tt.default(lambda: dval[1], "a"))
        self.assertEqual("a", tt.default(lambda: dval[2], "a"))
        self.assertEqual("a", tt.default(lambda: dval[3], "a"))

        with self.assertRaises(RuntimeError):
            tt.default(self.raises_runtime_error, self.raises_runtime_error)
        self.assertEqual("a", tt.default(self.raises_runtime_error, "a"))
        with self.assertRaises(RuntimeError):
            tt.default(self.raises_runtime_error, self.raises_value_error)
        self.assertEqual(None, tt.default(self.raises_runtime_error, self.raises_value_error, None))
        self.assertEqual("a", tt.default("a", self.raises_value_error))
        self.assertEqual("a", tt.default(None, lambda: lval[4], dval[3], lambda: dval[4], "a"))

        with self.assertRaises(ValueError):
            self.assertEqual("a", tt.default("a"))
        with self.assertRaises(ValueError):
            self.assertEqual(None, tt.default(None))

    @tt.strict_types
    def zero_args(self):
        """A near-zero arguments function. Tied to the testing class with a "self" argument."""
        pass

    @staticmethod
    @tt.strict_types  # TODO allow this to be applied before other decorators
    def static_zero_args():
        """A truly zero-arguments function. Tied to the testing class but without a "self" argument."""
        pass

    @tt.strict_types
    def int_arg(self, a: int):
        pass

    @tt.strict_types
    def int_default_arg(self, a=0):
        pass

    @tt.strict_types
    def float_arg(self, a: float):
        pass

    @tt.strict_types
    def str_arg(self, a: str):
        pass

    @tt.strict_types
    def complex_arg(self, a: complex):
        pass

    @tt.strict_types
    def list_arg(self, a: list):
        pass

    @tt.strict_types
    def tuple_arg(self, a: tuple):
        pass

    @tt.strict_types
    def tuple_default_arg(self, a=(1, 2, 3)):
        pass

    @tt.strict_types
    def dict_arg(self, a: dict):
        pass

    @tt.strict_types
    def class_arg(self, a: UserClass1):
        pass

    def test_zero_args(self):
        imperative_zero_args()
        self.zero_args()
        self.static_zero_args()

    def test_one_arg(self):
        self.int_arg(1)
        self.int_default_arg()
        self.float_arg(1.0)
        self.str_arg("1")
        self.complex_arg(1j)
        self.list_arg([])
        self.tuple_arg(tuple([]))
        self.tuple_default_arg()
        self.dict_arg({})
        self.class_arg(UserClass1())
        self.class_arg(UserClass1_1())

    def test_one_arg_bad(self):
        # for each test function, only one of these arguments should not raise an error
        all_args = [1, 1.0, "1", 1j, [], tuple([]), {}, UserClass1(), UserClass2()]
        all_one_arg_funcs = [
            self.int_arg,
            self.float_arg,
            self.str_arg,
            self.complex_arg,
            self.list_arg,
            self.tuple_arg,
            self.dict_arg,
            self.class_arg,
        ]

        for one_arg_func in all_one_arg_funcs:
            passed_args = []
            for arg in all_args:
                try:
                    one_arg_func(arg)
                    passed_args.append(arg)
                    # lt.info(f"Arg {(arg, type(arg))} passed for {one_arg_func.__name__} ({len(passed_args)} passed)")
                except TypeError:
                    # lt.info(f"Arg {(arg, type(arg))} failed for {one_arg_func.__name__} ({len(passed_args)} passed)")
                    pass

            if len(passed_args) == 0:
                self.fail(
                    f"In test_typing_tools.test_one_arg_bad: one of the argument types should have passed but none did for function {one_arg_func.__name__}"
                )
            self.assertEqual(
                len(passed_args),
                1,
                f"In test_typing_tools.test_one_arg_bad: exactly one of the argument types should have passed for function {one_arg_func.__name__} but {len(passed_args)} did. The passing arguments were {[(arg,type(arg)) for arg in passed_args]}",
            )


if __name__ == "__main__":
    unittest.main()
