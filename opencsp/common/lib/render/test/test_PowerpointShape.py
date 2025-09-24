import inspect
import pptx
import unittest
from typing import Iterable, overload
from opencsp.common.lib.render.lib.PowerpointShape import PowerpointShape


class test_PowerpointShape(unittest.TestCase):
    # Test of a single integer value
    def test_pptx_inches_single_integer(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=3)
        assert result == pptx.util.Inches(3)

    # Test of a single float value
    def test_pptx_inches_single_float(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=6.0)
        assert result == pptx.util.Inches(6.0)

    # Test for a None value
    def test_pptx_inches_none(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches(None)

    # Test for a list of integer values
    def test_pptx_inches_list_of_integers(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=[1, 2, 3, 4, 5])
        expected = [
            pptx.util.Inches(1),
            pptx.util.Inches(2),
            pptx.util.Inches(3),
            pptx.util.Inches(4),
            pptx.util.Inches(5),
        ]
        assert expected == result

    # Test for a list of float values
    def test_pptx_inches_list_of_floats(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=[1.0, 1.1, 1.2, 1.3, 1.4])
        expected = [
            pptx.util.Inches(1.0),
            pptx.util.Inches(1.1),
            pptx.util.Inches(1.2),
            pptx.util.Inches(1.3),
            pptx.util.Inches(1.4),
        ]
        assert expected == result

    # Test for a tuple of integers
    def test_pptx_inches_tuple_of_integers(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=(1, 2, 3, 4, 5))
        expected = [
            pptx.util.Inches(1),
            pptx.util.Inches(2),
            pptx.util.Inches(3),
            pptx.util.Inches(4),
            pptx.util.Inches(5),
        ]
        assert expected == result

    # Test for a tuple of floats
    def test_pptx_inches_tuple_of_floats(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=(1.0, 1.1, 1.2, 1.3, 1.4))
        expected = [
            pptx.util.Inches(1.0),
            pptx.util.Inches(1.1),
            pptx.util.Inches(1.2),
            pptx.util.Inches(1.3),
            pptx.util.Inches(1.4),
        ]
        assert expected == result

    # Test for a tuple of integers
    def test_pptx_inches_tuple_of_integers(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches(vals=(1, 2, "random string", 4, 5))

    # Test for an empty list
    def test_pptx_inches_empty_list(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=[])
        expected = []
        assert expected == result

    # Test for a set, not supported
    def test_unsupported_type_set(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches({1, 2, 3})

    # Test for a string, not supported
    def test_bad_string(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches("some string")

    # Test for a dictionary, not supported
    def test_unsupported_type_dictionary(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches({1: 3, 2: 4, 3: 5})

    # Test for a list of strings, not supported
    def test_unsupported_type_list_of_strings(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches(["unit", "test", "for", "string", "list"])

    # Test for a None value
    def test_dims_to_str_with_none(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._dims_to_str(dims=None)
        expected = None
        assert expected == result

    # Test converting tuple to string (csv)
    def test_dims_to_str_with_tuple(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._dims_to_str(dims=(5.1, 4.2, 3.3, 2.4, 1.5))
        expected = "5.1,4.2,3.3,2.4,1.5"
        assert expected == result

    # Test converting a csv string to a tuple of floats
    def test_str_to_dims_with_tuple(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._str_to_dims(sval="1.1,2.2,3.3,4.4,5.5")
        expected = (1.1, 2.2, 3.3, 4.4, 5.5)
        assert expected == result

    # Test with sval equal to None
    def test_str_to_dims_with_none(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._str_to_dims(sval=None)
        expected = None
        assert expected == result
