import inspect
import pptx
import unittest
from typing import Iterable, overload
from opencsp.common.lib.render.lib.PowerpointShape import PowerpointShape


class test_PowerpointShape(unittest.TestCase):
    def test_pptx_inches_single_integer(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=3)
        assert result == pptx.util.Inches(3)

    def test_pptx_inches_single_float(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals=6.0)
        assert result == pptx.util.Inches(6.0)

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

    def test_unsupported_type_set(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches({1, 2, 3})

    def test_bad_string(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(ValueError):
            ppt_shape_instance._pptx_inches("some string")

    def test_int_string(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals="1,2,3,4,5")
        expected = [
            pptx.util.Inches(1),
            pptx.util.Inches(2),
            pptx.util.Inches(3),
            pptx.util.Inches(4),
            pptx.util.Inches(5),
        ]
        assert expected == result

    def test_float_string(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._pptx_inches(vals="1.0,2.0,3.0,4.0,5.0")
        expected = [
            pptx.util.Inches(1.0),
            pptx.util.Inches(2.0),
            pptx.util.Inches(3.0),
            pptx.util.Inches(4.0),
            pptx.util.Inches(5.0),
        ]
        assert expected == result

    def test_mix_string(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches("1.0,2,3.0,4.0,5")

    def test_unsupported_type_dictionary(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches({1: 3, 2: 4, 3: 5})

    def test_unsupported_type_list_of_strings(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches(["unit", "test", "for", "string", "list"])

    def test_unsupported_type_tuple_of_strings(self):
        ppt_shape_instance = PowerpointShape()
        with self.assertRaises(TypeError):
            ppt_shape_instance._pptx_inches(("unit", "test", "for", "string", "list"))

    def test_dims_to_str_with_none(self):
        ppt_shape_instance = PowerpointShape()
        result = ppt_shape_instance._dims_to_str(dims=None)
        expected = None
        assert expected == result
