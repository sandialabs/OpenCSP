import os, unittest

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.lib.PowerpointShape as pps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.render.lib.PowerpointText import PowerpointText

class test_PowerpointText(unittest.TestCase):
    #def setUpClass(cls)
    def test_has_val_with_content(self):
        ppt_text_instance = PowerpointText(val="example text written for pytest")
        assert ppt_text_instance.has_val() == True

    def test_has_val_with_no_content(self):
        ppt_text_instance = PowerpointText(val=None)
        assert ppt_text_instance.has_val() == False

    def test_has_val_with_empty_string(self):
        ppt_text_instance = PowerpointText(val="")
        assert ppt_text_instance.has_val() == True

    def test_get_val_with_content(self):
        ppt_text_instance = PowerpointText(val="test get val")
        assert ppt_text_instance.get_val() == "test get val"

    def test_get_val_with_no_content(self):
        ppt_text_instance = PowerpointText(val=None)
        assert ppt_text_instance.get_val() == None

    def test_set_val(self): 
        val = "test_set_val test" 
        ppt_text_instance = PowerpointText(val) 
        ppt_text_instance.set_val(val) 
        assert ppt_text_instance._val == val 
 
    def test_set_val_none(self): 
        val = "test_set_val test" 
        ppt_text_instance = PowerpointText(val) 
        ppt_text_instance.set_val(val) 
        assert ppt_text_instance._saved_name_ext == None 
 
    def test_has_dims(self): 
        ppt_text_instance = PowerpointText() 
        assert ppt_text_instance.has_dims() == True 
 
    def test_dims_pptx(self): 
        powerpoint_var = [914400, 914400, 914400, 914400] # when dims are 1 the direct powerpoint equivalent is 914400 
        ppt_text_instance = PowerpointText(dims=(1.0, 1.0, 1.0, 1.0)) 
        assert ppt_text_instance.dims_pptx() == powerpoint_var 
 
    def test_compute_height(self): 
        test_font_pnt = 10 
        test_nlines = 1 
        value_should_be = 0.20666666666666667 
     
        ppt_text_instance = PowerpointText() 
        assert ppt_text_instance.compute_height(test_font_pnt, test_nlines) == value_should_be 
 
    def test_compute_and_assign_height(self): 
        font_pnt = 10 
        value_should_be = (1.0, 1.0, 1.0, 0.20666666666666667) 
        ppt_text_instance = PowerpointText(dims=(1.0, 1.0, 1.0, 1.0)) 
        ppt_text_instance.compute_and_assign_height(font_pnt) 
 
        assert ppt_text_instance.dims == value_should_be 

