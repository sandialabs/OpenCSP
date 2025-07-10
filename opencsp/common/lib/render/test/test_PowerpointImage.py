from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.render.lib.PowerpointImage as pi
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


class test_PowerpointImage(unittest.TestCase):
    dir_in = os.path.join(orp.opencsp_code_dir(), "common", "lib", "render", "test", "data", "input", "PowerpointImage")
    dir_out = os.path.join(
        orp.opencsp_code_dir(), "common", "lib", "render", "test", "data", "output", "PowerpointImage"
    )

    @classmethod
    def setUpClass(cls) -> None:
        ret = super().setUpClass()
        if ft.directory_exists(test_PowerpointImage.dir_out):
            ft.delete_files_in_directory(test_PowerpointImage.dir_out, "*")
        ft.create_directories_if_necessary(test_PowerpointImage.dir_out)
        return ret

    def gen_rand_ndarray(self, h=640, w=480) -> np.ndarray:
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def gen_rand_mpl_fig(self, n=480) -> plt.Figure:
        mplfig, mplax = plt.subplots(figsize=(6, 6))
        mpldata = {'a': np.random.randint(0, 256, n, dtype=np.uint8), 'b': np.random.randint(0, 256, n, dtype=np.uint8)}
        mplax.scatter('a', 'b', data=mpldata)
        return mplfig

    def setUp(self) -> None:
        # get the data paths, for easy access
        self.dir_in = test_PowerpointImage.dir_in
        self.dir_out = test_PowerpointImage.dir_out
        self.height = 640
        self.width = 480

        ndarray = self.gen_rand_ndarray(self.height, self.width)
        pil_image = Image.fromarray(self.gen_rand_ndarray(self.height, self.width))
        mplfig = self.gen_rand_mpl_fig()
        random_rcfr = rcfr.RenderControlFigureRecord("random_rcfr", "random_rcfr", "random_rcfr", 1, mplfig)

        self.test_str_obj = pi.PowerpointImage(os.path.join(self.dir_in, "example_image.png"))
        self.test_ndarray_obj = pi.PowerpointImage(ndarray)
        self.test_Image_obj = pi.PowerpointImage(pil_image)
        self.test_RenderControlFigureRecord_obj = pi.PowerpointImage(random_rcfr)

    def test_has_val(self):
        assert self.test_str_obj.has_val()
        assert self.test_ndarray_obj.has_val()
        assert self.test_Image_obj.has_val()
        assert self.test_RenderControlFigureRecord_obj.has_val()

    def test_get_val(self):
        assert self.test_str_obj.get_val() is not None
        assert self.test_ndarray_obj.get_val() is not None
        assert self.test_Image_obj.get_val() is not None
        assert self.test_RenderControlFigureRecord_obj.get_val() is not None

    def test_set_val(self):
        test_str = "dne.png"
        self.test_str_obj.set_val(test_str)
        assert self.test_str_obj.get_val() == test_str

        test_ndarray = self.gen_rand_ndarray()
        self.test_ndarray_obj.set_val(test_ndarray)
        assert np.array_equal(self.test_ndarray_obj.get_val(), test_ndarray)

        test_Image = Image.fromarray(self.gen_rand_ndarray())
        self.test_Image_obj.set_val(test_Image)
        assert self.test_Image_obj.get_val() == test_Image

        mplfig = self.gen_rand_mpl_fig()
        test_rcfr = rcfr.RenderControlFigureRecord("test_rcfr", "test_rcfr", "test_rcfr", 1, mplfig)
        self.test_RenderControlFigureRecord_obj.set_val(test_rcfr)
        assert self.test_RenderControlFigureRecord_obj.get_val() == test_rcfr

    def test_save(self):
        self.test_str_obj.save()
        self.test_ndarray_obj.save()
        self.test_Image_obj.save()
        self.test_RenderControlFigureRecord_obj.save()


if __name__ == "__main__":
    unittest.main()
