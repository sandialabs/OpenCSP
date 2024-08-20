import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.cv.CacheableImage as ci
import opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor as asaip
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable as sao


class DoNothingImageProcessor(asaip.AbstractSpotAnalysisImagesProcessor):
    def _execute(self, operable: sao.SpotAnalysisOperable, is_last: bool) -> list[sao.SpotAnalysisOperable]:
        return [operable]


class test_AbstractSpotAnalysisImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.in_dir = ft.join(path, 'data/input', name.split('test_')[-1])
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, '*')
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

        self.example_cache_path = ft.join(self.in_dir, "example_image.npy")
        self.cacheable_image = ci.CacheableImage(self.example_cache_path)
        self.example_operable = sao.SpotAnalysisOperable(self.cacheable_image)

        self.example_operables: list[sao.SpotAnalysisOperable] = []
        for i in range(3):
            ci_i = ci.CacheableImage(self.example_cache_path)
            sao_i = sao.SpotAnalysisOperable(ci_i)
            self.example_operables.append(sao_i)

    def test_name(self):
        """
        Verify that the auto-assigned name is the class name, and that it can be
        overwritten with a specific name.
        """
        instance = DoNothingImageProcessor()
        self.assertEqual(instance.name, "DoNothingImageProcessor")
        instance = DoNothingImageProcessor("Other Name")
        self.assertEqual(instance.name, "Other Name")


if __name__ == '__main__':
    unittest.main()
