import inspect
import unittest

# Assume opencsp is in PYHTONPATH
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.common.lib.cv import CacheableImage, SpotAnalysis
from opencsp.common.lib.cv.spot_analysis import ImagesStream, SpotAnalysisImagesStream, SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor

# TODO: import all user-facing classes here.


class test_Docstrings(unittest.TestCase):
    class_list = [
        Sofast,
        # Spot Analysis
        SpotAnalysis,
        SpotAnalysisOperable,
        ImagesStream,
        SpotAnalysisImagesStream,
        CacheableImage,
        AbstractAggregateImageProcessor,
        AbstractSpotAnalysisImageProcessor,
        AbstractVisualizationImageProcessor,
        AnnotationImageProcessor,
        AverageByGroupImageProcessor,
        BcsLocatorImageProcessor,
        ConvolutionImageProcessor,
        CroppingImageProcessor,
        EchoImageProcessor,
        ExposureDetectionImageProcessor,
        FalseColorImageProcessor,
        HotspotImageProcessor,
        LogScaleImageProcessor,
        NullImageSubtractionImageProcessor,
        PopulationStatisticsImageProcessor,
        SupportingImagesCollectorImageProcessor,
        View3dImageProcessor,
        ViewCrossSectionImageProcessor,
        # TODO: List all user-facing classes here.
    ]

    def test_docstrings_exist_for_methods(self):
        for class_module in self.class_list:
            method_list = [
                func
                for func in dir(class_module)
                if callable(getattr(class_module, func)) and not func.startswith("__") and not func.startswith("_")
            ]

            undocumented_methods: list[str] = []

            for method in method_list:
                doc_exists = True
                if inspect.getdoc(getattr(class_module, method)) is None:
                    doc_exists = False

                method_name = f"{class_module.__name__}.{method}"
                print(f"doc_exists({method_name}): " f"{doc_exists}")
                if not doc_exists:
                    undocumented_methods.append(method)

            self.assertEqual(
                len(undocumented_methods),
                0,
                f"Found undocumented methods in {class_module}:\n\t" + "\n\t".join(undocumented_methods),
            )


if __name__ == '__main__':
    unittest.main()
