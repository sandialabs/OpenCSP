from pathlib import Path

# Assume opencsp is in PYHTONPATH
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.common.lib.cv import CacheableImage, SpotAnalysis
from opencsp.common.lib.cv.spot_analysis import ImagesStream, SpotAnalysisImagesStream, SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor

# TODO: import all user-facing classes here.


def test_docstrings_exist_for_methods():
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

    for class_module in class_list:
        method_list = [
            func
            for func in dir(class_module)
            if callable(getattr(class_module, func)) and not func.startswith("__") and not func.startswith("_")
        ]

        for method in method_list:
            doc_exists = True
            if getattr(class_module, method).__doc__ is None:
                doc_exists = False

            print(f"doc_exists({class_module.__name__}.{method}): " f"{doc_exists}")
            assert doc_exists
