from typing import TYPE_CHECKING as _TYPE_CHECKING

from opencsp import LazyLoader as _LazyLoader

base_package = "opencsp.common.lib.cv.spot_analysis.image_processor."

# fmt: off
AbstractAggregateImageProcessor = _LazyLoader(base_package + "AbstractAggregateImageProcessor", "AbstractAggregateImageProcessor")
AbstractSpotAnalysisImageProcessor = _LazyLoader(base_package + "AbstractSpotAnalysisImageProcessor", "AbstractSpotAnalysisImageProcessor")
AbstractVisualizationImageProcessor = _LazyLoader(base_package + "AbstractVisualizationImageProcessor", "AbstractVisualizationImageProcessor")
AverageByGroupImageProcessor = _LazyLoader(base_package + "AverageByGroupImageProcessor", "AverageByGroupImageProcessor")
BcsLocatorImageProcessor = _LazyLoader(base_package + "BcsLocatorImageProcessor", "BcsLocatorImageProcessor")
ConvolutionImageProcessor = _LazyLoader(base_package + "ConvolutionImageProcessor", "ConvolutionImageProcessor")
CroppingImageProcessor = _LazyLoader(base_package + "CroppingImageProcessor", "CroppingImageProcessor")
EchoImageProcessor = _LazyLoader(base_package + "EchoImageProcessor", "EchoImageProcessor")
ExposureDetectionImageProcessor = _LazyLoader(base_package + "ExposureDetectionImageProcessor", "ExposureDetectionImageProcessor")
ViewFalseColorImageProcessor = _LazyLoader(base_package + "ViewFalseColorImageProcessor", "ViewFalseColorImageProcessor")
HotspotImageProcessor = _LazyLoader(base_package + "HotspotImageProcessor", "HotspotImageProcessor")
LogScaleImageProcessor = _LazyLoader(base_package + "LogScaleImageProcessor", "LogScaleImageProcessor")
NullImageSubtractionImageProcessor = _LazyLoader(base_package + "NullImageSubtractionImageProcessor", "NullImageSubtractionImageProcessor")
PopulationStatisticsImageProcessor = _LazyLoader(base_package + "PopulationStatisticsImageProcessor", "PopulationStatisticsImageProcessor")
SupportingImagesCollectorImageProcessor = _LazyLoader(base_package + "SupportingImagesCollectorImageProcessor", "SupportingImagesCollectorImageProcessor")
View3dImageProcessor = _LazyLoader(base_package + "View3dImageProcessor", "View3dImageProcessor")
ViewCrossSectionImageProcessor = _LazyLoader(base_package + "ViewCrossSectionImageProcessor", "ViewCrossSectionImageProcessor")
# fmt: on

if _TYPE_CHECKING:
    # fmt: off
    from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import AbstractAggregateImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import AbstractSpotAnalysisImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import AbstractVisualizationImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.AverageByGroupImageProcessor import AverageByGroupImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.BcsLocatorImageProcessor import BcsLocatorImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor import ConvolutionImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor import ExposureDetectionImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.ViewFalseColorImageProcessor import ViewFalseColorImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.LogScaleImageProcessor import LogScaleImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.NullImageSubtractionImageProcessor import NullImageSubtractionImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import PopulationStatisticsImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.SupportingImagesCollectorImageProcessor import SupportingImagesCollectorImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.View3dImageProcessor import View3dImageProcessor
    from opencsp.common.lib.cv.spot_analysis.image_processor.ViewCrossSectionImageProcessor import ViewCrossSectionImageProcessor
    # fmt: on

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    'AbstractAggregateImageProcessor',
    'AbstractSpotAnalysisImageProcessor',
    'AbstractVisualizationImageProcessor',
    'AverageByGroupImageProcessor',
    'BcsLocatorImageProcessor',
    'ConvolutionImageProcessor',
    'CroppingImageProcessor',
    'EchoImageProcessor',
    'ExposureDetectionImageProcessor',
    'HotspotImageProcessor',
    'LogScaleImageProcessor',
    'NullImageSubtractionImageProcessor',
    'PopulationStatisticsImageProcessor',
    'SupportingImagesCollectorImageProcessor',
    'View3dImageProcessor',
    'ViewCrossSectionImageProcessor',
    'ViewFalseColorImageProcessor',
]
