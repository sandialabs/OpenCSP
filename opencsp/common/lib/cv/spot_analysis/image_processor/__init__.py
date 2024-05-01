from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import (
    AbstractAggregateImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AnnotationImageProcessor import AnnotationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.AverageByGroupImageProcessor import (
    AverageByGroupImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.BcsLocatorImageProcessor import BcsLocatorImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor import ConvolutionImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor import (
    ExposureDetectionImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import FalseColorImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.LogScaleImageProcessor import LogScaleImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.NullImageSubtractionImageProcessor import (
    NullImageSubtractionImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import (
    PopulationStatisticsImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.SupportingImagesCollectorImageProcessor import (
    SupportingImagesCollectorImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.View3dImageProcessor import View3dImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ViewCrossSectionImageProcessor import (
    ViewCrossSectionImageProcessor,
)

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    'AbstractAggregateImageProcessor',
    'AbstractSpotAnalysisImagesProcessor',
    'AbstractVisualizationImageProcessor',
    'AnnotationImageProcessor',
    'AverageByGroupImageProcessor',
    'BcsLocatorImageProcessor',
    'ConvolutionImageProcessor',
    'CroppingImageProcessor',
    'EchoImageProcessor',
    'ExposureDetectionImageProcessor',
    'FalseColorImageProcessor',
    'HotspotImageProcessor',
    'LogScaleImageProcessor',
    'NullImageSubtractionImageProcessor',
    'PopulationStatisticsImageProcessor',
    'SupportingImagesCollectorImageProcessor',
    'View3dImageProcessor',
    'ViewCrossSectionImageProcessor',
]
