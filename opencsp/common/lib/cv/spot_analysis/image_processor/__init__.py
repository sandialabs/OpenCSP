from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import (
    AbstractAggregateImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor import (
    ExposureDetectionImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import FalseColorImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.LogScaleImageProcessor import LogScaleImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import (
    PopulationStatisticsImageProcessor,
)

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    'AbstractAggregateImageProcessor',
    'AbstractSpotAnalysisImagesProcessor',
    'CroppingImageProcessor',
    'EchoImageProcessor',
    'ExposureDetectionImageProcessor',
    'FalseColorImageProcessor',
    'LogScaleImageProcessor',
    'PopulationStatisticsImageProcessor',
]
