from contrib.common.lib.cv.spot_analysis.image_processor.CustomSimpleImageProcessor import CustomSimpleImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.BackgroundColorSubtractionImageProcessor import (
    BackgroundColorSubtractionImageProcessor,
)
from contrib.common.lib.cv.spot_analysis.image_processor.EnclosedEnergyImageProcessor import (
    EnclosedEnergyImageProcessor,
)
from contrib.common.lib.cv.spot_analysis.image_processor.InpaintImageProcessor import InpaintImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.MomentsImageProcessor import MomentsImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.PowerpointImageProcessor import PowerpointImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.SaveToFileImageProcessor import SaveToFileImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.SpotWidthImageProcessor import SpotWidthImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.StabilizationImageProcessor import StabilizationImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor.TargetBoardLocatorImageProcessor import (
    TargetBoardLocatorImageProcessor,
)
from contrib.common.lib.cv.spot_analysis.image_processor.ViewAnnotationsImageProcessor import (
    ViewAnnotationsImageProcessor,
)
from contrib.common.lib.cv.spot_analysis.image_processor.ViewHighlightImageProcessor import ViewHighlightImageProcessor

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    "BackgroundColorSubtractionImageProcessor",
    "CustomSimpleImageProcessor",
    "EnclosedEnergyImageProcessor",
    "InpaintImageProcessor",
    "MomentsImageProcessor",
    "PowerpointImageProcessor",
    "SaveToFileImageProcessor",
    "SpotWidthImageProcessor",
    "StabilizationImageProcessor",
    "TargetBoardLocatorImageProcessor",
    "ViewAnnotationsImageProcessor",
    "ViewHighlightImageProcessor",
]
