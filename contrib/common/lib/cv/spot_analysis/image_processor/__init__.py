from typing import TYPE_CHECKING as _TYPE_CHECKING

from opencsp import LazyLoader as _LazyLoader

base_package = "contrib.common.lib.cv.spot_analysis.image_processor."

# fmt: off
BackgroundColorSubtractionImageProcessor = _LazyLoader(base_package + "BackgroundColorSubtractionImageProcessor", "BackgroundColorSubtractionImageProcessor")
ColorConversionImageProcessor = _LazyLoader(base_package + "ColorConversionImageProcessor", "ColorConversionImageProcessor")
CustomSimpleImageProcessor = _LazyLoader(base_package + "CustomSimpleImageProcessor", "CustomSimpleImageProcessor")
DiscardAnnotationsImageProcessor = _LazyLoader(base_package + "DiscardAnnotationsImageProcessor", "DiscardAnnotationsImageProcessor")
EnclosedEnergyImageProcessor = _LazyLoader(base_package + "EnclosedEnergyImageProcessor", "EnclosedEnergyImageProcessor")
InpaintImageProcessor = _LazyLoader(base_package + "InpaintImageProcessor", "InpaintImageProcessor")
MomentsImageProcessor = _LazyLoader(base_package + "MomentsImageProcessor", "MomentsImageProcessor")
PowerpointImageProcessor = _LazyLoader(base_package + "PowerpointImageProcessor", "PowerpointImageProcessor")
SaveToFileImageProcessor = _LazyLoader(base_package + "SaveToFileImageProcessor", "SaveToFileImageProcessor")
SpotWidthImageProcessor = _LazyLoader(base_package + "SpotWidthImageProcessor", "SpotWidthImageProcessor")
StabilizationImageProcessor = _LazyLoader(base_package + "StabilizationImageProcessor", "StabilizationImageProcessor")
TargetBoardLocatorImageProcessor = _LazyLoader(base_package + "TargetBoardLocatorImageProcessor", "TargetBoardLocatorImageProcessor")
ViewAnnotationsImageProcessor = _LazyLoader(base_package + "ViewAnnotationsImageProcessor", "ViewAnnotationsImageProcessor")
ViewHighlightImageProcessor = _LazyLoader(base_package + "ViewHighlightImageProcessor", "ViewHighlightImageProcessor")
# fmt: on

if _TYPE_CHECKING:
    # fmt: off
    from contrib.common.lib.cv.spot_analysis.image_processor.BackgroundColorSubtractionImageProcessor import BackgroundColorSubtractionImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.ColorConversionImageProcessor import ColorConversionImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.CustomSimpleImageProcessor import CustomSimpleImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.DiscardAnnotationsImageProcessor import DiscardAnnotationsImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.EnclosedEnergyImageProcessor import EnclosedEnergyImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.InpaintImageProcessor import InpaintImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.MomentsImageProcessor import MomentsImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.PowerpointImageProcessor import PowerpointImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.SaveToFileImageProcessor import SaveToFileImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.SpotWidthImageProcessor import SpotWidthImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.StabilizationImageProcessor import StabilizationImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.TargetBoardLocatorImageProcessor import TargetBoardLocatorImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.ViewAnnotationsImageProcessor import ViewAnnotationsImageProcessor
    from contrib.common.lib.cv.spot_analysis.image_processor.ViewHighlightImageProcessor import ViewHighlightImageProcessor
    # fmt: on

# Make these classes available when importing cv.spot_analysis.image_processor.*
__all__ = [
    "BackgroundColorSubtractionImageProcessor",
    "ColorConversionImageProcessor",
    "CustomSimpleImageProcessor",
    "DiscardAnnotationsImageProcessor",
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
