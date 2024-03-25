from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class EchoImageProcessor(AbstractSpotAnalysisImagesProcessor):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        lt.debug(f"Processing image {operable.primary_image_name_for_logs}")
        return [operable]
