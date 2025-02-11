from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class EchoImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    A do-nothing processor that prints the image names to the console as they are encountered.
    """

    def __init__(self, log_level=lt.log.INFO, prefix=""):
        super().__init__()

        self.log_level = log_level
        self.prefix = prefix

        self.logger = lt.get_log_method_for_level(self.log_level)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        self.logger(f"{self.prefix}Processing image {operable.best_primary_nameext}")

        return [operable]
