from collections.abc import Iterator
from enum import Enum
import functools

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


@functools.total_ordering
class ImageType(Enum):
    PRIMARY = 1
    REFERENCE = 2
    NULL = 3
    COMPARISON = 4
    BACKGROUND_MASK = 5

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        raise NotImplementedError


class SpotAnalysisImagesStream(Iterator[dict[ImageType, CacheableImage]]):
    tt.strict_types

    def __init__(
        self,
        primary_iterator: ImagesIterable | ImagesStream,
        other_iterators: dict[ImageType, ImagesIterable | ImagesStream] = None,
    ):
        """This class combines the image streams for several ImageTypes into
        one convenient package. This helps to guarantee that images that are
        supposed to be processed together stay together for the entirety of the
        SpotAnalysis pipeline.

        Parameters
        ----------
        primary_iterator : ImagesIterable | ImagesStream
            The images of the spot to be analyzed.
        other_iterators : dict[ImageType,ImagesStream], optional
            Supporting images that provide extra information for spot analysis. Default None
        """
        if other_iterators == None:
            other_iterators = {}

        self.primary_iterator = primary_iterator
        self.other_iterators = other_iterators
        if ImageType.PRIMARY in other_iterators:
            lt.warn(
                "Warning in SpotAnalysisImagesStream: the other_iterators \"PRIMARY\" type will be ignored in favor of the primary_iterator"
            )
            del other_iterators[ImageType.PRIMARY]

        self.current_iterators: dict[ImageType, ImagesIterable | ImagesStream] = {ImageType.PRIMARY: None}

    def __iter__(self):
        self.current_iterators = {}
        for img_t in self.other_iterators:
            self.current_iterators[img_t] = iter(self.other_iterators[img_t])
        self.current_iterators[ImageType.PRIMARY] = iter(self.primary_iterator)
        return self

    def __next__(self):
        ret: dict[ImageType, CacheableImage] = {}
        for img_t in self.current_iterators:
            ret[img_t] = next(self.current_iterators[img_t])
        return ret
