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
    """
    Enumeration for different types of images used in analysis.

    This enumeration defines various image types that can be utilized in
    image processing and analysis workflows. Each type serves a specific
    purpose in the context of image comparison, background subtraction,
    and other analytical tasks.

    Attributes
    ----------
    PRIMARY : int
        The image we are trying to analyze.
    REFERENCE : int
        Contains a pattern to be compared or matched with in the PRIMARY image.
    NULL : int
        The same as the PRIMARY image, but without a beam on target.
        Likely used to subtract out the background.
    COMPARISON : int
        For multi-image comparison, such as for re-alignment to a previous
        position, motion characterization, or measuring wind effect.
    BACKGROUND_MASK : int
        A boolean image that indicates which pixels should be included in
        a computation (True to include, False to exclude).
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    PRIMARY = 1
    """ The image we are trying to analyze. """
    REFERENCE = 2
    """ Contains a pattern to be compared or matched with in the PRIMARY image. """
    NULL = 3
    """ The same as the PRIMARY image, but without a beam on target. Likely this will be used to subtract out the background. """
    COMPARISON = 4
    """ For multi-image comparison, such as for re-alignment to a previous position, motion characterization, or measuring wind effect. """
    BACKGROUND_MASK = 5
    """ A boolean image that indicates which pixels should be included in a computation (True to include, False to exclude). """

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        raise NotImplementedError


class SpotAnalysisImagesStream(Iterator[dict[ImageType, CacheableImage]]):
    """
    This class combines the image streams for several ImageTypes into
    one convenient package. This helps to guarantee that images that are
    supposed to be processed together stay together for the entirety of the
    SpotAnalysis pipeline.
    """

    def __init__(
        self,
        primary_iterator: ImagesIterable | ImagesStream,
        other_iterators: dict[ImageType, ImagesIterable | ImagesStream] = None,
    ):
        """
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
