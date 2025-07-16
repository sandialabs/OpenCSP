from enum import Enum
import functools


@functools.total_ordering
class ImageType(Enum):
    """
    The category of a single CacheableImage that is being passed through the
    image processing pipeline. Most images will be PRIMARY images, meaning that
    they will be the data being analyzed. Images of different types are grouped
    together by a :py:class:`SpotAnalysisImagesStream` or
    :py:class:`SpotAnalysisOperable`.
    """

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
    VISUALIZATION = 6
    """ Images used to view the results or state of image processing. """
    ALGORITHM = 7
    """ Explainer images that visualize how an image processing step was completed. """

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        raise NotImplementedError

    @staticmethod
    @functools.cache
    def ALL() -> tuple["ImageType"]:
        cls = ImageType
        return (
            cls.PRIMARY,
            cls.REFERENCE,
            cls.NULL,
            cls.COMPARISON,
            cls.BACKGROUND_MASK,
            cls.VISUALIZATION,
            cls.ALGORITHM,
        )
