import copy
from typing import Iterator

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import ImageType, SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class SpotAnalysisOperablesStream(Iterator[SpotAnalysisOperable]):
    """
    An iterator that streams SpotAnalysisOperable objects from a collection of images.

    This class allows for the iteration over a collection of images, converting them into
    `SpotAnalysisOperable` objects. It can handle various types of image sources, including
    `ImagesIterable`, `ImagesStream`, and `SpotAnalysisImagesStream`.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(
        self, images: ImagesIterable | ImagesStream | SpotAnalysisImagesStream | Iterator[SpotAnalysisOperable]
    ):
        """
        Initializes the SpotAnalysisOperablesStream with the provided image source.

        Parameters
        ----------
        images : ImagesIterable | ImagesStream | SpotAnalysisImagesStream | Iterator[SpotAnalysisOperable]
            The source of images to be processed.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.images = images
        self.images_iter = None
        self.default_support_images: dict[ImageType, CacheableImage] = None
        self.default_data: SpotAnalysisOperable = None

    def set_defaults(self, default_support_images: dict[ImageType, CacheableImage], default_data: SpotAnalysisOperable):
        """
        Sets default support images and data for the operables.

        Parameters
        ----------
        default_support_images : dict[ImageType, CacheableImage]
            A dictionary of default support images to be used in the operables.
        default_data : SpotAnalysisOperable
            Default data to be used in the operables.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.default_support_images = default_support_images
        self.default_data = default_data

    def __iter__(self):
        self.images_iter = iter(self.images)
        return self

    def __next_operable(self):
        val = next(self.images_iter)
        if isinstance(val, SpotAnalysisOperable):
            return val
        elif isinstance(val, dict):
            primary_image = val[ImageType.PRIMARY]
            supporting_images = copy.copy(val)
            return SpotAnalysisOperable(primary_image, supporting_images=supporting_images)
        else:
            primary_image = val
            return SpotAnalysisOperable(primary_image, {})

    def __next__(self):
        operable = self.__next_operable()
        operable = operable.replace_use_default_values(self.default_support_images, self.default_data)
        return operable
