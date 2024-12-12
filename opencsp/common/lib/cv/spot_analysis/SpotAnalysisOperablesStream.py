import copy
from typing import Iterator

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class _SpotAnalysisOperablesStream(Iterator[SpotAnalysisOperable]):
    """
    This class is meant for internal use. Users of the SpotAnalysis class or
    SpotAnalysisImageProcessor classes should not need to know the mechanics of
    this class.

    A stream that accepts images as input and provides SpotAnalysisOperables as output.

    This stream can be set up with default values for supporting images or other
    SpotAnalysisOperable data via :py:meth:`set_defaults`.
    """

    def __init__(
        self, images: ImagesIterable | ImagesStream | SpotAnalysisImagesStream | Iterator[SpotAnalysisOperable]
    ):
        """
        Initializes the SpotAnalysisOperablesStream with the provided image source.

        Parameters
        ----------
        images : ImagesIterable | ImagesStream | SpotAnalysisImagesStream | Iterator[SpotAnalysisOperable]
            The source of images to be processed. This will be used as the
            primary images for the produced operables. This
            SpotAnalysisOperablesStream stream will be restartable so long as
            the given 'images' stream is also restartable.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.images = images
        self.images_iter = None
        self.default_support_images: dict[ImageType, CacheableImage] = None
        self.default_data: SpotAnalysisOperable = None

    def set_defaults(self, default_support_images: dict[ImageType, CacheableImage], default_data: SpotAnalysisOperable):
        """
        This stream can be set up with default values for supporting images or
        other SpotAnalysisOperable data. If set, then all produced operables
        will have these default values applied.

        See also :py:meth:`SpotAnalysisOperable.replace_use_default_values`

        Parameters
        ----------
        default_support_images : dict[ImageType, CacheableImage]
            A dictionary of default support images to be used as defaults
            in the generated operables. Can be empty.
        default_data : SpotAnalysisOperable
            Additional data that can be assigned as defaults to the generated
            operables. Includes things that aren't supporting images, such as
            given_fiducials, found_fiducials, annotations,
            camera_intrinsics_characterization, light_sources.
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
