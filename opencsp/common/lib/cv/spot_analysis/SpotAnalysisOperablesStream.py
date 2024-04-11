import copy
from typing import Iterator

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import ImageType, SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable


class SpotAnalysisOperablesStream(Iterator[SpotAnalysisOperable]):
    def __init__(
        self, images: ImagesIterable | ImagesStream | SpotAnalysisImagesStream | Iterator[SpotAnalysisOperable]
    ):
        self.images = images
        self.images_iter = None
        self.default_support_images: dict[ImageType, CacheableImage] = None
        self.default_data: SpotAnalysisOperable = None

    def set_defaults(self, default_support_images: dict[ImageType, CacheableImage], default_data: SpotAnalysisOperable):
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
