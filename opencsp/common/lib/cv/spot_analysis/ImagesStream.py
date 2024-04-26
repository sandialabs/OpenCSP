from typing import Callable, Iterator

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.tool.log_tools as lt


class _StrToCacheableImagesIterator(Iterator[CacheableImage]):
    def __init__(self, iterator: Iterator[str | CacheableImage]):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        CacheableImage.from_single_source(next(self.iterator))


class ImagesStream(Iterator[CacheableImage]):
    """
    A one-time iterator over a list of images.

    Iterates over the given images. The next() method returns the next item
    from the input list.

    Note that calling iter() on this instance DOES NOT restart iteration.
    This is to maintain the interface for streamable image sources, such as
    webcams or networked cameras, which have no replay ability.

    Note that this does NOT build on the python asyncio stream library for
    networking. Such a class might be implemented later and will likely be
    called "ImagesStreamOverIP".
    """

    def __init__(
        self,
        images: (
            Callable[[int], CacheableImage]
            | list[str | CacheableImage]
            | vh.VideoHandler
            | Iterator[str | CacheableImage]
        ),
    ):
        """
        Parameters
        ----------
        images : Callable[[int],CacheableImage] | list[str|CacheableImage] | vh.VideoHandler | Iterator[str|CacheableImage]
            The images to iterate over.
        """
        self._images = images
        self._curr_iter_images: Iterator[CacheableImage] = None

        if isinstance(images, vh.VideoHandler):
            self._curr_iter_images = iter(ImagesIterable(images))
        elif isinstance(images, Callable):
            self._curr_iter_images = iter(ImagesIterable(images))
        elif isinstance(images, list):
            self._curr_iter_images = iter(ImagesIterable(images))
        elif isinstance(images, Iterator):
            self._curr_iter_images = iter(_StrToCacheableImagesIterator(images))
        else:
            lt.error_and_raise(TypeError)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._curr_iter_images)
