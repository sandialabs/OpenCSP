import os
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class _IndexableIterable(Iterable[CacheableImage]):
    """A restartable iterable (via an iter() call) that piggybacks off of an indexable object."""

    def __init__(self, src: list[str | Image.Image | np.ndarray | CacheableImage] | Callable[[int], CacheableImage]):
        self.src = src
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if isinstance(self.src, list):
            if len(self.src) <= self.idx:
                raise StopIteration
            ret = self.src[self.idx]
        else:
            ret = self.src(self.idx)

        self.idx += 1

        ret = CacheableImage.from_single_source(ret)
        return ret


class _VideoToFramesIterable(_IndexableIterable):
    """A restartable iterable (via an iter() call) that extracts video frames.

    Extracts all frames from a video, but only once the first frame is
    requested. This means that the first call to next() will be slow, but all
    subsequent calls should return immediately."""

    def __init__(self, handler: vh.VideoHandler):
        self.handler = handler
        self.frames_strs = None
        self.iterator = None

    def __iter__(self):
        self.iterator = None
        return self

    def __next__(self):
        if self.frames_strs == None:
            self.frames_strs = self._video_to_list()
        if self.iterator == None:
            self.iterator = iter(self.frames_strs)
        return CacheableImage.from_single_source(next(self.iterator))

    def _video_to_list(self):
        video = self.handler
        video.extract_frames()
        frame_format = video.get_extracted_frame_path_and_name_format()
        frame_dir, _, frame_ext = ft.path_components(frame_format)
        frame_names: list[str] = ft.files_in_directory_by_extension(frame_dir, [frame_ext])[frame_ext]
        return [os.path.join(frame_dir, frame_name) for frame_name in frame_names]


class ImagesIterable(Iterable[CacheableImage]):
    """
    A restartable iterable that returns one image at a time, for as long as images are still available.

    Iterates over an iterator or callable that returns one image at a time.
    Calling iter() on this instance forces iter() calls to all contained
    iterators.
    """

    def __init__(self, stream: Callable[[int], CacheableImage] | list[str | CacheableImage] | vh.VideoHandler):
        """
        Initializes the ImagesIterable with the provided stream.

        Parameters
        ----------
        stream : Callable[[int], CacheableImage] | list[str | CacheableImage] | vh.VideoHandler
            The stream of images to iterate over. If a callable, then will be passed
            the current iteration index as an argument.

        Raises
        ------
        TypeError
            If the provided stream is not one of the supported types.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if isinstance(stream, _IndexableIterable):
            self._images_iterable = stream
        elif isinstance(stream, vh.VideoHandler):
            vstream: vh.VideoHandler = stream
            self._images_iterable = _VideoToFramesIterable(vstream)
        elif isinstance(stream, list):
            lstream: list[str | CacheableImage] = stream
            self._images_iterable = _IndexableIterable(lstream)
        elif isinstance(stream, Callable):
            cstream: Callable[[int], CacheableImage] = stream
            self._images_iterable = _IndexableIterable(cstream)
        else:
            lt.error_and_raise(
                TypeError,
                f"Error in ImagesStream(): argument \"stream\" should be an iterator, callable, or list, but is instead of type \"{type(stream)}\"",
            )
        self._curr_iter_images: list[CacheableImage] = []

    def __iter__(self):
        self._curr_iter_images = None
        return self

    def __next__(self):
        # set up the iterator as necessary
        if self._curr_iter_images == None:
            self._curr_iter_images = iter(self._images_iterable)

        # load the next image
        ret: CacheableImage = next(self._curr_iter_images)
        return ret

    def to_list(self) -> list[CacheableImage]:
        """
        Converts the iterable to a list of images.

        Returns
        -------
        list[CacheableImage]
            A list containing all images from the iterable.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return [img for img in self._images_iterable]
