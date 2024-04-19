import sys
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from PIL import Image

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class CacheableImage:
    def __init__(self, array: np.ndarray = None, cache_path: str = None, source_path: str = None):
        """An image container that allows for caching an image when the image
        data isn't in use, or for retrieval of an image from the cached file
        when the data is in use.

        Only one of the inputs (image, cache_path, source_path) are required.
        However, if the image doesn't exist as a cache file (.npy) but does
        exists as an image file (.png), then both the cache_path and source_path
        can be provided. In this case the image will be loaded from the
        source_path and when cached will be saved to the cache_path.

        The intended use for this class is to reduce memory usage by caching
        images to disk while not in use. Therefore, there is an inherent
        priority order for the data that is returned from various methods:
        (1) in-memory array, (2) numpy cache file, (3) image source file.

        Parameters
        ----------
        array: np.ndarray, optional
            The image, as it exists in memory.
        cache_path: str, optional
            The cached version of the image. Should be a numpy (.npy) file.
        source_path: str, optional
            The source file for the image (an image file, for example jpg or png).
        """
        if array is None and cache_path == None and source_path == None:
            lt.error_and_raise(
                ValueError, "Error in CacheableImage.__init__(): must provide one of array, cache_path, or source_path!"
            )
        self.validate_cache_path(cache_path, "__init__")
        self._array = array
        self._image = None
        self.cache_path = cache_path
        self.source_path = source_path
        self.cached = False

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._array) + sys.getsizeof(self._image)

    @classmethod
    def from_single_source(cls, array_or_path: Union[np.ndarray, str, 'CacheableImage']) -> 'CacheableImage':
        """Generates a CacheableImage from the given numpy array, numpy '.npy' file, or image file."""
        if isinstance(array_or_path, CacheableImage):
            return array_or_path
        elif isinstance(array_or_path, str):
            path: str = array_or_path
            if path.lower().endswith(".npy"):
                return cls(cache_path=path)
            return cls(source_path=path)
        elif isinstance(array_or_path, np.ndarray):
            array: np.ndarray = array_or_path
            return cls(array=array)
        else:
            lt.error_and_raise(
                TypeError, f"Error in CacheableImage.from_single_source(): unexpected type {type(array_or_path)}"
            )

    def validate_cache_path(self, cache_path: Optional[str], caller_name: str):
        if cache_path == None:
            return
        if not cache_path.lower().endswith(".npy"):
            _, _, ext = ft.path_components(cache_path)
            lt.error_and_raise(
                ValueError,
                f"Error in CacheableImage.{caller_name}(): cache_path must end with '.npy' but instead the extension is {ext}",
            )

    @staticmethod
    def _load_image(im: str | np.ndarray) -> npt.NDArray[np.int_]:
        if isinstance(im, np.ndarray):
            return im
        elif im.lower().endswith(".npy"):
            return np.load(im)
        else:
            im = Image.open(im)
            return np.array(im)

    def __load_image(self) -> npt.NDArray[np.int_] | None:
        if self._array is not None:
            return self._load_image(self._array)
        elif self.cache_path is not None and ft.file_exists(self.cache_path):
            self.cached = True
            return self._load_image(self.cache_path)
        elif ft.file_exists(self.source_path):
            return self._load_image(self.source_path)
        else:
            lt.error_and_raise(
                RuntimeError,
                f"Error in CacheableImage.__load_image(): Can't load image! {self._array=}, {self.cache_path=}, {self.source_path=}",
            )

    @property
    def nparray(self) -> npt.NDArray[np.int_] | None:
        self._image = None

        if self._array is None:
            if not self.cached:
                self._array = self.__load_image()

        return self.__load_image()

    def to_image(self) -> Image.Image:
        if self._image == None:
            self._image = it.numpy_to_image(self.nparray)
        return self._image

    def cache(self, cache_path: str = None):
        """Stores this instance to the cache and releases the handle to the in-memory image.
        Note that the memory might not be abailable for garbage collection, if
        there are other parts of the code that still have references to the
        in-memory image or array."""
        # get the path to the numpy file
        if cache_path == None:
            if self.cache_path == None:
                lt.error_and_raise(
                    ValueError,
                    "Error in CacheableImage.cache(): "
                    + "this instance does not have a pre-programmed cache_path and the provided cache_path is None. "
                    + "Caching requires at least one path to be non-None!",
                )
            cache_path = self.cache_path
            self.validate_cache_path(cache_path, "cache")
        self.cache_path = cache_path

        # check that this instance isn't already cached
        if self._array is None and self._image == None:
            return

        # cache this instance
        np.save(cache_path, self.nparray)
        self._array = None
        self._image = None
        self.cached = True
