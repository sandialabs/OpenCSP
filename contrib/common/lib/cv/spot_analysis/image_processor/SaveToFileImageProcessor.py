from typing import Callable

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class SaveToFileImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Prints the image names to the console as they are encountered.
    """

    def __init__(
        self,
        save_dir: str,
        save_ext="png",
        prefix: str | Callable[[SpotAnalysisOperable], str] = None,
        suffix: str | Callable[[SpotAnalysisOperable], str] = None,
        save_primary: str | None = "",
        save_supporting: str | None = None,
        save_visualizations: str | None = None,
        save_algorithms: str | None = None,
        primary_log_level=lt.log.INFO,
    ):
        """
        Parameters
        ----------
        save_dir : str
            The directory where the images will be saved.
        save_ext : str, optional
            The file extension for the saved images (default is "png").
        prefix : str or callable, optional
            The prefix for the saved image file names. If a callable, it will be
            called with the SpotAnalysisOperable as an argument. Default is None.
        suffix : str or callable, optional
            The suffix for the saved image file names. If a callable, it will be
            called with the SpotAnalysisOperable as an argument. Default is None.
        save_primary : str, optional
            The subdirectory for saving primary images to (default is an empty string).
        save_supporting : str, optional
            The subdirectory for saving supporting images to (default is None).
        save_visualizations : str, optional
            The subdirectory for saving visualization images to (default is None).
        save_algorithms : str, optional
            The subdirectory for saving algorithm images to (default is None).
        primary_log_level : int, optional
            The log level for where the primary image is saved to (default is INFO).
        """
        super().__init__()

        # register parameters
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.prefix = prefix
        self.suffix = suffix
        self.save_primary = save_primary
        self.save_supporting = save_supporting
        self.save_visualizations = save_visualizations
        self.save_algorithms = save_algorithms
        self.primary_log_level = primary_log_level

    def get_save_dir(self, sub_dir: str) -> str:
        """
        Builds the full directory to save to, and creates the returned directory as necessary.

        Parameters
        ----------
        sub_dir : str
            The subdirectory to append to the base save directory.

        Notes
        -----
        - If the subdirectory does not exist, it will be created.
        """
        if sub_dir is None:
            return None

        # get the name of the directory
        if sub_dir == "":
            ret = self.save_dir
        else:
            ret = ft.join(self.save_dir, sub_dir)

        # create the directory, as necessary
        ft.create_directories_if_necessary(ret)

        return ret

    def save_image(self, save_path: str, image: CacheableImage, operable: SpotAnalysisOperable, always_number=False):
        """
        Saves an image to the specified path.

        Adds self.prefix, self.suffix to the image name. If the image file
        already exists then also add unique image numbers to the image name.

        Parameters
        ----------
        save_path : str
            The base directory to save the image in.
        image : CacheableImage
            The image to save.
        operable : SpotAnalysisOperable
            The operable associated with the image.
        always_number : bool, optional
            Whether to always add a number to the filename, even if the base
            filename does not exist (default is False).

        Returns
        -------
        str
            The full path/name.ext to the saved image.
        """
        _, save_name, _ = ft.path_components(operable.best_primary_nameext)

        # build the source name with the prefix and suffix
        if self.prefix is not None:
            prefix = self.prefix if isinstance(self.prefix, str) else self.prefix(operable)
            save_name = prefix + save_name
        if self.suffix is not None:
            suffix = self.suffix if isinstance(self.suffix, str) else self.suffix(operable)
            save_name += suffix

        # add numbers in the case that this image already exists
        save_path_name_ext = ft.join(save_path, "%s.%s" % (save_name, self.save_ext))
        i = 0
        if always_number:
            save_path_name_ext = ft.join(save_path, "%s%d.%s" % (save_name, i, self.save_ext))
            i += 1
        while ft.file_exists(save_path_name_ext):
            save_path_name_ext = ft.join(save_path, "%s%d.%s" % (save_name, i, self.save_ext))
            i += 1

        # save this image
        image.to_image().save(save_path_name_ext)

        return save_path_name_ext

    def save_images(self, save_path: str, images: list[CacheableImage], operable: SpotAnalysisOperable):
        """
        Saves a list of images to the specified path.

        Adds prefixes, suffixes, and numbers to the image name in self.save_image().

        Parameters
        ----------
        save_path : str
            The base directory to save the images in.
        images : list[CacheableImage]
            The images to save.
        operable : SpotAnalysisOperable
            The operable associated with the images.

        Returns
        -------
        list[str]
            A list of the full path/name.exts to the saved images.
        """
        ret: list[str] = []
        always_number = len(images) > 0

        for image in images:
            save_path_name_ext = self.save_image(save_path, image, operable, always_number)
            ret.append(save_path_name_ext)

        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.save_primary is not None:
            save_dir = self.get_save_dir(self.save_primary)
            primary_path_name_ext = self.save_image(save_dir, operable.primary_image, operable)
            lt.get_log_method_for_level(self.primary_log_level)(f"Saved primary image: {primary_path_name_ext}")

        if self.save_supporting is not None:
            save_dir = self.get_save_dir(self.save_supporting)
            images: list[CacheableImage] = []
            for image in operable.supporting_images:
                images.append(image)
            other_path_name_exts = self.save_images(save_dir, images, operable)
            lt.debug(f"Saved supporting images: {other_path_name_exts}")

        if self.save_algorithms is not None:
            save_dir = self.get_save_dir(self.save_algorithms)
            images: list[CacheableImage] = []
            for processor in operable.algorithm_images:
                images += operable.algorithm_images[processor]
            other_path_name_exts = self.save_images(save_dir, images, operable)
            lt.debug(f"Saved algorithm images: {other_path_name_exts}")

        if self.save_visualizations is not None:
            save_dir = self.get_save_dir(self.save_visualizations)
            images: list[CacheableImage] = []
            for processor in operable.visualization_images:
                images += operable.visualization_images[processor]
            other_path_name_exts = self.save_images(save_dir, images, operable)
            lt.debug(f"Saved visualization images: {other_path_name_exts}")

        return [operable]
