import copy
import dataclasses
from typing import Callable

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class CroppingImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Crops all input images to the given shape.

    Crops to either:

        - a specified left/right/top/bottom region
        - a specified width/height around a center x/y location.

    If the input image is too small, then an error will be thrown.
    """

    def __init__(
        self,
        x1: int = None,
        x2: int = None,
        y1: int = None,
        y2: int = None,
        centered_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]] = None,
        width: int = None,
        height: int = None,
    ):
        """It is suggested that users call either the :py:meth:`by_region` or
        :py:meth:`by_center_and_size` methods instead of this calling this
        default constructor directly."""
        super().__init__()

        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.centered_location = None
        self.width = None
        self.height = None

        # validate the inputs
        if (x1 is not None) and (x2 is not None) and (y1 is not None) and (y2 is not None):
            self.cropped_size_str = f"[left: {x1}, right: {x2}, top: {y1}, bottom: {y2}]"
            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                lt.error_and_raise(
                    ValueError,
                    "Error in CroppingImageProcessor(): " + f"all input values {self.cropped_size_str} must be >= 0",
                )
            if x1 >= x2 or y1 >= y2:
                lt.error_and_raise(
                    ValueError,
                    "Error in CroppingImageProcessor(): "
                    + f"x2 must be > x1, and y2 must be > y1, but {self.cropped_size_str}",
                )
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2

        elif (centered_location is not None) and (width is not None) and (height is not None):
            self.centered_location = centered_location
            self.width = width
            self.height = height

        else:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): "
                + "either all of x1/x2/y1/y2 must be set, or "
                + "all of centered_location/width/height must be set",
            )

    @classmethod
    def by_region(cls, x1: int, x2: int, y1: int, y2: int):
        """
        Parameters
        ----------
        x1 : int
            The left side of the box to crop to (inclusive).
        x2 : int
            The right side of the box to crop to (exclusive).
        y1 : int
            The top side of the box to crop to (inclusive).
        y2 : int
            The bottom side of the box to crop to (exclusive).
        """
        return cls(x1=x1, x2=x2, y1=y1, y2=y2)

    @classmethod
    def by_center_and_size(
        cls,
        centered_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
        width: int,
        height: int,
    ):
        """
        Parameters
        ----------
        centered_location : tuple[int, int] | Callabled[[Operable], tuple]
            The location around which to crop. If the location is too close to
            the edge of the image then the crop is done as close to the edge as
            possible while preserving the desired with and height.
        width : int
            The width to crop to.
        height : int
            The height to crop to.
        """
        return cls(centered_location=centered_location, width=width, height=height)

    def crop_by_static_location(self, operable: SpotAnalysisOperable) -> tuple[CacheableImage, list]:
        """
        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of SpotAnalysisOperable containing the primary image to be cropped
            and any associated image processor notes.
        """
        image = operable.primary_image.nparray

        # check the size of the image
        h, w = image.shape[0], image.shape[1]
        if w < self.x2 - 1 or h < self.y2 - 1:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor._execute(): "
                + f"given image '{operable.best_primary_pathnameext}' is smaller than the cropped size {self.cropped_size_str}",
            )

        # create the cropped image
        cropped = image[self.y1 : self.y2, self.x1 : self.x2]
        new_primary = CacheableImage(cropped)

        image_processor_notes = copy.copy(operable.image_processor_notes)
        image_processor_notes.append(
            ("CroppingImageProcessor", [f"{self.x1}", f"{self.x2}", f"{self.y1}", f"{self.y2}"])
        )

        return new_primary, image_processor_notes

    def crop_around_location(self, operable: SpotAnalysisOperable) -> tuple[CacheableImage, list]:
        """
        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of SpotAnalysisOperable containing the primary image to be cropped
            and any associated image processor notes.
        """
        image = operable.primary_image.nparray

        # Get the image dimensions
        h, w = image.shape[0], image.shape[1]

        # Verify that the width and height is smaller than the source image
        if self.width > w or self.height > h:
            lt.error_and_raise(
                RuntimeError,
                "Error in CroppingImageProcessor.crop_around_location(): "
                + f"cropping size [w: {self.width}, h: {self.height}] is smaller than "
                + f"the input image size [w: {w}, h: {h}]"
                + f"(for image {operable.best_primary_pathnameext})",
            )

        # Determine the centered location
        if callable(self.centered_location):
            center_x, center_y = self.centered_location(operable)
        else:
            center_x, center_y = self.centered_location

        # Verify that the center coordinates are within the image boundaries
        if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
            lt.error_and_raise(
                RuntimeError,
                "Error in CroppingImageProcessor.crop_around_location(): "
                + f"centered location ({center_x}, {center_y}) is out of image bounds (width: {w}, height: {h})",
            )

        # Calculate the cropping coordinates.
        # Remember that the width and height must match the requested value.
        half_height = self.height // 2
        half_width = self.width // 2

        y1 = max(center_y - half_height, 0)
        center_y = y1 + half_height
        y2 = min(center_y + half_height, h)
        center_y, y1 = y2 - half_height, y2 - self.height
        x1 = max(center_x - half_width, 0)
        center_x = x1 + half_width
        x2 = min(center_x + half_width, w)
        center_x, x1 = x2 - half_width, x2 - self.width

        # Sanity check
        assert x2 > x1
        assert y2 > y1
        assert x1 >= 0
        assert y1 >= 0

        # Create the cropped image
        lt.info("In CroppingImageProcessor(): " + f"cropping image from [0:{w},0:{h}] to [{x1}:{x2},{y1}:{y2}]")
        cropped = image[y1:y2, x1:x2]
        new_primary = CacheableImage.from_single_source(cropped)

        image_processor_notes = copy.copy(operable.image_processor_notes)
        image_processor_notes.append(
            (
                "CroppingImageProcessor",
                [f"centered at ({center_x}, {center_y})", f"width: {self.width}", f"height: {self.height}"],
            )
        )

        return new_primary, image_processor_notes

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.x1 is not None:
            new_primary, image_processor_notes = self.crop_by_static_location(operable)
        elif self.centered_location is not None:
            new_primary, image_processor_notes = self.crop_around_location(operable)
        else:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): " + "unknown cropping method encountered in _execute() method",
            )

        ret = dataclasses.replace(operable, primary_image=new_primary, image_processor_notes=image_processor_notes)
        return [ret]


if __name__ == "__main__":
    expdir = (
        orp.opencsp_scratch_dir()
        + "/solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01"
    )
    indir = expdir + "/raw_images"
    outdir = expdir + "/processed_images"

    x1, y1, x2, y2 = 120, 29, 1526, 1158
    x1, y1 = x1 + 20, y1 + 20
    x2, y2 = x2 - 20, y2 - 20

    ft.create_directories_if_necessary(outdir)
    ft.delete_files_in_directory(outdir, "*")

    processor = CroppingImageProcessor(x1, x2, y1, y2)
    for filename in ft.files_in_directory(indir):
        img = CacheableImage.from_single_source(indir + "/" + filename)
        result = processor.process_operable(SpotAnalysisOperable(img))[0]
        cropped = result.primary_image.save_image(outdir + "/" + filename)
