import copy
import dataclasses
from typing import Callable

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
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
        x1x2y1y2: tuple[int, int, int, int] | Callable[[SpotAnalysisOperable], tuple[int, int, int, int]] = None,
        centered_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]] = None,
        width_height: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]] = None,
    ):
        """It is suggested that users call either the :py:meth:`by_region` or
        :py:meth:`by_center_and_size` methods instead of this calling this
        default constructor directly."""
        super().__init__()

        self.x1x2y1y2 = x1x2y1y2
        self.centered_location = centered_location
        self.width_height = width_height

        # validate the inputs
        if x1x2y1y2 is not None:
            warn_msg = ""
            warn_msg += "ignoring centered_location" if centered_location is not None else ""
            warn_msg += "ignoring width_height" if width_height is not None else ""
            if warn_msg != "":
                lt.warning(warn_msg)

            self.cropped_size_str = f"[left: N/A, right: N/A, top: N/A, bottom: N/A]"

            if isinstance(x1x2y1y2, Callable):
                pass
            else:
                self.validate_x1x2y1y2(x1x2y1y2, __name__ + " constructor")

        elif (centered_location is not None) and (width_height is not None):
            self.centered_location = centered_location
            self.width_height = width_height

            if not isinstance(self.centered_location, Callable):
                self.validate_center(self.centered_location, None, __name__ + " constructor")
            if not isinstance(self.width_height, Callable):
                self.validate_center(self.width_height, None, __name__ + " constructor")

        else:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): "
                + "either x1x2y1y2 must be set, or "
                + "all of centered_location/width_height must be set",
            )

    @classmethod
    def by_region(cls, x1x2y1y2: tuple[int, int, int, int] | Callable[[SpotAnalysisOperable], tuple]):
        """
        Parameters
        ----------
        x1x2y1y2 : tuple | Callable
            Either the values to crop to, or a callable to produce said values.
            x1: The left side of the box to crop to (inclusive).
            x2: The right side of the box to crop to (exclusive).
            y1: The top side of the box to crop to (inclusive).
            y2: The bottom side of the box to crop to (exclusive).
        """
        return cls(x1x2y1y2=x1x2y1y2)

    @classmethod
    def by_center_and_size(
        cls,
        centered_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
        width_height: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
    ):
        """
        Parameters
        ----------
        centered_location : tuple[int, int] | Callable[[Operable], tuple]
            The location around which to crop. If the location is too close to
            the edge of the image then the crop is done as close to the edge as
            possible while preserving the desired with and height.
        width_height: tuple[int, int] | Callable[[Operable], tuple]
            The width and height to crop to.
        """
        return cls(centered_location=centered_location, width_height=width_height)

    def validate_x1x2y1y2(self, x1x2y1y2: tuple[int, int, int, int], debug_name: str):
        x1, x2, y1, y2 = x1x2y1y2

        self.cropped_size_str = f"[left: {x1}, right: {x2}, top: {y1}, bottom: {y2}]"
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): "
                + f"all input values {self.cropped_size_str} must be >= 0 for {debug_name}",
            )
        if x1 >= x2 or y1 >= y2:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): "
                + f"x2 must be > x1, and y2 must be > y1, but {self.cropped_size_str} for {debug_name}",
            )

    def validate_center(self, center_xy: tuple[int, int], operable: SpotAnalysisOperable | None, debug_name: str):
        center_x, center_y = center_xy

        # # verify that the center is inside the image boundaries
        # if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
        #     lt.error_and_raise(RuntimeError, err_msg)
        # if operable is not None:
        #     (h, w), _ = it.dims_and_nchannels(operable.primary_image.nparray)
        #     if center_x >= w or center_y >= h:
        #         lt.error_and_raise(RuntimeError, err_msg)

        # TODO RCB REVISED CODE BELOW
        # verify that the center is inside the image boundaries
        if center_x < 0 or center_y < 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in CroppingImageProcessor.crop_around_location(): "
                + f"centered location ({center_x}, {center_y}) is out of image bounds",
            )
        if operable is not None:
            (h, w), _ = it.dims_and_nchannels(operable.primary_image.nparray)
            err_msg = (
                "Error in CroppingImageProcessor.crop_around_location(): "
                + f"centered location ({center_x}, {center_y}) is out of image bounds (width: {w}, height: {h}) "
                + f"for {debug_name}"
            )
            if center_x >= w or center_y >= h:
                lt.error_and_raise(RuntimeError, err_msg)
            if center_x >= w or center_y >= h:
                lt.error_and_raise(RuntimeError, err_msg)

    def validate_width_height(
        self, width_height: tuple[int, int], operable: SpotAnalysisOperable | None, debug_name: str
    ):
        width, height = width_height

        # Verify that the width and height are both positive
        if width <= 0 or height <= 0:
            lt.error_and_raise(
                ValueError,
                f"Error in CroppingImageProcessor.validate_width_height(): "
                + "width and height must both be positive but "
                + f"{width=} and {height=} for {debug_name}!",
            )

        # Verify that the width and height is smaller than the source image
        if operable is not None:
            img = operable.primary_image.nparray
            (h, w), _ = it.dims_and_nchannels(img)
            if width > w or height > h:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in CroppingImageProcessor.crop_around_location(): "
                    + f"cropping size [w: {width}, h: {height}] is smaller than "
                    + f"the input image size [w: {w}, h: {h}]"
                    + f"for {debug_name}!",
                )

    def _crop_image(
        self,
        operable: SpotAnalysisOperable,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        additional_notes: list[tuple[str, str]],
    ) -> SpotAnalysisOperable:
        # crop the image
        image = operable.primary_image.nparray
        (h, w), _ = it.dims_and_nchannels(image)
        lt.debug("In CroppingImageProcessor(): " + f"cropping image from [0:{w},0:{h}] to [{x1}:{x2},{y1}:{y2}]")
        cropped = image[y1:y2, x1:x2]
        new_primary = CacheableImage.from_single_source(cropped)

        # apply the crop to the annotations
        given_fiducials = operable.given_fiducials
        found_fiducials = operable.found_fiducials
        annotations = operable.annotations
        for annots in [given_fiducials, found_fiducials, annotations]:
            for i, annot in enumerate(annots):
                annots[i] = annot.translate(p2.Pxy([-x1, -y1]))

        # apply the changes to the notes
        image_processor_notes = copy.copy(operable.image_processor_notes)
        image_processor_notes += additional_notes

        # build the new operable
        ret = dataclasses.replace(operable, primary_image=new_primary, image_processor_notes=image_processor_notes)

        return ret

    def crop_by_bounding_box(self, operable: SpotAnalysisOperable) -> SpotAnalysisOperable:
        """
        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of SpotAnalysisOperable containing the primary image to be cropped
            and any associated image processor notes.
        """
        image = operable.primary_image.nparray

        # get the x and y values
        if isinstance(self.x1x2y1y2, Callable):
            x1, x2, y1, y2 = self.x1x2y1y2(operable)
        else:
            x1, x2, y1, y2 = self.x1x2y1y2
        self.validate_x1x2y1y2((x1, x2, y1, y2), f"operable '{operable.best_primary_pathnameext}'")

        # check the size of the image
        (h, w), _ = it.dims_and_nchannels(image)
        if x1 >= w or y1 >= h or x2 > w or y2 > h:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor._execute(): "
                + f"given image '{operable.best_primary_pathnameext}' is smaller than the cropped size {self.cropped_size_str}",
            )

        # create the cropped image
        new_notes = [("CroppingImageProcessor", [f"{x1}", f"{x2}", f"{y1}", f"{y2}"])]
        new_operable = self._crop_image(operable, x1, x2, y1, y2, new_notes)

        return new_operable

    def crop_around_center(self, operable: SpotAnalysisOperable) -> SpotAnalysisOperable:
        """
        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of SpotAnalysisOperable containing the primary image to be cropped
            and any associated image processor notes.
        """
        image = operable.primary_image.nparray

        # Get the image dimensions
        (h, w), _ = it.dims_and_nchannels(image)

        # Get the cropping dimensions
        if isinstance(self.width_height, Callable):
            width, height = self.width_height(operable)
        else:
            width, height = self.width_height
        self.validate_width_height((width, height), operable, f"operable '{operable.best_primary_pathnameext}")

        # Determine the centered location
        if callable(self.centered_location):
            center_x, center_y = self.centered_location(operable)
        else:
            center_x, center_y = self.centered_location
        self.validate_center((center_x, center_y), operable, f"operable '{operable.best_primary_pathnameext}'")

        # Calculate the cropping coordinates.
        # Remember that the width and height must match the requested value.
        half_height = int(np.ceil(height / 2))
        half_width = int(np.ceil(width / 2))

        y2 = min(center_y + half_height, h)
        y1 = max(y2 - height, 0)
        y2 = min(y1 + height, h)
        x2 = min(center_x + half_width, w)
        x1 = max(x2 - width, 0)
        x2 = min(x1 + width, w)

        # Sanity check
        assert x2 > x1
        assert y2 > y1
        assert x1 >= 0
        assert y1 >= 0
        assert x2 - x1 == width
        assert y2 - y1 == height

        # Create the cropped image
        new_notes = [
            (
                "CroppingImageProcessor",
                [f"centered at ({center_x}, {center_y})", f"width: {width}", f"height: {height}"],
            )
        ]
        new_operable = self._crop_image(operable, x1, x2, y1, y2, new_notes)

        return new_operable

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.x1x2y1y2 is not None:
            ret = self.crop_by_bounding_box(operable)
        elif self.centered_location is not None:
            ret = self.crop_around_center(operable)
        else:
            lt.error_and_raise(
                ValueError,
                "Error in CroppingImageProcessor(): " + "unknown cropping method encountered in _execute() method",
            )

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
