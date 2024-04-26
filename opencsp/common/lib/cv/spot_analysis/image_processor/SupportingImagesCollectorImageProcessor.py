import dataclasses
from typing import Callable

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class NoPrimaryImageException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class SupportingImagesCollectorImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """
    Collects primary and supporting images together from a stream of mixed images.

    The basic algorithm is pretty simple:

        1. catagorize images based on the given supporting_images_map
        2. if the image type isn't already in the internal list, then add it and go back to step 1
        3. collect all images in the internal list together as a single operable
        4. clear the internal list
        5. start a new internal list with the current image
        6. return the new operable, go back to step 1
    """

    def __init__(
        self,
        supporting_images_map: dict[
            ImageType, Callable[[SpotAnalysisOperable, dict[ImageType, SpotAnalysisOperable]], bool]
        ],
    ):
        """
        Parameters
        ----------
        supporting_images_map : dict[ ImageType, Callable[[SpotAnalysisOperable, dict[ImageType, SpotAnalysisOperable]], bool] ]
            How to categorize images. If
            `supporting_images_map[ImageType.PRIMARY](operable, curr_mapped_images) == True`
            then the image will be assigned as a primary image. Otherwise it will be grouped with another primary image
            as a supporting image.
        """
        super().__init__(self.__class__.__name__)

        # register inputs
        self.supporting_images_map = supporting_images_map

        # list of images to be collected together
        self.collection: dict[ImageType, SpotAnalysisOperable] = {}
        self.prev_image_types: list[ImageType] = None

    def _update_collection(self) -> SpotAnalysisOperable:
        # 3. Turn the current collection into a new operable
        # 3.1. Check that there is a primary image
        if ImageType.PRIMARY not in self.collection:
            raise NoPrimaryImageException("No primary image registerd. Failed to update collection.")
        primary = self.collection[ImageType.PRIMARY]

        # 3.2. Check that we have as many image types as we expect
        image_types = sorted(list(self.collection.keys()))
        expected_image_types = sorted(list(self.supporting_images_map.keys()))
        if self.prev_image_types is not None:
            if image_types != self.prev_image_types:
                lt.warning(
                    "Warning in SupportingImagesCollectorImageProcessor._update_collection(): "
                    + f"expected to find images with types {self.prev_image_types}, but instead found {image_types}"
                )
        if image_types != expected_image_types:
            lt.debug(
                "In SupportingImagesCollectorImageProcessor._update_collection(): "
                + f"expected image types from input map {expected_image_types}, found image types {image_types}"
            )
        self.prev_image_types = image_types

        # 3.2. We have a primary, create the new operable
        lt.debug(
            "In SupportingImagesCollectorImageProcessor._update_collection(): "
            + f"collecting images {sorted(list(self.collection.keys()))}"
        )
        supporting_images: dict[ImageType, CacheableImage] = {}
        for it in self.collection:
            if it != ImageType.PRIMARY:
                supporting_images[it] = self.collection[it].primary_image
        new_operable = dataclasses.replace(primary, supporting_images=supporting_images)

        return new_operable

    def _execute(self, curr_operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # 1. get the image type
        curr_image_type = None
        for it in self.supporting_images_map:
            if self.supporting_images_map[it](curr_operable, self.collection):
                curr_image_type = it
        if curr_image_type is None:
            lt.error_and_raise(
                ValueError,
                "Error in SupportingImagesCollectorImageProcessor._execute(): "
                + f"unable to determine image type for operable {curr_operable.primary_image_source_path} ({curr_operable})",
            )

        # Handle is_last edge case
        if is_last:
            # add this operable to the collection, but first check if there's room in the collection
            if curr_image_type in self.collection:
                lt.warning(
                    "Warning in SupportingImagesCollectorImageProcessor._update_collection(): "
                    + "mismatched image types. "
                    + f"Removing {curr_image_type} '{self.collection[curr_image_type].primary_image_source_path}' and replacing it with '{curr_operable.primary_image_source_path}'"
                )
            self.collection[curr_image_type] = curr_operable

            # update the collection
            try:
                new_operable = self._update_collection()
            except NoPrimaryImageException as ex:
                lt.warning(repr(ex))
                lt.warning(
                    "Warning in SupportingImagesCollectorImageProcessor._update_collection(): "
                    + f"discarding {len(self.collection)} operables that don't have a matching primary image."
                )

            # 6. Return the new operable
            return [new_operable]

        # 2. If this image type isn't already in the collection, then add it and continue.
        elif curr_image_type not in self.collection:
            self.collection[curr_image_type] = curr_operable
            return []

        # Otherwise there is a duplicate.
        else:
            try:
                new_operable = self._update_collection()
            except NoPrimaryImageException as ex:
                lt.warning(repr(ex))
                lt.warning(
                    "Warning in SupportingImagesCollectorImageProcessor._update_collection(): "
                    + "no PRIMARY image is available, so we can't create new a new operable. "
                    + f"Removing {curr_image_type} '{self.collection[curr_image_type].primary_image_source_path}' and replacing it with '{curr_operable.primary_image_source_path}'"
                )
                self.collection[curr_image_type] = curr_operable
                return []

            # 4. Clear the collection
            self.collection.clear()

            # 5. Start a new collection with the current operable
            self.collection[curr_image_type] = curr_operable

            # 6. Return the new operable
            return [new_operable]
