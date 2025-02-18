import copy
from dataclasses import dataclass, field, replace
import numpy as np
import numpy.typing as npt
import os
import sys
from typing import TYPE_CHECKING, Optional, Union

import opencsp.common.lib.csp.LightSource as ls
import opencsp.common.lib.cv.annotations.AbstractAnnotations as aa
import opencsp.common.lib.cv.fiducials.AbstractFiducials as af
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisPopulationStatistics import SpotAnalysisPopulationStatistics
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

if TYPE_CHECKING:
    # Use the TYPE_CHECKING magic value to avoid cyclic imports at runtime.
    # This import is only here for type annotations.
    from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
        AbstractSpotAnalysisImagesProcessor,
    )


@dataclass(frozen=True)
class SpotAnalysisOperable:
    """
    Contains a list of all the pieces that might be used or generated during
    spot analysis. This helps to guarantee that related data stays together for
    the whole process. Note that although not all fields will be populated, the
    primary_image at least should be.

    This class should require very little memory, as there can be a great many
    instances of this class in flight at any given time."""

    primary_image: CacheableImage
    """ The input image to be processed, or the output processed image. """
    primary_image_source_path: str = None
    """ The path_name_ext of the primary image, if the source is a file on disk.
    This should be used as the secondary source of this information, after
    :py:meth:`best_primary_pathnameext` and :py:attr:`primary_image`.source_path. """
    supporting_images: dict[ImageType, CacheableImage] = field(default_factory=dict)
    """
    The supporting images, if any, that were provided with the associated input
    primary image. These images will be used as part of the computation.
    """
    previous_operables: (
        tuple[list["SpotAnalysisOperable"], "AbstractSpotAnalysisImagesProcessor"] | tuple[None, None]
    ) = (None, None)
    """
    The operable(s) that were used to generate this operable, and the image
    processor that they came from, if any. If this operable has no previous
    operables registered with it, then this will have the value (None, None).
    Does not include no-nothing image processors such as
    :py:class:`.EchoImageProcessor`.
    """
    given_fiducials: list[af.AbstractFiducials] = field(default_factory=list)
    """ Any fiducials handed to us in the currently processing image. """
    found_fiducials: list[af.AbstractFiducials] = field(default_factory=list)
    """ The identified fiducials in the currently processing image. """
    annotations: list[aa.AbstractAnnotations] = field(default_factory=list)
    """ The identified annotations in the currently processing image. """
    camera_intrinsics_characterization: any = (
        None  # TODO figure out how to specify information here, maybe using common/lib/camera/Camera
    )
    """ Distortion, color, bit depth, etc of the camera. Maybe also distortion properties of the lens system. """
    light_sources: list[ls.LightSource] = field(default_factory=list)
    """ The sources that produced the light that landed on the observed
    receiver surface. This can be used to indicate laser color, light intensity,
    or with simulations to provide a starting estimate of beam shape. """
    population_statistics: SpotAnalysisPopulationStatistics = None
    """ The population statistics, as populated by PopulationStatisticsImageProcessor. """
    image_processor_notes: list[tuple[str, list[str]]] = field(default_factory=list)
    """ Notes from specific image processors. These notes are generally intended for human use, but it is recommended
    that they maintain a consistent formatting so that they can also be used programmatically. """

    def __post_init__(self):
        # We use this method to sanitize the inputs to the constructor.
        #
        # This method will be called after __init__(parameters...) by the
        # @dataclass wrapper. When we detect that an input needs to be modified,
        # we can't modify it directly because of the frozen=True argument.
        # Instead, we use self.__init__(parameter=updated_value) to update the
        # parameter on this instance, which in turn calls __post_init__() again.
        primary_image = self.primary_image
        primary_image_source_path = self.primary_image_source_path
        supporting_images = copy.copy(self.supporting_images)
        requires_update = False

        # make sure that there isn't a primary image in the 'other_images' dict
        if ImageType.PRIMARY in supporting_images:
            if primary_image is None:
                primary_image = supporting_images[ImageType.PRIMARY]
            del supporting_images[ImageType.PRIMARY]
            requires_update = True

        # set the images to be CacheableImages, in the case that they aren't already
        if not isinstance(primary_image, CacheableImage):
            primary_image = CacheableImage.from_single_source(primary_image)
            requires_update = True
        for image_type in supporting_images:
            supporting_image = supporting_images[image_type]
            if not isinstance(supporting_image, CacheableImage):
                supporting_images[image_type] = CacheableImage(supporting_image)
                if isinstance(supporting_image, str):
                    supporting_images[image_type].source_path = supporting_image
                requires_update = True

        # record the primary image source, if not available already
        if primary_image_source_path is None:
            if primary_image.source_path is not None:
                primary_image_source_path = primary_image.source_path
            else:
                primary_image_source_path = primary_image.cache_path
            if primary_image_source_path is not None:
                requires_update = True

        if requires_update:
            # use __init__ to update frozen values
            self.__init__(
                primary_image,
                primary_image_source_path,
                supporting_images,
                self.previous_operables,
                self.given_fiducials,
                self.found_fiducials,
                self.annotations,
                self.camera_intrinsics_characterization,
                self.light_sources,
                self.population_statistics,
                self.image_processor_notes,
            )

    def get_all_images(self, primary=True, supporting=True, visualization=True, algorithm=True) -> list[CacheableImage]:
        """
        Get a list of all images tracked by this operable including all primary
        images, supporting images, visualization, and algorithm images. Does not
        include images from previous operables.

        Parameters
        ----------
        primary : bool, optional
            True to include the primary image in the list of returned images. By
            default True.
        supporting : bool, optional
            True to include the supporting images, if any, in the list of
            returned images. By default True.
        visualization : bool, optional
            True to include the visualization images in the list of returned
            images. By default True.
        algorithm : bool, optional
            True to include the algorithm images, if any, in the list of
            returned images. By default True.

        Returns
        -------
        list[CacheableImage]
            The images tracked by this operable.
        """
        ret: list[CacheableImage] = []

        if primary:
            ret.append(self.primary_image)

        if supporting:
            for image_type in self.supporting_images:
                ret.append(self.supporting_images[image_type])

        return ret

    def replace_use_default_values(
        self, supporting_images: dict[ImageType, CacheableImage] = None, data: "SpotAnalysisOperable" = None
    ) -> "SpotAnalysisOperable":
        """Sets the supporting_images and other data for an operable where they
        are None for this instance. Returns a new operable with the populated
        values."""
        ret = self

        if supporting_images is not None:
            for image_type in supporting_images:
                if (image_type in ret.supporting_images) and (ret.supporting_images[image_type] is not None):
                    supporting_images[image_type] = ret.supporting_images[image_type]
            ret = replace(ret, supporting_images=supporting_images)

        if data is not None:
            given_fiducials = data.given_fiducials if len(self.given_fiducials) == 0 else self.given_fiducials
            found_fiducials = data.found_fiducials if len(self.found_fiducials) == 0 else self.found_fiducials
            annotations = data.annotations if len(self.annotations) == 0 else self.annotations
            camera_intrinsics_characterization = (
                data.camera_intrinsics_characterization
                if self.camera_intrinsics_characterization is None
                else self.camera_intrinsics_characterization
            )
            light_sources = data.light_sources if len(self.light_sources) == 0 else self.light_sources
            ret = replace(
                ret,
                given_fiducials=given_fiducials,
                found_fiducials=found_fiducials,
                annotations=annotations,
                camera_intrinsics_characterization=camera_intrinsics_characterization,
                light_sources=light_sources,
            )

        # make sure we're returning a copy
        if ret == self:
            ret = replace(self)

        return ret

    def get_primary_path_nameext(self) -> tuple[str, str]:
        """
        Finds the best source path/name.ext for the primary image of this operable.

        The source path is chosen from among the primary_image's .source_path,
        the primary_image_source_path, and the primary_image's .cache_path.

        Returns
        -------
        source_path: str
            The path component of the source of the primary_image.
            "unknown_path" if there isn't an associated source.
        source_name_ext: str
            The name.ext component of the source of the primary_image.
            "unknown_image" if there isn't an associated source.
        """
        for image_name in [
            self.primary_image.source_path,
            self.primary_image_source_path,
            self.primary_image.cache_path,
        ]:
            if image_name is not None and image_name != "":
                break

        if image_name is None or image_name == "":
            ret_path, ret_name_ext = "unknown_path", "unknown_image"
        else:
            ret_path, name, ext = ft.path_components(image_name)
            ret_name_ext = name + ext

        return ret_path, ret_name_ext

    @property
    def best_primary_nameext(self) -> str:
        """The name.ext of the source of the primary_image."""
        return self.get_primary_path_nameext()[1]

    @property
    def best_primary_pathnameext(self) -> str:
        """The path/name.ext of the source of the primary_image."""
        return os.path.join(*self.get_primary_path_nameext())

    @property
    def max_popf(self) -> npt.NDArray[np.float_]:
        """Returns the maximum population float value, if it exists. Otherwise
        returns the maximum value for this instance's primary image."""
        if self.population_statistics is not None:
            return self.population_statistics.maxf
        else:
            return np.max(self.primary_image.nparray)

    @property
    def min_popf(self) -> npt.NDArray[np.float_]:
        """Returns the minimum population float value, if it exists. Otherwise
        returns the minimum value for this instance's primary image."""
        if self.population_statistics is not None:
            return self.population_statistics.minf
        else:
            return np.min(self.primary_image.nparray)

    def get_fiducials_by_type(
        self, fiducial_type: type[af.AbstractFiducials]
    ) -> list[aa.AbstractAnnotations | af.AbstractFiducials]:
        """
        Returns all fiducials from self.given_fiducials, self.found_fiducials,
        and self.annotations that match the given type.
        """
        matching_given_fiducials = filter(lambda f: isinstance(f, fiducial_type), self.given_fiducials)
        matching_found_fiducials = filter(lambda f: isinstance(f, fiducial_type), self.found_fiducials)
        matching_annotations = filter(lambda f: isinstance(f, fiducial_type), self.annotations)
        ret = list(matching_given_fiducials) + list(matching_found_fiducials) + list(matching_annotations)
        if len(ret) == 0:
            lt.debug(
                "In SpotAnalysisOperable.get_fiducials_by_type(): "
                + f"found 0 fiducials matching type {fiducial_type.__name__} for image {self.best_primary_pathnameext}"
            )
        return ret

    def is_ancestor_of(self, other: "SpotAnalysisOperable") -> bool:
        """
        Returns true if this operable is in the other operable's
        previous_operables tree. Does not match for equality between this and
        the other operable.
        """
        if other.previous_operables[0] is None:
            return False

        for prev in other.previous_operables[0]:
            if prev == self:
                return True
            elif self.is_ancestor_of(prev):
                return True

        return False

    def __sizeof__(self) -> int:
        """
        Get the size of this operable in memory including all primary images,
        supporting images, and visualization images.
        """
        all_images_size = sum([sys.getsizeof(img) for img in self.get_all_images()])
        return all_images_size

    def __str__(self):
        name = self.__class__.__name__
        image_shape = self.primary_image.nparray.shape
        imgsize = f"{image_shape[1]}w{image_shape[0]}h"
        source_path = self.best_primary_pathnameext
        nfiducials = len(self.given_fiducials) + len(self.found_fiducials)

        return f"<{name},{imgsize=},{source_path=},{nfiducials=}>"
