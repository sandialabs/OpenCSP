import copy
import dataclasses
from typing import Callable, Type

import cv2
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.LoopXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.image_tools as it


class DiscardAnnotationsImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Discards fiducials and/or annotations from operables that pass through
    based on given criteria.

    TODO: extend this class to discard single instances of fiducials for instances where multiple fiducials are included in a single "AbstractFiducials" instance.
    """

    def __init__(
        self,
        outside_boundaries: bool = False,
        matching_types: list[Type[AbstractFiducials]] = None,
        custom_criteria: Callable[[SpotAnalysisOperable], list[AbstractFiducials]] = None,
    ):
        """
        Parameters
        ----------
        outside_boundaries : bool, optional
            If True, then any fiducials and/or annotations that are centered
            outside the boundaries of the image are discarded.
        matching_types : list[Type[AbstractFiducials]], optional
            All fiducials and/or annotations that are instances of one of the
            given types are discarded.
        custom_criteria : Callable[[SpotAnalysisOperable], np.ndarray], optional
            The function to be provided that takes in an operable and returns a
            list of fiducials and/or annotations to be discarded.
        """
        super().__init__()

        self.outside_boundaries = outside_boundaries
        self.matching_types = matching_types or []
        self.custom_criteria = custom_criteria

    def get_fiducials_outside_boundaries(self, operable: SpotAnalysisOperable) -> list[AbstractFiducials]:
        """
        Get the annotations that fall outside the boundaries of the image. This can happen, for example, after a CroppingImageProcessor has trimmed down an image area and the new area excludes an existing fiducial. If self.outside_boundaries is False, then returns an empty list.
        """
        if not self.outside_boundaries:
            return []
        fiducials_to_discard: list[AbstractFiducials] = []

        (height, width), _ = it.dims_and_nchannels(operable.primary_image.nparray)
        image_bounds_tltrbrbl = p2.Pxy([[0, width, width, 0], [0, 0, height, height]])
        transformed_bounds_tltrbrbl = operable.transform_coordinates(image_bounds_tltrbrbl)[1]
        transformed_loop = l2.LoopXY.from_vertices(transformed_bounds_tltrbrbl)

        for fiducials_list in [operable.given_fiducials, operable.found_fiducials, operable.annotations]:
            for fiducials in fiducials_list:
                fiducials: AbstractFiducials = fiducials
                origins = operable.transform_coordinates(fiducials.origin)[1]
                origins_inside = transformed_loop.is_inside_or_on_border(origins)
                if not np.all(origins_inside):
                    fiducials_to_discard.append(fiducials)

        return fiducials_to_discard

    def get_fiducials_matching_types(self, operable: SpotAnalysisOperable):
        """Get the fiducials to discard based on the types of the fiducials."""
        fiducials_to_discard: list[AbstractFiducials] = []

        for fiducials_list in [operable.given_fiducials, operable.found_fiducials, operable.annotations]:
            for fiducials in fiducials_list:
                fiducials: AbstractFiducials = fiducials
                for fiducials_type in self.matching_types:
                    if isinstance(fiducials, fiducials_type):
                        fiducials_to_discard.append(fiducials)

        return fiducials_to_discard

    def get_fiducials_by_custom_criteria(self, operable: SpotAnalysisOperable) -> list[AbstractFiducials]:
        """Get the annotations to be discarded based on a custom function."""
        if self.custom_criteria is None:
            return []
        return self.custom_criteria(operable)

    def discard_fiducials(
        self, operable: SpotAnalysisOperable, fiducials_to_discard: list[AbstractFiducials]
    ) -> SpotAnalysisOperable:
        """
        Discards the fiducials in the given list and returns a new operable with
        newly updated annotations.
        """
        # Make copies so that we don't change the lists on the input operable
        given_fiducials = copy.copy(operable.given_fiducials)
        found_fiducials = copy.copy(operable.found_fiducials)
        annotations = copy.copy(operable.annotations)

        # Remove all fiducials in the list to be discarded
        for fiducials_list in [given_fiducials, found_fiducials, annotations]:
            fiducials_list: list[AbstractFiducials] = fiducials_list
            for fiducials in copy.copy(fiducials_list):
                if fiducials in fiducials_to_discard:
                    fiducials_list.remove(fiducials)

        # Update with a new operable instance
        return dataclasses.replace(
            operable, given_fiducials=given_fiducials, found_fiducials=found_fiducials, annotations=annotations
        )

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        fiducials_to_discard: list[AbstractFiducials] = []

        # Get the fiducials to discard
        fiducials_to_discard += self.get_fiducials_outside_boundaries(operable)
        fiducials_to_discard += self.get_fiducials_matching_types(operable)
        fiducials_to_discard += self.get_fiducials_by_custom_criteria(operable)

        # Build a new operable instance, keeping all fiducials not in the discard list
        operable = self.discard_fiducials(operable, fiducials_to_discard)

        return [operable]
