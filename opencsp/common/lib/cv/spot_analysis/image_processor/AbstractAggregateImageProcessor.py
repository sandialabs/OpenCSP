from abc import ABC, abstractmethod
import re
from typing import Callable

import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class AbstractAggregateImageProcessor(AbstractSpotAnalysisImageProcessor, ABC):
    """
    Detects and collects images that are part of the same group, so that they can be acted upon all at the same time.

    Each operator is assigned to an image group. Groups are determined by the
    images_group_assigner function. Any function with the correct signature can
    be used, or one of the builtin methods can be assigned. The builtin methods
    for this include AbstractAggregateImageProcessor.*, where "*" is one of:

        - :py:meth:`group_by_brightness`: groups are determined by the brightest pixel in the image
        - :py:meth:`group_by_name`: all images with the same name match are included as part of the same group

    When the assigned group number for the current operator is different than
    for the previous operator, the group execution is triggered.
    :py:meth:`_execute_aggregate` will be called for the entire group, and
    afterwards the group's list will be cleared. The trigger behavior can be
    changed by providing a value for the group_execution_trigger parameter.

    Inheriting classes need to implement the following method in liue of _execute():

        def _execute_aggregate(self, group: int, operables: list, is_last: bool) -> list:
    """

    def __init__(
        self,
        images_group_assigner: Callable[[SpotAnalysisOperable], int] = None,
        group_execution_trigger: Callable[[list[tuple[SpotAnalysisOperable, int]]], int | None] = None,
        name: str = None,
    ):
        """
        Parameters
        ----------
        images_group_assigner : Callable[[SpotAnalysisOperable], int], optional
            The function that determines which group a given operable should be
            assigned to. If None, then all images will be assigned to the same
            group. Default is None.
        group_execution_trigger : Callable[[], bool], optional
            The function that determines when a group of operators is executed
            on, by default group_trigger_on_change.
        """
        super().__init__(name)

        # normalize arguments
        if images_group_assigner is None:
            images_group_assigner = lambda o: 0
        if group_execution_trigger is None:
            group_execution_trigger = self.group_trigger_on_change()

        # register arguments
        self.images_group_assigner = images_group_assigner
        self.group_execution_trigger = group_execution_trigger

        # images groups dictionary
        self.image_groups: list[tuple[SpotAnalysisOperable, int]] = []
        """The lists of images and group assignments. The images are in the same order they were received."""

    @staticmethod
    def group_by_brightness(intensity_to_group: dict[int, int]) -> Callable[[SpotAnalysisOperable], int]:
        """
        Returns a group for the given operable based on the intensity mapping
        and the brightest pixel in the operable's primary image.

        Intended use is as the images_group_assigner parameter to this class.

        Parameters
        ----------
        intensity_to_group: dict[int, int]
            The mapping of minimum intensity for each group.

            Example intensity_to_group value::

                # All images with at least one pixel >= 200 will be assigned to group 2.
                # Images with at least one pixel >= 125 will be assigned to group 1.
                # All other images will be assigned to group 0.
                intensity_to_group = {
                    0: 0,
                    125: 1,
                    200: 2
                }
        """

        def group_by_brightness_inner(operable: SpotAnalysisOperable, intensity_to_group: dict[int, int]):
            image = operable.primary_image.nparray

            # get the brightest pixel's value
            max_pixel_value = np.max(image)

            # choose a group
            intensity_thresholds = sorted(list(intensity_to_group.keys()))
            assigned_group = intensity_to_group[intensity_thresholds[0]]
            for intensity_threshold in intensity_thresholds[1:]:
                if max_pixel_value >= intensity_threshold:
                    assigned_group = intensity_to_group[intensity_threshold]

            # lt.info(f"{operable.best_primary_nameext}: {max_pixel_value}/{assigned_group}")

            return assigned_group

        return lambda operable: group_by_brightness_inner(operable, intensity_to_group)

    @staticmethod
    def group_by_name(name_pattern: re.Pattern) -> Callable[[SpotAnalysisOperable], int]:
        """
        Returns a group for the given operable based on the groups matches "()" for the given pattern.

        Intended use is as the images_group_assigner parameter to this class.

        Example assignments::

            pattern = re.compile(r"(foo|bar)")

            foo_operable = SpotAnalaysisOperable(CacheableImage(source_path="hello_foo.png"))
            food_operable = SpotAnalaysisOperable(CacheableImage(source_path="hello_food.png"))
            bar_operable = SpotAnalaysisOperable(CacheableImage(source_path="hello_bar.png"))

            groups: list[str] = []
            images_group_assigner = group_by_name(pattern, groups)
            images_group_assigner(foo_operable)  # returns 0, groups=["foo"]
            images_group_assigner(food_operable) # returns 0, groups=["foo"]
            images_group_assigner(bar_operable)  # returns 1, groups=["foo", "bar"]
        """

        def group_by_name_inner(operable: SpotAnalysisOperable, name_pattern: re.Pattern, groups: list[str]) -> int:
            names_to_check = [
                operable.primary_image_source_path,
                operable.primary_image.source_path,
                operable.primary_image.cache_path_name_ext,
            ]
            names_to_check = list(filter(lambda name: name is not None, names_to_check))
            if len(names_to_check) == 0:
                lt.warning("Warning in AbstractAggregateImageProcessor.group_by_name(): operator has no image name")
                return 0

            # match the name_pattern against each of the names_to_check
            for name in names_to_check:
                m = name_pattern.search(name)
                if m is None:
                    continue
                match_groups = list(filter(lambda s: s is not None, m.groups()))
                if len(match_groups) == 0:
                    lt.debug(
                        "In AbstractAggregateImageProcessor.group_by_name(): "
                        + f"no groups found for pattern {name_pattern} when trying to match against name {name}"
                    )
                    continue

                # get the name match
                group_str = "".join(match_groups)

                # return the index of the existing group, or add a new group
                if group_str in groups:
                    return groups.index(group_str)
                else:
                    groups.append(group_str)
                    return len(groups) - 1

            # failed to find a match, assign to default group 0
            lt.warning(
                "Warning in AbstractAggregateImageProcessor.group_by_name(): "
                + f"failed to find a match to {name_pattern} in {names_to_check}"
            )
            return 0

        groups: list[str] = []
        return lambda operable: group_by_name_inner(operable, name_pattern, groups)

    @staticmethod
    def group_trigger_on_change() -> Callable[[list[tuple[SpotAnalysisOperable, int]]], int | None]:
        """
        Triggers anytime that the group assigned to the current operable is different than for the previous operable.

        Intended use is as the group_execution_trigger parameter to this class.
        """

        def group_trigger_on_change_inner(image_groups: list[tuple[SpotAnalysisOperable, int]]) -> int | None:
            if len(image_groups) <= 1:
                return None

            current_group = image_groups[-1][1]
            previous_group = image_groups[-2][1]
            if current_group != previous_group:
                return previous_group
            else:
                return None

        return group_trigger_on_change_inner

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # assign this operable to a group
        operable_group = self.images_group_assigner(operable)
        self.image_groups.append((operable, operable_group))

        # check if we should trigger execution
        triggered_groups: list[int] = []
        if is_last:
            triggered_groups_set = {operable_and_group[1]: None for operable_and_group in self.image_groups}
            triggered_groups = list(triggered_groups_set.keys())
        else:
            triggered_group = self.group_execution_trigger(self.image_groups)
            if triggered_group is not None:
                triggered_groups = [triggered_group]

        # execute the triggered groups
        ret: list[SpotAnalysisOperable] = []
        for group in triggered_groups:

            # get the operables to execute on
            operables: list[SpotAnalysisOperable] = []
            i = 0
            while i < len(self.image_groups):
                operable_and_group = self.image_groups[i]
                if operable_and_group[1] == group:
                    operables.append(operable_and_group[0])
                    del self.image_groups[i]
                else:
                    i += 1

            # collect the results of the execution
            ret += self._execute_aggregate(group, operables, is_last)

        return ret

    @abstractmethod
    def _execute_aggregate(
        self, group: int, operables: list[SpotAnalysisOperable], is_last: bool
    ) -> list[SpotAnalysisOperable]:
        pass
