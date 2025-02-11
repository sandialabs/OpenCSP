from typing import Callable

import opencsp.common.lib.cv.annotations.HotspotAnnotation as hsa
import contrib.common.lib.cv.annotations.MomentsAnnotation as ma
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable as sao
import opencsp.common.lib.tool.log_tools as lt


class PixelOfInterest:
    def __init__(
        self,
        locator_method: (
            Callable[[sao.SpotAnalysisOperable], tuple[float, float]] | tuple[float, float] | str | "PixelOfInterest"
        ),
    ):
        """
        Parameters
        ----------
        locator_method: Callable[[sao.SpotAnalysisOperable], tuple[float, float]] | tuple[float, float] | str
            The pixel location in the image to be returned in
            :py:meth:`get_location`. This can be an X/Y location, a function
            that returns an X/Y location, or one of the following string values:

            - "center": use the center of the image
            - "centroid": use the centroid value calculated from the MomentsImageProcessor
            - "hotspot": use the hotspot value calculated from the HotspotImageProcessor
        """
        # copy constructor support
        if isinstance(locator_method, PixelOfInterest):
            other: PixelOfInterest = locator_method
            locator_method = other.locator_method

        # validate the input
        self._check_locator_method_value(locator_method, "")

        self.locator_method = locator_method

    def _check_locator_method_value(self, locator_method_value, method_name: str):
        if isinstance(locator_method_value, str):
            allowed_str_vals = ["center", "centroid", "hotspot"]
            if locator_method_value not in allowed_str_vals:
                lt.error_and_raise(
                    ValueError,
                    f"Error in PixelOfInterest.{method_name}(): "
                    + f"the provided locator_method_value is '{locator_method_value}', "
                    + f"but it must be one of {allowed_str_vals}!",
                )
        elif isinstance(locator_method_value, tuple):
            if len(locator_method_value) != 2:
                lt.error_and_raise(
                    ValueError,
                    f"Error in PixelOfInterest.{method_name}(): "
                    + f"the provided locator_method_value is {locator_method_value} with length {len(locator_method_value)}, "
                    + "but it must have a length of 2!",
                )

    @staticmethod
    def _get_image_center(operable: sao.SpotAnalysisOperable) -> tuple[float, float]:
        image = operable.primary_image.nparray
        height, width = image.shape[0], image.shape[1]
        assert width == operable.primary_image.to_image().width
        return width / 2, height / 2

    @staticmethod
    def _get_centroid_location(operable: sao.SpotAnalysisOperable) -> tuple[float, float]:
        moments: list[ma.MomentsAnnotation] = operable.get_fiducials_by_type(ma.MomentsAnnotation)
        if len(moments) == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelOfInterest.._get_centroid_location(): "
                + "no MomentsAnnotation found for the current operable! "
                + "Maybe you need to run the MomentsImageProcessor first?",
            )
        return moments[-1].centroid.astuple()

    @staticmethod
    def _get_hotspot_location(operable: sao.SpotAnalysisOperable) -> tuple[float, float]:
        hotspots: list[hsa.HotspotAnnotation] = operable.get_fiducials_by_type(hsa.HotspotAnnotation)
        if len(hotspots) == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelOfInterest.._get_hotspot_location(): "
                + "no HotspotAnnotation found for the current operable! "
                + "Maybe you need to run the HotspotImageProcessor first?",
            )
        return hotspots[-1].origin.astuple()

    def get_location(self, operable: sao.SpotAnalysisOperable) -> tuple[float, float]:
        locator_method = self.locator_method

        if isinstance(locator_method, str):
            if locator_method == "center":
                locator_method = self._get_image_center
            if locator_method == "centroid":
                locator_method = self._get_centroid_location
            elif locator_method == "hotspot":
                locator_method = self._get_hotspot_location
            else:
                lt.error_and_raise(
                    f"Error in PixelOfInterest..get_location(): unknown locator method '{locator_method}'!"
                )

        elif isinstance(locator_method, tuple):
            return locator_method

        # at this point, the locator method should be a callable
        ret = locator_method(operable)
        return ret
