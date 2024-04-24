import copy
import dataclasses

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.BcsFiducial import BcsFiducial
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.AnnotationImageProcessor import AnnotationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor import ConvolutionImageProcessor
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render_control.RenderControlBcs as rcb
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class BcsLocatorImageProcessor(AbstractSpotAnalysisImagesProcessor):
    def __init__(self, min_radius_px=30, max_radius_px=150):
        """
        Locates the BCS by identifying a circle in the image.

        It is recommended this this processor be used after ConvolutionImageProcessor(kernel='gaussian').

        Parameters
        ----------
        min_radius_px : int, optional
            Minimum radius of the BSC circle, in pixels. By default 50
        max_radius_px : int, optional
            Maximum radius of the BSC circle, in pixels. By default 300
        """
        super().__init__(self.__class__.__name__)

        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray.squeeze()
        if image.ndim > 2:
            lt.error_and_raise(
                RuntimeError,
                "Error in BcsLocatorImageProcessor._execute(): image must be grayscale (2 dimensions), but "
                + f"the shape of the image is {image.shape} for '{operable.primary_image_source_path}'",
            )

        # find all possible matches
        method = cv.HOUGH_GRADIENT
        accumulator_pixel_size: float = 1
        circles: np.ndarray | None = cv.HoughCircles(
            image,
            method,
            accumulator_pixel_size,  # resolution for accumulators
            minDist=self.min_radius_px,  # distance between circles
            param1=70,  # upper threshold to Canny edge detector
            param2=20,  # minimum accumulations count for matching circles
            minRadius=self.min_radius_px,
            maxRadius=self.max_radius_px,
        )

        # opencv returns circles in order from best to worst matches, choose the first circle (best match)
        circle: BcsFiducial = None
        if circles is not None:
            circle_arr = circles[0][0]
            center = p2.Pxy([circle_arr[0], circle_arr[1]])
            radius = circle_arr[2]
            circle = BcsFiducial(center, radius, style=rcb.thin(color='m'))

        # assign to the operable
        new_found_fiducials = copy.copy(operable.found_fiducials)
        if circle != None:
            new_found_fiducials.append(circle)
        ret = dataclasses.replace(operable, found_fiducials=new_found_fiducials)
        return [ret]


if __name__ == "__main__":
    import os

    indir = ft.norm_path(
        os.path.join(
            orp.opencsp_scratch_dir(),
            "solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01/raw_images",
        )
    )
    image_file = ft.norm_path(os.path.join(indir, "20230512_114854.74 5E09_000_880_2890 Raw.JPG"))

    style = rcps.RenderControlPointSeq(markersize=10)
    operable = SpotAnalysisOperable(CacheableImage(source_path=image_file))

    processor0 = ConvolutionImageProcessor(kernel='gaussian', diameter=3)
    processor1 = BcsLocatorImageProcessor()
    processor2 = AnnotationImageProcessor()

    result0 = processor0.process_image(operable)[0]
    result1 = processor1.process_image(result0)[0]
    result2 = processor2.process_image(result1)[0]
    img = result2.primary_image.nparray

    plt.figure()
    plt.imshow(img)
    plt.show(block=True)
