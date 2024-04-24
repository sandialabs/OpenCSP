import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.PointFiducials import PointFiducials
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft


class AnnotationImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """
    Draws annotations on top of the input image. The annotations drawn are those in operable.given_fiducials and
    operable.found_fiducials.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        old_image = operable.primary_image.nparray
        new_image = np.array(old_image)

        for fiducials in operable.given_fiducials:
            new_image = fiducials.render_to_image(new_image)
        for fiducials in operable.found_fiducials:
            new_image = fiducials.render_to_image(new_image)

        cacheable_image = CacheableImage(new_image, source_path=operable.primary_image.source_path)
        ret = dataclasses.replace(operable, primary_image=cacheable_image)
        return [ret]


if __name__ == "__main__":
    import os

    indir = ft.norm_path(
        os.path.join(
            orp.opencsp_scratch_dir(),
            "solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01/processed_images",
        )
    )
    image_file = ft.norm_path(os.path.join(indir, "20230512_113032.81 5W01_000_880_2890 Raw_Testing_Peak_Flux.png"))

    style = rcps.RenderControlPointSeq(markersize=10)
    fiducials = PointFiducials(style, points=p2.Pxy(np.array([[0, 643, 1000], [0, 581, 1000]])))
    operable = SpotAnalysisOperable(CacheableImage(source_path=image_file), given_fiducials=[fiducials])

    processor = AnnotationImageProcessor()
    result = processor.process_image(operable)[0]
    img = result.primary_image.nparray

    plt.figure()
    plt.imshow(img)
    plt.show(block=True)
