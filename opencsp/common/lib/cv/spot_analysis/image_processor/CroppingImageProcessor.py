import copy
import dataclasses

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
    Crops all input images to the given shape. If the input image is too small, then an error will be thrown.
    """

    def __init__(self, x1: int, x2: int, y1: int, y2: int):
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
        super().__init__()

        # validate the inputs
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

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
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
