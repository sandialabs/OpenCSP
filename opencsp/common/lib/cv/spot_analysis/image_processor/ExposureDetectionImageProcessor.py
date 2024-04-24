import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class ExposureDetectionImageProcessor(AbstractSpotAnalysisImagesProcessor):
    def __init__(
        self,
        under_exposure_limit=0.99,
        under_exposure_threshold: int | float = 0.95,
        over_exposure_limit=0.97,
        max_pixel_value=255,
        log_level=lt.log.WARN,
    ):
        """
        Detects over and under exposure in images and adds the relavent tag to the image.

        Over or under exposure is determined by the proportion of pixels that are at near the max_pixel_value threshold.
        If more pixels than the over exposure limit is at the maximum level, then the image is considered over exposed. If
        more pixels than the under exposure limit is below the under_exposure_threshold, then the image is considered under
        exposed.

        For color images, the proportion of pixels across all color channels is used.

        Parameters
        ----------
        under_exposure_limit : float, optional
            Fraction of pixels allowed to be below the under_exposure_threshold, by default 0.99
        under_exposure_threshold : int | float, optional
            If a float, then this is the fraction of the max_pixel_value that is used to determine under exposure. If an
            int, then this is the pixel value. For example, 0.95 and 243 will produce the same results when
            max_pixel_value is 255. By default 0.95
        over_exposure_limit : float, optional
            Fraction of pixels that should be below the maximum value, by default 0.97
        max_pixel_value : int, optional
            The maximum possible value of the pixels, by default 255 to match uint8 images
        log_level : int, optional
            The level to print out warnings at, by default log.WARN
        """
        super().__init__(self.__class__.__name__)

        # validate the inputs
        val_err = lambda s: lt.error_and_raise(ValueError, "Error in ExposureDetectionImageProcessor: " + s)
        if under_exposure_limit < 0 or under_exposure_limit > 1 or over_exposure_limit < 0 or over_exposure_limit > 1:
            val_err(f"exposure limits must be between 0 and 1, but {under_exposure_limit=}, {over_exposure_limit=}")
        if max_pixel_value < 0:
            val_err(f"max_pixel_value should be the maximum possible value from the camera, but is {max_pixel_value=}")

        # register values
        self.under_exposure_limit = under_exposure_limit
        self.under_exposure_threshold = under_exposure_threshold
        self.over_exposure_limit = over_exposure_limit
        self.max_pixel_value = max_pixel_value
        self.log_level = log_level

        # internal variables
        self.log = lt.get_log_method_for_level(self.log_level)

        # variables for unit tests
        self._raise_on_error = False

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray
        notes = (
            'ExposureDetectionImageProcessor',
            [
                f'settings: {self.under_exposure_limit=}, {self.under_exposure_threshold=}, {self.over_exposure_limit=}, {self.max_pixel_value=}'
            ],
        )

        # check for under exposure
        if isinstance(self.under_exposure_threshold, float):
            under_exposure_threshold = int(np.ceil(self.max_pixel_value * self.under_exposure_threshold))
        else:  # isinstance(self.under_exposure_threshold, int)
            under_exposure_threshold: int = self.under_exposure_threshold
        num_dark_pixels = np.sum(image < under_exposure_threshold)
        proportion_dark_pixels = num_dark_pixels / image.size
        if proportion_dark_pixels > self.under_exposure_limit:
            self.log(
                "Warning in ExposureDetectionImageProcessor._execute(): image is under exposed. "
                + f"At most {self.under_exposure_limit*100:0.2f}% of pixels should have values less than {under_exposure_threshold}, "
                + f"but instead {proportion_dark_pixels*100:0.2f}% have a value less than that threshold."
            )
            notes[1].append(
                f"Image is under exposed. {proportion_dark_pixels*100:0.2f}% of pixels are below {under_exposure_threshold}"
            )
            if self._raise_on_error:
                raise RuntimeError("for unit testing: under exposed image")

        # check for over exposure
        over_exposure_threshold = self.max_pixel_value
        num_light_pixels = np.sum(image >= over_exposure_threshold)
        proportion_light_pixels = num_light_pixels / image.size
        if proportion_light_pixels > self.over_exposure_limit:
            self.log(
                "Warning in ExposureDetectionImageProcessor._execute(): image is over exposed. "
                + f"At most {self.over_exposure_limit*100:0.2f}% of pixels should have the value {over_exposure_threshold}, "
                + f"but instead {proportion_light_pixels*100:0.2f}% have a value greater than or equal to that threshold."
            )
            notes[1].append(
                f"Image is over exposed. {proportion_light_pixels*100:0.2f}% of pixels are above {over_exposure_threshold}"
            )
            if self._raise_on_error:
                raise RuntimeError("for unit testing: over exposed image")

        operable.image_processor_notes.append(notes)
        return [operable]


if __name__ == "__main__":
    expdir = (
        orp.opencsp_scratch_dir()
        + "/solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01"
    )
    indir = expdir + "/raw_images"
    outdir = expdir + "/processed_images"
    lt.logger(outdir + "/log.txt", level=lt.log.INFO)

    x1, y1, x2, y2 = 120, 29, 1526, 1158
    x1, y1 = x1 + 20, y1 + 20
    x2, y2 = x2 - 20, y2 - 20

    ft.create_directories_if_necessary(outdir)
    ft.delete_files_in_directory(outdir, "*")
    images_filenames = ft.files_in_directory_by_extension(indir, ["jpg"])["jpg"]
    images_path_name_ext = [indir + '/' + filename for filename in images_filenames]

    import opencsp.common.lib.cv.SpotAnalysis as sa
    from opencsp.common.lib.cv.spot_analysis.image_processor import *

    image_processors = [
        CroppingImageProcessor(x1, x2, y1, y2),
        ExposureDetectionImageProcessor(under_exposure_threshold=120),
    ]

    spot_analysis = sa.SpotAnalysis('ExposureDetectionImageProcessor test', image_processors, outdir)
    spot_analysis.set_primary_images(images_path_name_ext)

    for operable in spot_analysis:
        spot_analysis.save_image(operable)
