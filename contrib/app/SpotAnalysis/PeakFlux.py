import json
import os
import re

import numpy as np

import opencsp.common.lib.cv.SpotAnalysis as sa
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser as saoap
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt


class PeakFlux:
    """
    A class to process images from heliostat sweeps across a target, to find the spot of maximum flux from the
    heliostat.

    The input includes::

        - A series of images with the heliostat on target, and with the target in ambient light conditions. These images
          should be clearly labeled with the name of the heliostat under test, and whether the target is under ambient
          light or heliostat reflected light.
        - The pixel intensity to flux correction mapping.

    The generated output includes::

        - Over/under exposure warnings
        - Per-heliostat heatmap visualizations
        - Per-heliostat peak flux identification
    """

    def __init__(self, indir: str, outdir: str, experiment_name: str, settings_path_name_ext: str):
        self.indir = indir
        self.outdir = outdir
        self.experiment_name = experiment_name
        self.settings_path_name_ext = settings_path_name_ext

        settings_path, settings_name, settings_ext = ft.path_components(self.settings_path_name_ext)
        settings_dict = ft.read_json("PeakFlux settings", settings_path, settings_name + settings_ext)
        self.crop_box: list[int] = settings_dict['crop_box']
        self.bcs_pixel: list[int] = settings_dict['bcs_pixel_location']
        self.heliostate_name_pattern = re.compile(settings_dict['heliostat_name_pattern'])

        group_assigner = AverageByGroupImageProcessor.group_by_name(re.compile(r"(_off)?( Raw)"))
        group_trigger = AverageByGroupImageProcessor.group_trigger_on_change()
        supporting_images_map = {
            ImageType.PRIMARY: lambda operable, operables: "off" not in operable.primary_image_source_path,
            ImageType.NULL: lambda operable, operables: "off" in operable.primary_image_source_path,
        }
        # max_pixel_value_locator = AnnotationImageProcessor.AnnotationEngine(
        #     feature_locator=lambda operable: np.argmax(operable.primary_image.ndarray),
        #     color='k'
        # )
        # bcs_locator = AnnotationImageProcessor.AnnotationEngine(
        #     feature_locator=lambda operable: self.bcs_pixel,
        #     color='k'
        # )

        self.image_processors: list[AbstractSpotAnalysisImagesProcessor] = [
            CroppingImageProcessor(*self.crop_box),
            AverageByGroupImageProcessor(group_assigner, group_trigger),
            EchoImageProcessor(),
            SupportingImagesCollectorImageProcessor(supporting_images_map),
            # NullImageSubtractionImageProcessor(),
            # FilterImageProcessor(filter="box", diameter=3),
            PopulationStatisticsImageProcessor(initial_min=0, initial_max=255),
            FalseColorImageProcessor(),
            # AnnotationImageProcessor(max_pixel_value_locator, bcs_locator)
        ]
        self.spot_analysis = sa.SpotAnalysis(
            experiment_name, self.image_processors, save_dir=outdir, save_overwrite=True
        )

        filenames = ft.files_in_directory_by_extension(self.indir, [".jpg"])[".jpg"]
        source_path_name_exts = [os.path.join(self.indir, filename) for filename in filenames]
        self.spot_analysis.set_primary_images(source_path_name_exts)

    def run(self):
        # process all images from indir
        for result in self.spot_analysis:
            # save the processed image
            save_path = self.spot_analysis.save_image(
                result, self.outdir, save_ext="png", also_save_supporting_images=False, also_save_attributes_file=True
            )
            if save_path is None:
                lt.warn(
                    f"Warning in PeakFlux.run(): failed to save image. "
                    + "Maybe SpotAnalaysis.save_overwrite is False? ({self.spot_analysis.save_overwrite=})"
                )
            else:
                lt.info(f"Saved image to {save_path}")

            # Get the attributes of the processed image, to save the results we're most interested in into a single
            # condensed csv file.
            parser = saoap.SpotAnalysisOperableAttributeParser(result, self.spot_analysis)


# class PeakFluxOffsetImageProcessor(AbstractSpotAnalysisImagesProcessor):
#     def __init__(self, outfile_path_name_ext: str, max_pixel_value_locator: AnnotationImageProcessor.AnnotationEngine, bcs_pixel_location: tuple[int, int], heliostat_name_pattern: re.Pattern):
#         super().__init__("PeakFluxOffsetImageProcessor")

#         self.outfile_path_name_ext = outfile_path_name_ext
#         self.max_pixel_value_locator = max_pixel_value_locator
#         self.bcs_pixel_location = bcs_pixel_location
#         self.heliostat_name_pattern = heliostat_name_pattern

#         with open(outfile_path_name_ext, "w") as fout:
#             fout.writelines(["Heliostat,Peak Flux Pixel,Pixels Offset"])

#     def _execute(self, operable: sa.SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
#         # get the heliostat name
#         names_to_search = [
#             operable.primary_image_source_path,
#             operable.primary_image.source_path,
#             operable.primary_image.cache_path
#         ]

#         heliostat_name = None
#         for name in names_to_search:
#             m = self.heliostat_name_pattern.search(name)
#             if m is not None:
#                 groups = list(filter(lambda s: s is not None, m.groups()))
#                 if len(groups) > 0:
#                     heliostat_name = "".join(groups)
#                     break

#         if heliostat_name is None:
#             lt.error("Error in PeakFluxOffsetImageProcessor._execute(): " +
#                      f"failed to find heliostat name in {names_to_search}")
#             return [operable]

#         # get the peak pixel location
#         peak_flux_pixel = max_pixel_value_locator.feature_locator(operable.primary_image.nparray)[0]
#         pixels_offset = peak_flux_pixel - np.array(self.bcs_pixel_location)

#         # write the results
#         peak_flux_pixel_str = f"{peak_flux_pixel[0]} {peak_flux_pixel[1]}"
#         pixels_offset_str = f"{pixels_offset[0]} {pixels_offset[1]}"
#         with open(self.outfile_path_name_ext, "a") as fout:
#             fout.writelines([f"{heliostat_name},{peak_flux_pixel_str},{pixels_offset_str}"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"), description='Processes images to find the point of peak flux.'
    )
    parser.add_argument('indir', type=str, help="Directory with images to be processed.")
    parser.add_argument('outdir', type=str, help="Directory for where to put processed images and computed results.")
    parser.add_argument('experiment_name', type=str, help="A description of the current data collection.")
    parser.add_argument('settings_file', type=str, help="Path to the settings JSON file for this PeakFlux evaluation.")
    args = parser.parse_args()

    # create the output directory
    ft.create_directories_if_necessary(args.outdir)
    ft.delete_files_in_directory(args.outdir, "*")

    # create the log file
    log_path_name_ext = os.path.join(args.outdir, "PeakFlux_" + tdt.current_date_time_string_forfile() + ".log")
    lt.logger(log_path_name_ext)

    # validate the rest of the inputs
    if not ft.directory_exists(args.indir):
        lt.error_and_raise(FileNotFoundError, f"Error in PeakFlux.py: input directory '{args.indir}' does not exist!")

    PeakFlux(args.indir, args.outdir, args.experiment_name, args.settings_file).run()
