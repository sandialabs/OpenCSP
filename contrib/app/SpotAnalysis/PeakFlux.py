import os


import opencsp.common.lib.cv.SpotAnalysis as sa
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

    def __init__(self, indir: str, outdir: str, experiment_name: str):
        self.indir = indir
        self.outdir = outdir

        self.image_processors: list[AbstractSpotAnalysisImagesProcessor] = [
            # TODO
        ]
        self.spot_analysis = sa.SpotAnalysis(
            experiment_name, self.image_processors, save_dir=outdir, save_overwrite=True
        )

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

            # TODO append these results to the csv file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"), description='Processes images to find the point of peak flux.'
    )
    parser.add_argument('indir', type=str, help="Directory with images to be processed.")
    parser.add_argument('outdir', type=str, help="Directory for where to put processed images and computed results.")
    parser.add_argument('experiment_name', type=str, help="A description of the current data collection.")
    args = parser.parse_args()

    # create the output directory
    ft.create_directories_if_necessary(args.outdir)

    # create the log file
    log_path_name_ext = os.path.join(args.outdir, "PeakFlux_" + tdt.current_date_time_string_forfile() + ".log")
    lt.logger(log_path_name_ext)

    # validate the rest of the inputs
    if not ft.directory_exists(args.indir):
        lt.error_and_raise(FileNotFoundError, f"Error in PeakFlux.py: input directory '{args.indir}' does not exist!")

    PeakFlux(args.indir, args.outdir).run()
