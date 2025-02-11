import os
import re

import numpy as np

from contrib.app.SpotAnalysis.PeakFluxSettings import PeakFluxSettings
import opencsp.common.lib.cv.SpotAnalysis as sa
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
from opencsp.common.lib.cv.fiducials.BcsFiducial import BcsFiducial
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser as saoap
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


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

    def __init__(self, outdir: str, experiment_name: str, settings: PeakFluxSettings):
        self.outdir = outdir
        self.experiment_name = experiment_name
        self.settings = settings

        group_assigner = AverageByGroupImageProcessor.group_by_name(re.compile(r"([0-9]{1,2}[EW][0-9]{1,2})"))
        group_trigger = AverageByGroupImageProcessor.group_trigger_on_change()
        supporting_images_map = {
            ImageType.PRIMARY: lambda operable, operables: "off" not in operable.get_primary_path_nameext(),
            ImageType.NULL: lambda operable, operables: "off" in operable.get_primary_path_nameext(),
        }

        self.image_processors: list[AbstractSpotAnalysisImageProcessor] = {
            'Crop': CroppingImageProcessor(*self.crop_box),
            'AvgG': AverageByGroupImageProcessor(group_assigner, group_trigger),
            'Echo': EchoImageProcessor(),
            'Coll': SupportingImagesCollectorImageProcessor(supporting_images_map),
            'Null': NullImageSubtractionImageProcessor(),
            'Noiz': ConvolutionImageProcessor(kernel="box", diameter=3),
            'Targ': TargetBoardLocatorImageProcessor(
                None,
                None,
                settings.target_board_size_wh[0],
                settings.target_board_size_wh[1],
                settings.target_canny_gradients[0],
                settings.target_canny_gradients[1],
                edge_coarse_width=30,
                canny_test_gradients=settings.target_canny_test_gradients,
            ),
            'Fill': InpaintImageProcessor(settings.infill_mask_file, settings.tmpdir),
            'Save': SaveToFileImageProcessor(ft.join(settings.outdir, "filled_targets")),
            'Ve3d': View3dImageProcessor(crop_to_threshold=20, max_resolution=(100, 100)),
            'HotL': HotspotImageProcessor(
                desired_shape=49, draw_debug_view=False, record_visualization=False, record_debug_view=6
            ),
            'Vcx2': ViewCrossSectionImageProcessor(
                self.get_peak_origin, 'Hotspot', single_plot=False, crop_to_threshold=20
            ),
            'Stat': PopulationStatisticsImageProcessor(min_pop_size=1, initial_min=0, initial_max=255),
            'Fclr': ViewFalseColorImageProcessor(),
            'Anno': ViewAnnotationsImageProcessor(),
        }
        self.pptx_processor = PowerpointImageProcessor(
            save_dir=outdir,
            save_name="processing_pipeline",
            overwrite=True,
            processors_per_slide=[
                [self.image_processors['Noiz'], self.image_processors['Targ']],
                [self.image_processors['Fill']],
                [self.image_processors['Ve3d']],
                [self.image_processors['HotL']],
                [self.image_processors['Vcx2']],
                [self.image_processors['Fclr'], self.image_processors['Anno']],
            ],
        )
        image_processor_list = list(self.image_processors.values()) + [self.pptx_processor]
        self.image_processors_list: list[AbstractSpotAnalysisImageProcessor] = image_processor_list

        self.spot_analysis = sa.SpotAnalysis(
            experiment_name, self.image_processors_list, save_dir=outdir, save_overwrite=True
        )

    def assign_inputs(self, dirs_or_files: str | list[str]):
        """
        Assigns the given image files or the image files in the given
        directories as input to this instance's SpotAnalysis. Must be done
        before iterating through this instance's SpotAnalysis.

        Parameters
        ----------
        dirs_or_files : str | list[str]
            The image files to be analyzed, or directories that contain image
            files. Can be mixed files and directores.
        """
        file_path_name_exts: list[str] = []

        for dir_or_file in dirs_or_files:
            if os.path.isfile(dir_or_file):
                file_path_name_exts.append(dir_or_file)
            elif os.path.isdir(dir_or_file):
                dir_files = it.image_files_in_directory(dir_or_file)
                file_path_name_exts += [os.path.join(dir_or_file, filename) for filename in dir_files]
            else:
                lt.error_and_raise(
                    FileNotFoundError,
                    "Error in PeakFlux.assign_inputs(): "
                    + f"given dir_or_file {dir_or_file} is neither a directory nor a file.",
                )

        self.spot_analysis.set_primary_images(file_path_name_exts)

    def assign_target_board_reference_image(self, reference_image_dir_or_file: str):
        """
        Assigns the given image file or directory containing image files as the
        reference for the TargetBoardLocatorImageProcessor.

        Parameters
        ----------
        reference_image_dir_or_file : str
            The image file, or the directory containing image files.
        """
        targ_processor: TargetBoardLocatorImageProcessor = self.image_processors['Targ']
        targ_processor.set_reference_images(reference_image_dir_or_file)

    def assign_inputs_from_settings(self):
        """
        Use the values from the PeakFluxSettings instance to find input images
        and reference images, and assign them to this PeakFlux instance.
        """
        # Build the filters for the directories
        is_reference_dir = lambda dn: any([rdn in dn for rdn in self.settings.target_reference_images_dirnames])
        is_missed_helio_dir = lambda dn: self.get_helio_name(dn) in self.settings.missed_heliostats
        is_issue_helio_dir = lambda dn: self.get_helio_name(dn) in self.settings.issue_heliostats
        has_helio_name = lambda dn: self.get_helio_name(dn) is not None

        # Get the target reference directories. These should be directories that
        # contain images without any sun on them.
        dirnames = ft.files_in_directory(self.settings.indir, files_only=False)
        reference_dirnames = list(filter(lambda dn: is_reference_dir(dn), dirnames))
        target_reference_path = ft.norm_path(os.path.join(self.settings.indir, reference_dirnames[0], "Raw Images"))

        # Get the heliostat image directories. Since the directories names start
        # with a datetime string, sorting will put them all into collection order.
        heliostat_dirnames = list(filter(lambda dn: not is_reference_dir(dn), dirnames))
        heliostat_dirnames = list(filter(lambda dn: not is_missed_helio_dir(dn), heliostat_dirnames))
        heliostat_dirnames = list(filter(lambda dn: not is_issue_helio_dir(dn), heliostat_dirnames))
        heliostat_dirnames = list(filter(lambda dn: has_helio_name(dn), heliostat_dirnames))
        # heliostat_dirnames = list(filter(lambda dn: "07E01" in dn, heliostat_dirnames))
        if self.settings.limit_num_heliostats >= 0:
            heliostat_dirnames = heliostat_dirnames[
                : np.min([self.settings.limit_num_heliostats, len(heliostat_dirnames)])
            ]
        heliostat_dirs = [ft.norm_path(os.path.join(self.settings.indir, dn)) for dn in heliostat_dirnames]

        # Compile a list of images from the image directories.
        source_images_path_name_exts: list[str] = []
        for dirname in heliostat_dirs:
            raw_images_path = ft.norm_path(os.path.join(self.settings.indir, dirname, "Raw Images"))
            if not ft.directory_exists(raw_images_path):
                lt.error_and_raise(
                    FileNotFoundError,
                    "Error in peak_flux: " + f"raw images directory \"{raw_images_path}\" does not exist",
                )
            for file_name_ext in ft.files_in_directory(raw_images_path, files_only=True):
                source_images_path_name_exts.append(ft.norm_path(os.path.join(raw_images_path, file_name_ext)))

        self.assign_inputs(source_images_path_name_exts)
        self.assign_target_board_reference_image(target_reference_path)

    def get_helio_name(self, dirname: str) -> str | None:
        """Uses the heliostat_name_pattern from the peak flux settings to parse
        the heliostat name out of the given directory name."""
        helio_names = self.settings.heliostat_name_pattern.findall(dirname)
        if len(helio_names) == 0:
            return None
        return helio_names[0]

    def get_bcs_origin(self, operable: SpotAnalysisOperable):
        fiducials = operable.get_fiducials_by_type(BcsFiducial)
        if len(fiducials) == 0:
            return None
        fiducial = fiducials[0]
        origin_fx, origin_fy = fiducial.origin.astuple()
        origin_ix, origin_iy = int(np.round(origin_fx)), int(np.round(origin_fy))
        return origin_ix, origin_iy

    def get_peak_origin(self, operable: SpotAnalysisOperable):
        fiducials = operable.get_fiducials_by_type(HotspotAnnotation)
        if len(fiducials) == 0:
            return None
        fiducial = fiducials[0]
        origin_fx, origin_fy = fiducial.origin.astuple()
        origin_ix, origin_iy = int(np.round(origin_fx)), int(np.round(origin_fy))
        return origin_ix, origin_iy
