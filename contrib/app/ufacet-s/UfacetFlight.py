import glob
import os
import re

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.system_tools as st


class UfacetFlight:
    # name, extracted frames name format, and extracted frames directory for each ufacet flight
    _known = [
        (
            "2021-05-13_FastScan2_C0039",
            "C0039_s3m15_d825.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1210_F03_C0039_s3m15_d825/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0040",
            "C0040_s1m25_d165.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1237_F04_C0040_s1m25_d165/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0042",
            "C0042_s2m05_d920.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1325_F06_C0042_s2m05_d920/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0043",
            "C0043_s1m31_d875.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1410_F07_C0043_s1m31_d875/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0044",
            "C0044_s2m25_d870.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1514_F08_C0044_s2m25_d870/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0045",
            "C0045_s1m50_d880.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1644_F10_C0045_s1m50_d880/sony/M4ROOT/CLIP",
        ),
        (
            "2021-05-13_FastScan2_C0046",
            "C0046_s2m42_d870.MP4.%06d.JPG",
            "Experiments/2021-05-13_FastScan2/3_Frames/20210513/1720_F11_C0046_s2m42_d870/sony/M4ROOT/CLIP",
        ),
    ]
    _flights: dict[str, "UfacetFlight"] = {}

    def __init__(self, name: str, extracted_frames_name_format: str, extracted_frames_dir: str):
        """This class holds onto relevant information for a variety of UFACET flights."""
        self._name = name

        self._frames_name_format_extracted = extracted_frames_name_format
        self._frames_dir_extracted = os.path.join(orp.opencsp_scratch_dir("ufacetnio"), extracted_frames_dir)
        self._num_frames_extracted: int = None

        self._frames_name_format_deduplicated = None
        self._frames_dir_deduplicated = None
        self._num_frames_deduplicated: int = None
        self._warned_no_deduplicated_yes_extracted = False

        self.__class__._flights[name] = self

    @classmethod
    def get_by_name(cls, name_subset: str):
        """Searches through the list of known flights for a flight matching the given name_subset.

        Args:
            name_subset (str): A part of the full name of the flight. Example "C0045" will return the flight with the name "2021-05-13_FastScan2_C0045".

        Returns:
            flight (UfacetFlight|None): The flight, if one is found with a matching name. If there are no matching flights then this will be None.
        """
        for name, *other in cls._known:
            if name_subset in name:
                return cls(name, *other)
        return None

    @classmethod
    def get_all(cls):
        for name, _frames_name_format_extracted, extracted_frames_dir in cls._known:
            if name not in cls._flights:
                cls(name, _frames_name_format_extracted, extracted_frames_dir)
        return list(cls._flights.values())

    @property
    def name(self):
        """The long name for this flight (ex "2021-05-13_FastScan2_C0045")"""
        return self._name

    @property
    def short_name(self):
        """The short name for this flight (example "C0045")"""
        cpattern = re.compile(".*(C[0-9]+).*")
        matches = cpattern.match(self._name)
        if matches == None:
            return self._name
        groups = matches.groups()
        return groups[0]

    def _get_frame_name_prefix_suffix(self, file_name_ext: str):
        prefix = file_name_ext.split("%")[0]
        suffix = file_name_ext.split("%")[-1].lstrip("0123456789d")
        return prefix, suffix

    def video_dir_name_ext(self, allow_ppt_compressed_videos=False):
        """Find the (best matching) video file for this flight.

        This method searches for a video named something similar to the flight name,
        in a directory that mirrors the frames directory but in the "2_Data" path
        instead of the "3_Frames" path. The returned video might be a trimmed down
        version of the original video, in the case that the original isn't found.

        Example::

            flight = uf.UfacetFlight.get_by_name("C0045")
            lt.info(flight.video_dir_name_ext())
            # output "<experiment_dir>/2021-05-13_FastScan2/2_Data/20210513/1644_F10_C0045_s1m50_d880/sony/M4ROOT/CLIP/C0045_s1m50_d880.MP4"

        Args:
            allow_ppt_compressed_videos (bool, optional): If True, then videos that
                have been compressed for use in PowerPoint (has the "_ppt" tag in
                the name) will be returned if they're the only ones available.
                Defaults to False.

        Returns:
            video_dir_name_ext (str): The path + name + extension to the video.
               Returns a default value if the video file can't be found.
        """
        frames_dir = self.frames_dir_extracted()
        video_dir = frames_dir.replace("3_Frames", "2_Data")
        video_extension = ".MP4"
        video_name = self._name.split("_")[-1]
        video_path_name_ext = os.path.join(video_dir, video_name + video_extension)

        # simple case, the file already exists
        if os.path.exists(video_path_name_ext):
            return video_path_name_ext

        # check for a similar name
        else:
            exp = f"{video_dir}/{video_name}*{video_extension}"
            lt.debug(f'In UfacetFlight.video_dir_name_ext(): searching for videos matching the expression "{exp}"')
            matches = list(glob.glob(exp))
            to_remove, to_append_to_end = [], []
            for m in matches:
                if "ppt" in m:
                    to_remove.append(m)
                    if allow_ppt_compressed_videos:
                        to_append_to_end.append(m)
            for m in to_remove:
                matches.remove(m)
            for m in to_append_to_end:
                matches.append(m)

            if len(matches) == 1:
                return matches[0]
            else:
                lt.debug(
                    f"In UfacetFlight.video_dir_name_ext(): unable to find a matching video, returning a default value"
                )

        return video_path_name_ext

    def frames_dir_extracted(self):
        """Get the directory for the extracted frames of this flight.

        Example::

            flight = uf.UfacetFlight.get_by_name("C0045")
            lt.info(flight.frames_dir_extracted())
            # output "<experiment_dir>/2021-05-13_FastScan2/3_Frames/20210513/1644_F10_C0045_s1m50_d880/sony/M4ROOT/CLIP"

        Returns:
            results_dir (str): The "3_Frames" directory for extracted frames.
        """
        return self._frames_dir_extracted

    def frames_dir_deduplicated(self):
        """Get the directory for the deduplicated extracted frames of this flight.

        Example::

            flight = uf.UfacetFlight.get_by_name("C0045")
            lt.info(flight.frames_dir_deduplicated())
            # output "<experiment_dir>/2021-05-13_FastScan2/3_Frames/20210513/1644_F10_C0045_s1m50_d880/sony/M4ROOT/CLIP_deduplicated"

        Returns:
            results_dir (str): The "3_Frames" directory for deduplicated extracted frames.
        """
        if self._frames_dir_deduplicated != None:
            return self._frames_dir_deduplicated

        if not self._frames_dir_extracted.endswith("/CLIP"):
            lt.error_and_raise(
                f'Error: in UfacetFlight.frames_dir_extracted, unexpected directory name "{self._frames_dir_extracted}" for flight "{self._name}". Extracted directory name should end in "/CLIP"!'
            )
        self._frames_dir_deduplicated = self._frames_dir_extracted[: -len("CLIP")]
        self._frames_dir_deduplicated += "CLIP_deduplicated"

        return self._frames_dir_deduplicated

    def frames_name_format_extracted(self):
        return self._frames_name_format_extracted

    def frames_name_format_deduplicated(self):
        if self._frames_name_format_deduplicated != None:
            return self._frames_name_format_deduplicated
        parts = self._frames_name_format_extracted.split("%")
        self._frames_name_format_deduplicated = parts[0] + "id_%" + parts[1]
        return self._frames_name_format_deduplicated

    def frame_name_ext_extracted(self, frame_number: int):
        return self.frames_name_format_extracted() % frame_number

    def frame_name_ext_deduplicated(self, frame_number: int):
        return self.frames_name_format_deduplicated() % frame_number

    def frame_path_name_ext_extracted(self, frame_number: int):
        return os.path.join(self.frames_dir_extracted(), self.frame_name_ext_extracted(frame_number))

    def frame_path_name_ext_deduplicated(self, frame_number: int):
        return os.path.join(self.frames_dir_deduplicated(), self.frame_name_ext_deduplicated(frame_number))

    def _count_frames(self, path: str, name_format: str):
        if ft.directory_exists(path, follow_symlinks=True):
            if not st.is_cluster():
                prefix, suffix = self._get_frame_name_prefix_suffix(name_format)
                return ft.count_items_in_directory(path, prefix, suffix)
            else:
                # we expect the cluster computers to have all frames available, so we can use binary_count
                return ft.binary_count_items_in_directory(path, name_format)
        else:
            return 0

    def num_frames_extracted(self):
        if self._num_frames_extracted != None:
            return self._num_frames_extracted
        self._num_frames_extracted = self._count_frames(
            self.frames_dir_extracted(), self.frames_name_format_extracted()
        )
        return self._num_frames_extracted

    def num_frames_deduplicated(self):
        if self._num_frames_deduplicated != None:
            return self._num_frames_deduplicated

        self._num_frames_deduplicated = self._count_frames(
            self.frames_dir_deduplicated(), self.frames_name_format_deduplicated()
        )
        if self._num_frames_deduplicated == 0:
            num_extracted = self.num_frames_extracted()
            if num_extracted > 0:
                if not self._warned_no_deduplicated_yes_extracted:
                    lt.warn(
                        f"Warning: in UfacetFlight.num_frames_deduplicated(), attempting to access frames for flight {self._name}, which have been extracted but not deduplicated"
                    )
                    self._warned_no_deduplicated_yes_extracted = True

        return self._num_frames_deduplicated

    def results_base_dir(self):
        """Get the directory for processed results for this flight.

        Example::

            flight = uf.UfacetFlight.get_by_name("C0045")
            lt.info(flight.results_base_dir())
            # output "<experiment_dir>/2021-05-13_FastScan2/4_PostProcess/20210513/1644_F10_C0045_s1m50_d880/"

        Returns:
            results_dir (str): The "4_PostProcess" directory for this flight.
        """
        extracted_dir = self.frames_dir_extracted()
        if not (("3_Frames" in extracted_dir) and ("sony" in extracted_dir)):
            lt.error_and_raise(
                ValueError,
                f"Error: in UfacetFlight.results_base_dir(), unexpected extracted_dir structure for frames: {extracted_dir}",
            )

        experiment_dir, day_time_camera_dir = extracted_dir.split("3_Frames")
        day_time_dir, _ = day_time_camera_dir.strip("/").split("sony")
        return os.path.join(experiment_dir, "4_PostProcess", day_time_dir)
