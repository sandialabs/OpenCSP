import os

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.lib.PowerpointShape as pps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class PowerpointText(pps.PowerpointShape):
    """
    A class representing text elements in a PowerPoint presentation.

    This class allows for the creation, manipulation, and storage of text elements
    that can be added to PowerPoint slides. It supports saving to and loading from
    text files for persistence.

    Attributes
    ----------
    dims : tuple[float, float, float, float]
        The dimensions of the text element in the PowerPoint slide.
    is_title : bool
        A flag indicating whether this text element is a title.
    parent_slide : PowerpointSlide
        The parent slide to which this text element belongs.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    _tmp_save_path = os.path.join(orp.opencsp_temporary_dir(), "PowerpointTexts", "texts", "tmp")

    def __init__(
        self,
        val: str = None,
        dims: tuple[float, float, float, float] = None,
        cell_dims: tuple[float, float, float, float] = None,
        is_title=False,
        parent_slide=None,
    ):
        """
        Initializes the PowerpointText instance.

        Parameters
        ----------
        val : str, optional
            The text value of the PowerPoint text element. Defaults to None.
        dims : tuple[float, float, float, float], optional
            The dimensions of the text element (left, top, width, height). Defaults to None.
        cell_dims : tuple[float, float, float, float], optional
            The dimensions of the cell containing the text. Defaults to None.
        is_title : bool, optional
            A flag indicating whether this text element is a title. Defaults to False.
        parent_slide : PowerpointSlide, optional
            The parent slide to which this text element belongs. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super().__init__(cell_dims)
        self._val = val
        self.dims = dims
        self.is_title = is_title
        self._tmp_save_path = self.__class__._tmp_save_path
        self._saved_name_ext = None

        # do this import here to avoid a circular import loop
        import opencsp.common.lib.render.PowerpointSlide as pps

        self.parent_slide: pps.PowerpointSlide = parent_slide

    def has_val(self):
        """
        Checks if the text element has a value.

        Returns
        -------
        has_value : bool
            True if the text element has a value, False otherwise.
        """
        return self._val != None

    def get_val(self) -> str | None:
        """
        Gets the value of the text element.

        Returns
        -------
        val : str | None
            The value of the text element, or None if it has no value.
        """
        return self._val

    def set_val(self, val: str):
        """
        Sets the value of the text element.

        Parameters
        ----------
        val : str
            The new value of the text element.
        """
        self._val = val
        self._saved_name_ext = None

    def has_dims(self):
        """
        Checks if the text element has dimensions (top, left, width, height).

        Returns
        -------
        has_dims : bool
            True if the text element has dimensions, False otherwise.
        """
        return self.dims is not None

    def dims_pptx(self):
        """Returns the powerpoint-style inches to place this text at (left, top, width, height)."""
        return self._pptx_inches(self.dims)

    @staticmethod
    def compute_height(font_pnt: int, nlines: int = 1):
        """Attempts to calculate a height, in inches, based on the given font size."""
        # experimental results:
        #     single line of pnt 30 font: 0.62"
        #     each additional line: 0.5"
        per_line = font_pnt / 30 * 0.5
        padding_top_bot = font_pnt / 30 * (0.62 - 0.5)
        return (nlines * per_line) + padding_top_bot

    def compute_and_assign_height(self, font_pnt: int):
        """Attempts to calculate a height, in inches, for this text based on the given font size."""
        if self.has_val():
            nlines = len(self._val.split("\n"))
        else:
            nlines = 1

        x, y, w, _ = self.dims
        h = self.compute_height(font_pnt, nlines)
        self.dims = x, y, w, h

    def is_saved_to_file(self):
        """
        Checks if the text element has been serialized and saved to a file.

        Returns
        -------
        is_saved : bool
            True if the text element is saved to a file, False otherwise.
        """
        return self._saved_name_ext != None

    def get_text_file_path(self) -> str:
        """
        Gets the file path of the saved text element.

        Returns
        -------
        path : str
            The file path of the saved text element, or None if it is not saved.
        """
        if not self.is_saved_to_file():
            return None
        return os.path.join(self._tmp_save_path, self._saved_name_ext)

    def _to_text_file(self, path_name_ext: str):
        """Serializes this instance to a file."""
        path, _, _ = ft.path_components(path_name_ext)
        ft.create_directories_if_necessary(path)
        with open(path_name_ext, "w") as fout:
            fout.write("PowerpointText\n")
            fout.write("v1\n")
            for v in [self._dims_to_str(self.dims), self._dims_to_str(self.cell_dims), self.is_title, self.has_val()]:
                fout.write(f"{v}\n")
            fout.write(f"{self.get_val()}")

    @classmethod
    def from_txt_file(cls, path_name_ext: str):
        """Given the text file from get_text_file_path(), this reconstructs a PowerpointText instance."""
        path, _, _ = ft.path_components(path_name_ext)

        with open(path_name_ext, "r") as fin:
            lines = fin.readlines()
        slines = [line.strip() for line in lines]
        if len(slines) < 2:
            lt.error_and_raise(
                RuntimeError, f"Error: in PowerpointSlide.from_txt_file(), not enough lines in file {path_name_ext}"
            )

        file_type = slines[0]
        if file_type != "PowerpointText":
            lt.error_and_raise(
                RuntimeError,
                f"Error: in PowerpointText.from_txt_file(), bad file type {file_type} in {path_name_ext}, expected type PowerpointText",
            )
        version = slines[1]
        if version != "v1":
            lt.error_and_raise(
                RuntimeError,
                f"Error: in PowerpointText.from_txt_file(), bad version {version} in {path_name_ext}, expected version v1",
            )
        dims = cls._str_to_dims(None, slines[2])
        cell_dims = cls._str_to_dims(None, slines[3])
        is_title = slines[4] == "True"
        has_val = slines[5] == "True"
        val = None if not has_val else "\n".join(lines[6:])

        return cls(val, dims, cell_dims, is_title)

    def update_save_path(self, save_path: str):
        """
        Updates the save file path of the text element. Moves the existing
        save file if there is one.

        Parameters
        ----------
        save_path : str
            The new save path of the text element.
        """
        if self.is_saved_to_file():
            to_rename = [self.get_text_file_path()]
            for path_name_ext in to_rename:
                _, name, ext = ft.path_components(path_name_ext)
                ft.rename_file(path_name_ext, os.path.join(save_path, name + ext))

        self._tmp_save_path = save_path

    def save(self):
        """Saves this instance to a text file, as necessary.
        It can then be reconstructed by calling from_text_file() with the returned path_name_ext.

        Returns:
        --------
            ppt_path_name_ext (str|None): The path to the serialized instance. None if saving failed.
        """
        # first check if we need to save
        if self.is_saved_to_file():
            return self.get_text_file_path()

        # get the slide range from the parent, if any
        slide_idx = -1
        if self.parent_slide != None:
            if self.parent_slide.slide_control.slide_index >= 0:
                slide_idx = self.parent_slide.slide_control.slide_index
        if slide_idx < 0:
            return None

        # find an available slot to save to
        found_open_spot = False
        for i in range(20):  # probably don't need more than 20 texts per slide
            path_name_ext = os.path.join(self._tmp_save_path, f"{slide_idx}_{i}.txt")
            if not ft.file_exists(path_name_ext):
                found_open_spot = True
                break
        if not found_open_spot:
            lt.error_and_raise(
                RuntimeError,
                f"Unable to find a spot in {self._tmp_save_path} to save this powerpoint text for slide {slide_idx} to.",
            )

        _, name, ext = ft.path_components(path_name_ext)
        self._to_text_file(path_name_ext)
        self._saved_name_ext = name + ext
        return self.get_text_file_path()

    def clear_tmp_save(self):
        """
        Clears the temporary save of the text element.
        """
        if self._saved_name_ext != None:
            ft.delete_file(self._saved_name_ext, error_on_not_exists=False)
            self._saved_name_ext = None

    @classmethod
    def clear_tmp_save_all(cls):
        """
        Clears all temporary saves of text elements.
        """
        if ft.directory_exists(cls._tmp_save_path, error_if_exists_as_file=False):
            ft.delete_files_in_directory(cls._tmp_save_path, "*.txt")
