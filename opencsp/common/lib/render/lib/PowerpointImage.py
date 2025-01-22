from PIL import Image
import numpy as np
import time

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.lib.PowerpointShape as pps
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class PowerpointImage(pps.PowerpointShape):
    """OpenCSP representation of a pptx powerpoint image, for more control over layouts.

    Like all PowerpointShapes, this class can be initialized as a placeholder.
    Being a placeholder means that it does not have a specific image assigned to
    it yet. You can check if an image has been assigned with the method
    :py:meth:`has_val`.

    Typical lifecycle for this class:

        1. Initialization (possibly including image assignment)
        2. Assignment to a PowerpointSlide instance
        3. Assignment of an image with :py:meth:`set_val`
        4. Serialize the instance to a file with :py:meth:`_to_text_file` to reduce memory usage
        5. De-serialize into a new instance with :py:meth:`from_text_file`
        6. Clean up temporary files with :py:meth:`clear_tmp_save`
    """

    _tmp_save_path = ft.join(orp.opencsp_temporary_dir(), "PowerpointImage/images/tmp")

    def __init__(
        self,
        val: str | np.ndarray | Image.Image | rcfr.RenderControlFigureRecord = None,
        dims: tuple[float, float, float, float] = None,
        cell_dims: tuple[float, float, float, float] = None,
        caption_is_above=False,
        caption: str = None,
        stretch=False,
        parent_slide=None,
    ):
        """
        Parameters
        ----------
        image: str | np.ndarray | rcfr.RenderControlFigureRecord
            The image to add. If an array, image, or figure record, then it will
            be saved to a temporary file with save(). Defaults to None.
        dims: tuple[float,float,float,float]
            The (left, top, width, height) of the image, or None to fill in one of
            the cells from the template slides. Defaults to None.
        dims: tuple[float,float,float,float]
            The (left, top, width, height) of the bounds for this shape, or None
            to fill in one of the cells from the template slides. Defaults to None.
        caption_is_above: bool
            True to put the image caption (if any) above the image, False to put
            the caption below the image. Defaults to False.
        caption: str
            The image caption. Defaults to None.
        stretch: bool
            True to stretch the image to fit the entire cell, False to fit
            within cell. Most useful with template slides. Default false.
        parent_slide: PowerpointSlide
            The slide containing this image. Used for fitting within the slide
            format. If None then the default format will be used. Default None.
        """
        super().__init__(cell_dims)
        self._val: str | np.ndarray | Image.Image | rcfr.RenderControlFigureRecord | None = None
        """ The image data for this instance, or the "path/name.ext" to the
        image file, or None if not yet set.
         
        Data types (Numpy arrays and Pillow Images) are treated as one type of image data.
        Reference types (strings and RenderControlFigureRecords) are treated as a different type of image data. """
        self._saved_name_ext = None
        """ Name+ext of this image in the temporary path, or None if not yet saved. """
        self.width: int = -1
        """ Width in pixels (-1 if not yet saved) """
        self.height: int = -1
        """ height in pixels (-1 if not yet saved) """
        self.dims = dims
        """ The left, top, width, and height of this image on the powerpoint slide, in inches. """
        self.caption_is_above = caption_is_above
        self.caption = caption
        self.stretch = stretch

        # do this import here to avoid a circular import loop
        import opencsp.common.lib.render.PowerpointSlide as pps

        self.parent_slide: pps.PowerpointSlide = parent_slide

        self._tmp_save_path = self.__class__._tmp_save_path

        self.set_val(val)

    def has_val(self) -> bool:
        """
        Returns true if an image has been assigned to this instance, or False if
        this instance was created without an assigned image and has not had an
        image assigned yet.
        """
        return self._val is not None

    def get_val(self) -> None | str | np.ndarray | Image.Image | rcfr.RenderControlFigureRecord:
        """
        Get the image assigned to this instance. What you probably actually want
        is :py:meth:`get_saved_path`. Returns None if :py:meth:`has_val` is False.
        """
        return self._val

    def set_val(self, image: str | np.ndarray | Image.Image | rcfr.RenderControlFigureRecord):
        """
        Assigns an image to this instance. If this instance already has an image
        assigned then the old value is overwritten without any checking.
        """
        self._val = image
        self.width = -1
        self.height = -1
        self._saved_name_ext = None

        if isinstance(self._val, str):
            self._val = self._val.strip()
            path, _, _ = ft.path_components(self._val)
            if ft.norm_path(path) == ft.norm_path(self._tmp_save_path):
                self._saved_name_ext = self._val

            if not ft.file_exists(self._val, error_if_exists_as_dir=False):
                lt.warn(
                    f"Warning in PowerpointImage.set_val: "
                    + f"reference type value \"{self._val}\" should be a path to an image file, but no such file exists!"
                )

    def _test_saved_path(self):
        """Verification check that I (BGB) haven't goofed up how images are
        saved to temporary files. This method is called after saving."""
        if isinstance(self._val, str) and (
            ft.path_to_cmd_line(self.get_saved_path()) == ft.path_to_cmd_line(self._val)
        ):
            if "tmp" in str(self._val):
                # lt.info(f"Image val and save path are the same:\nval: {self._val}\nsave path: {self.get_saved_path()}")
                pass
            else:
                lt.warn(
                    f"Image val and save path are the same:\n\tval: {self._val}\n\tsave path: {self.get_saved_path()}"
                )
        elif isinstance(self._val, str):
            if "tmp" in str(self._val):
                lt.warn(f"'tmp' found in image val:\n\t\"{self._val}\"")
            else:
                pass  # lt.info(f"Reference style image has a different save path than its reference path.")

    def replace_with_save(self):
        """Replace this instance's image with its save file path.

        We do this mainly to save on memory by releasing images."""
        if not self.has_val():
            return
        if not self.is_saved_to_file():
            return

        self._val = self.get_saved_path()
        self._test_saved_path()

    @property
    def shape(self) -> tuple[int, int] | tuple[None, None]:
        """
        Returns the (width, height) of the assigned image in pixels, or (None, None) if no
        image is assigned.

        Calls save() in the case that the assigned image is a string or
        RenderControlFigureRecord.
        """
        if not self.has_val():
            return None, None
        if self.width >= 0 and self.height >= 0:
            # return cached values
            return self.width, self.height

        # get the image size, calling save() as necessary
        if isinstance(self._val, Image.Image):
            pil_val: Image.Image = self._val
            self.width, self.height = pil_val.width, pil_val.height
        elif isinstance(self._val, np.ndarray):
            nd_val: np.ndarray = self._val
            self.width, self.height = nd_val.shape[1], nd_val.shape[0]
        else:  # reference-type and figure records
            if not self.is_saved_to_file():
                self.save()
            pil_val = Image.open(self.get_saved_path())
            self.width, self.height = pil_val.width, pil_val.height

        return self.width, self.height

    def has_dims(self):
        return self.dims is not None

    def dims_pptx(self):
        """Returns the powerpoint-style inches to place this image at (left, top, width, height)."""
        return self._pptx_inches(self.dims)

    @staticmethod
    def _image_dims_relative_to_cell(
        cell_dims: tuple[float, float, float, float], image_width: int, image_height: int, stretch=False
    ):
        """Returns the x, y, width, and height of an image fitted to the dimensions of the given cell."""
        cell_x, cell_y, cell_width, cell_height = cell_dims

        # compute image area for individual images with the given aspect ratio
        limit_by_width = (image_width / image_height) > (cell_width / cell_height)
        if stretch:
            limit_by_width = not limit_by_width
        if limit_by_width:
            # limit by width & center image vertically
            w = cell_width
            h = (cell_width / image_width) * image_height
            x_offset = 0
            y_offset = (cell_height - h) / 2
        else:
            # limit by height & center image horizontally
            w = (cell_height / image_height) * image_width
            h = cell_height
            x_offset = (cell_width - w) / 2
            y_offset = 0
        x = cell_x + x_offset
        y = cell_y + y_offset

        # lt.info(f"Image w,h: ({image_width},{image_height}), cell_dims: ({','.join([str(v) for v in cell_dims])}), image_dims: ({','.join([str(v) for v in [x,y,w,h]])}), limit_by_width: {limit_by_width}")

        return x, y, w, h

    def fit_to_cell_dimensions(self, cell_dims: tuple[float, float, float, float]):
        width, height = self.shape
        self.cell_dims = cell_dims
        self.dims = self._image_dims_relative_to_cell(self.cell_dims, width, height, self.stretch)

    def stretch_to_cell_dimensions(self, cell_dims: tuple[float, float, float, float]):
        width, height = self.shape
        self.cell_dims = cell_dims
        self.dims = self._image_dims_relative_to_cell(self.cell_dims, width, height, stretch=True)

    def reduce_size(self, reduced_image_size_scale: float = -1):
        """If the given image is significantly bigger than its rendered size,
        then resize the image to take up less disk space.
        This will save the image to disk first if necessary."""
        if reduced_image_size_scale < 0:
            return

        # save the image, if not saved already
        if not self.is_saved_to_file():
            self.save()

        # get the width and height
        image_width, image_height = self.shape

        # check if the size is reasonable (within reduced_image_size_scale% of the target size)
        reasonable = reduced_image_size_scale
        dpi = 300  # dots per inch
        if self.parent_slide != None:
            dpi = self.parent_slide.slide_control.slide_dpi
        expected_width_pixels = int(self.dims[2] * dpi * reasonable)
        expected_height_pixels = int(self.dims[3] * dpi * reasonable)
        if image_width <= expected_width_pixels * reasonable:
            return

        # image is larger than is reasonable, shrink it
        pil_image: Image.Image = Image.open(self.get_saved_path())
        lt.debug(f"Resizing from ({image_width,image_height}) to ({expected_width_pixels,expected_height_pixels})")
        pil_image = pil_image.resize((expected_width_pixels, expected_height_pixels))
        pil_image.save(self.get_saved_path())

    def get_saved_path(self) -> str:
        """
        Get the path/name.ext to the temporary file for the assigned image.
        Calls :py:meth:`save` if not yet saved to the temporary directory.
        """
        if not self.is_saved_to_file():
            self.save()
        return ft.join(self._tmp_save_path, self._saved_name_ext)

    def get_text_file_path(self) -> str:
        """Get the path to the PowerpointImage metadata for the image at get_saved_path().

        See also:"""
        return self.get_saved_path() + ".txt"

    def is_saved_to_file(self):
        if not self.has_val():
            return False
        return self._saved_name_ext != None

    def _move_file(self, from_dir_name_ext: str, to_dir_name_ext: str):
        try:
            ft.rename_file(from_dir_name_ext, to_dir_name_ext)
        except OSError:
            ft.copy_and_delete_file(from_dir_name_ext, to_dir_name_ext)

    def _save(self, path_name_ext: str):
        """Saves this image value to the given path+name+ext.

        Returns
        -------
            path (str): The directory of the saved file.
            body_ext (str): The body+ext of the saved file."""
        path, _, ext = ft.path_components(path_name_ext)
        ft.create_directories_if_necessary(path)

        if isinstance(self._val, Image.Image):
            pil_val: Image.Image = self._val
            pil_val.save(path_name_ext)

        elif isinstance(self._val, rcfr.RenderControlFigureRecord):
            # Figure records add extra stuffs to the image names, save them to
            # a temporary file and then move to our desired location.
            rec_val: rcfr.RenderControlFigureRecord = self._val
            format = ext.lstrip(".")
            _, tmp_path_name_ext = ft.get_temporary_file(suffix=ext, text=False)
            tmp_path, tmp_name, tmp_ext = ft.path_components(tmp_path_name_ext)
            tmp_path_name_ext_rcfr, _ = rec_val.save(tmp_path, tmp_name, format)
            try:
                self._move_file(tmp_path_name_ext_rcfr, path_name_ext)
            except PermissionError:
                if ft.file_exists(path_name_ext):
                    raise
                # probably just need to wait for matplotlib to release its stranglehold...
                time.sleep(1)
                self._move_file(tmp_path_name_ext_rcfr, path_name_ext)

        elif isinstance(self._val, np.ndarray):
            pil_val = it.numpy_to_image(self._val)
            pil_val.save(path_name_ext)

        elif isinstance(self._val, str):
            path, name, ext = ft.path_components(path_name_ext)
            ft.copy_file(self._val, path, name + ext)

        else:
            lt.error_and_raise(
                ValueError,
                f"Error: in PowerpointImage.save: unrecognized type for image "
                + f"(type \"{self._val.__class__.__name__}\")",
            )

        return path, ft.body_ext_given_file_dir_body_ext(path_name_ext)

    def _to_txt_file(self):
        """Serializes the non-image values of this instance to a file alongside the saved image file."""
        image_name_ext = None
        if self.has_val():
            image_name_ext = ft.body_ext_given_file_dir_body_ext(self.get_saved_path())

        path_name_ext = self.get_text_file_path()

        with open(path_name_ext, "w") as fout:
            fout.write("PowerpointImage\n")
            fout.write("v1\n")
            for v in [
                self.has_val(),
                image_name_ext,
                self._dims_to_str(self.dims),
                self._dims_to_str(self.cell_dims),
                self.caption_is_above,
                self.caption,
                self.caption == None,
                self.stretch,
            ]:
                fout.write(f"{v}\n")

    @classmethod
    def from_txt_file(cls, path_name_ext: str):
        """Given the text file from get_text_file_path(), this reconstructs a PowerpointImage instance."""
        path, _, _ = ft.path_components(path_name_ext)

        with open(path_name_ext, "r") as fin:
            lines = fin.readlines()
        slines = [line.strip() for line in lines]

        file_type = slines[0]
        if file_type != "PowerpointImage":
            lt.error_and_raise(
                RuntimeError,
                f"Error: in PowerpointImage.from_txt_file(), bad file type {file_type} in {path_name_ext}, expected type PowerpointImage",
            )
        version = slines[1]
        if version != "v1":
            lt.error_and_raise(
                RuntimeError,
                f"Error: in PowerpointImage.from_txt_file(), bad version {version} in {path_name_ext}, expected version v1",
            )
        has_val = slines[2] == 'True'
        image_name_ext = slines[3]
        dims = cls._str_to_dims(None, slines[4])
        cell_dims = cls._str_to_dims(None, slines[5])
        caption_is_above = slines[6] == 'True'
        caption = slines[7]
        caption_is_none = slines[8] == 'True'
        stretch = slines[9] == 'True'

        image_path_name_ext = None if not has_val else ft.join(path, slines[3])
        caption = None if caption_is_none else caption

        return cls(image_path_name_ext, dims, cell_dims, caption_is_above, caption, stretch)

    @classmethod
    def _get_save_dir_name_ext_pattern(cls, slide_idx: int = None, for_glob=False) -> str:
        """
        Get the temporary file path/name.ext pattern to :py:meth:`save` image
        data to. This name will also be used for the text files that hold the
        metadata for PowerpointImage serialization.

        Parameters
        ----------
        slide_idx : int, optional
            The slide index in the powerpoint deck, or None to leave as a fill
            in '%d' value. By default None.
        for_glob : bool, optional
            True to replace all fill in '%d' values with '*', for matching
            filenames with a glob expression. By default False.

        Returns
        -------
        dir_name_ext_pattern: str
            A name.ext pattern with 2 '%d' fill values, or 1 '%d' if slide_idx
            is not None, or 0 '%d' if for_glob is True.
        """
        if slide_idx == None:
            ret = "%d_%d.png"
        else:
            ret = f"{slide_idx}_%d.png"
        if for_glob:
            ret = ret.replace("%d", "*")
        return ft.join(cls._tmp_save_path, ret)

    def update_save_path(self, save_path: str):
        """
        Updates the path of the saved image data and associated serialized text
        file to the given save_path, moving any saved files from the old path to
        the new path.
        """
        if self.is_saved_to_file():
            to_rename = [self.get_saved_path(), self.get_text_file_path()]
            for path_name_ext in to_rename:
                _, name, ext = ft.path_components(path_name_ext)
                ft.copy_and_delete_file(path_name_ext, save_path, name + ext)

        self._tmp_save_path = save_path

        if self.is_saved_to_file():
            self._test_saved_path()

    def save(self) -> None | str:
        """
        Saves the metadata for this instance to a text file in the temporary
        directory, and saves the assigned image (as necessary) to an image file
        in the same directory.

        After saving, this instance can be reconstructed by calling
        :py:meth:`from_text_file` with the returned path+name+ext.

        Returns
        --------
        ppi_path_name_ext: str | None
            The path to the serialized instance. None if no image is assigned or saving failed.
        """
        # import inspect
        # frame = inspect.currentframe().f_back
        # to_print = []
        # while frame != None:
        #     to_print.append(str(frame))
        #     frame = frame.f_back
        # lt.info("In PowerpointImage.save()\n\t" + "\n\t".join(reversed(to_print)))

        # check if there is an assigned image, and thus if this instance can be saved
        if not self.has_val():
            return None

        # check if the assigned image has already been saved
        if self.is_saved_to_file():
            self._to_txt_file()  # update serialization text file with the latest values
            return self.get_text_file_path()

        # get the slide range from the parent, if any, for saving the image data
        slide_idx_range = range(1000)
        if self.parent_slide != None:
            if self.parent_slide.slide_control.slide_index >= 0:
                slide_idx_range = [self.parent_slide.slide_control.slide_index]

        # get a temporary name to save to
        found_unused_name = False
        for tmp_slide_idx in slide_idx_range:
            max_img_idx = 20  # probably shouldn't need more than 20 images in a slide
            dir_name_ext_pattern = self._get_save_dir_name_ext_pattern(tmp_slide_idx)
            max_image_path_name_ext = dir_name_ext_pattern % (max_img_idx - 1)
            if ft.file_exists(max_image_path_name_ext):
                continue

            for img_idx in range(max_img_idx):
                image_path_name_ext = dir_name_ext_pattern % img_idx
                if not ft.file_exists(image_path_name_ext):
                    found_unused_name = True
                    break
            break
        if not found_unused_name:
            lt.error_and_raise(
                RuntimeError,
                "Failed to find an empty spot to save this image to. Try using PowerpointImage.clear_tmp_save_all() to make more room.",
            )
        lt.info(f"saving image to {image_path_name_ext}")

        # save the image
        saved_path, body_ext = self._save(image_path_name_ext)
        if saved_path != self._tmp_save_path:
            lt.error_and_raise(
                RuntimeError,
                f"Error: in PowerpointImage.save(): programmer error, "
                + f"should have saved the temporary image to \"{self._tmp_save_path}\""
                + f"but instead saved it to \"{saved_path}\"!",
            )
        self._saved_name_ext = body_ext
        self._test_saved_path()

        # save the serialization values for this class
        self._to_txt_file()

        return self.get_text_file_path()

    def clear_tmp_save(self):
        """
        Reloads the image for this instance from the temporary save file, then
        deletes the temporary save file.
        """
        if not self.is_saved_to_file():
            return
        path_name_ext = self.get_saved_path()
        path_name_ext_serialized = self.get_text_file_path()

        # replace this instance's value with the saved-to-disk version
        self.set_val(np.array(Image.open(path_name_ext)))
        self._saved_name_ext = None

        # delete the saved files
        ft.delete_file(path_name_ext, error_on_not_exists=False)
        ft.delete_file(path_name_ext_serialized, error_on_not_exists=False)

    @classmethod
    def clear_tmp_save_all(cls):
        """Remove all temporary save files for all saved PowerpointImages"""
        if ft.directory_exists(cls._tmp_save_path, error_if_exists_as_file=False):
            ft.delete_files_in_directory(cls._tmp_save_path, "*.png", error_on_dir_not_exists=False)
            ft.delete_files_in_directory(cls._tmp_save_path, "*.png.txt", error_on_dir_not_exists=False)
