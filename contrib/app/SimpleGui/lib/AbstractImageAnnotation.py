from abc import ABC, abstractmethod

import tkinter
from typing import Callable, Optional

import opencsp.common.lib.file.SimpleCsv as scsv
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class AbstractImageAnnotation(ABC):
    """Simple annotations that get displayed on top of images."""

    _registered_annotation_classes: set[type["AbstractImageAnnotation"]] = set()
    """ Register of all available simple annotation classes. """

    def __init__(self, is_preview=False):
        """
        Parameters
        ----------
        is_preview : bool, optional
            True if this instance is waiting to be finished (such as when
            drawing a line and the mouse button hasn't been released yet). False
            otherwise. By default False.
        """
        self.is_preview = is_preview
        """
        True if this instance is waiting to be finished (such as when drawing a
        line and the mouse button hasn't been released yet). False otherwise.
        """
        self.canvas: tkinter.Canvas = None
        """ The canvas instance on which to draw this instance. """
        self._canvas_items: list[int] = []
        """ Handles to the canvas items used to draw this instance. """

    @property
    def canvas_items(self) -> list[int]:
        """List of handles to the graphics items on the canvas."""
        return self._canvas_items

    def clear(self):
        """Removes all graphics representing this instance from the canvas."""
        for canvas_item in self.canvas_items:
            with et.ignored(Exception):
                self.canvas.delete(canvas_item)
        self.canvas_items.clear()

    @classmethod
    def on_mouse_down(
        cls, coord_translator: Callable[[p2.Pxy], p2.Pxy], mouse_down_event: tkinter.Event
    ) -> Optional["AbstractImageAnnotation"]:
        """
        Creates an instance of this class when the mouse button is pressed. If
        no instance is created, then return None.

        Parameters
        ----------
        coord_translator : Callable[[p2.Pxy], p2.Pxy]
            Function to translate from event x and y coordinates to image coordinates.
        """
        return None

    @classmethod
    def on_mouse_move(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_move_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """
        Creates an instance of this class when the mouse is moved. If no
        instance is created, then return None.

        Parameters
        ----------
        coord_translator : Callable[[p2.Pxy], p2.Pxy]
            Function to translate from event x and y coordinates to image coordinates.
        """
        return None

    @classmethod
    def on_mouse_up(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_up_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """
        Creates an instance of this class when the mouse button is pressed. If
        no instance is created, then return None.

        Parameters
        ----------
        coord_translator : Callable[[p2.Pxy], p2.Pxy]
            Function to translate from event x and y coordinates to image coordinates.
        """
        return None

    @staticmethod
    def save_annotations_to_csv(annotations: list["AbstractImageAnnotation"], file_path_name_ext: str, overwrite=False):
        """
        Saves the given list of simple annotations to the given CSV file using
        each annotations built-in CSV conversion methods.

        Parameters
        ----------
        annotations : list[AbstractAnnotation]
            The annotations to be saved.
        file_path_name_ext : str
            The CSV file to be saved to.
        overwrite : bool, optional
            True to replace the current contents of the CSV file at
            file_path_name_ext, by default False

        Raises
        ------
        FileExistsError
            If file_path_name_ext exists and overwrite is False.
        FileNotFoundError
            If the directory of file_path_name_ext doesn't exist.
        """
        if ft.file_exists(file_path_name_ext):
            if not overwrite:
                raise FileExistsError
        file_path, file_name, file_ext = ft.path_components(file_path_name_ext)
        if not ft.directory_exists(file_path):
            raise FileNotFoundError

        # build the list of columns
        columns: list[str] = ["class"]
        for annotation in annotations:
            for aheader in annotation.csv_columns():
                if aheader not in columns:
                    columns.append(aheader)
        header = ",".join(columns)

        # add a row for each annotation
        rows: list[str] = []
        for annotation in annotations:
            row = [""] * len(columns)
            row[0] = annotation.class_descriptor()
            for aheader, sval in zip(annotation.csv_columns(), annotation.csv_values()):
                row[columns.index(aheader)] = sval
            rows.append(",".join(row))

        # save all values to a csv file
        lt.info(f"Saving annotations csv {file_name+file_ext}")
        with open(file_path_name_ext, "w") as fout:
            fout.write(header + "\n")
            for row in rows:
                fout.write(row + "\n")

    @staticmethod
    def load_annotations_from_csv(file_path_name_ext: str, is_preview=False) -> list["AbstractImageAnnotation"]:
        """
        Loads simple annotations from the given CSV file.

        Parameters
        ----------
        file_path_name_ext: str
            The CSV file to load the annotations from.

        Returns
        -------
        annotations: list[AbstractAnnotation]
            The loaded annotations.
        """
        ret: list[AbstractImageAnnotation] = []
        file_path, file_name, file_ext = ft.path_components(file_path_name_ext)

        parser = scsv.SimpleCsv("annotations csv", file_path, file_name + file_ext)
        for row_dict in parser:
            descriptor = row_dict["class"]

            for aclass in AbstractImageAnnotation._registered_annotation_classes:
                if aclass.class_descriptor() == descriptor:
                    aheaders = aclass.csv_columns()
                    svals = [row_dict[aheader] for aheader in aheaders]
                    inst = aclass.from_csv(svals, is_preview)
                    ret.append(inst)
                    break

        return ret

    @abstractmethod
    def draw(self, coord_translator: Callable[[p2.Pxy], p2.Pxy], canvas: tkinter.Canvas):
        """
        Adds the graphics to represent this instance to the canvas. Modifies self.canvas_items.

        Implementations of this class should call super().draw().

        Parameters
        ----------
        coord_translator : Callable[[p2.Pxy], p2.Pxy]
            Function to translate from image coordinates to screen coordinates.
        canvas : tkinter.Canvas
            The canvas to draw this instance onto.
        """
        self.clear()
        self.canvas = canvas

    @classmethod
    @abstractmethod
    def class_descriptor(self) -> str:
        """A string used to identify this class in an annotations file."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def csv_columns(cls) -> list[str]:
        """
        The names of the columns used to represent this annotation.

        These column names can potentially be shared with other annotations.
        """
        raise NotImplementedError

    @abstractmethod
    def csv_values(self) -> list[str]:
        """Get the values that represent this instance. Should match the order from csv_columns."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_csv(cls, data: list[str], is_preview=False) -> "AbstractImageAnnotation":
        """Construct an instance of this class from the columns matching the column names for this class."""
        raise NotImplementedError()
