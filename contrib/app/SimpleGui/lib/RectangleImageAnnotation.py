import tkinter
from typing import Callable, Optional

from contrib.app.SimpleGui.lib.AbstractImageAnnotation import AbstractImageAnnotation
import opencsp.common.lib.geometry.Pxy as p2


class RectangleImageAnnotation(AbstractImageAnnotation):
    """
    A simple annotation that indicates a rectangular area of interest.
    """

    def __init__(self, corner_1: p2.Pxy, corner_2: p2.Pxy, is_preview=False):
        """
        Create a rectangular annotation to be drawn onto a canvas instance.

        Parameters
        ----------
        corner_1 : p2.Pxy
            One corner of the rectangle (typically the top-left corner).
        corner_2 : p2.Pxy
            The opposite corner to corner_1 (typically the bottom-right corner).
        """
        super().__init__(is_preview)

        # normalize input
        tlx, tly, brx, bry = corner_1.x[0], corner_1.y[0], corner_2.x[0], corner_2.y[0]
        if tlx > brx:
            tlx = corner_2.x[0]
            brx = corner_1.x[0]
        if tly > bry:
            tly = corner_2.y[0]
            bry = corner_1.y[0]
        top_left, bottom_right = p2.Pxy((tlx, tly)), p2.Pxy((brx, bry))

        # register inputs
        self.top_left = top_left
        self.bottom_right = bottom_right

    def draw(self, coord_translator: Callable[[p2.Pxy], p2.Pxy], canvas: tkinter.Canvas):
        """Adds the graphics to represent this instance to the canvas. Modifies self.canvas_items."""
        super().draw(coord_translator, canvas)
        tlx, tly, brx, bry = self.top_left.x[0], self.top_left.y[0], self.bottom_right.x[0], self.bottom_right.y[0]
        tl = coord_translator(p2.Pxy((tlx, tly)))
        br = coord_translator(p2.Pxy((brx, bry)))
        self.canvas_items.append(self.canvas.create_rectangle(tl.x[0], tl.y[0], br.x[0], br.y[0], outline="red"))

    @classmethod
    def on_mouse_move(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_move_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """Creates an instance of this class when the mouse is moved. If no instance is created, then return None."""
        if mouse_down_event is None:
            return None
        mouse_down_loc = coord_translator(p2.Pxy((mouse_down_event.x, mouse_down_event.y)))
        mouse_move_loc = coord_translator(p2.Pxy((mouse_move_event.x, mouse_move_event.y)))
        return cls(mouse_down_loc, mouse_move_loc, is_preview=True)

    @classmethod
    def on_mouse_up(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_up_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """Creates an instance of this class when the mouse button is pressed. If no instance is created, then return None."""
        if mouse_down_event is None:
            return None
        mouse_down_loc = coord_translator(p2.Pxy((mouse_down_event.x, mouse_down_event.y)))
        mouse_up_loc = coord_translator(p2.Pxy((mouse_up_event.x, mouse_up_event.y)))
        return cls(mouse_down_loc, mouse_up_loc, is_preview=False)

    @classmethod
    def class_descriptor(self) -> str:
        return "rect"

    @classmethod
    def csv_columns(cls) -> list[str]:
        """
        The names of the columns used to represent this annotation.

        These column names can potentially be shared with other annotations.
        """
        return ["p1x", "p1y", "p2x", "p2y"]

    def csv_values(self) -> list[str]:
        """Get the values that represent this instance. Should match the order from csv_columns."""
        tlx, tly, brx, bry = self.top_left.x[0], self.top_left.y[0], self.bottom_right.x[0], self.bottom_right.y[0]
        return [str(tlx), str(tly), str(brx), str(bry)]

    @classmethod
    def from_csv(cls, data: list[str], is_preview=False) -> tuple["AbstractImageAnnotation"]:
        """Construct an instance of this class from the columns matching the column names for this class."""
        tlx, tly, brx, bry = float(data[0]), float(data[1]), float(data[2]), float(data[3])
        corner_1, corner_2 = p2.Pxy((tlx, tly)), p2.Pxy((brx, bry))
        return cls(corner_1, corner_2, is_preview)


# register this class with AbstractAnnotation, so that it can be created from
# various triggers.
AbstractImageAnnotation._registered_annotation_classes.add(RectangleImageAnnotation)
