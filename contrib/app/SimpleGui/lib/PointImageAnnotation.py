import tkinter
from typing import Callable, Optional

from contrib.app.SimpleGui.lib.AbstractImageAnnotation import AbstractImageAnnotation
import opencsp.common.lib.geometry.Pxy as p2


class PointImageAnnotation(AbstractImageAnnotation):
    """
    A simple annotation that indicates a pixel point on an image.
    """

    def __init__(self, point: p2.Pxy, is_preview=False):
        """
        Create a point annotation to be drawn onto a canvas instance.

        Parameters
        ----------
        point : p2.Pxy
            The pixel location of this instance.
        """
        super().__init__(is_preview)

        # register inputs
        self.point = point

    def draw(self, coord_translator: Callable[[p2.Pxy], p2.Pxy], canvas: tkinter.Canvas):
        """Adds the graphics to represent this instance to the canvas. Modifies self.canvas_items."""
        super().draw(coord_translator, canvas)
        x, y = coord_translator(self.point).astuple()
        x0, y0, x1, y1 = x - 3, y - 3, x + 3, y + 3
        self.canvas_items.append(self.canvas.create_oval(x0, y0, x1, y1, outline="magenta"))

    @classmethod
    def on_mouse_down(
        cls, coord_translator: Callable[[p2.Pxy], p2.Pxy], mouse_down_event: tkinter.Event
    ) -> Optional["AbstractImageAnnotation"]:
        """Creates an instance of this class when the mouse is moved. If no instance is created, then return None."""
        mouse_down_loc = coord_translator(p2.Pxy((mouse_down_event.x, mouse_down_event.y)))
        return cls(mouse_down_loc, is_preview=True)

    @classmethod
    def on_mouse_move(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_move_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """Creates an instance of this class when the mouse is moved. If no instance is created, then return None."""
        mouse_move_loc = coord_translator(p2.Pxy((mouse_move_event.x, mouse_move_event.y)))
        return cls(mouse_move_loc, is_preview=True)

    @classmethod
    def on_mouse_up(
        cls,
        coord_translator: Callable[[p2.Pxy], p2.Pxy],
        mouse_down_event: tkinter.Event | None,
        mouse_up_event: tkinter.Event,
    ) -> Optional["AbstractImageAnnotation"]:
        """Creates an instance of this class when the mouse button is pressed. If no instance is created, then return None."""
        mouse_up_loc = coord_translator(p2.Pxy((mouse_up_event.x, mouse_up_event.y)))
        return cls(mouse_up_loc, is_preview=False)

    @classmethod
    def class_descriptor(self) -> str:
        return "point"

    @classmethod
    def csv_columns(cls) -> list[str]:
        """
        The names of the columns used to represent this annotation.

        These column names can potentially be shared with other annotations.
        """
        return ["p1x", "p1y"]

    def csv_values(self) -> list[str]:
        """Get the values that represent this instance. Should match the order from csv_columns."""
        x, y = self.point.x[0], self.point.y[0]
        return [str(x), str(y)]

    @classmethod
    def from_csv(cls, data: list[str], is_preview=False) -> tuple["AbstractImageAnnotation"]:
        """Construct an instance of this class from the columns matching the column names for this class."""
        x, y = float(data[0]), float(data[1])
        return cls(p2.Pxy((x, y)), is_preview)


# register this class with AbstractAnnotation, so that it can be created from
# various triggers.
AbstractImageAnnotation._registered_annotation_classes.add(PointImageAnnotation)
