import tkinter as tk

import opencsp.common.lib.tool.log_tools as lt


class TkToolTip(object):
    """
    A tooltip class for Tkinter widgets.

    This class provides a tooltip that appears when the mouse hovers over a specified widget.
    The tooltip displays a text message and automatically hides when the mouse leaves the widget.

    Attributes
    ----------
    waittime : int
        The time in milliseconds to wait before showing the tooltip.
    wraplength : int
        The maximum width of the tooltip in pixels.
    id_ : int or None
        The identifier for the scheduled tooltip display.
    tw : tk.Toplevel or None
        The tooltip window instance.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, widget, text='widget info'):
        """
        A tooltip class for Tkinter widgets.

        This class provides a tooltip that appears when the mouse hovers over a specified widget.
        The tooltip displays a text message and automatically hides when the mouse leaves the widget.

        Parameters
        ----------
        widget : tk.Widget
            The Tkinter widget to which the tooltip is attached.
        text : str, optional
            The text to display in the tooltip. Default is 'widget info'.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.waittime = 500  # miliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id_ = None
        self.tw = None

    def enter(self, event=None):
        """Schedule the tooltip to be displayed."""
        # "ChatGPT 4o" assisted with generating this docstring.
        self.schedule()

    def leave(self, event=None):
        """Unschedule the tooltip and hide it."""
        # "ChatGPT 4o" assisted with generating this docstring.
        self.unschedule()
        self.hidetip()

    def schedule(self):
        """Schedule the tooltip to be shown after the wait time."""
        # "ChatGPT 4o" assisted with generating this docstring.
        self.unschedule()
        self.id_ = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        """Cancel the scheduled tooltip display."""
        # "ChatGPT 4o" assisted with generating this docstring.
        id_ = self.id_
        self.id_ = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
        """Create and display the tooltip window."""
        # "ChatGPT 4o" assisted with generating this docstring.
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + self.widget.winfo_width()
        y += self.widget.winfo_rooty() + self.widget.winfo_height()
        # Creates a toplevel window
        self.tw = window(self.widget, TopLevel=True)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x:d}+{y:d}")
        label = tk.Label(
            self.tw,
            text=self.text,
            justify="left",
            background="#ffffff",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
        )
        label.pack(ipadx=1)

    def hidetip(self):
        """Hide the tooltip window."""
        # "ChatGPT 4o" assisted with generating this docstring.
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


def window(*vargs, TopLevel=False, **kwargs):
    """
    Initialize and return a new Tkinter window or Toplevel instance.

    This function attempts to create a Tkinter window. If the creation fails, it retries up to two more times.
    This is useful in cases where initializing a Tkinter window may fail due to environmental issues.

    Parameters
    ----------
    *vargs : tuple
        Positional arguments to pass to the Tkinter window constructor.
    TopLevel : bool, optional
        If True, create a Toplevel window instead of a Tk window. Default is False.
    **kwargs : dict
        Keyword arguments to pass to the Tkinter window constructor.

    Returns
    -------
    tk.Tk or tk.Toplevel
        A new Tkinter window or Toplevel instance.

    Raises
    ------
    Exception
        If the window creation fails after three attempts.

    Notes
    -----
    If the window creation fails, a warning is logged, and the function will wait for a second before retrying.

    Examples
    --------
    >>> main_window = window()
    >>> top_window = window(TopLevel=True)
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    window_class = tk.Tk
    if TopLevel:
        window_class = tk.Toplevel

    try:
        # try to create a window
        return window_class(*vargs, **kwargs)
    except Exception:
        try:
            lt.warn("Failed to create a tkinter.Tk() window. Trying again (2nd attempt).")
            # first attempt failed, try again
            return window_class(*vargs, **kwargs)
        except Exception:
            # second attempt failed, give the system a second to stabalize and
            # try a third time
            lt.warn("Failed to create a tkinter.Tk() window. Trying again (3rd attempt).")
            import time

            time.sleep(1)
            return window_class(*vargs, **kwargs)
