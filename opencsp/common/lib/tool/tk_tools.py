import tkinter as tk

import opencsp.common.lib.tool.log_tools as lt


class TkToolTip(object):
    def __init__(self, widget, text='widget info'):
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
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id_ = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id_ = self.id_
        self.id_ = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
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
            justify='left',
            background="#ffffff",
            relief='solid',
            borderwidth=1,
            wraplength=self.wraplength,
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


def window(*vargs, TopLevel=False, **kwargs):
    """Initializes and returns a new tkinter.Tk (or tkinter.TopLevel) instance.

    If creating the window fails, tries again (up to two more times).

    Sometimes initializing a tk window fails. But if you try to
    initialize the window again, it seems to always succeed.
    When it fails, it is often with an error about not being able to find a
    file. Something like::

        _tkinter.TclError: Can't find a usable init.tcl in the following directories
    """
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
