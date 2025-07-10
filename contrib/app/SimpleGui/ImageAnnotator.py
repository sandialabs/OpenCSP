import copy
import os
import tkinter.filedialog
import tkinter.messagebox

from PIL import Image
import tkinter
import tkinter.ttk

from contrib.app.SimpleGui.lib.AbstractImageAnnotation import AbstractImageAnnotation
from contrib.app.SimpleGui.lib.PointImageAnnotation import PointImageAnnotation
from contrib.app.SimpleGui.lib.RectangleImageAnnotation import RectangleImageAnnotation
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.tk_tools as tkt


class ImageAnnotator:
    """
    This class presents a GUI for adding annotations to an image and recording
    them to a CSV file. After loading an image, the user draws annotations on
    top of the image and chooses a file to record the annotations to.
    """

    def __init__(self):
        """Initializes the GUI"""

        # Create tkinter object
        self.root = tkt.window()

        # Set title
        self.root.title('Annotation Selector')

        # Set size of GUI
        self.root.geometry('600x640+200+100')

        # Internal representation and tracking of the loaded image
        self.image_path_name_ext: str = None
        """ The path to the chosen image to be annotated """
        self.image: Image.Image = None
        """ The pillow representation of the image """
        self.tkimage: tkinter.PhotoImage = None
        """ The tkinter representation of the image """
        self.canvas_image: int = None
        """ The internal canvas id of the tkimage. """

        # User input state
        self.mouse_down_event: tkinter.Event = None
        """ Tracks when the mouse has been pressed by not released yet. None if
        the mouse is not currently pressed. """
        self._zoom_level: int = 0
        """ The zoom level, between -10 and 10. 0 for no zoom. """
        self._zoom_loc: p2.Pxy = p2.Pxy((0, 0))
        """ The image coordinate to draw in the center of the screen. Defaults to the center of the image. """

        # Annotations
        self.annotations: list[AbstractImageAnnotation] = []
        """ List of all annotations, both preview and regular. """
        self.save_path_name_ext: str = None
        """ File to autosave the annotations to. Same as the last file saved. """

        # Add all buttons/widgets to window
        self._create_layout()
        self._on_tool_select()

        # Run window infinitely
        self.root.mainloop()

    @property
    def preview_annotations(self) -> list[AbstractImageAnnotation]:
        return list(filter(lambda a: a.is_preview, self.annotations))

    @property
    def regular_annotations(self) -> list[AbstractImageAnnotation]:
        return list(filter(lambda a: not a.is_preview, self.annotations))

    @property
    def tool(self) -> str:
        """String representing the currently active tool. For example "zoom" or "rect"."""
        return self.var_tool_sel.get()

    @property
    def zoom_level(self) -> int:
        """The zoom_level. This is an integer value between -10 and 10. See also zoom_scale."""
        return self._zoom_level

    @zoom_level.setter
    def zoom_level(self, val: int):
        """Sets the zoom level and redraws the canvas."""
        self._zoom_level = val
        self.draw()

    @property
    def zoom_loc(self) -> p2.Pxy:
        """The zoom focus location, which will be located in the center of the canvas."""
        return self._zoom_loc

    @zoom_loc.setter
    def zoom_loc(self, val: p2.Pxy):
        """Sets the zoom focus location and redraws the canvas."""
        self._zoom_loc = val
        self.draw()

    @property
    def zoom_scale(self) -> float:
        """The scale of the zoom, between 1/11 and 6. Returns 1 for no zoom."""
        if self.zoom_level == 0:
            return 1.0

        elif self.zoom_level < 0:
            return 1.0 / abs(self.zoom_level - 1)

        else:
            return (self.zoom_level + 2) / 2.0

    @property
    def image_zero_offset(self) -> p2.Pxy:
        """Canvas screen-pixel at which the image (0,0) pixel is located."""
        if self.image is None:
            return p2.Pxy((0, 0))

        scale = self.zoom_scale
        center = self.zoom_loc
        can_w, can_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_w, img_h = self.image.width, self.image.height
        can_pixel_center = p2.Pxy((can_w / 2 / scale, can_h / 2 / scale))

        if img_w * scale <= can_w:
            offset_screen_x = 0
        else:
            offset_screen_x = (can_pixel_center.x[0] - center.x[0]) * scale

        if img_h * scale <= can_h:
            offset_screen_y = 0
        else:
            offset_screen_y = (can_pixel_center.y[0] - center.y[0]) * scale

        return p2.Pxy((offset_screen_x, offset_screen_y))

    def _create_layout(self):
        assets_dir = ft.norm_path(os.path.join(orp.opencsp_code_dir(), "../contrib/app/SimpleGui/assets"))

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1, minsize="110")
        self.root.grid_columnconfigure(1, weight=1000)

        label_frame_controls = tkinter.LabelFrame(self.root, text='Controls')
        label_frame_controls.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)

        label_frame_image = tkinter.LabelFrame(self.root, text='Image')
        label_frame_image.grid(row=0, column=1, sticky='nesw', padx=5, pady=5)
        label_frame_image.grid_rowconfigure(0, weight=1)
        label_frame_image.grid_columnconfigure(0, weight=1)

        # ======================= File Controls ========================== #
        r = 0

        self.btn_load = tkinter.Button(label_frame_controls, text='Load Image', command=self.on_load_image)
        self.btn_load.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky='nesw')
        r += 1

        self.btn_save = tkinter.Button(label_frame_controls, text='Save Annotations', command=self.on_save_annotations)
        self.btn_save.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky='nesw')
        r += 1

        self.checkbtn_auto_save_checked = tkinter.BooleanVar()
        self.checkbtn_auto_save = tkinter.Checkbutton(
            label_frame_controls, text='Auto Save', variable=self.checkbtn_auto_save_checked
        )
        self.checkbtn_auto_save.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky='nesw')
        r += 1

        self.btn_load = tkinter.Button(label_frame_controls, text='Load Annotations', command=self.on_load_annotations)
        self.btn_load.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky='nesw')
        r += 1

        self.separator_file_controls = tkinter.ttk.Separator(label_frame_controls)
        self.separator_file_controls.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky="nesw")
        r += 1

        # ======================= Tools ========================== #
        self.var_tool_sel = tkinter.StringVar(value="zoom")

        self.zoom_image = tkinter.PhotoImage(file=ft.norm_path(os.path.join(assets_dir, "magnifying_glass.gif")))
        self.btn_load = tkinter.Radiobutton(
            label_frame_controls,
            image=self.zoom_image,
            indicatoron=0,
            command=self._on_tool_select,
            value="zoom",
            variable=self.var_tool_sel,
        )
        self.btn_load.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        self.rect_image = tkinter.PhotoImage(file=ft.norm_path(os.path.join(assets_dir, "square.gif")))
        self.btn_load = tkinter.Radiobutton(
            label_frame_controls,
            image=self.rect_image,
            indicatoron=0,
            command=self._on_tool_select,
            value="rect",
            variable=self.var_tool_sel,
        )
        self.btn_load.grid(row=r, column=1, pady=2, padx=2, sticky='nesw')

        self.point_image = tkinter.PhotoImage(file=ft.norm_path(os.path.join(assets_dir, "crosshair.gif")))
        self.btn_load = tkinter.Radiobutton(
            label_frame_controls,
            image=self.point_image,
            indicatoron=0,
            command=self._on_tool_select,
            value="point",
            variable=self.var_tool_sel,
        )
        self.btn_load.grid(row=r, column=2, pady=2, padx=2, sticky='nesw')
        r += 1

        self.separator_file_controls = tkinter.ttk.Separator(label_frame_controls)
        self.separator_file_controls.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky="nesw")
        r += 1

        # ======================= Info ========================== #
        self.label_mouse_coord = tkinter.Label(label_frame_controls, justify="left")
        self.label_mouse_coord.grid(row=r, column=0, columnspan=4, pady=2, padx=2, sticky='nesw')

        # ======================= Image Pane ========================== #
        r = 0

        self.canvas = tkinter.Canvas(label_frame_image)
        self.canvas.grid(sticky="nesw")
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def draw(self):
        """Updates the canvas by redrawing everything."""
        can_w, can_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        draw_loc = self.image_zero_offset

        # clear preview annotations
        for annotation in self.annotations:
            annotation.clear()

        # clear the old image
        if self.canvas_image is not None:
            self.canvas.delete(self.canvas_image)
            self.canvas_image = None

        # Load the new image.
        # This sets the zoom level of the image.
        # Also, TKinter is only documented to support gif, pgm, and ppm.
        _, tmpname = ft.get_temporary_file(".gif", text=False)
        image = self.image
        img_w, img_h = image.width, image.height
        new_w, new_h = int(img_w * self.zoom_scale), int(img_h * self.zoom_scale)
        sampling = Image.Resampling.NEAREST if self.zoom_scale > 1.0 else Image.Resampling.HAMMING
        image = image.resize((new_w, new_h), resample=sampling)
        image = image.crop(
            (int(-draw_loc.x[0]), int(-draw_loc.y[0]), int(-draw_loc.x[0] + can_w), int(-draw_loc.y[0] + can_h))
        )
        image.save(tmpname)
        self.tkimage = tkinter.PhotoImage(file=tmpname)

        # draw the new image
        self.canvas_image = self.canvas.create_image((0, 0), anchor=tkinter.NW, image=self.tkimage)

        # draw all annotations
        for annotation in self.annotations:
            annotation.draw(self.image_to_canvas_coordinate, self.canvas)

    def on_load_image(self):
        filetypes = [("Image", it.pil_image_formats_readable), ("Any", "*")]
        file_path_name_ext = tkinter.filedialog.askopenfilename(filetypes=filetypes, title="Load Image")
        if file_path_name_ext == "":
            return
        if not ft.file_exists(file_path_name_ext):
            tkinter.messagebox.showerror("File Load Error", f"Can't find image at \"{file_path_name_ext}\"")
            return

        try:
            image = Image.open(file_path_name_ext)
            self.image_path_name_ext = file_path_name_ext
            self.image = image
            self.tkimage = None

        except Exception as ex:
            tkinter.messagebox.showerror("File Load Error", f"Error loading image \"{file_path_name_ext}\": {repr(ex)}")
            self.image_path_name_ext = None
            self.image = None
            self.tkimage = None
            return

        # reset the zoom
        self._zoom_level = 0
        self._zoom_loc = p2.Pxy((self.image.width / 2, self.image.height / 2))

        # draw the new image
        self.draw()

    def _add_annotations(self, *annotations: AbstractImageAnnotation):
        """
        Adds the given annotations to be drawn onto the current image.

        The annotations are drawn and saved, as appropriate.
        """
        # add and draw the new annotations
        for annotation in annotations:
            self.annotations.append(annotation)
            annotation.draw(self.image_to_canvas_coordinate, self.canvas)

        # save, as necessary
        has_regular_annotations = any([not annotation.is_preview for annotation in annotations])
        if self.checkbtn_auto_save_checked.get():
            if has_regular_annotations:
                self._save()

    def _on_event_tool_handler(self, handler_function: str, *events: tkinter.Event):
        """
        For the active tool type of simple annotation, check if an instance of
        that type should be added based on the current event(s) and handler.

        Parameters
        ----------
        handler_function : str
            The name of the AbstractAnnotation function to call to create new annotations.
        events : list[tkinter.Event]
            One or more event(s) to be passed to the handler_function.
        """
        # clear out old annotations
        self.clear_annotations(preview_only=True)

        # choose the appropriate tool
        if self.tool == "zoom":
            self.zoom(handler_function, *events)
            return

        elif self.tool == "rect":
            tool = RectangleImageAnnotation

        elif self.tool == "point":
            tool = PointImageAnnotation

        # apply the tool
        if issubclass(tool, AbstractImageAnnotation):
            handler = getattr(tool, handler_function)
            new_annotation = handler(self.canvas_to_image_coordinate, *events)
            if new_annotation is not None:
                self._add_annotations(new_annotation)

    def on_mouse_down(self, event: tkinter.Event):
        self.mouse_down_event = event
        self._on_event_tool_handler("on_mouse_down", self.mouse_down_event)

    def on_mouse_move(self, event: tkinter.Event):
        if self.mouse_down_event is not None:
            self._on_event_tool_handler("on_mouse_move", self.mouse_down_event, event)

        mouse_coord = p2.Pxy((event.x, event.y))
        image_coord = self.canvas_to_image_coordinate(mouse_coord)
        self.label_mouse_coord.config(text=f"x: {int(image_coord.x[0])}, y: {int(image_coord.y[0])}")

    def on_mouse_up(self, event: tkinter.Event):
        if self.mouse_down_event is not None:
            self._on_event_tool_handler("on_mouse_up", self.mouse_down_event, event)
            self.mouse_down_event = None

    def clear_annotations(self, preview_only=False, force=False) -> bool:
        """
        Clears and removes (some or all) annotations.

        Parameters
        ----------
        preview_only : bool, optional
            If True then only remove preview annotations that haven't been
            finished yet. If False then remove all annotations. By default
            False.
        force : bool, optional
            If True then don't ask before removing all annotations, by default False

        Returns
        -------
        annotations_cleared: bool
            True if some number of annotations have been removed, or False if no
            annotations have been removed.
        """
        # clear the annotations from the canvas
        if preview_only:
            for annotation in copy.copy(self.preview_annotations):
                annotation.clear()
                self.annotations.remove(annotation)

        else:
            if not force:
                if len(self.regular_annotations) > 0:
                    ok = tkinter.messagebox.askokcancel(
                        title="Continue?",
                        message=f"This will remove {len(self.regular_annotations)} existing annotations. "
                        + "Are you sure you want to continue?",
                    )
                    if not ok:
                        return False

            for annotation in self.annotations:
                annotation.clear()
            self.annotations.clear()

        # stop drawing
        self.mouse_down_loc = None

        # redraw the canvas
        self.canvas.update()

        return True

    def on_save_annotations(self):
        """
        Callback for the save annotations button. Asks for a file to save to,
        then saves the current annotations to that file.
        """
        self.clear_annotations(preview_only=True)

        # get the save path
        old_path, old_name_ext = None, None
        if self.save_path_name_ext is not None:
            old_path, old_name, old_ext = ft.path_components(self.save_path_name_ext)
            old_name_ext = old_name + old_ext
        filetypes = [("CSV", "*.csv"), ("Any", "*")]
        file_path_name_ext = tkinter.filedialog.asksaveasfilename(
            filetypes=filetypes, title="Save Annotations", initialdir=old_path, initialfile=old_name_ext
        )
        if file_path_name_ext == "":
            return
        file_path, file_name, file_ext = ft.path_components(file_path_name_ext)

        # normalize the file extension
        if file_ext.lower() == "":
            file_ext = ".csv"
            file_path_name_ext = ft.norm_path(os.path.join(file_path, file_name + file_ext))

        # check for errors
        if not ft.directory_exists(file_path):
            tkinter.messagebox.showerror("File Save Error", f"Can't find directory \"{file_path}\"")
            return

        # save the file
        self.save_path_name_ext = file_path_name_ext
        self._save()

    def _save(self):
        """Forces a save of the current annotations to the previously selected save file."""
        if self.save_path_name_ext is not None:
            AbstractImageAnnotation.save_annotations_to_csv(
                self.regular_annotations, self.save_path_name_ext, overwrite=True
            )

    def on_load_annotations(self):
        """
        Callback for the load annotations button. Requests a file to load
        annotations from, unloads the existing annotations, and replaces them
        with the annotations from the chosen file.
        """
        filetypes = [("CSV", "*.csv"), ("Any", "*")]
        file_path_name_ext = tkinter.filedialog.askopenfilename(filetypes=filetypes, title="Load Annotations")
        if file_path_name_ext == "":
            return
        if not ft.file_exists(file_path_name_ext):
            tkinter.messagebox.showerror("File Load Error", f"Can't find annotations file at \"{file_path_name_ext}\"")
            return

        # try to read the file
        try:
            if self.clear_annotations(force=False):
                annotations = AbstractImageAnnotation.load_annotations_from_csv(file_path_name_ext, self.canvas)
                self._add_annotations(*annotations)
            else:
                return
        except Exception as ex:
            tkinter.messagebox.showerror("File Load Error", f"Can't read file at \"{file_path_name_ext}\": {repr(ex)}")
            return

    def _on_tool_select(self):
        if self.tool == "zoom":
            self.canvas.config({"cursor": "double_arrow"})

        elif self.tool == "rect":
            self.canvas.config({"cursor": "dotbox"})

        elif self.tool == "point":
            self.canvas.config({"cursor": "crosshair"})

    def canvas_to_image_coordinate(self, canvas_coord: p2.Pxy, clip_to_zero=False, clip_to_image_size=False) -> p2.Pxy:
        """
        Converts the given canvas coordinate (in screen pixels) to an image coordinate (in image pixels).

        Parameters
        ----------
        canvas_coord : p2.Pxy
            The screen-pixel coordinate on the canvas.
        clip_to_zero : bool, optional
            True to limit the returned value to 0+, by default False
        clip_to_image_size : bool, optional
            True to limit the returned value to image_size-, by default False

        Returns
        -------
        image_coordinate : p2.Pxy
            The corresponding image-pixel coordinate
        """
        offset = self.image_zero_offset
        canvas_pixel_xy = canvas_coord.x[0] / self.zoom_scale, canvas_coord.y[0] / self.zoom_scale
        image_pixel_offset_xy = -offset.x[0] / self.zoom_scale, -offset.y[0] / self.zoom_scale
        image_pixel_x, image_pixel_y = (
            canvas_pixel_xy[0] + image_pixel_offset_xy[0],
            canvas_pixel_xy[1] + image_pixel_offset_xy[1],
        )

        if clip_to_zero:
            image_pixel_x = max(image_pixel_x, 0)
            image_pixel_y = max(image_pixel_y, 0)

        if clip_to_image_size:
            image_pixel_x = min(image_pixel_x, self.image.width - 1)
            image_pixel_y = min(image_pixel_y, self.image.height - 1)

        return p2.Pxy((image_pixel_x, image_pixel_y))

    def image_to_canvas_coordinate(self, image_coord: p2.Pxy, clip_to_zero=False, clip_to_canvas_size=False):
        """
        Converts the given image coordinate (in image pixels) to a canvas coordinate (in screen pixels).

        Parameters
        ----------
        image_coord : p2.Pxy
            The screen-pixel coordinate on the canvas.
        clip_to_zero : bool, optional
            True to limit the returned value to 0+, by default False
        clip_to_canvas_size : bool, optional
            True to limit the returned value to canvas_size-, by default False

        Returns
        -------
        canvas_coordinate : p2.Pxy
            The corresponding screen-pixel coordinate
        """
        offset = self.image_zero_offset
        image_screen_xy = image_coord.x[0] * self.zoom_scale, image_coord.y[0] * self.zoom_scale
        canvas_screen_xy = image_screen_xy[0] + offset.x[0], image_screen_xy[1] + offset.y[0]

        if clip_to_zero:
            canvas_screen_xy[0] = max(canvas_screen_xy[0], 0)
            canvas_screen_xy[1] = max(canvas_screen_xy[1], 0)

        if clip_to_canvas_size:
            canvas_screen_xy[0] = min(canvas_screen_xy[0], self.canvas.winfo_width())
            canvas_screen_xy[1] = min(canvas_screen_xy[1], self.canvas.winfo_height())

        return p2.Pxy((canvas_screen_xy[0], canvas_screen_xy[1]))

    def zoom(self, method: str, *events: tkinter.Event):
        """Adjusts the zoom level and location based on the given event, then redraws the canvas."""
        event = events[-1]

        if event.type == tkinter.EventType.ButtonPress:
            if event.num == 1:
                image_coord = self.canvas_to_image_coordinate(
                    p2.Pxy((event.x, event.y)), clip_to_zero=True, clip_to_image_size=True
                )
                self._zoom_loc = image_coord
                self.zoom_level = max(min(self.zoom_level + 1, 10), -10)
                self.draw()

            elif event.num == 3:
                self.zoom_level = max(min(self.zoom_level - 1, 10), -10)
                self.draw()


if __name__ == "__main__":
    selector = ImageAnnotator()
