"""
GUI used to view annotated checkerboard corners.
"""

import tkinter

from numpy import ndarray
from PIL import Image, ImageTk


class ViewAnnotatedImages:
    """
    Class that controls a window used to view images with a next and previous
    button
    """

    def __init__(self, root: tkinter.Tk, images: list[ndarray], image_names: list[str]):
        """Instantiates the window from a given tkinter root

        Parameters
        ----------
        root : _type_
            Tkinter root
        images : list
            List of images
        image_names : list
            List of image names
        """
        # Save window parameters
        self.root = root
        self.images = images
        self.image_names = image_names
        self.idx_im = 0

        # Set title
        self.root.title('View Annotated Images')

        # Get screen height/width
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set height/width of image canvas
        self.width = screen_width - 80
        self.height = screen_height - 160

        # Set size of GUI
        self.root.geometry(f'{screen_width - 60:d}x{screen_height - 80:d}+20+0')

        # Set escape to exit window
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<Right>", lambda e: self.show_next())
        self.root.bind("<Left>", lambda e: self.show_prev())

        # Create image title
        self.var_title = tkinter.StringVar(value=image_names[self.idx_im])
        title = tkinter.Label(root, textvariable=self.var_title, font=('calibre', 15, 'bold'))

        # Create drawing canvas
        self.canvas = tkinter.Canvas(root, width=self.width, height=self.height)
        self.canvas.configure(background='white')

        self.canvas_image = self.canvas.create_image(0, 0, anchor='nw')

        # Create buttons
        btn_1 = tkinter.Button(root, text='Previous', width=20, command=self.show_prev)
        btn_2 = tkinter.Button(root, text='Next', width=20, command=self.show_next)

        # Place widgets
        title.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.canvas.grid(row=1, column=0, columnspan=2)
        btn_1.grid(row=2, column=0, sticky='e')
        btn_2.grid(row=2, column=1, sticky='w')

        # Show first image
        self.update_image()

        # Start window
        self.root.mainloop()

    def update_image(self):
        """
        Updates displayed image and image label
        """
        # Update image title
        self.var_title.set(self.image_names[self.idx_im])

        # Get current image
        image_array = self.images[self.idx_im]

        # Format RGB image array
        asp = self.width / self.height
        asp_cur = image_array.shape[1] / image_array.shape[0]
        if asp < asp_cur:
            width = self.width
            height = int(self.width / asp_cur)
        else:
            width = int(self.height * asp_cur)
            height = self.height

        image = Image.fromarray(image_array, 'RGB').resize(size=(width, height))
        image_tk = ImageTk.PhotoImage(image)

        # Display TK image
        self.canvas.imgref = image_tk
        self.canvas.itemconfig(self.canvas_image, image=image_tk)

    def show_next(self):
        """
        Show the next image.
        """
        # Update index
        self.idx_im += 1
        if self.idx_im >= len(self.images):
            self.idx_im = 0

        # Show image
        self.update_image()

    def show_prev(self):
        """
        Show the previous image.
        """
        # Update index
        self.idx_im -= 1
        if self.idx_im < 0:
            self.idx_im = len(self.images) - 1

        # Show index
        self.update_image()

    def close(self) -> None:
        """
        Closes window
        """
        self.root.destroy()
