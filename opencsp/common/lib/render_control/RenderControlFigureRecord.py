"""


"""

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import PIL.Image as PilImage

from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class RenderControlFigureRecord:
    """
    Tracks figures that have been generated.
    """

    def __init__(
        self,
        name: str,  # Figure handle and title of figure window.
        title: str,  # Title of plot.
        caption: str,  # Caption for plot.
        figure_num: int,  # Number of this figure in generated sequence.  A unique key.
        figure: Figure,  # Matplotlib figure object.
        axis_control=None,
    ):  # Axis control instance used in figure_management.setup_figure
        """Register for a render figure.

        You probably don't want to instantiate this class directly.
        Most likely what you want is one of:
         - figure_management.setup_figure()
         - figure_management.setup_figure_for_3d_data()"""

        # in-situ imports to avoid import cycles
        import opencsp.common.lib.render_control.RenderControlAxis as rca

        axis_control: rca.RenderControlAxis = axis_control

        super(RenderControlFigureRecord, self).__init__()

        # Initialization.
        self.name = name
        self._title = title
        self.caption = caption
        self.figure_num = figure_num
        self.figure = figure
        self.axis_control = axis_control
        """ Axis control instance used in figure_management.setup_figure. Can be None|RenderControlAxis. """
        self.metadata: list[str] = []  # A list of standard string fields -- name, figure number, file path, etc.
        self.comments: list[str] = []  # A list of caller-defined strings, to be filled in later.
        self.axis: plt.Axes = None  # Matplotlib plot axes object.  Set later.
        self.view: View3d = None  # View3d object.                Set later.
        self.equal = None  # Whether to make axes equal.   Set later.
        self.x_limits = None  # X-axis limits (optional).     Set later.
        self.y_limits = None  # Y-axis limits (optional).     Set later.
        self.z_limits = None  # Z-axis limits (optional).     Set later.

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, new_title: str):
        self._title = new_title
        self.axis.set_title(new_title)

    def close(self):
        """Closes any matplotlib window opened with this instance's view"""
        if self.view != None:
            self.view.close()

    def clear(self):
        """
        Clears the old plot data without deleting the window, listeners, or orientation. Useful for updating a plot
        interactively.
        """
        # self.fig_record.figure.clear(keep_observers=True) <-- not doing this, clears everything except window

        # Register data to be re-assigned
        xlabel = self.axis.get_xlabel()
        ylabel = self.axis.get_ylabel()
        if hasattr(self.axis, 'get_zlabel'):
            zlabel = self.axis.get_zlabel()

        # Clear the previous graph
        self.axis.clear()

        # Clear the previous title
        if self.axis.title is not None:
            try:
                self.axis.title.remove()
            except Exception:
                pass

        # Re-assign necessary data
        self.axis.set_xlabel(xlabel)
        self.axis.set_ylabel(ylabel)
        if hasattr(self.axis, 'get_zlabel'):
            self.axis.set_zlabel(zlabel)

    def add_metadata_line(self, metadata_line: str) -> None:
        self.metadata.append(metadata_line)

    def add_comment_line(self, comment_line: str) -> None:
        self.comments.append(comment_line)

    def print_comments(self):
        for comment_line in self.comments:
            lt.info(comment_line)

    def to_array(self):
        return self.figure_to_array(self.figure)

    @staticmethod
    def figure_to_array(figure: Figure):
        # Force matplotlib to render to it's internal buffer
        figure.canvas.draw()

        # Convert the buffer to a numpy array
        return np.asarray(figure.canvas.buffer_rgba())

    def save(self, output_dir: str, output_file_body: str = None, format: str = None, dpi=600, close_after_save=True):
        """Saves this figure record to an image file.

        Args:
            - output_dir (str): The directory to save to.
            - output_file_body (str): Name of the file to save to. None for self.name. Defaults to None.
            - format (str): The file format to save with. None for "svg". Defaults to None.
            - dpi (int): Dots per inch used to format the figure. Defaults to 600.
            - close_after_save (bool): False: keep the matplotlib plot open after saving. True: close the plot after saving. Defaults to True.

        Returns:
            - str: output_figure_dir_body_ext, the image file
            - str: output_figure_text_dir_body_ext, the description of the image file (metadata, title, caption, comments, etc)
        """
        try:
            if format is None:
                format = 'svg'  # Default format was previously 'png'.
            orig_format = format
            if format.lower() == "gif":
                format = "png"

            # Ensure the output destination is available.
            if not (os.path.exists(output_dir)):
                os.makedirs(output_dir)

            # Contruct the output filename.
            if not output_file_body:
                # Add a prefix to ensure unique filenames.      # This is currently done in figure_management.py
                # prefix = '{0:03d}_'.format(self.figure_num)   #
                # output_file_body = prefix + self.name         #
                output_file_body = ft.convert_string_to_file_body(self.name)
            output_figure_body = output_file_body
            # Join with output directory.
            output_figure_dir_body = os.path.join(output_dir, output_figure_body)

            # If this is a 3-d plot, add the projection choice.
            if self.view != None:
                output_figure_dir_body_ext = self.view.save(output_dir, output_figure_body, format=format, dpi=dpi)
            else:
                # Make the figure current.
                plt.figure(self.name)
                # Save the current figure.
                output_figure_dir_body_ext = output_figure_dir_body + '.' + format
                # TODO RCB: THIS CODE IS DEPRECATED AS OF 11/20/2022.  ONCE IT'S CLEAR WE DON'T WANT IT, DELETE THE FOLLOWING COMMENTED-OUT LINES.
                # if ft.file_exists(output_figure_dir_body_ext):
                #     print('Skipping save of existing figure: ' + output_figure_dir_body_ext)
                # else:
                output_figure_dir_body_ext = output_figure_dir_body + '.' + format
                lt.info('In RenderControlFigureRecord.save(), saving figure: ' + output_figure_dir_body_ext)
                plt.savefig(output_figure_dir_body_ext, format=format, dpi=dpi)
        finally:
            # Close figure if desired.
            if self.view is not None:
                if close_after_save:
                    plt.close()

        # Convert to gif, as necessary
        # Matplotlib doesn't support the gif format, so we convert to it after the fact
        if orig_format.lower() == "gif":
            im = PilImage.open(output_figure_dir_body_ext)
            png_file = output_figure_dir_body_ext
            output_figure_dir_body_ext = png_file.rstrip("." + format) + "." + orig_format
            im.save(output_figure_dir_body_ext)
            ft.delete_file(png_file)

        # Save the figure explanation.
        output_figure_dir, output_figure_body, output_figure_ext = ft.path_components(output_figure_dir_body_ext)
        output_figure_text_body_ext = output_figure_body + '.txt'
        output_figure_text_dir_body_ext = os.path.join(output_figure_dir, output_figure_text_body_ext)
        lt.info('Saving figure text: ' + output_figure_text_dir_body_ext)
        with open(output_figure_text_dir_body_ext, 'w') as output_stream:
            # Save the figure metadata.
            if len(self.metadata) > 0:
                output_stream.write('Metadata:\n')
                for metadata_line in self.metadata:
                    output_stream.write(metadata_line + '\n')
                output_stream.write('\n')

            # Save the figure title.
            if self.title is not None:
                output_stream.write('Title:\n')
                output_stream.write(self.title + '\n')
                output_stream.write('\n')

            # Save the figure caption.
            if self.caption is not None:
                output_stream.write('Caption:\n')
                output_stream.write(self.caption + '\n')
                output_stream.write('\n')

            # Save the figure comments.
            if len(self.comments) > 0:
                output_stream.write('Comments:\n')
                for comment_line in self.comments:
                    output_stream.write(comment_line + '\n')
                output_stream.write('\n')

        # Return the files created.
        return output_figure_dir_body_ext, output_figure_text_dir_body_ext
