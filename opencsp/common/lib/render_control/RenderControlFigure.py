"""


"""

import matplotlib.pyplot
import numpy as np


class RenderControlFigure:
    """
    Render control for figures.
    """

    def __init__(
        self,
        tile=True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
        tile_array: tuple[int, int] | None = (3, 2),  # (n_x, n_y)
        tile_square=False,  # Set to True for equal-axis 3d plots.
        num_figures: int = None,
        figsize=(6.4, 4.8),  # inch.
        upper_left_xy=None,  # pixel.  (0,0) --> Upper left corner of screen.
        grid=True,
        draw_whitespace_padding=True,
        maximize=False,
    ):  # Whether or not to draw grid lines.
        """Set of controls for how to render figures.

        Example::

            plt.close('all')
            fm.reset_figure_management()
            figure_control = rcfg.RenderControlFigure()
            axis_control = rca.RenderControlAxis(x_label='Time (s)', y_label='Collected Energy (W)')
            view_spec_2d = vs.view_spec_xy()

            fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control, view_spec_2d, name="energy_over_time", code_tag=f"{__file__}.main()")
            style = rcps.RenderControlPointSeq(color=None, marker=None)
            view = fig_record.view

            view.draw_pq_list(energy_values, style=style)
            view.show(block=True)

        Params:
        -------
        tile : bool, optional
            True => Lay out figures in grid. False => Place at upper_left or
            default screen center. If True, then figsize, upper_left_xy, and
            maximize are ignored. Default True
        tile_array : tuple[int] | None, optional
            How many tiles across and down (n_x, n_y). If None then the tiling
            pattern is guessed based on num_figures. Default (3, 2)
        tile_square : bool, optional
            Set to True for equal-axis 3d plots. Default False
        figsize : tuple[float], optional
            Size of the figure in inches. Ignored if tile is True. Default (6.4, 4.8)
        upper_left_xy : tuple[int], optional
            Pixel placement for the first tile. (0,0) --> Upper left corner of
            screen. Ignored if tile is True. Default None
        grid : bool, optional
            Whether or not to draw grid lines. Note: this value seems to be
            inverted. Default True
        draw_whitespace_padding : bool, optional
            If False then don't pad the figure with whitespace. Default True
        maximize : bool, optional
            Whether the figure should be maximized (made full screen) as soon as
            it is made visible. Ignored if tile is True. Default False.
        """

        super(RenderControlFigure, self).__init__()

        # Standardize arguments
        if figsize is None:
            figsize = (6.4, 4.8)
        if tile_array is None:
            if num_figures is not None:
                tile_array = self.num_tiles_4x3aspect(num_figures)

        # Figure management.
        self.figure_names = []
        self.tile = tile
        self.tile_array = tile_array
        self.tile_square = tile_square
        self.num_figures = num_figures

        # Figure size and placement.
        self.figsize = figsize
        self.upper_left_xy = upper_left_xy
        self.maximize = maximize

        # Axis control.
        self.x_label = 'x (m)'
        self.y_label = 'y (m)'
        self.z_label = 'z (m)'
        self.grid = grid

        # Whitespace control
        self.draw_whitespace_padding = draw_whitespace_padding

    @staticmethod
    def num_tiles_4x3aspect(num_figures: int):
        if num_figures <= 1:
            n_rows = 1
            n_cols = 1
        elif num_figures <= 2:
            n_rows = 1
            n_cols = 2
        elif num_figures <= 8:
            n_rows = 2
            n_cols = int(np.ceil(num_figures / 2))
        elif num_figures <= 12:
            n_rows = 3
            n_cols = int(np.ceil(num_figures / 3))
        else:
            n_rows = int(np.floor(np.sqrt(num_figures)))
            n_cols = int(np.ceil(num_figures / n_rows))

        return n_rows, n_cols

    @staticmethod
    def full_screen_inches(remove_taskbar=True) -> tuple[float, float]:
        """Returns the number of inches (width and height) necessary to show a graph fullscreen."""
        dpi: float = matplotlib.pyplot.rcParams['figure.dpi']

        # get the size of the screen
        fig = matplotlib.pyplot.figure('RenderControlFigure.full_screen')
        try:
            import tkinter

            window: tkinter.Tk = fig.canvas.manager.window
            screen_y = window.winfo_screenheight()
            screen_x = window.winfo_screenwidth()
        except Exception as ex:
            return matplotlib.pyplot.rcParams['figure.figsize']
        finally:
            matplotlib.pyplot.close(fig)

        # account for the windows taskbar
        if remove_taskbar:
            # TODO detect the location and height of the taskbar
            screen_y -= 100

        return screen_x / dpi, screen_y / dpi

    @staticmethod
    def pixel_resolution_inches(width: int, height: int) -> tuple[float, float]:
        """Returns the number of inches (width and height) necessary to show the given pixel dimensions."""
        dpi: float = matplotlib.pyplot.rcParams['figure.dpi']
        return width / dpi, height / dpi
