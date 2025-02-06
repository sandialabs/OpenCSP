"""


"""


class RenderControlFigure:
    """
    Render control for figures.
    """

    def __init__(
        self,
        tile=True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
        tile_array: tuple[int, int] = (3, 2),  # (n_x, n_y)
        tile_square=False,  # Set to True for equal-axis 3d plots.
        figsize=(6.4, 4.8),  # inch.
        upper_left_xy=None,  # pixel.  (0,0) --> Upper left corner of screen.
        grid=True,
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
            How many tiles across and down (n_x, n_y).
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
        maximize : bool, optional
            Whether the figure should be maximized (made full screen) as soon as
            it is made visible. Ignored if tile is True. Default False.
        """

        super(RenderControlFigure, self).__init__()

        # Figure management.
        self.figure_names = []
        self.tile = tile
        self.tile_array = tile_array
        self.tile_square = tile_square

        # Figure size and placement.
        self.figsize = figsize
        self.upper_left_xy = upper_left_xy
        self.maximize = maximize

        # Axis control.
        self.x_label = 'x (m)'
        self.y_label = 'y (m)'
        self.z_label = 'z (m)'
        self.grid = grid
