"""


"""


class RenderControlFigure:
    """
    Render control for figures.
    """

    def __init__(
        self,
        tile=True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
        tile_array=(3, 2),  # (n_x, n_y)
        tile_square=False,  # Set to True for equal-axis 3d plots.
        figsize=(6.4, 4.8),  # inch.
        upper_left_xy=None,  # pixel.  (0,0) --> Upper left corner of screen.
        grid=True,
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

        Args:
            - tile (bool): True => Lay out figures in grid.  False => Place at upper_left or default screen center.  Default True
            - tile_array (tuple[int]): How many tiles across and down (n_x, n_y).  Default (3, 2)
            - tile_square (bool): Set to True for equal-axis 3d plots.  Default False
            - figsize (tuple[float]): Size of the figure in inches.  Default (6.4, 4.8)
            - upper_left_xy (tuple[int]): Pixel placement for the first tile.  (0,0) --> Upper left corner of screen.  Default None
            - grid (bool): Whether or not to draw grid lines.  Note: this value seems to be inverted.  Default True
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

        # Axis control.
        self.x_label = 'x (m)'
        self.y_label = 'y (m)'
        self.z_label = 'z (m)'
        self.grid = grid
