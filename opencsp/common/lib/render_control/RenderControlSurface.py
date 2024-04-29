import opencsp.common.lib.render.color as cl
import opencsp.common.lib.tool.log_tools as lt


class RenderControlSurface:
    def __init__(
        self,
        draw_title=True,
        color: str | None = None,
        color_map: str | None = "viridis",
        alpha: float = 1.0,
        edgecolor='black',
        linewidth=0.05,
        contour: None | bool | str = True,
        contour_color_map: str | None = None,
    ) -> None:
        """
        Render control information for how to style surface plots (see View3d function plot_surface and plot_trisurface).

        Parameters
        ----------
        draw_title : bool, optional
            If True then the title will be drawn on graph, default is True
        color : str | None, optional
            The color of the plot if not using a color map, example "silver", by default _PlotColors().blue
        color_map : str | None, optional
            The color map of the plot to help decern different plot values. See
            https://matplotlib.org/stable/gallery/color/colormap_reference.html for common options. By default "viridis".
        alpha : float, optional
            The opacity of the plot between 0 (fully transparent) and 1 (fully opaque), by default 1.0
        edgecolor: str, optional
            The color to use for the lines between the faces, default is 'black'
        linewidth: float, optional
            The width of the edge lines between the faces in points, default is 0.3
        contour : None | bool | str, optional
            If False or None, then don't include a contour plot alongside the 3d plot. If True, then draw a 2D contour
            plot below the 3D surface plot (on z-axis). If a string, can be any combination of 'x', 'y', and 'z'.
            Default is True.
        contour_color_map : str, optional
            If set, then this determines the color map for the contour. If None, then use the same color map for the
            contour as for the surface. If None and the color_map argument is also None, then we should create a custom
            color map based on the given color. Default is None.
        """
        self.draw_title = draw_title
        self.alpha = alpha
        self.antialiased = False if self.alpha > 0.99 else None
        self.color = color
        self.color_map = color_map
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.contour = False
        self.contour_color_map = contour_color_map
        self.contour_alpha = 0.7
        self.contours = {'x': False, 'y': False, 'z': False}

        # make sure one of "color" or "color_map" is set
        if self.color is None:
            if self.color_map is None:
                self.color = cl._PlotColors().blue

        # determine the type of contour to be drawn
        if contour is None or contour == False:
            self.contour = False
        elif contour == True:
            self.contours['x'] = True
        elif isinstance(contour, str):
            self.contour = True
            for axis in contour:
                axis = axis.replace('p', 'x').replace('q', 'y').replace('r', 'z')
                if axis not in self.contours:
                    lt.error_and_raise(
                        ValueError,
                        "Error in RenderControlSurface(): "
                        + f"unknown axis in {contour=} (only 'x', 'y', 'z', 'p', 'q', or 'r' are allowed)",
                    )
                self.contours[axis] = True

        # set the "contour_color_map" based on "color_map" or "color"
        if self.contour:
            if self.contour_color_map is None:
                if self.color_map is not None:
                    self.contour_color_map = self.color_map
                else:
                    # TODO create a custom color map based on the color
                    raise NotImplementedError()
