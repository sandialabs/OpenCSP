class RenderControlFunctionXY:
    """
    Render control for visualizing functions in XY space.

    This class manages the rendering settings for visualizing functions using heatmaps and contours.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(
        self,
        draw_heatmap: bool = True,
        draw_contours: bool = False,
        cmap: str = "jet",
        colorbar: bool = False,
        colorbar_min_max: tuple[float, float] = None,
        bounds: tuple[float, float, float, float] = None,  # unverified
    ) -> None:
        """
        Render control for visualizing functions in XY space.

        This class manages the rendering settings for visualizing functions using heatmaps and contours.

        Parameters
        ----------
        draw_heatmap : bool, optional
            Whether to draw a heatmap representation of the function. By default, True.
        draw_contours : bool, optional
            Whether to draw contour lines for the function. By default, False.
        cmap : str, optional
            Colormap to use for the heatmap. By default, "jet".
        colorbar : bool, optional
            Whether to display a colorbar alongside the heatmap. By default, False.
        colorbar_min_max : tuple[float, float], optional
            Minimum and maximum values for the colorbar. By default, None.
        bounds : tuple[float, float, float, float], optional
            Bounds for the visualization area (xmin, xmax, ymin, ymax). By default, None (unverified).
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.draw_heatmap = draw_heatmap
        self.draw_contours = draw_contours
        self.cmap = cmap
        self.colorbar = colorbar
        self.colorbar_min_max = colorbar_min_max
        self.bounds = bounds  # unverified


def countours(**kwargs):
    """
    Create a render control for drawing contours only.

    This function returns a `RenderControlFunctionXY` instance configured to draw only contour lines,
    without the heatmap representation.

    Parameters
    ----------
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFunctionXY`.

    Returns
    -------
    RenderControlFunctionXY
        An instance of `RenderControlFunctionXY` configured to display contours only.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    kwargs["draw_heatmap"] = False
    kwargs["draw_contours"] = True
    return RenderControlFunctionXY(**kwargs)


def heatmap(**kwargs):
    """
    Create a render control for drawing a heatmap.

    This function returns a `RenderControlFunctionXY` instance configured to draw a heatmap representation
    of the function.

    Parameters
    ----------
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFunctionXY`.

    Returns
    -------
    RenderControlFunctionXY
        An instance of `RenderControlFunctionXY` configured to display a heatmap.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    kwargs["draw_heatmap"] = True
    return RenderControlFunctionXY(**kwargs)


def heatmap_and_contours(**kwargs):
    """
    Create a render control for drawing both heatmap and contours.

    This function returns a `RenderControlFunctionXY` instance configured to draw both the heatmap
    representation and contour lines for the function.

    Parameters
    ----------
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFunctionXY`.

    Returns
    -------
    RenderControlFunctionXY
        An instance of `RenderControlFunctionXY` configured to display both heatmap and contours.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    kwargs["draw_contours"] = True
    kwargs["draw_heatmap"] = True
    return RenderControlFunctionXY(**kwargs)
