import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf


class RenderControlFramesNoDuplicates(rcvf.RenderControlVideoFrames):
    """
    Render control for the UFACET pipeline step FramesNoDuplicates.
    """

    def __init__(
        self,
        frame_format="JPG",
        clear_dir=True,  # Remove any existing files in the designated output directory.
        draw_example_frames=True,
        example_dpi=200,
        **kwargs
    ):
        """
        Render control for the UFACET pipeline step FramesNoDuplicates.

        This class manages the rendering settings for the FramesNoDuplicates step in the UFACET pipeline,
        allowing customization of frame formats and rendering options.

        Parameters
        ----------
        frame_format : str, optional
            The format of the frames to be rendered. By default, "JPG".
        clear_dir : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_example_frames : bool, optional
            Whether to draw example frames during rendering. By default, True.
        example_dpi : int, optional
            DPI (dots per inch) for the example frames. By default, 200.
        **kwargs : keyword arguments
            Additional keyword arguments passed to the parent class `RenderControlVideoFrames`.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if "inframe_format" not in kwargs:
            kwargs["inframe_format"] = frame_format
        if "outframe_format" not in kwargs:
            kwargs["outframe_format"] = frame_format
        super().__init__(
            clear_dir=clear_dir, draw_example_frames=draw_example_frames, example_dpi=example_dpi, **kwargs
        )


# COMMON CASES


def default():
    """
    Create a default render control for frames without duplicates.

    This function returns a `RenderControlFramesNoDuplicates` instance with default settings.

    Returns
    -------
    RenderControlFramesNoDuplicates
        An instance of `RenderControlFramesNoDuplicates` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFramesNoDuplicates()


def fast(**kwargs):
    """
    Create a fast render control for frames without duplicates.

    This function returns a `RenderControlFramesNoDuplicates` instance configured to skip drawing
    example frames, which can speed up the rendering process.

    Parameters
    ----------
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFramesNoDuplicates`.

    Returns
    -------
    RenderControlFramesNoDuplicates
        An instance of `RenderControlFramesNoDuplicates` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if "draw_example_frames" not in kwargs:
        kwargs["draw_example_frames"] = False
    return RenderControlFramesNoDuplicates(**kwargs)
