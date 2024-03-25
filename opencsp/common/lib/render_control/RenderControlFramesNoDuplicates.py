"""


"""

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
        if "inframe_format" not in kwargs:
            kwargs["inframe_format"] = frame_format
        if "outframe_format" not in kwargs:
            kwargs["outframe_format"] = frame_format
        super().__init__(
            clear_dir=clear_dir, draw_example_frames=draw_example_frames, example_dpi=example_dpi, **kwargs
        )


# COMMON CASES


def default():
    return RenderControlFramesNoDuplicates()


def fast(**kwargs):
    if "draw_example_frames" not in kwargs:
        kwargs["draw_example_frames"] = False
    return RenderControlFramesNoDuplicates(**kwargs)
