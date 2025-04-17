import opencsp.common.lib.render_control.RenderControlLightPath as rclp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class RenderControlRayTrace:
    """
    A class for controlling the rendering of ray traces in a graphical environment.

    This class manages the rendering settings for light paths during ray tracing.

    Parameters
    ----------
    light_path_control : RenderControlLightPath, optional
        The control settings for rendering light paths (default is the default light path control).
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, light_path_control=rclp.default_path()) -> None:
        self.light_path_control = light_path_control


# Common Configurations


def init_current_lengths(init_len=1, current_len=1, line_render: float = rcps.thin()):
    """
    Initializes a RenderControlRayTrace object with specified initial and current lengths for light paths.

    This function creates a RenderControlRayTrace object with a RenderControlLightPath configured
    with the provided initial and current lengths.

    Parameters
    ----------
    init_len : float, optional
        The initial length of the light paths (default is 1).
    current_len : float, optional
        The current length of the light paths (default is 1).

    Returns
    -------
    RenderControlRayTrace
        A RenderControlRayTrace object initialized with the specified lengths.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RenderControlRayTrace(
        rclp.RenderControlLightPath(init_length=init_len, current_length=current_len, line_render_control=line_render)
    )
