import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


class RenderControlLightPath:
    """
    Render control for visualizing light paths.

    This class manages the rendering settings for light paths, allowing customization of various
    parameters related to the light path's length and rendering.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(
        self,
        init_length: float = 1,
        current_length: float = 1,
        end_at_plane: tuple[Pxyz, Uxyz] = None,
        line_render_control: float = rcps.thin(),
    ) -> None:
        """
        Render control for visualizing light paths.

        This class manages the rendering settings for light paths, allowing customization of various
        parameters related to the light path's length and rendering.

        Parameters
        ----------
        init_length : float, optional
            The initial length of the light path. By default, 1.0.
        current_length : float, optional
            The current length of the light path. By default, 1.0.
        end_at_plane : tuple[Pxyz, Uxyz], optional
            A tuple containing the position (Pxyz) and orientation (Uxyz) of the plane where the light path ends.
            By default, None.
        line_render_control : RenderControlPointSeq, optional
            Control settings for rendering the light path line. By default, `rcps.thin()`.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.init_length = init_length
        self.current_length = current_length
        self.end_at_plane = end_at_plane
        self.line_render_control: rcps.RenderControlPointSeq = line_render_control


# Common Configurations


def default_path() -> RenderControlLightPath:
    """
    Create a default render control for light paths.

    This function returns a `RenderControlLightPath` instance with default settings.

    Returns
    -------
    RenderControlLightPath
        An instance of `RenderControlLightPath` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlLightPath()


def end_at_plane(plane: tuple[Pxyz, Uxyz]):
    """
    Create a render control for a light path that ends at a specified plane.

    This function returns a `RenderControlLightPath` instance configured to end at the specified plane.

    Parameters
    ----------
    plane : tuple[Pxyz, Uxyz]
        A tuple containing the position (Pxyz) and orientation (Uxyz) of the plane where the light path ends.

    Returns
    -------
    RenderControlLightPath
        An instance of `RenderControlLightPath` configured to end at the specified plane.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlLightPath(end_at_plane=plane)
