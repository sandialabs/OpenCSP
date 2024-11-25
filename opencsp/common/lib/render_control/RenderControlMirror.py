import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface
from opencsp.common.lib.geometry.RegionXY import Resolution


class RenderControlMirror:
    """
    A class for controlling the rendering of a mirror in a graphical environment.

    This class allows for the configuration of various visual aspects of a mirror,
    including its centroid, surface normals, resolution, and styles for rendering.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(
        self,
        centroid: bool = False,
        surface_normals: bool = False,
        resolution: Resolution = None,
        norm_len: float = 1,
        norm_res: int = 5,
        norm_base_style: RenderControlPointSeq = rcps.marker(markersize=2),
        surface_style: RenderControlSurface = RenderControlSurface(),
        point_styles: RenderControlPointSeq = None,
        number_of_edge_points: int = 20,
    ) -> None:
        """
        Initializes a RenderControlMirror object with the specified parameters.

        Parameters
        ----------
        centroid : bool, optional
            If True, renders the centroid of the mirror (default is False).
        surface_normals : bool, optional
            If True, renders the surface normals of the mirror (default is False).
        resolution : Resolution, optional
            The resolution of the rendering, specified as a Resolution object (default is None).
        norm_len : float, optional
            The length of the normals to be rendered (default is 1).
        norm_res : int, optional
            The resolution of the normals (default is 5).
        norm_base_style : RenderControlPointSeq, optional
            The style for rendering the normals (default is a marker with size 2).
        surface_style : RenderControlSurface, optional
            The style for rendering the surface of the mirror (default is a new RenderControlSurface object).
        point_styles : RenderControlPointSeq, optional
            The styles for rendering points on the mirror (default is None, which sets it to a marker with size 2).
        number_of_edge_points : int, optional
            The number of edge points to be rendered (default is 20).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if point_styles == None:
            self.point_styles = rcps.marker(markersize=2)
        if resolution is None:
            resolution = Resolution.pixelX(number_of_edge_points)
        self.resolution = resolution
        self.number_of_edge_points = number_of_edge_points
        self.centroid = centroid
        self.surface_normals = surface_normals
        self.norm_len = norm_len
        self.norm_res = norm_res
        self.norm_base_style = norm_base_style
        self.surface_style = surface_style
        self.point_styles = point_styles


# Common Configurations


def normal_mirror():
    """
    Creates a default RenderControlMirror object with standard settings.

    Returns
    -------
    RenderControlMirror
        A RenderControlMirror object initialized with default parameters.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RenderControlMirror()


def low_res_mirror():
    """
    Creates a RenderControlMirror object with low resolution.

    Returns
    -------
    RenderControlMirror
        A RenderControlMirror object initialized with a resolution of 5 pixels in the x direction.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RenderControlMirror(resolution=Resolution.pixelX(5))
