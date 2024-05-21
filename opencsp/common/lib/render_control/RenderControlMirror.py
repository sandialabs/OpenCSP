import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface
from opencsp.common.lib.geometry.RegionXY import Resolution


class RenderControlMirror():
    def __init__(self,
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
    return RenderControlMirror()


def low_res_mirror():
    return RenderControlMirror(resolution=Resolution.pixelX(5))
