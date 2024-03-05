import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface


class RenderControlMirror():
    def __init__(self,
                 centroid: bool = False,
                 surface_normals: bool = False,
                 resolution: int = 20,
                 norm_len: float = 1,
                 norm_res: int = 5,
                 norm_base_style: RenderControlPointSeq = rcps.marker(markersize=2),
                 surface_style: RenderControlSurface = RenderControlSurface(),
                 point_styles: RenderControlPointSeq = None
                 ) -> None:
        if point_styles == None:
            self.point_styles = rcps.marker(markersize=2)
        self.centroid = centroid
        self.surface_normals = surface_normals
        self.resolution = resolution
        self.norm_len = norm_len
        self.norm_res = norm_res
        self.norm_base_style = norm_base_style
        self.surface_style = surface_style
        self.point_styles = point_styles

# Common Configurations


def normal_mirror():
    return RenderControlMirror()


def low_res_mirror():
    return RenderControlMirror(resolution=5)
