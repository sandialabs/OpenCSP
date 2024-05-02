from opencsp.common.lib.cv.annotations.PointAnnotations import PointAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class HotspotAnnotation(PointAnnotations):
    def __init__(self, style: rcps.RenderControlPointSeq = None, point: p2.Pxy = None):
        if style is None:
            style = rcps.RenderControlPointSeq(color='blue', marker='x', markersize=1)
        super().__init__(style, point)
