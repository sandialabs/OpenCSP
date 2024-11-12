from opencsp.common.lib.cv.annotations.PointAnnotations import PointAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class HotspotAnnotation(PointAnnotations):
    """
    A class representing a hotspot annotation in a graphical context.

    This class extends the `PointAnnotations` class to create a specific type of annotation
    that represents a hotspot, which can be rendered with a specific style and point location.

    Attributes
    ----------
    style : rcps.RenderControlPointSeq
        The rendering style of the hotspot annotation.
    point : p2.Pxy
        The point location of the hotspot annotation.
    """

    def __init__(self, style: rcps.RenderControlPointSeq = None, point: p2.Pxy = None):
        """
        A class representing a hotspot annotation in a graphical context.

        This class extends the `PointAnnotations` class to create a specific type of annotation
        that represents a hotspot, which can be rendered with a specific style and point location.

        Parameters
        ----------
        style : rcps.RenderControlPointSeq, optional
            The rendering style for the hotspot annotation. If not provided, a default style with
            blue color, 'x' marker, and a marker size of 1 will be used.
        point : p2.Pxy, optional
            The point location of the hotspot annotation, represented as a Pxy object. If not provided,
            the annotation will not have a specific point location.

        Examples
        --------
        >>> hotspot = HotspotAnnotation()
        >>> print(hotspot.style.color)
        'blue'

        >>> point = p2.Pxy(10, 20)
        >>> hotspot_with_point = HotspotAnnotation(point=point)
        >>> print(hotspot_with_point.point)
        Pxy(10, 20)
        """
        if style is None:
            style = rcps.RenderControlPointSeq(color='blue', marker='x', markersize=1)
        super().__init__(style, point)
