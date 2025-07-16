from opencsp.common.lib.cv.annotations.PointAnnotations import PointAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class HotspotAnnotation(PointAnnotations):
    """
    A class representing a hotspot annotation, likely created from a :py:class:`opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor` instance.

    The hotspot is the overall hottest location in an image, when accounting for the surrounding area. It may be the different from the centroid location or the single hottest pixel location.

    This class extends the `PointAnnotations` class to create a specific type of annotation
    that represents a hotspot, which can be rendered with a specific style and point location.

    Attributes
    ----------
    style : rcps.RenderControlPointSeq
        The rendering style of the hotspot annotation.
    point : p2.Pxy
        The point location of the hotspot annotation.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(self, style: rcps.RenderControlPointSeq = None, point: p2.Pxy = None):
        """
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
        >>> processor = HotspotImageProcessor(desired_shape=(30, 30))
        >>> input_image = CacheableImage.from_single_source("C:/path/to/image.png")
        >>> input_operable = SpotAnalysisOperable(input_image)
        >>> result = processor.process_operable(input_operable, is_last=True)[0]
        >>> hotspot = result.get_fiducials_by_type(HotspotAnnotation)[0]
        >>> lt.info(str(type(hotspot)))
        <class 'opencsp.common.lib.cv.annotations.HotspotAnnotation.HotspotAnnotation'>
        >>> lt.info(str(hotspot.origin))
        2D Point:
        array([[2517.],
               [2733.]])
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if style is None:
            style = rcps.RenderControlPointSeq(color="blue", marker="x", markersize=1)
        super().__init__(style, point)
