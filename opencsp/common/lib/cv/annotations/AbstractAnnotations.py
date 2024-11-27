import opencsp.common.lib.cv.fiducials.AbstractFiducials as af


class AbstractAnnotations(af.AbstractFiducials):
    """
    Annotations are applied to images to mark specific points of interest. Some
    examples of annotations might include:

    - The hotspot in a beam where light is the brightest
    - The power envelope for 90% of the light of a beam
    - Distances between two pixels
    - Measurement overlays

    This class has all the same properties as Fiducials.
    """

    pass
