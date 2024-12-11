from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
import opencsp.common.lib.cv.fiducials.PointFiducials as pf


class PointAnnotations(pf.PointFiducials, AbstractAnnotations):
    """
    A class representing point annotations.

    An example of this class is :py:class:`HotspotAnnotation`.

    This class extends both `PointFiducials` and `AbstractAnnotations` to provide functionality
    for managing and rendering point annotations, which can be used to mark specific locations
    in a visual representation.

    Inherits from:
    ---------------
    pf.PointFiducials : Implements methods from AbstractFiducial for point fiducials, and provides related attributes.
    AbstractAnnotations : Provides an abstract base for annotation classes.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    pass
