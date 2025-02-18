import numpy as np
from warnings import warn

from opencsp.common.lib.geometry.Vxyz import Vxyz


class TranslationXYZ:
    """
    DEPRECATED: A class representing a translation in 3D space.

    This class is deprecated and should be replaced with the Vxyz class for handling translations.

    Attributes
    ----------
    trans_mtrx : np.ndarray
        A 3x1 matrix representing the translation vector in 3D space.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self) -> None:
        """
        DEPRECATED: Initializes the TranslationXYZ instance.

        This constructor creates a zero translation matrix and issues a deprecation warning.

        Raises
        ------
        DeprecationWarning
            Indicates that TranslationXYZ is deprecated and should be replaced with Vxyz.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        warn("TranslationXYZ is deprecated. Replace with Vxyz.", DeprecationWarning, stacklevel=2)
        self.trans_mtrx = np.zeros((3, 1))

    def from_vector(v: Vxyz):
        """
        DEPRECATED: Initializes the translation matrix from a Vxyz vector.

        Parameters
        ----------
        v : Vxyz
            A Vxyz object representing the translation vector.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        pass
