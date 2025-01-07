import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxyz import Vxyz


class TransformXYZ:
    """
    Representation of a 3D homogeneous spatial transform.

    A TransformXYZ object encapsulates a 4x4 homogeneous transformation matrix,
    which includes both rotation and translation components. This class provides methods
    for creating transformations, applying them to vectors, and obtaining their inverse.

    Properties
    ----------
    matrix : np.ndarray
        The 4x4 matrix representation of the transformation.
    R : Rotation
        The rotation component of the transformation.
    R_matrix : np.ndarray
        The 3x3 rotation matrix.
    V : Vxyz
        The translation component of the transformation.
    V_matrix : np.ndarray
        The translation vector as a length 3 array.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, matrix: np.ndarray):
        """
        Initializes a TransformXYZ object with the given transformation matrix.

        Parameters
        ----------
        matrix : np.ndarray
            A 4x4 homogeneous transformation matrix.

        Raises
        ------
        ValueError
            If the input matrix does not have shape (4, 4).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Check 4x4 shape
        if matrix.shape != (4, 4):
            raise ValueError('Input matrix must have shape 4x4.')

        # Save matrix data
        self._matrix = matrix.astype(float)

    def __repr__(self):
        return '3D Transform:\n' + self._matrix.__repr__()

    def __mul__(self, T):
        # Check input type
        if not isinstance(T, TransformXYZ):
            raise TypeError(f'Type, {type(self)}, cannot be multipled by type, {type(T)}.')

        return TransformXYZ(self._matrix @ T._matrix)

    @classmethod
    def from_zero_zero(cls):
        """
        Returns zero translation and zero rotation TransformXYZ.

        Returns
        -------
        TransformXYZ.

        """
        return cls(np.eye(4))

    @classmethod
    def identity(cls):
        """
        Returns the identity tranformation.
        Alias for TransformXYZ.from_zero_zero().

        Returns
        -------
        TransformXYZ.

        """
        return cls(np.eye(4))

    @classmethod
    def from_R_V(cls, R: Rotation, V: Vxyz):
        """
        TransformXYZ from a 3D rotation and a 3D translation.

        Parameters
        ----------
        R : Rotation
            3D rotation.
        V : Vxyz
            3D translation.

        Returns
        -------
        TransformXYZ.

        """
        # Check V is length 1
        if len(V) != 1:
            raise ValueError('Input V must be a length 1 vector.')
        # Create matrix
        matrix = np.eye(4)
        # Add rotation and translation components
        matrix[:3, :3] = R.as_matrix()
        matrix[:3, 3:4] = V.data

        return cls(matrix)

    @classmethod
    def from_R(cls, R: Rotation):
        """
        TransformXYZ from a 3D rotation. Assumes zero translation.

        Parameters
        ----------
        R : Rotation
            3D rotation.

        Returns
        -------
        TransformXYZ.

        """
        V = Vxyz((0, 0, 0))
        return cls.from_R_V(R, V)

    @classmethod
    def from_V(cls, V: Vxyz):
        """
        TransformXYZ from a 3D translation. Assumes zero rotation.

        Parameters
        ----------
        V : Vxyz
            3D translation.

        Returns
        -------
        TransformXYZ.

        """
        R = Rotation.from_rotvec([0, 0, 0])
        return cls.from_R_V(R, V)

    @property
    def matrix(self):
        """
        4x4 matrix representation of transform.

        Returns
        -------
        np.ndarray
            4x4 transform data.

        """
        return self._matrix

    @property
    def R(self) -> Rotation:
        """
        Rotation component of 3D transform.

        Returns
        -------
        Rotation.

        """
        return Rotation.from_matrix(self.R_matrix)

    @property
    def R_matrix(self) -> np.ndarray:
        """
        Rotation component of 3D transform.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        """
        return self._matrix[:3, :3]

    @property
    def V(self) -> Vxyz:
        """
        Translation component of 3D transform

        Returns
        -------
        Vxyz
            3D translation vector.

        """
        return Vxyz(self.V_matrix)

    @property
    def V_matrix(self) -> np.ndarray:
        """
        Translation component of 3D transform

        Returns
        -------
        np.ndarray
            3D translation vector as a length 3 array.

        """
        return self._matrix[:3, 3:4]

    def apply(self, V: Vxyz) -> Vxyz:
        """
        Applies 3D spatial transform to input vector by rotating then
        translating. Returns a rotated copy of the input vector.

        Parameters
        ----------
        V : Vxyz
            Length n input vector.

        Returns
        -------
        Vxyz
            Transformed vector.

        """
        # Rotation
        V_out = V.rotate(self.R)
        # Translation
        V_out += self.V
        return V_out

    def inv(self) -> 'TransformXYZ':
        """Returns inverse transformation

        Returns
        -------
        TransformXYZ
            Inverse transformation
        """
        mat_inv = np.linalg.inv(self.matrix)
        return TransformXYZ(mat_inv)

    def copy(self) -> 'TransformXYZ':
        """Returns a copy of the transform"""
        return TransformXYZ(self.matrix.copy())
