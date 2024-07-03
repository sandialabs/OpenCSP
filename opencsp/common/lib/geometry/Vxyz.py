"""Three dimensional vector representation
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxy import Vxy


class Vxyz:
    """
    3D vector class to represent 3D points/vectors.

    Parameters
    ----------
    data : array-like
        The 3d point data: 3xN array, length 3 tuple, length 3 list
    dtype : data type, optional
        Data type. The default is float.

    """

    def __init__(self, data, dtype=float):
        """
        3D vector class to represent 3D points/vectors.

        To represent a single vector::

            x = 1
            y = 2
            z = 3
            vec = Vxyz(np.array([[x], [y], [z]])) # same as vec = Vxyz([x, y, z])
            print(vec.x) # [1.]
            print(vec.y) # [2.]
            print(vec.z) # [3.]

        To represent a set of vectors::

            vec1 = [1, 2, 3]
            vec2 = [4, 5, 6]
            vec3 = [7, 8, 9]
            zipped = list(zip(vec1, vec2, vec3))
            vecs = Vxyz(np.array(zipped))
            print(vec.x) # [1. 4. 7.]
            print(vec.y) # [2. 5. 8.]
            print(vec.z) # [3. 6. 9.]

            # or this equivalent method
            xs = [1, 4 ,7]
            ys = [2, 5, 8]
            zs = [3, 6, 9]
            vecs = Vxyz((xs, ys, zs))

        Parameters
        ----------
        data : array-like
            The 3d point data: 3xN array, length 3 tuple, length 3 list. If a Vxy, then the data will be padded with 0s
            for 'z'.
        dtype : data type, optional
            Data type. The default is float.

        """
        # Check input shape
        if isinstance(data, np.ndarray):
            data = data.squeeze()
            if np.ndim(data) not in [1, 2]:
                raise ValueError('Input data must have 1 or 2 dimensions if ndarray.')
            elif np.ndim(data) == 2 and data.shape[0] != 3:
                raise ValueError('First dimension of 2-dimensional data must be length 3 if ndarray.')
        elif isinstance(data, Vxy):
            data = np.pad(data.data, ((0, 1), (0, 0)))
        elif len(data) != 3:
            raise ValueError('Input data must have length 3.')

        # Save and format data
        self._data = np.array(data, dtype=dtype).reshape((3, -1))

    @property
    def data(self):
        """
        An array with shape (3, N), where N is the number of 3D vectors in this instance.
        """
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def x(self) -> np.ndarray:
        return self._data[0, :]

    @property
    def y(self) -> np.ndarray:
        return self._data[1, :]

    @property
    def z(self) -> np.ndarray:
        return self._data[2, :]

    def as_Vxyz(self):
        return self

    @classmethod
    def _from_data(cls, data, dtype=float):
        return cls(data, dtype)

    def _check_is_Vxyz(self, v_in):
        """
        Checks if input data is instance of Vxyz for methods that require this
        type.

        """
        if not isinstance(v_in, Vxyz):
            raise TypeError(f'Input operand must be {Vxyz}, not {type(v_in)}')

    def __add__(self, v_in):
        """
        Element wise addition. Operand 1 type must be Vxyz
        """
        self._check_is_Vxyz(v_in)
        return self._from_data(self._data + v_in.data)

    def __sub__(self, v_in):
        """
        Element wise subtraction. Operand 1 type must be Vxyz
        """
        self._check_is_Vxyz(v_in)
        return self._from_data(self._data - v_in.data)

    def __mul__(self, data_in):
        """
        Element wise addition. Operand 1 type must be int, float, or Vxyz.
        """
        if type(data_in) in [int, float, np.float32, np.float64, np.int32, np.int64]:
            return self._from_data(self._data * data_in)
        elif isinstance(data_in, Vxyz):
            return self._from_data(self._data * data_in.data)
        elif type(data_in) is np.ndarray:
            return self._from_data(self._data * data_in)
        else:
            raise TypeError(f'Vxyz cannot be multipled by type, {type(data_in)}.')

    def __getitem__(self, key) -> 'Vxyz':
        # Check that only one dimension is being indexed
        if np.size(key) > 1 and any(isinstance(x, slice) for x in key):
            raise ValueError('Can only index over one dimension.')

        return self._from_data(self._data[:, key], dtype=self.dtype)

    def __repr__(self):
        return '3D Vector:\n' + self._data.__repr__()

    def __len__(self):
        return self._data.shape[1]

    def __neg__(self):
        return self._from_data(-self._data, dtype=self.dtype)

    def _magnitude_with_zero_check(self) -> np.ndarray:
        """
        Returns ndarray of normalized vector data.

        Returns
        -------
        ndarray
            1d vector of normalized data

        """
        mag = self.magnitude()

        if np.any(mag == 0):
            raise ValueError('Vector contains zero vector, cannot normalize.')

        return mag

    def normalize(self):
        """
        Returns copy of normalized vector.

        Returns
        -------
        Vxyz
            Normalized vector.

        """
        V_out = self._from_data(self._data.copy())
        V_out.normalize_in_place()
        return V_out

    def normalize_in_place(self) -> None:
        """
        Normalizes vector. Replaces data in Vxyz object with normalized data.

        """
        self._data /= self._magnitude_with_zero_check()

    def magnitude(self) -> npt.NDArray[np.float_]:
        """
        Returns magnitude of each vector.

        Returns
        -------
        np.ndarray
            Length n ndarray of vector magnitudes.

        """
        return np.sqrt(np.sum(self._data**2, 0))

    def rotate(self, R: Rotation):
        """
        Returns a copy of the rotated vector rotated about the coordinate
        system origin.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.

        Returns
        -------
        Vxyz
            Rotated vector.

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f'Rotaion must be type {Rotation}, not {type(R)}')

        V_out = self._from_data(self._data.copy())
        V_out.rotate_in_place(R)
        return V_out

    def rotate_about(self, R: Rotation, V_pivot):
        """
        Returns a copy of the rotated vector rotated about the given pivot
        point.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.
        V_pivot : Vxyz
            Pivot point to rotate about.

        Returns
        -------
        Vxyz
            Rotated vector.

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f'Rotaion must be type {Rotation}, not {type(R)}')
        self._check_is_Vxyz(V_pivot)

        V_out = self._from_data(self._data.copy())
        V_out.rotate_about_in_place(R, V_pivot)
        return V_out

    def rotate_in_place(self, R: Rotation) -> None:
        """
        Rotates vector about the coordinate system origin. Replaces data in Vxyz
        object with rotated data.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.

        Returns
        -------
        None

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f'Rotation must be type {Rotation}, not {type(R)}')

        self._data = R.apply(self._data.T).T

    def rotate_about_in_place(self, R: Rotation, V_pivot) -> None:
        """
        Rotates about the given pivot point. Replaces data in Vxyz object with
        rotated data.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.
        V_pivot : Vxyz
            Pivot point to rotate about.

        Returns
        -------
        None

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f'Rotaion must be type {Rotation}, not {type(R)}')
        self._check_is_Vxyz(V_pivot)

        # Center pivot to origin
        self._data -= V_pivot.data
        # Rotate about origin
        self.rotate_in_place(R)
        # Recenter pivot point
        self._data += V_pivot.data

    def dot(self, V) -> np.ndarray:
        """
        Calculated dot product. Size of input data must broadcast with size of
        data.

        Parameters
        ----------
        V : Vxyz
            Input vector.

        Returns
        -------
        np.ndarray
            Length n array of dot product values.

        """
        # Check inputs
        self._check_is_Vxyz(V)

        return (self._data * V.data).sum(axis=0)

    def cross(self, V):
        """
        Calculates cross product. Operands 0 and 1 must have data sizes that
        can broadcast together.

        Parameters
        ----------
        V : Vxyz
            Input vector.

        Returns
        -------
        Vxyz
            Cross product.

        """
        # Check inputs
        self._check_is_Vxyz(V)
        if not (len(self) == 1 or len(V) == 1 or len(self) == len(V)):
            raise ValueError('Operands must be same same length, or at least one must have length 1.')

        # Calculate
        return self._from_data(np.cross(self._data.T, V.data.T).T)

    def align_to(self, V) -> Rotation:
        """
        Calculate shortest rotation that aligns current vector to input vector.

        Parameters
        ----------
        V : Vxyz
            3D vector to align current vector to.

        Returns
        -------
        Rotation
            Rotation object to align current vector to given vector.

        """
        # Check inputs
        self._check_is_Vxyz(V)
        if len(self) != 1 or len(V) != 1:
            raise ValueError('Can only align vectors with length 1.')

        # Normlize
        A = self.normalize()
        B = V.normalize()

        # Calculate
        v = A.cross(B).data.squeeze()
        c = A.dot(B)
        C = 1.0 / (1.0 + c)
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rmat = np.eye(3) + Vx + np.matmul(Vx, Vx) * C
        return Rotation.from_matrix(Rmat)

    def concatenate(self, V: 'Vxyz') -> 'Vxyz':
        """Concatenates Vxyz to end of current vector.

        Parameters
        ----------
        V : Vxyz
            Vxyz to concatenate.

        Returns
        -------
        Vxyz
            Concatenated vector
        """
        x = np.concatenate((self.x, V.x))
        y = np.concatenate((self.y, V.y))
        z = np.concatenate((self.z, V.z))
        return Vxyz(np.array([x, y, z]))

    def copy(self) -> 'Vxyz':
        """Returns copy of vector"""
        return Vxyz(self.data.copy())

    def projXY(self) -> Vxy:
        """Returns the x and y components of self as a Vxy

        The components are deep copied.
        """
        return Vxy([self.x.copy(), self.y.copy()])

    @classmethod
    def from_lifted_points(cls, v: Vxy, func: Callable) -> 'Vxyz':
        """Returns Vxyz from a Vxy and a function of form: z = func(x, y)

        Parameters
        ----------
        v : Vxy
            X/Y points
        func : Callable
            Z coordinate function of form z = func(x, y)

        Returns
        -------
        Vxyz
            Output XYZ points
        """
        zs = []
        for x, y in zip(v.x, v.y):
            zs.append(func(x, y))
        return Vxyz([v.x.copy(), v.y.copy(), zs])

    @classmethod
    def empty(cls):
        """Returns an empty Vxyz object

        Returns
        -------
        Vxyz
            Empty (length 0) Vxyz
        """
        return Vxyz([[], [], []])

    def hasnan(self):
        """returns True if there is a nan in self\n
        Note: this method exists because:
        ```python
        >>> isinstance(np.nan, numbers.Number)
        True
        ```"""
        return np.isnan(self.data).any()

    @classmethod
    def merge(cls, V_list: list['Vxyz']):
        """Merges list of multiple Vxyz objects into one Vxyz.

        Parameters
        ----------
        v_list : list[Vxyz]
            List of Vxyz objects to merge

        Returns
        -------
        Vxyz
            Merged Vxyz object
        """
        if len(V_list) == 0:
            return cls.empty()
        data = np.concatenate([v_i.data for v_i in V_list], 1)
        return cls(data)

    @classmethod
    def origin(cls):
        return cls([0, 0, 0])

    # @classmethod
    # def lift(cls, v: Vxy.Vxy, func: Callable):
    #     """Takes in a Vxy and and Callable that takes in 2 arguments.
    #     Returns the Vxyz where the z values correspond to the outputs of the x and y values."""

    #     xs = copy.deepcopy(v.x)
    #     ys = copy.deepcopy(v.y)
    #     zs = []
    #     for x, y in zip(xs, ys):
    #         zs.append(func(x, y))
    #     return cls([xs, ys, zs])
