import numbers

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import opencsp.common.lib.tool.log_tools as lt


class Vxy:
    """
    2D vector class to represent 2D points/vectors.

    Parameters
    ----------
    data : array-like
        The 2d point data: 2xN array, length 2 tuple, length 2 list
    dtype : data type, optional
        Data type. The default is float.

    """

    def __init__(self, data, dtype=float):
        """
        2D vector class to represent 2D points/vectors.

        To represent a single vector::

            x = 1
            y = 2
            vec = Vxy(np.array([[x], [y])) # same as vec = Vxy([x, y])
            print(vec.x) # [1.]
            print(vec.y) # [2.]

        To represent a set of vectors::

            vec1 = [1, 2]
            vec2 = [4, 5]
            vec3 = [7, 8]
            zipped = list(zip(vec1, vec2, vec3))
            vecs = Vxy(np.array(zipped))
            print(vec.x) # [1. 4. 7.]
            print(vec.y) # [2. 5. 8.]

            # or this equivalent method
            xs = [1, 4 ,7]
            ys = [2, 5, 8]
            vecs = Vxy((xs, ys))

        Parameters
        ----------
        data : array-like
            The 2d point data: 2xN array, length 2 tuple, length 2 list
        dtype : data type, optional
            Data type. The default is float.

        """
        # Check input shape
        if type(data) is np.ndarray:
            data = data.squeeze()
            if np.ndim(data) not in [1, 2]:
                raise ValueError('Input data must have 1 or 2 dimensions if ndarray.')
            elif np.ndim(data) == 2 and data.shape[0] != 2:
                raise ValueError('First dimension of 2-dimensional data must be length 2 if ndarray.')
        elif len(data) != 2:
            raise ValueError('Input data must have length 2.')

        # Save and format data
        self._data = np.array(data, dtype=dtype).reshape((2, -1))

    @property
    def data(self):
        """
        An array with shape (2, N), where N is the number of 2D vectors in this instance.
        """
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def x(self):
        return self._data[0, :]

    @property
    def y(self):
        return self._data[1, :]

    @classmethod
    def _from_data(cls, data, dtype=float):
        return cls(data, dtype)

    def _check_is_Vxy(self, v_in):
        """
        Checks if input data is instance of Vxy for methods that require this
        type.

        """
        if not isinstance(v_in, Vxy):
            raise TypeError('Input operand must be {}, not {}'.format(Vxy, type(v_in)))

    def __add__(self, v_in):
        """
        Element wise addition. Operand 1 type must be Vxy
        """
        self._check_is_Vxy(v_in)
        return self._from_data(self._data + v_in.data)

    def __sub__(self, v_in):
        """
        Element wise subtraction. Operand 1 type must be Vxy
        """
        self._check_is_Vxy(v_in)
        return self._from_data(self._data - v_in.data)

    def __mul__(self, data_in):
        """
        Element wise addition. Operand 1 type must be int, float, or Vxy.
        """
        if type(data_in) in [int, float, np.float32, np.float64]:
            return self._from_data(self._data * data_in)
        elif isinstance(data_in, Vxy):
            return self._from_data(self._data * data_in.data)
        elif type(data_in) is np.ndarray:
            return self._from_data(self._data * data_in)
        else:
            raise TypeError('Vxy cannot be multipled by type, {}.'.format(type(data_in)))

    def __getitem__(self, key):
        # Check that only one dimension is being indexed
        if np.size(key) > 1 and any(isinstance(x, slice) for x in key):
            raise ValueError('Can only index over one dimension.')

        return self._from_data(self._data[:, key], dtype=self.dtype)

    def __repr__(self):
        return '2D Vector:\n' + self._data.__repr__()

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
        Vxy
            Normalized vector.

        """
        V_out = self._from_data(self._data.copy())
        V_out.normalize_in_place()
        return V_out

    def normalize_in_place(self) -> None:
        """
        Normalizes vector. Replaces data in Vxy object with normalized data.

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

    def rotate(self, R: np.ndarray):
        """
        Rotates vector about coordinate system origin. Returns a copy of the
        rotated vector.

        Parameters
        ----------
        R : np.ndarray
            2x2 rotation matrix.

        Returns
        -------
        Vxy
            Rotated vector.

        """
        V_out = self._from_data(self._data.copy())
        V_out.rotate_in_place(R)
        return V_out

    def rotate_about(self, R: np.ndarray, V_pivot):
        """
        Rotates vector about a pivot point. Returns a copy of the rotated
        vector.

        Parameters
        ----------
        R : np.ndarray
            2x2 rotation matrix.
        V_pivot : Vxy
            Pivot point to rotate about

        Returns
        -------
        Vxy
            Rotated vector.

        """
        V_out = self._from_data(self._data.copy())
        V_out.rotate_about_in_place(R, V_pivot)
        return V_out

    def rotate_in_place(self, R: np.ndarray) -> None:
        """
        Rotates vector about coordinate system origin. Replaces data in Vxy
        object with rotated data.

        Parameters
        ----------
        R : np.ndarray
            2x2 rotation matrix.

        """
        # Check inputs
        if type(R) is not np.ndarray:
            raise TypeError('Rotation must be type ndarray, not {}'.format(type(R)))
        if R.shape != (2, 2):
            raise ValueError('Rotation matrix must be shape (2, 2), not {}'.format(R.shape))

        self._data = R @ self._data

    def rotate_about_in_place(self, R: np.ndarray, V_pivot) -> None:
        """
        Rotates vector about given pivot point. Replaces data in Vxy object
        with rotated data.

        Parameters
        ----------
        R : np.ndarray
            2x2 rotation matrix.
        V_pivot : Vxy
            Pivot point to rotate about

        """
        # Center pivot point to origin
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
        V : Vxy
            Input vector.

        Returns
        -------
        np.ndarray
            Length n array of dot product values.

        """
        self._check_is_Vxy(V)
        return (self._data * V.data).sum(axis=0)

    def cross(self, V) -> np.ndarray:
        """
        Calculates cross product. Operands 0 and 1 must have data sizes that
        can broadcast together.

        Parameters
        ----------
        V : Vxy
            Input vector.

        Returns
        -------
        ndarray
            1D ndarray, cross product.

        """
        self._check_is_Vxy(V)
        return np.cross(self._data.T, V.data.T)

    def draw(self, ax=None):
        """
        Draws points on axis via plt.scatter.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to draw on. The default is None.

        """
        if ax is None:
            ax = plt.gca()

        ax.scatter(*self._data)

        return ax

    def concatenate(self, V: 'Vxy') -> 'Vxy':
        """Concatenates Vxy to end of current vector.

        Parameters
        ----------
        V : Vxy
            Vxy to concatenate.

        Returns
        -------
        Vxy
            Concatenated vector
        """
        x = np.concatenate((self.x, V.x))
        y = np.concatenate((self.y, V.y))
        return Vxy(np.array([x, y]))

    @classmethod
    def merge(cls, v_list: list['Vxy']) -> 'Vxy':
        """Merges list of multiple Vxy objects into one Vxy.

        Parameters
        ----------
        v_list : list[Vxy]
            List of Vxy objects to merge

        Returns
        -------
        Vxy
            Merged Vxy object
        """
        if len(v_list) == 0:
            return cls([[], []])

        data = np.concatenate([v_i.data for v_i in v_list], 1)
        return cls(data)

    def astuple(self) -> tuple[numbers.Number, numbers.Number]:
        """Get this instance as a tuple (x, y). Only works for single-value vectors.

        Raises:
        -------
        RuntimeError:
            This vector has more than one value.
        """
        if len(self) > 1:
            lt.error_and_raise(
                RuntimeError,
                "Error in Vxy.astuple(): " + f"can't convert a Vxy with {len(self)} sets of values to a single tuple",
            )
        return self.x[0], self.y[0]
