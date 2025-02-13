"""Three dimensional vector representation
"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class Vxyz:
    """
    3D vector class to represent 3D points/vectors. Contains N 3D vectors where len == N.

    The values for the contained vectors can be retrieved with
    :py:meth:`data`, or individual vectors can be retrieved with the indexing
    or x/y/z methods. For example, the following can both be used to get the first contained vector::

    .. code-block:: python

        vec = v3.Vxyz([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        v0 = Vxyz([vec.x()[0], vec.y()[0], vec.z()[0]])
        # v0 == Vxyz([0, 3, 6])
        v0 = vec[0]
        # v0 == Vxyz([0, 3, 6])

    """

    def __init__(
        self, data_in: Union[np.ndarray, tuple[float, float, float], tuple[list, list, list], Vxy, "Vxyz"], dtype=float
    ):
        """
        To represent a single vector:

        .. code-block:: python

            x = 1
            y = 2
            z = 3
            vec = Vxyz(np.array([[x], [y], [z]])) # same as vec = Vxyz([x, y, z])
            print(vec.x) # [1.]
            print(vec.y) # [2.]
            print(vec.z) # [3.]

        To represent a set of vectors:

        .. code-block:: python

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
        data_in : array-like
            The 3d point data: 3xN array, length 3 tuple, length 3 list. If a Vxy, then the data will be padded with 0s
            for 'z'.
        dtype : data type, optional
            Data type. The default is float.

        """
        # Check input shape
        data_tmp = data_in
        if isinstance(data_in, np.ndarray):
            data_tmp = data_in.squeeze()
            if np.ndim(data_tmp) not in [1, 2]:
                raise ValueError("Input data must have 1 or 2 dimensions if ndarray.")
            elif np.ndim(data_tmp) == 2 and data_tmp.shape[0] != 3:
                raise ValueError("First dimension of 2-dimensional data must be length 3 if ndarray.")
        elif isinstance(data_in, Vxy):
            data_tmp = np.pad(data_in.data, ((0, 1), (0, 0)))
        elif isinstance(data_in, Vxyz):
            data_tmp = data_in.data
        elif len(data_in) != 3:
            raise ValueError("Input data must have length 3.")

        # Save and format data
        self._data = np.array(data_tmp, dtype=dtype).reshape((3, -1))

    @property
    def data(self) -> np.ndarray:
        """
        An array with shape (3, N), where N is the number of 3D vectors in this instance.
        """
        return self._data

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def x(self) -> np.ndarray:
        """The x-coordinates of the vectors."""
        return self._data[0, :]

    @property
    def y(self) -> np.ndarray:
        """The y-coordinates of the vectors."""
        return self._data[1, :]

    @property
    def z(self) -> np.ndarray:
        """The z-coordinates of the vectors."""
        return self._data[2, :]

    def as_Vxyz(self) -> "Vxyz":
        """
        Returns
        -------
        Vxyz
            This instance
        """
        return self

    @classmethod
    def _from_data(cls, data, dtype=None) -> "Vxyz":
        """
        Builds a new instance with the given data and data type.

        Parameters
        ----------
        data : array-like | list-like
            The data to build this class with. Acceptable input types are
            enumerated in the constructor type hints.
        dtype : Literal[float] | Literal[int], optional
            The data type used for representing the input data, by default float
        """
        return cls(data, dtype)

    @classmethod
    def from_list(cls, vals: list["Vxyz"]):
        """Builds a single Vxyz instance from a list of Vxyz instances."""
        xs, ys, zs = [], [], []
        for val in vals:
            xs += val.x.tolist()
            ys += val.y.tolist()
            zs += val.z.tolist()
        return cls((xs, ys, zs))

    def _check_is_Vxyz(self, v_in):
        """
        Checks if input data is instance of Vxyz for methods that require this
        type.

        Raises
        ------
        TypeError:
            If the input v_in is not a Vxyz type object.
        """
        if not isinstance(v_in, Vxyz):
            raise TypeError(f"Input operand must be {Vxyz}, not {type(v_in)}")

    def __add__(self, v_in):
        """
        Element wise addition. Operand 1 type must be Vxyz. Returns a new Vxyz.
        """
        self._check_is_Vxyz(v_in)
        return self._from_data(self._data + v_in.data)

    def __sub__(self, v_in):
        """
        Element wise subtraction. Operand 1 type must be Vxyz. Returns a new Vxyz.
        """
        self._check_is_Vxyz(v_in)
        return self._from_data(self._data - v_in.data)

    def __mul__(self, data_in):
        """
        Element wise multiplication. Operand 1 type must be int, float, or Vxyz. Returns a new Vxyz.
        """
        if type(data_in) in [int, float, np.float32, np.float64, np.int32, np.int64]:
            return self._from_data(self._data * data_in)
        elif isinstance(data_in, Vxyz):
            return self._from_data(self._data * data_in.data)
        elif type(data_in) is np.ndarray:
            return self._from_data(self._data * data_in)
        else:
            raise TypeError(f"Vxyz cannot be multipled by type, {type(data_in)}.")

    def __getitem__(self, key) -> "Vxyz":
        # Check that only one dimension is being indexed
        if np.size(key) > 1 and any(isinstance(x, slice) for x in key):
            raise ValueError("Can only index over one dimension.")

        return self._from_data(self._data[:, key], dtype=self.dtype)

    def __repr__(self):
        return "3D Vector:\n" + self._data.__repr__()

    def __len__(self):
        return self._data.shape[1]

    def __neg__(self):
        return self._from_data(-self._data, dtype=self.dtype)

    def _magnitude_with_zero_check(self) -> np.ndarray:
        """
        Returns magnitude of each vector as a new array.

        Returns
        -------
        ndarray
            1d vector of normalized data. Shape is (n).

        Raises
        ------
        ValueError:
            If the magnitude of any of the contained vectors is 0.
        """
        mag = self.magnitude()

        if np.any(mag == 0):
            raise ValueError("Vector contains zero vector, cannot normalize.")

        return mag

    def normalize(self) -> "Vxyz":
        """
        Creates a copy of this instance and normalizes it.

        Returns
        -------
        Vxyz
            Normalized vector copy with the same shape.

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
        Returns magnitude of each vector as a new array.

        Returns
        -------
        np.ndarray
            Vector magnitudes copy. Shape is (n).

        """
        return np.sqrt(np.sum(self._data**2, 0))

    def rotate(self, R: Rotation) -> "Vxyz":
        """
        Returns a copy of the rotated vector rotated about the coordinate system
        origin. The rotation is applied to each of the contained 3d coordinates.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.

        Returns
        -------
        Vxyz
            Rotated vector copy.

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f"Rotation must be type {Rotation}, not {type(R)}")

        V_out = self._from_data(self._data.copy())
        V_out.rotate_in_place(R)
        return V_out

    def rotate_about(self, R: Rotation, V_pivot: "Vxyz") -> "Vxyz":
        """
        Returns a copy of the rotated vector rotated about the given pivot
        point. The rotation is applied to each of the contained 3d coordinates.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.
        V_pivot : Vxyz
            Pivot point to rotate about. Must broadcast with the size of this
            instance (must have length 1 or N, where N is the length of this
            instance).

        Returns
        -------
        Vxyz
            Rotated vector copy.

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f"Rotaion must be type {Rotation}, not {type(R)}")
        self._check_is_Vxyz(V_pivot)

        V_out = self._from_data(self._data.copy())
        V_out.rotate_about_in_place(R, V_pivot)
        return V_out

    def rotate_in_place(self, R: Rotation) -> None:
        """
        Rotates vector about the coordinate system origin. Replaces data in Vxyz
        object with rotated data. The rotation is applied to each of the
        contained 3d coordinates.

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
            raise TypeError(f"Rotation must be type {Rotation}, not {type(R)}")

        self._data = R.apply(self._data.T).T

    def rotate_about_in_place(self, R: Rotation, V_pivot: "Vxyz") -> None:
        """
        Rotates about the given pivot point. Replaces data in Vxyz object with
        rotated data. The rotation is applied to each of the contained 3d
        coordinates.

        Parameters
        ----------
        R : Rotation
            Rotation object to apply to vector.
        V_pivot : Vxyz
            Pivot point to rotate about. Must broadcast with the size of this
            instance (must have length 1 or N, where N is the length of this
            instance).

        Returns
        -------
        None

        """
        # Check inputs
        if not isinstance(R, Rotation):
            raise TypeError(f"Rotaion must be type {Rotation}, not {type(R)}")
        self._check_is_Vxyz(V_pivot)

        # Center pivot to origin
        self._data -= V_pivot.data
        # Rotate about origin
        self.rotate_in_place(R)
        # Recenter pivot point
        self._data += V_pivot.data

    def dot(self, V: "Vxyz") -> np.ndarray:
        """
        Calculated dot product. Size of input data must broadcast with size of
        data.

        Parameters
        ----------
        V : Vxyz
            Input vector to compute the dot product with. Must broadcast with
            the size of this instance (must have length 1 or N, where N is the
            length of this instance).

        Returns
        -------
        np.ndarray
            Array of dot product values with shape (N).

        """
        # Check inputs
        self._check_is_Vxyz(V)

        return (self._data * V.data).sum(axis=0)

    def cross(self, V: "Vxyz") -> "Vxyz":
        """
        Calculates cross product. Operands 0 and 1 must have data sizes that
        can broadcast together.

        Parameters
        ----------
        V : Vxyz
            Input vector to computer the cross product with. Must broadcast with
            the size of this instance (must have length 1 or N, where N is the
            length of this instance).

        Returns
        -------
        Vxyz
            Cross product copy with shape (P), where O is the length of the
            input V and P is the greater of N and O.

        """
        # Check inputs
        self._check_is_Vxyz(V)
        if not (len(self) == 1 or len(V) == 1 or len(self) == len(V)):
            raise ValueError("Operands must be same same length, or at least one must have length 1.")

        # Calculate
        return self._from_data(np.cross(self._data.T, V.data.T).T)

    def align_to(self, V: "Vxyz") -> Rotation:
        """
        Calculate shortest rotation that aligns current vector to input vector.
        Both vectors must have length 1. The returned rotation can be applied to
        the current vector so that it then aligns with the input vector. For
        example::

            vec = Vxyz([1, 2, 3])
            R = vec.align_to(Vxyz([1, 0, 0]))
            vec_r = vec.rotate(R)
            vec_r_n = vec_r.normalize()

            # vec.magnitude() == 3.74165739
            # vec_r == [ 3.74165739, -4.44089210e-16, -2.22044605e-16 ]
            # vec_r_n == [ 1.00000000, -1.18687834e-16, -5.93439169e-17 ]

        Parameters
        ----------
        V : Vxyz
            3D vector to align current vector to. Must have length 1.

        Returns
        -------
        Rotation
            Rotation object to align current vector to given vector.

        """
        # Check inputs
        self._check_is_Vxyz(V)
        if len(self) != 1 or len(V) != 1:
            raise ValueError("Can only align vectors with length 1.")

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

    def concatenate(self, V: "Vxyz") -> "Vxyz":
        """
        Concatenates Vxyz to the end of current vector. Returns a copy as the
        new vector.

        Parameters
        ----------
        V : Vxyz
            Vxyz to concatenate.

        Returns
        -------
        Vxyz
            Concatenated vector copy. Shape is (3,N+O), where O is the length of
            the input vector V.
        """
        x = np.concatenate((self.x, V.x))
        y = np.concatenate((self.y, V.y))
        z = np.concatenate((self.z, V.z))
        return Vxyz(np.array([x, y, z]))

    def copy(self) -> "Vxyz":
        """Returns copy of vector"""
        return Vxyz(self.data.copy())

    def projXY(self) -> Vxy:
        """Returns the x and y components of self as a Vxy.

        The components are not a view via indexing but rather a copy.

        Returns
        -------
        Vxy
            Output XY points as a new vector. Shape is (2,N).
        """
        return Vxy([self.x.copy(), self.y.copy()])

    @classmethod
    def from_lifted_points(cls, v: Vxy, func: Callable) -> "Vxyz":
        """Returns Vxyz from a Vxy and a function of form: z = func(x, y).

        Parameters
        ----------
        v : Vxy
            X/Y points with shape (2,N).
        func : Callable
            Z coordinate function of form z = func(x, y)

        Returns
        -------
        Vxyz
            Output XYZ points as a new vector. Shape is (3,N).
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
        """Returns True if there is a single NaN in the current vector.

        Note: this method exists because of the unintuitive behavior of isinstance in Python::

            isinstance(np.nan, numbers.Number) # True
        """
        return np.isnan(self.data).any()

    @classmethod
    def merge(cls, V_list: list["Vxyz"]):
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

    def draw_line(
        self,
        figure: rcfr.RenderControlFigureRecord | v3d.View3d,
        close: bool = None,
        style: rcps.RenderControlPointSeq = None,
        label: str = None,
    ) -> None:
        """
        Calls figure.draw_xyz_list(self.data.T) to draw all xyz points in a
        single series. Uses the default arguments for
        :py:meth:`View3d.draw_xyz_list` in place of any None arguments.

        Parameters
        ----------
        figure : rcfr.RenderControlFigureRecord or v3d.View3d
            The figure to draw to.
        close : bool, optional
            True to add the first point again at the end of the plot, thereby
            drawing this set of points as a closed polygon. None or False to not
            add another point at the end (draw_xyz_list default)
        style : rcps.RenderControlPointSeq, optional
            The style to use for the points and lines, or None for
            :py:meth:`RenderControlPointSequence.default`.
        label : str, optional
            A string used to label this plot in the legend, or None for no label.
        """
        kwargs = dict()
        for key, val in [("close", close), ("style", style), ("label", label)]:
            if val is not None:
                kwargs[key] = val

        view = figure if isinstance(figure, v3d.View3d) else figure.view
        view.draw_xyz_list(self.data.T, **kwargs)

    def draw_points(
        self,
        figure: rcfr.RenderControlFigureRecord | v3d.View3d,
        style: rcps.RenderControlPointSeq = None,
        labels: list[str] = None,
    ) -> None:
        """
        Calls figure.draw_xyz(p) to draw all xyz points in this instance
        individually. Uses the default arguments for :py:meth:`View3d.draw_xyz`
        in place of any None arguments.

        Parameters
        ----------
        figure : rcfr.RenderControlFigureRecord | v3d.View3d
            The figure to draw to.
        close : bool, optional
            True to add the first point again at the end of the plot, thereby
            drawing this set of points as a closed polygon. None or False to not
            add another point at the end (draw_xyz_list default).
        style : rcps.RenderControlPointSeq, optional
            The style to use for the points and lines, or None for
            :py:meth:`RenderControlPointSequence.default`.
        label : str, optional
            A string used to label this plot in the legend, or None for no label.
        """
        if labels is None:
            labels = [None] * len(self)
        view = figure if isinstance(figure, v3d.View3d) else figure.view
        for x, y, z, label in zip(self.x, self.y, self.z, labels):
            view.draw_xyz((x, y, z), style, label)
