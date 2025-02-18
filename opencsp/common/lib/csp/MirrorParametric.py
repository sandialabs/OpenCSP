"""Parametric mirror representing a single reflective surface defined
by an algebraic function.
"""

import inspect
from typing import Callable

import numpy as np
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify

from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.FunctionXYContinuous import FunctionXYContinuous


class MirrorParametric(MirrorAbstract):
    """
    Mirror implementation defined by a parametric function and a 2d region.
    """

    def __init__(
        self, surface_function: Callable[[np.ndarray, np.ndarray], np.ndarray], shape: RegionXY
    ) -> "MirrorParametric":
        """Instantiates MirrorParametric class

        Parameters
        ----------
        surface_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Callable which takes in two ndarrays (the x and y sample coordinates
            respectively), and outputs the corresponding z height of the mirror from
            the z=0 plane in ndaray format. surface_function(x, y) = z
        shape : RegionXY
            The 2d region of the mirror when looking along the z axis. These 2d points
            are 'lifted' from the z=0 plane to where those points are located as defined
            by 'surface_function'.
        """
        super().__init__(shape)  # initalizes the attributes universal to all mirrors

        # Define surface z and surface normal vector functions.
        self._surface_function = surface_function
        self._normals_function = self._define_normals_function(surface_function)

    def __repr__(self) -> str:
        if isinstance(self._surface_function, FunctionXYContinuous):
            return f"Parametricly defined mirror defined by the function {self._surface_function}"
        return (
            f"Parametricly defined mirror defined by the function {inspect.getsourcelines(self._surface_function)[0]}"
        )

    def _define_normals_function(self, surface_function: Callable[[float, float], float]) -> Callable:
        """Returns a normal vector generating function given a surface z coordinate
        function

        Parameters
        ----------
        surface_function : Callable[[float, float], float]
            Callable z surface height function

        Returns
        -------
        Callable
            Normal vector function
        """
        # Create X/Y symbolic variables
        x_s = Symbol("x")
        y_s = Symbol("y")

        # Take derivative of surface function in X and Y
        sym_func = surface_function(x_s, y_s)
        dfdx = diff(sym_func, x_s)
        dfdy = diff(sym_func, y_s)

        # Evaluate function at XY coordinates
        func_dfdx = lambdify([x_s, y_s], dfdx, "numpy")
        func_dfdy = lambdify([x_s, y_s], dfdy, "numpy")

        def _normals_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Returns normal vectors for given xy points

            Parameters
            ----------
            x/y : np.ndarray
                XY sample points

            Returns
            -------
            np.ndarray
                Normal vectors array, shape (x.shape, 3)
            """
            dfdx_n = func_dfdx(x, y)
            dfdy_n = func_dfdy(x, y)
            # Check if surface function outputs constant value
            if isinstance(dfdx_n, (float, int)):
                dfdx_n *= np.ones(x.shape)
            if isinstance(dfdy_n, (float, int)):
                dfdy_n *= np.ones(y.shape)
            # Create constant z coordinate
            z_norm = np.ones(x.shape)
            return np.concatenate((-dfdx_n[..., None], -dfdy_n[..., None], z_norm[..., None]), axis=-1)

        return _normals_function

    def _check_in_bounds(self, p_samp: Pxyz) -> None:
        """Checks that points are within mirror bounds"""
        if not all(self.in_bounds(p_samp)):
            raise ValueError("Not all points are within mirror perimeter.")

    def surface_norm_at(self, p: Pxy) -> Vxyz:
        if not issubclass(type(p), Vxy):
            raise TypeError(f"Sample point must be type {Vxy}, not type {type(p)}")
        self._check_in_bounds(p)
        pts = self._normals_function(p.x, p.y)
        return Vxyz(pts.T).normalize()

    def surface_displacement_at(self, p: Pxy) -> np.ndarray[float]:
        self._check_in_bounds(p)
        return self._surface_function(p.x, p.y)

    @classmethod
    def generate_symmetric_paraboloid(cls, focal_length: float, shape: RegionXY) -> "MirrorParametric":
        """Generate a symmetric parabolic mirror with the given focal length

        Parameters
        ----------
        focal_length : float
            Focal length
        shape : RegionXY
            Mirror top-down region.

        Returns
        -------
        MirrorParametric
        """
        # Create surface function
        a = 1.0 / (4 * focal_length)

        def surface_function(x, y):
            return a * (x**2 + y**2)

        return cls(surface_function, shape)

    @classmethod
    def generate_flat(cls, shape: RegionXY) -> "MirrorParametric":
        """Generate a flat, z=0 mirror

        Parameters
        ----------
        shape : RegionXY
            Mirror top-down region.

        Returns
        -------
        MirrorParametric
        """

        # Create a surface function
        def surface_function(x, y):
            return x * y * 0

        return cls(surface_function, shape)
