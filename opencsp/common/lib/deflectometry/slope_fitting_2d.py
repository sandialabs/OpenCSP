import numpy as np

from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


def propagate_rays_to_plane(u_ray: Uxyz, v_origin: Vxyz, v_plane: Vxyz, u_plane: Uxyz) -> Vxyz:
    """
    Propagates rays to their intersection with a plane

    Parameters
    ----------
    u_ray : Uxyz
        Length 1 or N ray direction vector.
    v_origin : Vxyz
        Length 1 or N ray origin vector.
    v_plane : Vxyz
        Length 1 vector, location of plane from coordinate origin.
    u_plane: Uxyz
        Length 1 vector, orientation of plane normal.

    Returns
    -------
    Vxyz
        Intersection points of rays with plane.

    """
    if type(u_ray) is not Uxyz:
        raise TypeError('u_ray must be type {} not type {}.'.format(Uxyz, type(u_ray)))
    if type(u_plane) is not Uxyz:
        raise TypeError('u_plane must be type {} not type {}.'.format(Uxyz, type(u_plane)))

    v_origin_plane = v_plane - v_origin
    w_dot = u_plane.dot(v_origin_plane)
    v_dot = u_plane.dot(u_ray)
    scales = w_dot / v_dot  # scaling factor
    int_pts = Vxyz(v_origin.data + u_ray.data * scales[np.newaxis, :])
    return int_pts


def calc_slopes(v_surf_int_pts_optic: Vxyz, v_optic_cam_optic: Vxyz, v_screen_points_optic: Vxyz) -> np.ndarray:
    """
    Calculate slopes of every measurement point. The normal of the surface is
    calculated sa the vector between the camera-to-optic vector and the
    optic-to-screen vector.

    Parameters
    ----------
    v_surf_int_pts_optic : Vxyz
        Measurement points on optic surface.
    v_optic_cam_optic : Vxyz
        Optic to camera vector in optic coordinates.
    v_screen_points_optic : Vxyz
        Optic to screen points vector in optic coordinates.

    Returns
    -------
    slopes : ndarray
        Measurement point slopes.

    """
    # Calculate normal vectors
    u_out = (v_screen_points_optic - v_surf_int_pts_optic).normalize()
    u_in = (v_optic_cam_optic - v_surf_int_pts_optic).normalize()
    u_norm = (u_out + u_in).normalize()

    # Convert normals to slopes
    slope_x = -u_norm.x / u_norm.z
    slope_y = -u_norm.y / u_norm.z

    return np.array((slope_x, slope_y))


def fit_slope_robust_ls(
    slope_fit_poly_order: int, slope: np.ndarray, weights: np.ndarray, v_surf_int_pts_optic: Vxyz
) -> np.ndarray:
    """
    Fits a slope using robust least squares fitting with weighted residuals.

    This function performs a robust least squares fit to the provided slope data,
    adjusting weights iteratively based on the residuals to minimize the influence
    of outliers.

    Parameters
    ----------
    slope_fit_poly_order : int
        The order of the polynomial used for fitting the slope.
    slope : np.ndarray
        A 1D array of slope measurements.
    weights : np.ndarray
        A 1D array of weights corresponding to the slope measurements.
    v_surf_int_pts_optic : Vxyz
        An object containing the x and y coordinates of the surface intersection points.

    Returns
    -------
    np.ndarray
        The coefficients of the fitted slope.

    Raises
    ------
    ValueError
        If the lengths of the input arrays do not match or if the fitting process does not converge
        within the maximum number of iterations.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Check lengths match
    if slope.size != weights.size or slope.size != len(v_surf_int_pts_optic):
        raise ValueError(
            f'Input data lengths must be same size, but sizes were {slope.size:d}, {weights.size:d}, {len(v_surf_int_pts_optic):d}.'
        )

    # Calculate number of points
    num_pts = len(v_surf_int_pts_optic)

    # Create terms
    terms = poly_terms(slope_fit_poly_order, v_surf_int_pts_optic.x, v_surf_int_pts_optic.y)

    # Robust least squares fit
    c1 = 4.685
    c2 = 0.6745
    delta = 1e-9  # change in residual magnitude that signals convergence
    max_iter = 1000  # maximum number of iterations to perform
    res_mag_prev = 0  # instantiate variable

    for loop_idx in range(max_iter):
        # Perform weighted least squares fit
        terms_weighted = terms * weights[:, np.newaxis]  # shape (N, 3)
        a = terms_weighted.T @ terms  # shape (3, 3)
        A = np.linalg.lstsq(a, terms_weighted.T, rcond=None)[0]  # shape (3, N)
        coefficients = A @ slope  # shape (3,)
        # Compute hat matrix
        hat = (terms * A.T).sum(1)  # shape (N,)
        # Compute residuals
        res = slope - terms @ coefficients  # shape (N,)
        res_mag = np.linalg.norm(res) / num_pts  # float
        # Check convergence
        if loop_idx > 0 and np.abs(res_mag_prev - res_mag) < delta:
            break
        # Update residual error
        res_mag_prev = np.copy(res_mag)
        # Compute adjusted residuals
        res_adj = res / np.sqrt(1 - hat)  # shape (N,)
        # Compute median adjusted deviation
        mad = np.median(np.abs(res_adj - np.median(res_adj)))  # float
        # Compute robust variance estimate
        robust_var = mad / c2  # float
        # Compute standardized residuals
        res_sta = res_adj / (c1 * robust_var)  # shape (N,)
        # Compute weights
        weights = (1 - res_sta**2) ** 2
        weights[np.abs(res_sta) >= 1] = 0

    if loop_idx == max_iter:
        raise ValueError('Robust least squares slope fitting could not converge.')

    return coefficients, weights


def fit_slope_ls(slope_fit_poly_order: int, slope: np.ndarray, v_surf_int_pts_optic: Vxyz) -> np.ndarray:
    """
    Fits a slope using ordinary least squares fitting.

    This function computes the best fit slope coefficients for the provided slope data
    using the least squares method.

    Parameters
    ----------
    slope_fit_poly_order : int
        The order of the polynomial used for fitting the slope.
    slope : np.ndarray
        A 1D array of slope measurements.
    v_surf_int_pts_optic : Vxyz
        An object containing the x and y coordinates of the surface intersection points.

    Returns
    -------
    np.ndarray
        The coefficients of the fitted slope.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Create terms
    terms = poly_terms(slope_fit_poly_order, v_surf_int_pts_optic.x, v_surf_int_pts_optic.y)

    # Simple least squares fit
    #   a @ x = b
    #       a = XY terms
    #       b = slopes
    #       x = slope coefficients
    a = terms.T @ terms
    b = terms.T @ slope
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]

    return coefficients


def poly_terms(poly_order: int, x: np.ndarray, y: np.ndarray):
    """
    Creates terms for polynomial fitting given polynomial order.

    Parameters
    ----------
    poly_order : int
        Order or polynomial.
    x : length N array
        X points.
    y : length N array
        Y points.

    Returns
    -------
    terms : ndarray
        Nxn array.

    """
    # Check input sizes are the same
    if x.size != y.size:
        raise ValueError('X and Y sizes must be equal.')

    # Build array to contain terms
    n = np.arange(1, poly_order + 2).sum()
    terms = np.zeros((x.size, n))
    col_n = 0
    for yi in range(poly_order + 1):
        for xi in range(poly_order - yi + 1):
            terms[:, col_n] = y**yi * x**xi
            col_n += 1

    return terms


def coef_to_points(Pxy: Vxyz, coefficients: np.ndarray, poly_order: int) -> np.ndarray:
    """
    Converts x/y coordinates and polynomial coefficients to XYZ points.

    Parameters
    ----------
    Pxy : Vxyz or Vxy
        Input XY points.
    coefficients : np.ndarray
        Optic surface fit polynomial coefficients.
    poly_order : int
        Order of polynomial.

    Returns
    -------
    np.ndarray
        Z coordinate corresponding to input XY points.

    """
    terms = poly_terms(poly_order, Pxy.x, Pxy.y)
    return np.matmul(terms, coefficients.reshape((coefficients.size, -1))).squeeze()


def dist_optic_screen_error(
    scale: float,
    dist_meas: float,
    v_align_point_optic: Vxyz,
    v_optic_cam_optic: Vxyz,
    v_optic_screen_optic: Vxyz,
    v_meas_pts_surf_int_optic: Vxyz,
) -> float:
    """
    Calculates the optic to screen distance error.

    Parameters
    ----------
    scale : float
        Scale factor to apply to position vector.
    dist_meas : float
        Measured optic to screen distance.
    v_align_point_optic : Vxyz
        Align point in optic coordinates.
    v_optic_cam_optic : Vxyz
        Optic to camera vector in optic coordinates.
    v_optic_screen_optic : Vxyz
        Optic to screen vector in optic coordinates.
    v_meas_pts_surf_int_optic : Vxyz
        Location of the "measure point" XYZ location in optic coordinates.

    Returns
    -------
    float
        Position error, meters.

    """

    v_align_point_cam = v_optic_cam_optic - v_align_point_optic
    dv_align_point_cam = v_align_point_cam * (scale - 1)
    v_optic_screen_new = v_optic_screen_optic + dv_align_point_cam
    error = np.linalg.norm((v_optic_screen_new - v_meas_pts_surf_int_optic).data) - dist_meas

    return np.abs(error)
