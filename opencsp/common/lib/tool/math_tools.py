"""
Various math routines.



"""

from functools import reduce
import operator as op
import copy
import numbers
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from typing import Callable
from opencsp.common.lib.tool.typing_tools import strict_types

import opencsp.common.lib.tool.log_tools as lt


INVERSE_TRIG_TOLERANCE: float = 1.0e-6


def robust_arccos(x: float) -> float:
    """
    Inverse cosine function, but with a small margin for values outside of [-1,1] to allow for numerical roundoff error.
    """
    if (x < (-1.0 - INVERSE_TRIG_TOLERANCE)) or (x > (1.0 + INVERSE_TRIG_TOLERANCE)):
        # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
        print("ERROR: In robust_arccos(), input value " + str(x) + " is out of valid range [-1,1].")
        assert False
    elif x < -1.0:
        return np.pi
    elif x > 1.0:
        return 0.0
    else:
        return np.arccos(x)


def robust_arcsin(x: float) -> float:
    """
    Inverse sine function, but with a small margin for values outside of [-1,1] to allow for numerical roundoff error.
    """
    if (x < (-1.0 - INVERSE_TRIG_TOLERANCE)) or (x > (1.0 + INVERSE_TRIG_TOLERANCE)):
        # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
        print("ERROR: In robust_arcsin(), input value " + str(x) + " is out of valid range [-1,1].")
        assert False
    elif x < -1.0:
        return -(np.pi / 2.0)
    elif x > 1.0:
        return np.pi / 2.0
    else:
        return np.arcsin(x)


def clamp(x, x_min, x_max):
    """
    Assure that x is in [x_min, x_max].

    For numpy arrays, use clip():
    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html
    """
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x


def overlapping_range(
    range1: tuple[float, float] | list[float],
    range2: tuple[float, float] | list[float],
    *rangeN: tuple[float, float] | list[float],
    default=None,
) -> list[float]:
    """Find the overlapping region of two or more ranges of numbers.

    Example::

        overlap = overlapping_range([1,5], [3,8])
        # overlap == [3,5]
        overlap = overlapping_range([1,5], [3,8], [0,4])
        # overlap == [3,4]
        overlap = overlapping_range([6,7], [7,8])
        # overlap == [7,7]
        overlap = overlapping_range([6,7], [8,9])
        # overlap == []
        overlap = overlapping_range([], [])
        # overlap == []

    Arguments:
    ----------
        - range1 (tuple[float,float]|list[float]): Two values that define a range.
        - range2 (tuple[float,float]|list[float]): Two values that define a range.
        - rangeN (tuple[float,float]|list[float]): Any other number of tuples or lists of two values that define a range.
        - default: The default value to return when there's an error with the input or there is no overlap. Empty list [] when None. Defaults to None.

    Returns:
    --------
        - overlap (list[float,float]|list): The overlapping range. If there is no overlap, returns an empty list.
    """
    if (len(range1) == 0 or len(range2) == 0) or (range2[0] > range1[1] or range1[0] > range2[1]):
        if default is None:
            default = []
        return default

    ret = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    ret = reduce(overlapping_range, rangeN, ret)
    return ret


def string_is_integer(input_str):
    """
    Returns True if the string can be parsed as an integer.
    If this routine returns True, then int(input_str) should succeed.

    See string_is_float() below for inspiration.

    Another method:
        return input_str.isnumeric()
    """
    try:
        int(input_str)
        return True
    except ValueError:
        return False


def string_is_float(input_str):
    """
    Returns True if the string can be parsed as a floating-point value.
    If this routine returns True, then float(input_str) should succeed.

    Thanks to StackOverflow for this answer:
        https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    """
    try:
        float(input_str)
        return True
    except ValueError:
        return False


def is_number(x):
    """
    Determines whether an input variable is numeric type.
    """
    return isinstance(x, numbers.Number)


def is_nan(x):
    """
    Determines whether an input variable is NaN.
    Unlike the numpy isnan() function, this deos not thrown an error if the input parameter is not a number.
    """
    return is_number(x) and np.isnan(x)


def zero_if_nan(x):
    """
    If x is NaN, then returns zero.  Else returns x.
    """
    if is_nan(x):
        return 0
    else:
        return x


def none_if_nan(x):
    """
    If x is NaN, then returns None.  Else returns x.
    """
    if is_nan(x):
        return None
    else:
        return x


# I have no idea how this works.  Thanks to to the helpful Stack Overlow entry:
#    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
#


def ncr(n, r):
    """
    Calculate the binomial coefficient, also known as "n choose r".

    The binomial coefficient is defined as:

    C(n, r) = n! / (r! * (n - r)!)

    where 'n' is the total number of items, and 'r' is the number of items to choose.

    Parameters
    ----------
    n : int
        The total number of items. Must be a non-negative integer.
    r : int
        The number of items to choose. Must be a non-negative integer less than or equal to \( n \).

    Returns
    -------
    float
        The binomial coefficient \( C(n, r) \).

    Raises
    ------
    ValueError
        If \( n \) or \( r \) are negative, or if \( r \) is greater than \( n \).

    Examples
    --------
    >>> ncr(5, 2)
    10.0
    >>> ncr(10, 3)
    120.0
    >>> ncr(0, 0)
    1.0
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def n_combinations(n, k):
    """
    Returns (n choose k).
    All combinations of k length.
    """
    return ncr(n, k)


def n_combinations_upto(n, k):
    """
    Returns (n choose 1) + (n choose 2) + ... + (n choose k)
    All combinations of up to k length.
    """
    sum = 0
    for i in range(1, k + 1):
        sum += n_combinations(n, i)
    return sum


def rms(error_list):
    """
    Computes the root mean square of the input list of error values.
    """
    # Check input.
    if len(error_list) == 0:
        print("ERROR: In rms(), enncountered null error list.")
        assert False
    # Compute RMS.
    n = len(error_list)
    squares = [x * x for x in error_list]
    sum_squares = sum(squares)
    mean_sum_squares = sum_squares / n
    return np.sqrt(mean_sum_squares)


def gaussian_convolution(num_points: int):
    """Get the probability-density-function (y-values) of the normal distribution covering 98+% of all probabilities.

    Example::

        timestep = 0.1                         # 10 Hz measurement frequency for "spiky_data"
        gaussian = mt.gaussian_convolution(int(1.0 / timestep))
        print(["%0.2f" % p for p in gaussian]) # ['0.01', '0.04', '0.09', '0.15', '0.20', '0.20', '0.15', '0.09', '0.04', '0.01']
        print(sum(gaussian))                   # 0.9902573253069845
        smooth_data = np.convolve(spiky_data, gaussian, 'same')

    Parameters:
    -----------
        - num_points (int): How many bins should be used to represent the pdf?

    Returns:
    --------
        - pdf (list[float]): probability that the value at index i is chosen, where:
                             i follows a gaussian curve, i is greatest at len/2, and 0.98 < sum(pdf) <= 1.00
    """
    y_vals: list[float] = []

    # validate input
    if num_points < 1:
        lt.error_and_raise(
            ValueError,
            f"Error: in math_tools.gaussian_convolution(), num_points must be positive >= 1 but is {num_points}",
        )
    if num_points == 1:
        return [1.0]

    # generate the convolution values
    range98 = [norm.ppf(0.01), norm.ppf(0.99)]
    x_vals = np.linspace(range98[0], range98[1], num_points)
    step_size = x_vals[1] - x_vals[0]
    for x in x_vals:
        y_vals.append(norm.cdf(x + step_size / 2) - norm.cdf(x - step_size / 2))

    return y_vals


def rolling_average(data: list[float] | npt.NDArray[np.float_], window_size: int):
    """Compute the rolling average over all values in the given list.

    For each point x at index i, assign a new value equal to sum([i-w/2:i+w/2]) / 2.
    That is, assign the average of the surrounding window_size values.

    Edge cases:
        - If window_size is odd, then use an equal number of values above and below x.
        - If window_size is even, then use one more value below than above.
        - At edges, divide by the window size to avoid edge effects.

    Parameters:
    -----------
        - data (list|np.ndarray): one-dimensional list of values to compute the rolling average for TODO support multiple dimensions
        - window_size (int): size of the rolling average window (how many values to average around each data point)

    Returns:
    --------
        - averaged_data (list|np.ndarray): the averaged values, of the same length and type as the input data
    """
    # validate input
    if len(data) == 0 or window_size == 1:
        return data
    if window_size < 1:
        lt.error_and_raise(
            ValueError, f"Error: in math_tools.rolling_average(), window_size must be >= 1, but is {window_size}"
        )
    window_size = min(window_size, len(data))

    # build the output list
    data_cp = copy.copy(data)

    # take the rolling average
    avg = np.convolve(data_cp, np.ones((window_size)), "same") / window_size

    # fix edge effects
    half_win_size = int(window_size / 2)
    even_fix = 1 if window_size % 2 == 0 else 0
    for i in range(half_win_size):
        window = max(0, i - half_win_size), min(len(data) - 1, i + half_win_size)
        edge_size = window[1] - window[0] + 1
        edge_size -= even_fix
        avg[i] *= window_size / edge_size
    for j in range(max(i + 1, len(data) - half_win_size), len(data)):
        window = max(0, j - half_win_size), min(len(data) - 1, j + half_win_size)
        edge_size = window[1] - window[0] + 1
        avg[j] *= window_size / edge_size

    # return same data type
    if isinstance(data, np.ndarray):
        return avg
    else:
        return avg.tolist()


# @strict_types
def lambda_symmetric_paraboloid(focal_length: numbers.Number) -> Callable[[float, float], float]:
    """
    Create a lambda function representing a symmetric paraboloid.

    The symmetric paraboloid is defined by the equation:

    z = (1 / (4 * f)) * (x^2 + y^2)

    where 'f' is the focal length of the paraboloid.

    Parameters
    ----------
    focal_length : numbers.Number
        The focal length of the paraboloid. Must be a positive number.

    Returns
    -------
    Callable[[float, float], float]
        A lambda function that takes two float arguments (x, y) and returns the corresponding z value
        of the symmetric paraboloid.

    Raises
    ------
    ValueError
        If the focal_length is not positive.

    Examples
    --------
    >>> paraboloid = lambda_symmetric_paraboloid(2.0)
    >>> z_value = paraboloid(1.0, 1.0)
    >>> print(z_value)
    0.125
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    a = 1.0 / (4 * focal_length)
    return lambda x, y: a * (x**2 + y**2)
    # return FunctionXYContinuous(f"{a} * (x**2 + y**2)")
