"""
Utilities for list processing.



"""

import numpy as np
import re
from typing import Callable, TypeVar

import opencsp.common.lib.tool.log_tools as lt

_T = TypeVar("_T")
_V = TypeVar("_V")


def remove_duplicates(list):
    """
    Remove duplicate elements from a list while preserving the original order.

    Parameters
    ----------
    list : list
        The input list from which duplicates will be removed.

    Returns
    -------
    list
        A new list containing the unique elements from the input list, in the order they first appeared.

    Examples
    --------
    >>> remove_duplicates([1, 2, 2, 3, 4, 4, 5])
    [1, 2, 3, 4, 5]
    >>> remove_duplicates(['a', 'b', 'a', 'c'])
    ['a', 'b', 'c']
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    result = []
    for x in list:
        if x not in result:
            result.append(x)
    return result


def contains_duplicates(list):
    """
    Check if a list contains any duplicate elements.

    Parameters
    ----------
    list : list
        The input list to check for duplicates.

    Returns
    -------
    bool
        True if the list contains duplicates, False otherwise.

    Examples
    --------
    >>> contains_duplicates([1, 2, 3, 4])
    False
    >>> contains_duplicates([1, 2, 2, 3])
    True
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return len(list) != len(remove_duplicates(list))


def zamboni_shuffle(segment_list):
    """
    Shuffle elements so input consecutive elements are separated.
    Inspired by the path of a Zamboni in an ice rink.

    Examples:
        zamboni_shuffle([1,2,3,4,5,6])  -->  [1, 4, 2, 5, 3, 6]

        zamboni_shuffle([1,2,3,4,5,6,7]) --> [1, 5, 2, 6, 3, 7, 4]
    """
    if len(segment_list) < 2:
        return segment_list
    else:
        n_segments = len(segment_list)
        idx_offset = int(np.ceil(n_segments / 2.0))
        shuffled_segment_list = []
        for idx in range(0, idx_offset):
            shuffled_segment_list.append(segment_list[idx])
            if (idx + idx_offset) < n_segments:
                shuffled_segment_list.append(segment_list[idx + idx_offset])
        return shuffled_segment_list


def binary_search(
    sorted_list: list[_T],
    search_val: _V,
    comparator: Callable[[_T, _V], int] = None,
    key: Callable[[_T], _V] = None,
    err_if_not_equal=False,
):
    """Does a binary search to get the index and item in the list corresponding to the given search_val.

    This only works when sorted_list is sorted by the desired search key. If it is not,
    then there is no guarantee that this will return anything useful and list.indexof
    should probably be used instead.

    If _T is a number, then this functionality is extended to return the closest item
    when there isn't an exact match.

    Args:
        - sorted_list (list[_T]): A list that is sorted by its natural order (by the desired attribute if a list of objects).
        - search_val (_V): The desired value. Must either be of type _T, or comparator must be defined.
        - comparator (Callable, optional): A function to order list values compared to the search_val (-1 if T<V, 1 if T>V, 0 if equal). Useful for lists of objects. Defaults to None.
        - key (Callable, optional): A function to return the numeric value used for sorting. Useful for lists of objects when you don't need a fancy comparator. Defaults to None.
        - err_if_not_equal (bool): If the value to be returned is not equal to the search value, then error out. Defaults to false.

    Returns:
        - int: index of the list for the closest matching
        - _T: the closest matching item
    """
    left, middle, right = 0, 0, len(sorted_list) - 1
    if key == None:
        key = lambda t: t
    if comparator == None:
        comparator = lambda t, v: -1 if key(t) < v else (1 if key(t) > v else 0)

    # base case
    if len(sorted_list) == 0:
        if err_if_not_equal:
            lt.error_and_raise(
                RuntimeError, f"Error: in list_tools.binary_search: empty list, can't find search value {search_val}"
            )
        return -1, None

    # binary search
    while right > left + 1:
        # find the middle index
        middle = int((right - left) / 2)
        middle = max(middle, left + 1)

        # determine which way to split
        if comparator(sorted_list[middle], search_val) < 0:
            left = middle
        else:
            right = middle

    ret = None, None

    # if there's only one value left, return it
    if right == left:
        ret = right, sorted_list[middle]

    # if split between two values, choose the closer one
    else:
        if isinstance(sorted_list[left], (int, float, complex)) and not isinstance(sorted_list[left], bool):
            # list instances are numbers, so we can evaluate which is closer
            lv = sorted_list[left]
            rv = sorted_list[right]
            if (rv - search_val) < (search_val - lv):
                ret = right, rv
            else:
                ret = left, lv
        else:
            # list instances aren't numbers, either we (a) have an exact match, (b) search_val is outside this range, or (c) return the left value
            if comparator(sorted_list[left], search_val) >= 0:
                ret = left, sorted_list[left]
            elif comparator(sorted_list[right], search_val) <= 0:
                ret = right, sorted_list[right]
            else:
                ret = left, sorted_list[left]

    # check for equality
    if err_if_not_equal:
        if ret[1] != search_val:
            lt.error_and_raise(
                RuntimeError,
                f"Error: in list_tools.binary_search: Found value {ret[1]} at index {ret[0]} != search value {search_val}",
            )

    return ret


def get_range(
    data_keys: list[float], data_values: list[_T], key_subset_range: tuple[float, float], exclude_outside_range=False
) -> tuple[list[float], list[_T]]:
    """Select a subset of the data_keys[] and data_values[], limited to the given key_subset_range.
    Chooses the keys closest to the given range start and end points, inclusively.

    Ideally, the input data_keys and data_values lists should be the same length.
    The returned data_keys and data_values lists should be the same length.

    Parameters:
    -----------
        - data_keys (list[float]): The list of keys for the corresponding data_values points in \"data_values\". This list must be sorted.
        - data (list): The data_values to get a subset of.
        - key_subset_range (tuple[float,float]): The start and end range to include in the returned subset.
        - exclude_outside_range (bool, optional): If False, then values outside the range could be included based on the closest key match.
                                                  If True, then only values in the range will be returned. Default False.

    Returns:
    --------
        - data_keys_subset (list[float]): A subset of the input data_keys, approximately limited to the given key_subset_range
        - data_values_subset (list[float]): A subset of the input data_values, approximately limited to the given key_subset_range
    """
    # validate input
    if len(data_keys) == 0:
        return data_keys, data_values

    # search for the range
    start_idx, _ = binary_search(data_keys, key_subset_range[0])
    stop_idx, _ = binary_search(data_keys, key_subset_range[1])
    if stop_idx < len(data_keys):
        stop_idx += 1

    # inclusive only?
    if exclude_outside_range:
        while (start_idx < len(data_keys)) and (data_keys[start_idx] < key_subset_range[0]):
            start_idx += 1
        while (stop_idx > 0) and (data_keys[stop_idx - 1] > key_subset_range[1]):
            stop_idx -= 1

    return data_keys[start_idx:stop_idx], data_values[start_idx:stop_idx]


def rindex(values: list, needle):
    """Like values.index(needle), but search for the last occurance."""
    for i in range(len(values) - 1, -1, -1):
        if needle == values[i]:
            return i
    return -1


def natural_sort(values: list[str]):
    """Sorts the given list naturally, so that numbers are sorted from lowest to highest.

    Adapted from https://stackoverflow.com/questions/11150239/natural-sorting"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(values, key=alphanum_key)


# PRINT


def print_list(
    input_list,  # List to print.
    max_items=10,  # Maximum number of elements to print.  Elipsis after that.
    max_item_length=70,  # Maximum item length to print.  Elipsis after that.
    indent=None,
):  # Number of blanks to print at the beginning of each line.
    """
    Prints a list, limiting print-out length both laterally and vertically.
    """
    # Check input.
    if not isinstance(input_list, list):
        print("ERROR: In print_list(), Non-list input_list:", input_list)
        assert False
    # Content.
    for item in input_list[0:max_items]:
        # Convert to string, and limit its length.
        item_str = str(item)
        trimmed_item_str = item_str[0:max_item_length]
        if len(item_str) > len(trimmed_item_str):
            trimmed_item_str += "..."
        # Print item.
        if indent == None:
            indent_str = ""
        else:
            indent_str = " " * indent
        print(indent_str + trimmed_item_str)
    # Postamble.
    if max_items < len(input_list):
        print(indent_str + "...")
