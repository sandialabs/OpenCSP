"""
Files for string manipulation.
"""

import re

import opencsp.common.lib.tool.log_tools as lt


def add_to_last_sentence(base: str, add: str) -> str:
    """
    Adds "add" string to "base" string, inserting before final period if present.

    Example 1:
        base:  "My initial caption."
        add:   ", plus more"
        -->    "My initial caption, plus more."

    Example 2:
        base:  "Characters ABC"
        add:   "DEF"
        -->    "Characters ABCDEF"
    """
    if len(base) == 0:
        return add
    elif base[-1] != ".":
        return base + add
    else:
        return base[0:-1] + add + '.'


def camel_case_split(to_split: str) -> list[str]:
    """
    Splits the given string into pieces at leading uppercase letters.

    For example:

        camel_case_split("TheABCsOfPython")
        # ['The', 'ABCs', 'Of', 'Python']

    Parameters
    ----------
    to_split : str
        The CamelCase string to be split

    Returns
    -------
    list[str]
        The to_split string, split into camel case sections.
    """
    return re.findall(r'([a-z]+|[A-Z]+[^A-Z]+)', to_split)


def convert_true_false_string_to_boolean(true_or_false_str: str) -> bool:
    """
    Accepts string with value either "True" or "False" and returns corresponding Boolean value.
    Throws an error if not one of these two values.

    Parameters
    ----------
    true_or_false_str : str
        The string to convert to Boolean.  Must be "True" or "False".

    Returns
    -------
    bool
        The corresponding value as a Boolean type.
    """
    if true_or_false_str == "True":
        return True
    elif true_or_false_str == "False":
        return False
    else:
        lt.error_and_raise(ValueError, f'True/False string {true_or_false_str} is not "True" or "False"')


def float_or_none(num_or_none_str: str) -> bool:
    """
    Accepts string with value either "None" or a valid floating-point number (e.g., "-1.234")
    and returns either the corresponding number, or None.
    Throws an error if not one of these two cases.

    Parameters
    ----------
    num_or_none_str : str
        The string to convert to floating point.  Must be either a valid number, or "None".

    Returns
    -------
    float | None
        The corresponding value as a float type, or None.
    """
    if num_or_none_str == "None":
        return None
    else:
        try:
            return float(num_or_none_str)
        except ValueError:
            lt.error_and_raise(
                ValueError, f'Number/None string {num_or_none_str} is not either a valid number or "None"'
            )


def verify_contiguous(token_str: str) -> str:
    """
    Checks string to ensure that is a contiguous string with no white space, newlines, etc.
    Throws an error if any white space is found.

    Parameters
    ----------
    token_str : str
        The string to check.

    Returns
    -------
    str
        The input string, assuming it passes the check.
        (Throws an error if not.)
    """
    if token_str != token_str.strip():
        lt.error_and_raise(ValueError, f'Token string {token_str} contains leading and/or trailing whitespace.')
    if len(token_str.split()) != 1:
        lt.error_and_raise(ValueError, f'Token string {token_str} contains internal whitespace.')
    return token_str
