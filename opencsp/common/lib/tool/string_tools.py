"""
Files for string manipulation.
"""

import re


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

    For example::

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
