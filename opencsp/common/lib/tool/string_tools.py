"""
Files for string manipulation.
"""


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
        return base[0:-1] + add + "."
