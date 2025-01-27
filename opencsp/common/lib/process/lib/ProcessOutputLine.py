from dataclasses import dataclass


@dataclass
class ProcessOutputLine:
    """
    Represents a single line of stdout or stderr output.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    val: str
    """ The value of the line of output """
    is_err: bool = False
    """ Whether this line originated from stderr. (Default False) """
    lineno: int = 0
    """ The index of this line from the mixed stderr/stdout of a process. (Default 0)"""
