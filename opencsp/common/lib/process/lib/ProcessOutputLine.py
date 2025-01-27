from dataclasses import dataclass


@dataclass
class ProcessOutputLine:
    """
    Represents a single line of stdout or stderr output.

    Attributes
    ----------
    val : str
        The value of the line of output.
    is_err : bool, optional
        Whether this line originated from stderr. Defaults to False.
    lineno : int, optional
        The index of this line from the mixed stderr/stdout of a process. Defaults to 0.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    val: str
    """ The value of the line """
    is_err: bool = False
    """ Whether this line originated from stderr. """
    lineno: int = 0
    """ The index of this line from the mixed stderr/stdout of a process. """
