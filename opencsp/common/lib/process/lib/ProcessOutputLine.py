from dataclasses import dataclass

@dataclass
class ProcessOutputLine():
    """ Represents a single line of stdout or stderr output. """
    val: str
    """ The value of the line """
    is_err: bool = False
    """ Whether this line originated from stderr. """
    lineno: int = 0
    """ The index of this line from the mixed stderr/stdout of a process. """