import subprocess


class CalledProcessError(subprocess.CalledProcessError):
    """
    Exception raised when a subprocess call fails.

    This class extends the standard subprocess.CalledProcessError to include
    additional information about the standard error output captured during the
    subprocess execution.

    Parameters
    ----------
    returncode : int
        The exit status of the process.
    cmd : str or list
        The command that was executed.
    output : bytes, optional
        The standard output captured from the process. Defaults to None.
    stderr : bytes, optional
        The standard error output captured from the process. Defaults to None.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __str__(self):
        ret = super().__str__()
        if self.stderr != None:
            ret += f' Captured stderr:\n"{self.stderr}"'
        return ret
