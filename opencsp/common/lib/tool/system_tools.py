import os
import re
import socket

from opencsp import opencsp_settings


def is_solo():
    """Determines if this computer is one of the Solo HPC nodes.

    Returns:
        bool: True if running on Solo
    """
    hn = socket.gethostname()
    re_solo = re.compile(r"solo(-login)?[0-9]+")
    if re_solo.match(hn):
        return True
    return False


def is_cluster():
    """Determines if this computer is a HPC nodes.

    Returns:
        bool: True if running on a HPC cluster node
    """
    return opencsp_settings['system'].getboolean('is_cluster')


__is_production_run = is_cluster() or not __debug__


def is_production_run():
    """Determines if this code is being evaluated as "production".
    Defaults to False, or True if running on solo or not __debug__.

    An exmple use case for this code is in verifying that a "verbose"
    flag isn't accidentally left on when we go to ship our code."""
    return __is_production_run


def set_is_production_run(is_production_run: bool):
    __is_production_run = is_production_run


def mem_status():
    """Get the memory status of this computer.  Also consider using the psutil package.

    Returns:
        tuple[float,float,float]: Total system memory, used system memory, available system memory (in GB)
    """
    # TODO should we just use the psutil package?
    if os.name == "nt" or os.name == "posix":
        import psutil

        ret = psutil.virtual_memory()

        total = ret.total
        avail = ret.available
        percent = ret.percent
        used = ret.used
        free = ret.free

        return total / 10e8, used / 10e8, (avail) / 10e8
    else:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        return total_memory / 1000, used_memory / 1000, free_memory / 1000
